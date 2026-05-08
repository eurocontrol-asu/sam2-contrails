# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import glob
import numpy as np
import torch

from PIL import Image as PILImage

try:
    from pycocotools import mask as mask_utils
except ImportError:
    pass


class JSONSegmentLoader:
    def __init__(self, video_json_path, ann_every=1, frames_fps=24, valid_obj_ids=None):
        # Annotations in the json are provided every ann_every th frame
        self.ann_every = ann_every
        # Ids of the objects to consider when sampling this video
        self.valid_obj_ids = valid_obj_ids
        with open(video_json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                self.frame_annots = data
            elif isinstance(data, dict):
                masklet_field_name = "masklet" if "masklet" in data else "masks"
                self.frame_annots = data[masklet_field_name]
                if "fps" in data:
                    if isinstance(data["fps"], list):
                        annotations_fps = int(data["fps"][0])
                    else:
                        annotations_fps = int(data["fps"])
                    assert frames_fps % annotations_fps == 0
                    self.ann_every = frames_fps // annotations_fps
            else:
                raise NotImplementedError

    def load(self, frame_id, obj_ids=None):
        assert frame_id % self.ann_every == 0
        rle_mask = self.frame_annots[frame_id // self.ann_every]

        valid_objs_ids = set(range(len(rle_mask)))
        if self.valid_obj_ids is not None:
            # Remove the masklets that have been filtered out for this video
            valid_objs_ids &= set(self.valid_obj_ids)
        if obj_ids is not None:
            # Only keep the objects that have been sampled
            valid_objs_ids &= set(obj_ids)
        valid_objs_ids = sorted(list(valid_objs_ids))

        # Construct rle_masks_filtered that only contains the rle masks we are interested in
        id_2_idx = {}
        rle_mask_filtered = []
        for obj_id in valid_objs_ids:
            if rle_mask[obj_id] is not None:
                id_2_idx[obj_id] = len(rle_mask_filtered)
                rle_mask_filtered.append(rle_mask[obj_id])
            else:
                id_2_idx[obj_id] = None

        # Decode the masks
        raw_segments = torch.from_numpy(mask_utils.decode(rle_mask_filtered)).permute(
            2, 0, 1
        )  # （num_obj, h, w）
        segments = {}
        for obj_id in valid_objs_ids:
            if id_2_idx[obj_id] is None:
                segments[obj_id] = None
            else:
                idx = id_2_idx[obj_id]
                segments[obj_id] = raw_segments[idx]
        return segments

    def get_valid_obj_frames_ids(self, num_frames_min=None):
        # For each object, find all the frames with a valid (not None) mask
        num_objects = len(self.frame_annots[0])

        # The result dict associates each obj_id with the id of its valid frames
        res = {obj_id: [] for obj_id in range(num_objects)}

        for annot_idx, annot in enumerate(self.frame_annots):
            for obj_id in range(num_objects):
                if annot[obj_id] is not None:
                    res[obj_id].append(int(annot_idx * self.ann_every))

        if num_frames_min is not None:
            # Remove masklets that have less than num_frames_min valid masks
            for obj_id, valid_frames in list(res.items()):
                if len(valid_frames) < num_frames_min:
                    res.pop(obj_id)

        return res


class LazySegments:
    """
    Only decodes segments that are actually used.
    """

    def __init__(self):
        self.segments = {}
        self.cache = {}

    def __setitem__(self, key, item):
        self.segments[key] = item

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        rle = self.segments[key]
        mask = torch.from_numpy(mask_utils.decode([rle])).permute(2, 0, 1)[0]
        self.cache[key] = mask
        return mask

    def __contains__(self, key):
        return key in self.segments

    def __len__(self):
        return len(self.segments)

    def keys(self):
        return self.segments.keys()


class SA1BSegmentLoader:
    def __init__(
        self,
        video_mask_path,
        mask_area_frac_thresh=1.1,
        video_frame_path=None,
        uncertain_iou=-1,
    ):
        with open(video_mask_path, "r") as f:
            self.frame_annots = json.load(f)

        if mask_area_frac_thresh <= 1.0:
            # Lazily read frame
            orig_w, orig_h = PILImage.open(video_frame_path).size
            area = orig_w * orig_h

        self.frame_annots = self.frame_annots["annotations"]

        rle_masks = []
        for frame_annot in self.frame_annots:
            if not frame_annot["area"] > 0:
                continue
            if ("uncertain_iou" in frame_annot) and (
                frame_annot["uncertain_iou"] < uncertain_iou
            ):
                # uncertain_iou is stability score
                continue
            if (
                mask_area_frac_thresh <= 1.0
                and (frame_annot["area"] / area) >= mask_area_frac_thresh
            ):
                continue
            rle_masks.append(frame_annot["segmentation"])

        self.segments = LazySegments()
        for i, rle in enumerate(rle_masks):
            self.segments[i] = rle

    def load(self, frame_idx):
        return self.segments


class MultiplePNGSegmentLoader:
    def __init__(
        self,
        video_png_root,
        single_object_mode=False,
        keep_empty_masks=True,
        use_ternary_prompts=False,
    ):
        """
        Args:
            video_png_root (str): The path to a specific video folder.
            keep_empty_masks (bool): If False, objects with no GT pixels in the
                                     current frame are excluded.
            use_ternary_prompts (bool): If True, load union PNG and compute
                ternary prompt values {-1, 0, +1}. Default: False (binary {0, 1}).
        """
        self.video_png_root = video_png_root
        self.single_object_mode = single_object_mode
        self.keep_empty_masks = keep_empty_masks
        self.use_ternary_prompts = use_ternary_prompts

        self.H, self.W = 1024, 1024

        # Build a stable folder-name → obj_id mapping once at init time.
        # Sorting lexicographically over all subfolders (any name, e.g. flight_id hex
        # strings) and assigning 1-based integer IDs gives a deterministic mapping that
        # is consistent across the segment loader, dataset discovery, and sampler.
        obj_dirs = sorted(
            d for d in glob.glob(os.path.join(video_png_root, "*"))
            if os.path.isdir(d)
        )
        self._folder_to_obj_id = {
            os.path.basename(d): idx + 1
            for idx, d in enumerate(obj_dirs)
        }

    def load(self, frame_id):
        binary_segments = {}

        # Optionally load the all-prompts union mask (for ternary prompt construction).
        union_np = None
        if self.use_ternary_prompts:
            union_path = os.path.join(
                self.video_png_root, f"{frame_id:05d}_all_prompts_union.png"
            )
            if os.path.exists(union_path):
                union_np = np.array(PILImage.open(union_path)).astype(np.float32) / 255.0

        for folder_name, obj_id in self._folder_to_obj_id.items():
            obj_folder = os.path.join(self.video_png_root, folder_name)

            prompt_path = os.path.join(obj_folder, f"{frame_id:05d}_prompt.png")
            mask_path = os.path.join(obj_folder, f"{frame_id:05d}_mask.png")

            prompt_exists = os.path.exists(prompt_path)
            mask_exists = os.path.exists(mask_path)

            # Skip if neither prompt nor mask exists for this frame
            if not prompt_exists and not mask_exists:
                if self.use_ternary_prompts and union_np is not None:
                    h, w = union_np.shape[:2]
                    binary_segments[obj_id] = torch.stack(
                        [
                            torch.from_numpy(np.zeros((h, w), dtype=bool)),
                            torch.from_numpy(-union_np),
                        ],
                        dim=-1,
                    )
                continue

            # Load whichever files exist to determine spatial size
            mask_np = np.array(PILImage.open(mask_path)) if mask_exists else None
            prompt_np = np.array(PILImage.open(prompt_path)).astype(np.float32) / 255.0 if prompt_exists else None

            # Determine H, W from whichever file was loaded
            if mask_np is not None:
                h, w = mask_np.shape[:2]
            else:
                h, w = prompt_np.shape[:2]

            # Fill in missing arrays with correctly sized zeros
            if mask_np is None:
                mask_np = np.zeros((h, w), dtype=bool)

            # Skip if no GT, no prompt, and keep_empty_masks is False
            is_empty = not mask_exists or not np.any(mask_np > 0)
            if not self.keep_empty_masks and is_empty and not prompt_exists:
                continue

            # Build prompt channel
            if prompt_np is not None:
                if union_np is not None:
                    # Ternary: own prompt takes priority (positive, age-weighted).
                    # Where own prompt is absent but union > 0, apply negative age-weighted signal.
                    prompt_channel = np.where(prompt_np > 0, prompt_np, -union_np)
                else:
                    prompt_channel = prompt_np
            else:
                if self.use_ternary_prompts and union_np is not None:
                    prompt_channel = -union_np
                else:
                    prompt_channel = np.zeros((h, w), dtype=np.float32)

            binary_segments[obj_id] = torch.stack(
                [
                    torch.from_numpy(mask_np > 0),
                    torch.from_numpy(prompt_channel),
                ],
                dim=-1,
            )

        return binary_segments


class PalettisedPNGSegmentLoader:
    def __init__(
        self,
        video_png_root,
        keep_empty_masks=True,
    ):
        """
        Args:
            video_png_root (str): Path to folder with stacked PNGs.
            keep_empty_masks (bool): If False, objects with no GT pixels are skipped.
        """
        self.video_png_root = video_png_root
        self.keep_empty_masks = keep_empty_masks

        all_files = os.listdir(self.video_png_root)
        self.frame_ids = sorted(
            [
                int(f.split("_")[0])
                for f in all_files
                if f.endswith("_mask.png") and f.split("_")[0].isdigit()
            ]
        )

    def load(self, frame_id):
        mask_path = os.path.join(self.video_png_root, f"{frame_id:05d}_mask.png")
        prompt_path = os.path.join(self.video_png_root, f"{frame_id:05d}_prompt.png")

        masks = np.array(PILImage.open(mask_path))
        prompts = np.array(PILImage.open(prompt_path))

        # Detect unique IDs
        unique_ids = np.unique(prompts)
        unique_ids = unique_ids[unique_ids != 0]

        binary_segments = {}

        for obj_id in unique_ids:
            # Check if mask is empty for this specific object
            has_mask = np.any(masks == obj_id)

            if not self.keep_empty_masks and not has_mask:
                continue

            tensor_gt = torch.from_numpy(masks == obj_id)
            tensor_prompt = torch.from_numpy(prompts == obj_id)

            binary_segments[int(obj_id)] = torch.stack(
                [
                    tensor_gt,
                    tensor_prompt,
                ],
                dim=-1,
            )

        return binary_segments
