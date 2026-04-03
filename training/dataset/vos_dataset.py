# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
import random
from copy import deepcopy

import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.vos_raw_dataset import VOSRawDataset
from training.dataset.vos_sampler import VOSSampler
from training.dataset.vos_segment_loader import JSONSegmentLoader

from training.utils.data_utils import Frame, Object, VideoDatapoint

MAX_RETRIES = 100


class VOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
        target_segments_available=True,
        object_centric=False,
    ):
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler
        self.object_centric = object_centric

        if self.object_centric:
            self._object_pairs = self._discover_object_pairs()
            self.repeat_factors = torch.ones(len(self._object_pairs), dtype=torch.float32)
            self.repeat_factors *= multiplier
            logging.info(
                "Object-centric mode: %d (video, object) pairs from %d videos",
                len(self._object_pairs), len(self.video_dataset),
            )
        else:
            self._object_pairs = None
            self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
            self.repeat_factors *= multiplier
            logging.info("Raw dataset length = %d", len(self.video_dataset))

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target
        self.target_segments_available = target_segments_available

    def _discover_object_pairs(self):
        """
        Scan all videos' gt_folder to discover (video_idx, obj_id) pairs.
        Mirrors LifecycleSampler's filesystem scan logic.
        """
        pairs = []
        gt_folder = self.video_dataset.gt_folder
        for video_idx, video_name in enumerate(self.video_dataset.video_names):
            video_root = os.path.join(gt_folder, video_name)
            if not os.path.isdir(video_root):
                continue
            all_obj_dirs = sorted(
                d for d in glob.glob(os.path.join(video_root, "*"))
                if os.path.isdir(d)
            )
            for obj_id, obj_dir in enumerate(all_obj_dirs, start=1):
                if glob.glob(os.path.join(obj_dir, "*_prompt.png")):
                    pairs.append((video_idx, obj_id))
        return pairs

    def _get_datapoint(self, idx):
        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()

                # In object-centric mode, map idx to (video_idx, object_id)
                if self.object_centric:
                    video_idx, target_obj_id = self._object_pairs[idx]
                else:
                    video_idx = idx
                    target_obj_id = None

                # sample a video
                video, segment_loader = self.video_dataset.get_video(video_idx)
                # sample frames and object indices to be used in a datapoint
                sample_kwargs = dict(
                    video=video,
                    segment_loader=segment_loader,
                    epoch=self.curr_epoch,
                )
                if target_obj_id is not None:
                    sample_kwargs["object_id"] = target_obj_id
                sampled_frms_and_objs = self.sampler.sample(**sample_kwargs)
                break  # Succesfully loaded video
            except Exception as e:
                if self.training:
                    logging.warning(
                        f"Loading failed (id={idx}); Retry {retry} with exception: {e}"
                    )
                    if self.object_centric:
                        idx = random.randrange(0, len(self._object_pairs))
                    else:
                        idx = random.randrange(0, len(self.video_dataset))
                else:
                    # Shouldn't fail to load a val video
                    raise e

        datapoint = self.construct(
            video,
            sampled_frms_and_objs,
            segment_loader,
        )

        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)

        return datapoint

    def construct(self, video, sampled_frms_and_objs, segment_loader):
        """
        Constructs a VideoDatapoint sample to pass to transforms
        """
        sampled_frames = sampled_frms_and_objs.frames
        sampled_object_ids = sampled_frms_and_objs.object_ids

        images = []
        rgb_images = load_images(sampled_frames)

        # Iterate over the sampled frames and store their rgb data and object data (bbox, segment)
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size

            images.append(
                Frame(
                    data=rgb_images[frame_idx],
                    objects=[],
                )
            )

            # We load the gt segments associated with the current frame
            if isinstance(segment_loader, JSONSegmentLoader):
                segments = segment_loader.load(
                    frame.frame_idx,
                    obj_ids=sampled_object_ids,
                )
            else:
                # This is what is called with the Palettised PNG loader
                segments = segment_loader.load(frame.frame_idx)

            for obj_id in sampled_object_ids:
                # Extract the segment
                if obj_id in segments:
                    assert segments[obj_id] is not None, (
                        "None targets are not supported"
                    )
                    # segment [H, W, 2]: channel 0 = GT mask, channel 1 = prompt
                    # Prompt channel may be float (ternary {-1,0,+1}) or bool (binary)
                    segment = segments[obj_id]
                    # has_prompt = True if prompt channel has non-zero values.
                    # False for GT-only frames (mask exists but no prompt).
                    has_prompt = bool(segment[..., 1].any())
                else:
                    # No prompt AND no GT mask for this object on this frame.
                    # With always_target=True, pad with zeros; otherwise skip.
                    if not self.always_target:
                        continue

                    segment = torch.zeros(h, w, 2, dtype=torch.float32)
                    has_prompt = False

                images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                        has_prompt=has_prompt,
                    )
                )

        return VideoDatapoint(
            frames=images,
            video_id=video.video_id,
            size=(h, w),
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        if self.object_centric:
            return len(self._object_pairs)
        return len(self.video_dataset)


def load_images(frames):
    all_images = []
    cache = {}
    for frame in frames:
        if frame.data is None:
            # Load the frame rgb data from file
            path = frame.image_path
            if path in cache:
                all_images.append(deepcopy(all_images[cache[path]]))
                continue
            with g_pathmgr.open(path, "rb") as fopen:
                all_images.append(PILImage.open(fopen).convert("RGB"))
            cache[path] = len(all_images) - 1
        else:
            # The frame rgb data has already been loaded
            # Convert it to a PILImage
            all_images.append(tensor_2_PIL(frame.data))

    return all_images


def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)
