# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import random
from dataclasses import dataclass
from typing import List
from training.dataset.vos_segment_loader import LazySegments

MAX_RETRIES = 1000


@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids: List[int]


class VOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()


class RandomUniformSampler(VOSSampler):
    def __init__(
        self,
        num_frames,
        max_num_objects,
        reverse_time_prob=0.0,
        negative_proportion=0.0,
    ):
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob
        self.negative_proportion = negative_proportion

    def sample(self, video, segment_loader, epoch=None):
        for retry in range(MAX_RETRIES):
            if len(video.frames) < self.num_frames:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {len(video.frames)} annotated frames."
                )

            start = random.randrange(0, len(video.frames) - self.num_frames + 1)
            frames = [video.frames[start + step] for step in range(self.num_frames)]

            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = frames[::-1]

            # Get first frame object ids
            loaded_segms = segment_loader.load(frames[0].frame_idx)
            visible_object_ids = []

            if isinstance(loaded_segms, LazySegments):
                # LazySegments for SA1BRawDataset
                visible_object_ids = list(loaded_segms.keys())
            else:
                positve_object_ids = set()
                all_object_ids = set()

                # Detect all positive objects in the video
                # and all objects present in the video
                for frame in frames:
                    for object_id, segment in segment_loader.load(
                        frame.frame_idx
                    ).items():
                        if object_id in positve_object_ids:
                            continue

                        mask = segment[..., 0]

                        if mask.any():  # Object is visible in the frame (whichever frame in the video)
                            positve_object_ids.add(object_id)

                        all_object_ids.add(object_id)

                # Collect negative object ids
                negative_object_ids = all_object_ids - positve_object_ids

                # Collect object ids present in the first frame of the video
                obj_in_first_frame = set(list(loaded_segms.keys()))

                # Collect positive object ids in the first frame
                positive_object_ids_in_first_frame = list(
                    positve_object_ids & obj_in_first_frame
                )

                # Collect negative object ids in the first frame
                negative_object_ids_in_first_frame = list(
                    negative_object_ids & obj_in_first_frame
                )

                # How many negatives to sample based on the proportion
                negatives_to_sample = int(
                    len(positive_object_ids_in_first_frame) * self.negative_proportion
                )

                # how many remaining objects should be sampled
                negatives_to_sample = max(
                    self.max_num_objects - len(positive_object_ids_in_first_frame),
                    negatives_to_sample,
                )

                # Sample visible object ids in the first frame
                visible_object_ids = positive_object_ids_in_first_frame + random.sample(
                    negative_object_ids_in_first_frame,
                    min(
                        len(negative_object_ids_in_first_frame),
                        negatives_to_sample,
                    ),
                )

            # First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break

            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        object_ids = random.sample(
            visible_object_ids,
            min(
                len(visible_object_ids),
                self.max_num_objects,
            ),
        )
        return SampledFramesAndObjects(
            frames=frames,
            object_ids=object_ids,
        )


class DensePromptSampler(VOSSampler):
    """
    Sampler for dense prompt training (e.g., contrail detection).

    Key differences from RandomUniformSampler:
    - No first-frame constraint: objects from ANY frame in the window are eligible.
      This matters when objects (flights) enter/exit mid-video.
    - Prioritizes positive objects (those with non-empty GT in at least one frame),
      then fills remaining slots with negatives (flights without contrails).
    - Better suited for heavily negative datasets where most objects have prompts
      but empty ground truth (e.g., most flights don't produce contrails).

    Sampling logic:
    1. Sample a random window of num_frames consecutive frames
    2. Scan ALL frames in the window to find all objects with prompts
    3. Classify each object as positive (has GT mask somewhere) or negative (never)
    4. Include all positives (up to max_num_objects), fill rest with negatives
    """

    def __init__(
        self,
        num_frames,
        max_num_objects,
        reverse_time_prob=0.0,
    ):
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob

    def sample(self, video, segment_loader, epoch=None):
        for retry in range(MAX_RETRIES):
            if len(video.frames) < self.num_frames:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video "
                    f"{video.video_name} as it only has {len(video.frames)} "
                    f"annotated frames."
                )

            start = random.randrange(0, len(video.frames) - self.num_frames + 1)
            frames = [video.frames[start + step] for step in range(self.num_frames)]

            if random.uniform(0, 1) < self.reverse_time_prob:
                frames = frames[::-1]

            # Scan ALL frames in the window (no first-frame restriction)
            positive_ids = set()
            all_ids = set()

            for frame in frames:
                for obj_id, segment in segment_loader.load(
                    frame.frame_idx
                ).items():
                    all_ids.add(obj_id)
                    # Positive = has non-empty GT mask in at least one frame
                    if obj_id not in positive_ids and segment[..., 0].any():
                        positive_ids.add(obj_id)

            negative_ids = all_ids - positive_ids

            # Need at least some objects (positive or negative) in the window
            if len(all_ids) > 0:
                break

            if retry >= MAX_RETRIES - 1:
                raise Exception("No objects found in any sampled window")

        # Prioritize positives, fill remaining slots with negatives.
        # This naturally reflects the dataset distribution: if most flights
        # don't produce contrails, most sampled objects will be negative.
        positive_list = list(positive_ids)
        negative_list = list(negative_ids)
        random.shuffle(positive_list)
        random.shuffle(negative_list)

        # Include all positives (up to max_num_objects)
        sampled = positive_list[: self.max_num_objects]

        # Fill remaining slots with negatives
        remaining = self.max_num_objects - len(sampled)
        if remaining > 0:
            sampled += negative_list[:remaining]

        # Shuffle so the model doesn't learn positional bias
        random.shuffle(sampled)

        return SampledFramesAndObjects(
            frames=frames,
            object_ids=sampled,
        )


class LifecycleSampler(VOSSampler):
    """
    Sampler for single-object, full-lifecycle training.

    Samples ONE object at a time and tracks its full lifecycle:
    from the first prompt frame through num_frames consecutive frames.
    Frames beyond the object's last prompt are included without prompts,
    teaching the model to propagate from memory and eventually stop predicting.

    Args:
        num_frames: Number of frames to return per sample.
        frame_stride: Take every N-th frame (1=consecutive, 2=every other, etc.).
            Covers num_frames * frame_stride actual video frames while only
            returning num_frames to the model, saving memory.
        reverse_time_prob: Probability of reversing the frame order.

    Sampling logic:
    1. Discover all objects and their prompt frame ranges from the filesystem
    2. Pick a random object uniformly (no bias toward positive/negative)
    3. Start from the object's first prompt frame
    4. Take num_frames frames with frame_stride spacing
    """

    def __init__(
        self,
        num_frames=20,
        frame_stride=1,
        reverse_time_prob=0.0,
        post_evidence_frames=0,
    ):
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.reverse_time_prob = reverse_time_prob
        self.post_evidence_frames = post_evidence_frames

    def sample(self, video, segment_loader, epoch=None, object_id=None):
        # Discover per-object prompt ranges from the filesystem.
        # Use sorted enumeration (same as MultiplePNGSegmentLoader) so obj_ids are consistent.
        video_root = segment_loader.video_png_root
        all_obj_dirs = sorted(
            d for d in glob.glob(os.path.join(video_root, "*"))
            if os.path.isdir(d)
        )

        obj_prompt_info = {}  # {obj_id: (first_prompt, last_evidence)}
        for obj_id, obj_dir in enumerate(all_obj_dirs, start=1):

            prompt_files = sorted(glob.glob(os.path.join(obj_dir, "*_prompt.png")))
            if not prompt_files:
                continue

            first_prompt = int(os.path.basename(prompt_files[0]).split("_")[0])
            last_prompt = int(os.path.basename(prompt_files[-1]).split("_")[0])

            mask_files = sorted(glob.glob(os.path.join(obj_dir, "*_mask.png")))
            last_mask = int(os.path.basename(mask_files[-1]).split("_")[0]) if mask_files else last_prompt

            # The object's lifecycle ends at whichever comes last: prompt or mask
            last_evidence = max(last_prompt, last_mask)

            obj_prompt_info[obj_id] = (first_prompt, last_evidence)

        if not obj_prompt_info:
            raise Exception(f"No objects with prompts found in {video_root}")

        # Use specified object_id (object-centric mode) or random choice (default)
        if object_id is not None:
            obj_id = object_id
            if obj_id not in obj_prompt_info:
                raise Exception(
                    f"Object {obj_id} not found in {video_root}. "
                    f"Available objects: {list(obj_prompt_info.keys())}"
                )
        else:
            obj_id = random.choice(list(obj_prompt_info.keys()))

        first_prompt_frame, last_evidence_frame = obj_prompt_info[obj_id]

        # Build frame_idx -> position mapping
        frame_idx_to_pos = {f.frame_idx: i for i, f in enumerate(video.frames)}

        if first_prompt_frame not in frame_idx_to_pos:
            raise Exception(
                f"First prompt frame {first_prompt_frame} for object {obj_id} "
                f"not found in video frames"
            )

        start_pos = frame_idx_to_pos[first_prompt_frame]

        # Sample with stride from first prompt, stopping at last evidence frame.
        # The lifecycle window covers exactly [first_prompt, last_evidence].
        candidate_positions = list(range(start_pos, len(video.frames), self.frame_stride))

        # Keep positions up to last evidence frame + post_evidence_frames margin,
        # capped at num_frames for GPU memory safety.
        # The margin adds frames beyond all evidence so the model learns to
        # propagate from memory and stop predicting when the object is gone.
        selected_positions = []
        frames_past_evidence = 0
        for pos in candidate_positions:
            selected_positions.append(pos)
            if video.frames[pos].frame_idx >= last_evidence_frame:
                frames_past_evidence += 1
                if frames_past_evidence > self.post_evidence_frames:
                    break
            if len(selected_positions) >= self.num_frames:
                break

        frames = [video.frames[p] for p in selected_positions]

        if random.uniform(0, 1) < self.reverse_time_prob:
            frames = frames[::-1]

        return SampledFramesAndObjects(
            frames=frames,
            object_ids=[obj_id],
        )


class EvalSampler(VOSSampler):
    """
    VOS Sampler for evaluation: sampling all the frames and all the objects in a video
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(self, video, segment_loader, epoch=None):
        """
        Sampling all the frames and all the objects
        """
        if self.sort_frames:
            # ordered by frame id
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            # use the original order
            frames = video.frames

        object_ids = segment_loader.load(frames[0].frame_idx).keys()

        if len(object_ids) == 0:
            raise Exception("First frame of the video has no objects")

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
