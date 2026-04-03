#!/usr/bin/env python3
"""
Test object-centric exhaustive dataset sampling:
- Object pair discovery from filesystem
- Exhaustive coverage (every object seen exactly once)
- Deterministic object selection via LifecycleSampler(object_id=X)
- Backward compatibility (object_centric=False preserves old behavior)
"""

import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image as PILImage

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_multi_video_dataset(tmp_dir, video_configs):
    """
    Create a fake dataset with multiple videos, each with multiple objects.

    Args:
        video_configs: list of dicts, each with:
            - name: video directory name
            - num_frames: number of frames
            - objects: list of dicts with:
                - id: object folder name (int, 0-indexed on disk)
                - prompt_frames: list of frame indices with prompts
                - mask_frames: list of frame indices with GT masks
    """
    img_folder = os.path.join(tmp_dir, "img_folder")
    gt_folder = os.path.join(tmp_dir, "per_object_data")

    H, W = 64, 64

    for vc in video_configs:
        video_name = vc["name"]
        img_dir = os.path.join(img_folder, video_name)
        gt_dir = os.path.join(gt_folder, video_name)
        os.makedirs(img_dir, exist_ok=True)

        # Create frame images
        for f in range(vc["num_frames"]):
            img = PILImage.fromarray(
                np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
            )
            img.save(os.path.join(img_dir, f"{f:05d}.jpg"))

        # Create per-object data
        for obj in vc["objects"]:
            obj_dir = os.path.join(gt_dir, str(obj["id"]))
            os.makedirs(obj_dir, exist_ok=True)

            for f in obj["prompt_frames"]:
                prompt = np.zeros((H, W), dtype=np.uint8)
                prompt[10:30, 10:30] = 255
                PILImage.fromarray(prompt).save(
                    os.path.join(obj_dir, f"{f:05d}_prompt.png")
                )

                mask = np.zeros((H, W), dtype=np.uint8)
                if f in obj.get("mask_frames", []):
                    mask[15:25, 15:25] = 255
                PILImage.fromarray(mask).save(
                    os.path.join(obj_dir, f"{f:05d}_mask.png")
                )

    return img_folder, gt_folder


def test_object_pair_discovery():
    """Test that _discover_object_pairs finds all (video, object) pairs."""
    print("\n" + "=" * 60)
    print("TEST 1: Object pair discovery")
    print("=" * 60)

    from training.dataset.vos_dataset import VOSDataset
    from training.dataset.vos_raw_dataset import PNGRawDataset
    from training.dataset.vos_sampler import LifecycleSampler
    from training.dataset.transforms import ComposeAPI, ToTensorAPI

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_configs = [
            {
                "name": "vid_A",
                "num_frames": 10,
                "objects": [
                    {"id": 0, "prompt_frames": [0, 1, 2, 3], "mask_frames": [1, 2]},
                    {"id": 1, "prompt_frames": [0, 1, 2], "mask_frames": []},
                    {"id": 2, "prompt_frames": [2, 3, 4], "mask_frames": [3]},
                ],
            },
            {
                "name": "vid_B",
                "num_frames": 8,
                "objects": [
                    {"id": 0, "prompt_frames": [0, 1], "mask_frames": [0]},
                ],
            },
            {
                "name": "vid_C",
                "num_frames": 6,
                "objects": [
                    {"id": 0, "prompt_frames": [0, 1, 2], "mask_frames": [1]},
                    {"id": 1, "prompt_frames": [1, 2, 3], "mask_frames": [2]},
                ],
            },
        ]
        img_folder, gt_folder = create_multi_video_dataset(tmp_dir, video_configs)

        raw_dataset = PNGRawDataset(
            img_folder=img_folder,
            gt_folder=gt_folder,
            is_palette=False,
            single_object_mode=False,
        )
        sampler = LifecycleSampler(num_frames=5)
        transforms = [ComposeAPI([ToTensorAPI()])]

        dataset = VOSDataset(
            transforms=transforms,
            training=True,
            video_dataset=raw_dataset,
            sampler=sampler,
            multiplier=1,
            object_centric=True,
        )

        # Expected pairs: vid_A has 3 objects, vid_B has 1, vid_C has 2 = 6 total
        expected_count = 3 + 1 + 2
        print(f"  Discovered {len(dataset._object_pairs)} pairs (expected {expected_count})")
        print(f"  Pairs: {dataset._object_pairs}")

        assert len(dataset._object_pairs) == expected_count, (
            f"Expected {expected_count} pairs, got {len(dataset._object_pairs)}"
        )
        assert len(dataset) == expected_count, (
            f"__len__ should be {expected_count}, got {len(dataset)}"
        )

        # Verify object IDs use +1 convention (folder "0" → obj_id 1)
        obj_ids = [obj_id for _, obj_id in dataset._object_pairs]
        assert all(oid >= 1 for oid in obj_ids), f"All obj_ids should be >= 1, got {obj_ids}"

        print("\n  PASSED: All (video, object) pairs discovered correctly")


def test_exhaustive_coverage():
    """Test that iterating through all indices covers every object exactly once."""
    print("\n" + "=" * 60)
    print("TEST 2: Exhaustive coverage")
    print("=" * 60)

    from training.dataset.vos_dataset import VOSDataset
    from training.dataset.vos_raw_dataset import PNGRawDataset
    from training.dataset.vos_sampler import LifecycleSampler
    from training.dataset.transforms import ComposeAPI, ToTensorAPI

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_configs = [
            {
                "name": "vid_A",
                "num_frames": 10,
                "objects": [
                    {"id": 0, "prompt_frames": list(range(8)), "mask_frames": [2, 3]},
                    {"id": 1, "prompt_frames": list(range(10)), "mask_frames": []},
                ],
            },
            {
                "name": "vid_B",
                "num_frames": 8,
                "objects": [
                    {"id": 0, "prompt_frames": list(range(6)), "mask_frames": [1, 2]},
                    {"id": 1, "prompt_frames": list(range(5)), "mask_frames": [3]},
                    {"id": 2, "prompt_frames": list(range(7)), "mask_frames": []},
                ],
            },
        ]
        img_folder, gt_folder = create_multi_video_dataset(tmp_dir, video_configs)

        raw_dataset = PNGRawDataset(
            img_folder=img_folder,
            gt_folder=gt_folder,
            is_palette=False,
            single_object_mode=False,
        )
        sampler = LifecycleSampler(num_frames=5)
        transforms = [ComposeAPI([ToTensorAPI()])]

        dataset = VOSDataset(
            transforms=transforms,
            training=True,
            video_dataset=raw_dataset,
            sampler=sampler,
            multiplier=1,
            object_centric=True,
        )

        # Iterate through all indices and collect (video_id, object_id) pairs
        seen_pairs = set()
        for idx in range(len(dataset)):
            datapoint = dataset[idx]
            video_id = datapoint.video_id
            # Each datapoint should have exactly 1 object per frame
            obj_id = datapoint.frames[0].objects[0].object_id
            seen_pairs.add((video_id, obj_id))
            print(f"  idx={idx}: video_id={video_id}, obj_id={obj_id}")

        expected_pairs = set(dataset._object_pairs)
        # Convert expected pairs to (video_id, obj_id) - video_id is the index
        expected_set = set()
        for vidx, oid in expected_pairs:
            expected_set.add((vidx, oid))

        assert seen_pairs == expected_set, (
            f"Coverage mismatch!\n  Seen: {seen_pairs}\n  Expected: {expected_set}"
        )
        print(f"\n  Covered all {len(seen_pairs)} (video, object) pairs exactly once")
        print("\n  PASSED: Exhaustive coverage verified")


def test_deterministic_object_selection():
    """Test that LifecycleSampler.sample(object_id=X) always returns object X."""
    print("\n" + "=" * 60)
    print("TEST 3: Deterministic object selection")
    print("=" * 60)

    from training.dataset.vos_sampler import LifecycleSampler
    from training.dataset.vos_segment_loader import MultiplePNGSegmentLoader
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class FakeFrame:
        frame_idx: int
        image_path: str
        data: Optional[torch.Tensor] = None
        is_conditioning_only: Optional[bool] = False

    @dataclass
    class FakeVideo:
        video_name: str
        video_id: int
        frames: list

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_configs = [
            {
                "name": "video_0",
                "num_frames": 10,
                "objects": [
                    {"id": 0, "prompt_frames": list(range(8)), "mask_frames": [2, 3]},
                    {"id": 1, "prompt_frames": list(range(10)), "mask_frames": []},
                    {"id": 2, "prompt_frames": list(range(5)), "mask_frames": [1, 2]},
                ],
            },
        ]
        img_folder, gt_folder = create_multi_video_dataset(tmp_dir, video_configs)
        img_dir = os.path.join(img_folder, "video_0")
        gt_dir = os.path.join(gt_folder, "video_0")

        frames = [FakeFrame(f, os.path.join(img_dir, f"{f:05d}.jpg")) for f in range(10)]
        video = FakeVideo("video_0", 0, frames)
        segment_loader = MultiplePNGSegmentLoader(gt_dir)

        sampler = LifecycleSampler(num_frames=5)

        # Request specific objects
        for target_obj_id in [1, 2, 3]:  # folder 0→1, folder 1→2, folder 2→3
            result = sampler.sample(video, segment_loader, object_id=target_obj_id)
            assert result.object_ids == [target_obj_id], (
                f"Expected [{target_obj_id}], got {result.object_ids}"
            )
            print(f"  object_id={target_obj_id}: got {result.object_ids}, "
                  f"frames={[f.frame_idx for f in result.frames]}")

        # Test invalid object_id raises
        try:
            sampler.sample(video, segment_loader, object_id=99)
            assert False, "Should have raised exception for invalid object_id"
        except Exception as e:
            print(f"  object_id=99: correctly raised: {e}")

        print("\n  PASSED: Deterministic object selection works correctly")


def test_backward_compatibility():
    """Test that object_centric=False (default) preserves old behavior."""
    print("\n" + "=" * 60)
    print("TEST 4: Backward compatibility (object_centric=False)")
    print("=" * 60)

    from training.dataset.vos_dataset import VOSDataset
    from training.dataset.vos_raw_dataset import PNGRawDataset
    from training.dataset.vos_sampler import LifecycleSampler
    from training.dataset.transforms import ComposeAPI, ToTensorAPI

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_configs = [
            {
                "name": "vid_A",
                "num_frames": 10,
                "objects": [
                    {"id": 0, "prompt_frames": list(range(8)), "mask_frames": [2, 3]},
                    {"id": 1, "prompt_frames": list(range(10)), "mask_frames": []},
                ],
            },
            {
                "name": "vid_B",
                "num_frames": 8,
                "objects": [
                    {"id": 0, "prompt_frames": list(range(6)), "mask_frames": [1, 2]},
                ],
            },
        ]
        img_folder, gt_folder = create_multi_video_dataset(tmp_dir, video_configs)

        raw_dataset = PNGRawDataset(
            img_folder=img_folder,
            gt_folder=gt_folder,
            is_palette=False,
            single_object_mode=False,
        )

        # Default: object_centric=False
        sampler = LifecycleSampler(num_frames=5)
        transforms = [ComposeAPI([ToTensorAPI()])]

        dataset = VOSDataset(
            transforms=transforms,
            training=True,
            video_dataset=raw_dataset,
            sampler=sampler,
            multiplier=50,
        )

        # __len__ should be number of videos, not object pairs
        assert len(dataset) == len(raw_dataset), (
            f"Expected len={len(raw_dataset)}, got {len(dataset)}"
        )
        assert dataset._object_pairs is None, "Should not have _object_pairs"
        assert dataset.object_centric is False, "Should default to False"

        # repeat_factors should match video count
        assert len(dataset.repeat_factors) == len(raw_dataset)
        assert dataset.repeat_factors[0] == 50

        # Can still load a datapoint (random object selected)
        datapoint = dataset[0]
        assert datapoint is not None
        print(f"  len(dataset) = {len(dataset)} (== num_videos={len(raw_dataset)})")
        print(f"  repeat_factors shape = {dataset.repeat_factors.shape}")
        print(f"  object_centric = {dataset.object_centric}")
        print(f"  _object_pairs = {dataset._object_pairs}")

        print("\n  PASSED: Backward compatibility preserved")


def test_multiplier_with_object_centric():
    """Test that multiplier works with object-centric mode (repeat_factors length)."""
    print("\n" + "=" * 60)
    print("TEST 5: Multiplier with object-centric mode")
    print("=" * 60)

    from training.dataset.vos_dataset import VOSDataset
    from training.dataset.vos_raw_dataset import PNGRawDataset
    from training.dataset.vos_sampler import LifecycleSampler
    from training.dataset.transforms import ComposeAPI, ToTensorAPI

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_configs = [
            {
                "name": "vid_A",
                "num_frames": 10,
                "objects": [
                    {"id": 0, "prompt_frames": list(range(8)), "mask_frames": [2]},
                    {"id": 1, "prompt_frames": list(range(5)), "mask_frames": []},
                ],
            },
        ]
        img_folder, gt_folder = create_multi_video_dataset(tmp_dir, video_configs)

        raw_dataset = PNGRawDataset(
            img_folder=img_folder,
            gt_folder=gt_folder,
            is_palette=False,
            single_object_mode=False,
        )
        sampler = LifecycleSampler(num_frames=5)
        transforms = [ComposeAPI([ToTensorAPI()])]

        # multiplier=3 with object_centric: repeat_factors has length = num_pairs
        dataset = VOSDataset(
            transforms=transforms,
            training=True,
            video_dataset=raw_dataset,
            sampler=sampler,
            multiplier=3,
            object_centric=True,
        )

        num_pairs = len(dataset._object_pairs)
        assert len(dataset.repeat_factors) == num_pairs, (
            f"repeat_factors length should be {num_pairs}, got {len(dataset.repeat_factors)}"
        )
        assert all(rf == 3.0 for rf in dataset.repeat_factors), (
            f"All repeat_factors should be 3.0, got {dataset.repeat_factors}"
        )
        print(f"  num_pairs = {num_pairs}")
        print(f"  repeat_factors = {dataset.repeat_factors.tolist()}")

        print("\n  PASSED: Multiplier correctly applied to object pairs")


def test_post_evidence_frames():
    """Test that post_evidence_frames extends sampling beyond last evidence."""
    print("\n" + "=" * 60)
    print("TEST 6: post_evidence_frames extends sampling window")
    print("=" * 60)

    from training.dataset.vos_sampler import LifecycleSampler
    from training.dataset.vos_segment_loader import MultiplePNGSegmentLoader
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class FakeFrame:
        frame_idx: int
        image_path: str
        data: Optional[torch.Tensor] = None
        is_conditioning_only: Optional[bool] = False

    @dataclass
    class FakeVideo:
        video_name: str
        video_id: int
        frames: list

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a video with 20 frames, object has prompts on 0-4 and masks on 0-6
        # So last_evidence = 6 (frame index 00006)
        video_configs = [
            {
                "name": "video_0",
                "num_frames": 20,
                "objects": [
                    {
                        "id": 0,
                        "prompt_frames": list(range(5)),  # 0..4
                        "mask_frames": list(range(7)),     # 0..6
                    },
                ],
            },
        ]
        img_folder, gt_folder = create_multi_video_dataset(tmp_dir, video_configs)
        img_dir = os.path.join(img_folder, "video_0")
        gt_dir = os.path.join(gt_folder, "video_0")

        frames = [FakeFrame(f, os.path.join(img_dir, f"{f:05d}.jpg")) for f in range(20)]
        video = FakeVideo("video_0", 0, frames)
        segment_loader = MultiplePNGSegmentLoader(gt_dir)

        obj_id = 1  # folder "0" -> obj_id 1

        # Without post_evidence_frames: should stop at or shortly after last evidence
        sampler_no_post = LifecycleSampler(num_frames=20, frame_stride=1, post_evidence_frames=0)
        result_no_post = sampler_no_post.sample(video, segment_loader, object_id=obj_id)
        frames_no_post = [f.frame_idx for f in result_no_post.frames]
        # Last selected frame should be at last_evidence (6) + 1 frame past = 7
        # (the frame AT last_evidence counts as 1 past, then break since > 0)
        print(f"  post_evidence_frames=0: frames={frames_no_post}")

        # With post_evidence_frames=3: should extend 3 frames beyond last evidence
        sampler_with_post = LifecycleSampler(num_frames=20, frame_stride=1, post_evidence_frames=3)
        result_with_post = sampler_with_post.sample(video, segment_loader, object_id=obj_id)
        frames_with_post = [f.frame_idx for f in result_with_post.frames]
        print(f"  post_evidence_frames=3: frames={frames_with_post}")

        # The window WITH post_evidence should be strictly longer
        assert len(frames_with_post) > len(frames_no_post), (
            f"post_evidence_frames=3 should produce more frames than 0: "
            f"{len(frames_with_post)} vs {len(frames_no_post)}"
        )

        # The extra frames should be beyond last_evidence_frame (6)
        last_evidence = 6
        frames_beyond = [f for f in frames_with_post if f > last_evidence]
        assert len(frames_beyond) > 0, "Should have frames beyond last evidence"
        assert len(frames_beyond) <= 4, (  # 3 post + 1 at evidence
            f"Should have at most 4 frames at/beyond evidence, got {len(frames_beyond)}"
        )

        print(f"  Frames beyond last_evidence ({last_evidence}): {frames_beyond}")
        print("\n  PASSED: post_evidence_frames correctly extends window")


if __name__ == "__main__":
    test_object_pair_discovery()
    test_exhaustive_coverage()
    test_deterministic_object_selection()
    test_backward_compatibility()
    test_multiplier_with_object_centric()
    test_post_evidence_frames()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
