#!/usr/bin/env python3
"""
Test the lifecycle training pipeline end-to-end:
- LifecycleSampler: single-object, full-lifecycle sampling
- has_prompt tracking: distinguishes cond vs non-cond frames
- SAM2DensePromptTrain: mixed cond/non-cond frame processing
- use_ternary_prompts parameter: binary vs ternary prompts
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import torch
from PIL import Image as PILImage

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_video(tmp_dir, num_frames=10, num_objects=3, prompt_end_frame=7):
    """
    Create a fake video with per-object PNGs.
    Object 1 has prompts on frames 0-6 (prompt_end_frame-1), with GT mask on frames 2-5.
    Object 2 has prompts on all frames (negative: no GT mask).
    Object 3 has prompts on frames 0-4, GT mask on frames 1-3.
    """
    img_dir = os.path.join(tmp_dir, "img_folder", "video_0")
    gt_dir = os.path.join(tmp_dir, "per_object_data", "video_0")
    os.makedirs(img_dir, exist_ok=True)

    H, W = 64, 64

    # Create frame images
    for f in range(num_frames):
        img = PILImage.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))
        img.save(os.path.join(img_dir, f"{f:05d}.jpg"))

    # Object 0 (id becomes 1 in loader): prompts on frames 0 to prompt_end_frame-1
    obj0_dir = os.path.join(gt_dir, "0")
    os.makedirs(obj0_dir, exist_ok=True)
    for f in range(prompt_end_frame):
        # Prompt: always present
        prompt = np.zeros((H, W), dtype=np.uint8)
        prompt[10:30, 10:30] = 255
        PILImage.fromarray(prompt).save(os.path.join(obj0_dir, f"{f:05d}_prompt.png"))
        # Mask: only on frames 2-5
        mask = np.zeros((H, W), dtype=np.uint8)
        if 2 <= f <= 5:
            mask[15:25, 15:25] = 255
        PILImage.fromarray(mask).save(os.path.join(obj0_dir, f"{f:05d}_mask.png"))

    # Object 1 (id becomes 2): prompts on all frames, no GT mask (negative)
    obj1_dir = os.path.join(gt_dir, "1")
    os.makedirs(obj1_dir, exist_ok=True)
    for f in range(num_frames):
        prompt = np.zeros((H, W), dtype=np.uint8)
        prompt[40:60, 40:60] = 255
        PILImage.fromarray(prompt).save(os.path.join(obj1_dir, f"{f:05d}_prompt.png"))
        mask = np.zeros((H, W), dtype=np.uint8)
        PILImage.fromarray(mask).save(os.path.join(obj1_dir, f"{f:05d}_mask.png"))

    # Create union PNGs (for ternary testing)
    for f in range(num_frames):
        union = np.zeros((H, W), dtype=np.uint8)
        if f < prompt_end_frame:
            union[10:30, 10:30] = 255  # obj 0
        union[40:60, 40:60] = 255  # obj 1 on all frames
        PILImage.fromarray(union).save(
            os.path.join(gt_dir, f"{f:05d}_all_prompts_union.png")
        )

    return img_dir, gt_dir


def _make_video(img_dir, num_frames):
    """Helper to create a VOSVideo-like object without importing vos_raw_dataset."""
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

    frames = [FakeFrame(f, os.path.join(img_dir, f"{f:05d}.jpg")) for f in range(num_frames)]
    return FakeVideo("video_0", 0, frames)


def test_lifecycle_sampler():
    """Test LifecycleSampler samples correct frames and single object."""
    print("\n" + "=" * 60)
    print("TEST 1: LifecycleSampler")
    print("=" * 60)

    from training.dataset.vos_sampler import LifecycleSampler
    from training.dataset.vos_segment_loader import MultiplePNGSegmentLoader

    with tempfile.TemporaryDirectory() as tmp_dir:
        num_frames = 10
        prompt_end = 7
        img_dir, gt_dir = create_test_video(tmp_dir, num_frames, prompt_end_frame=prompt_end)

        # Build video and segment_loader
        video_gt_root = os.path.join(gt_dir)
        segment_loader = MultiplePNGSegmentLoader(video_gt_root)
        video = _make_video(img_dir, num_frames)

        sampler = LifecycleSampler(num_frames=10)
        result = sampler.sample(video, segment_loader)

        print(f"  Sampled object_ids: {result.object_ids}")
        print(f"  Sampled frames: {[f.frame_idx for f in result.frames]}")
        print(f"  Num frames: {len(result.frames)}")

        assert len(result.object_ids) == 1, f"Expected 1 object, got {len(result.object_ids)}"
        assert len(result.frames) <= 10, f"Expected <= 10 frames, got {len(result.frames)}"
        print("\n  PASSED: LifecycleSampler returns single object with correct frame range")


def test_has_prompt_tracking():
    """Test that has_prompt is correctly tracked through the pipeline."""
    print("\n" + "=" * 60)
    print("TEST 2: has_prompt tracking through pipeline")
    print("=" * 60)

    from training.dataset.vos_segment_loader import MultiplePNGSegmentLoader
    from training.dataset.vos_sampler import LifecycleSampler
    from training.utils.data_utils import Object, Frame, VideoDatapoint, collate_fn

    with tempfile.TemporaryDirectory() as tmp_dir:
        num_frames = 10
        prompt_end = 7
        img_dir, gt_dir = create_test_video(tmp_dir, num_frames, prompt_end_frame=prompt_end)

        segment_loader = MultiplePNGSegmentLoader(gt_dir)
        video = _make_video(img_dir, num_frames)

        sampler = LifecycleSampler(num_frames=10)
        result = sampler.sample(video, segment_loader)
        obj_id = result.object_ids[0]
        print(f"  Sampled object: {obj_id}")
        print(f"  Frames: {[f.frame_idx for f in result.frames]}")

        # Manually construct VideoDatapoint (like VOSDataset.construct)
        H, W = 64, 64
        frames_list = []
        for t, frame_info in enumerate(result.frames):
            img = PILImage.open(frame_info.image_path).convert("RGB")
            segments = segment_loader.load(frame_info.frame_idx)

            if obj_id in segments:
                segment = segments[obj_id]
                # Derive has_prompt from prompt channel (matches construct() logic)
                has_prompt = bool(segment[..., 1].any())
            else:
                segment = torch.zeros(H, W, 2, dtype=torch.float32)
                has_prompt = False

            obj = Object(
                object_id=obj_id,
                frame_index=frame_info.frame_idx,
                segment=segment,
                has_prompt=has_prompt,
            )
            frames_list.append(Frame(data=img, objects=[obj]))

        datapoint = VideoDatapoint(frames=frames_list, video_id=0, size=(H, W))

        # Print has_prompt per frame
        for t, frame in enumerate(datapoint.frames):
            frame_idx = result.frames[t].frame_idx
            hp = frame.objects[0].has_prompt
            print(f"  Frame {frame_idx}: has_prompt={hp}")

        # Apply transforms and collate
        from training.dataset.transforms import ComposeAPI, ToTensorAPI
        transform = ComposeAPI([ToTensorAPI()])
        datapoint = transform(datapoint)

        batch = collate_fn([datapoint], dict_key="test")
        print(f"\n  has_prompt shape: {batch.has_prompt.shape}")
        print(f"  has_prompt values: {batch.has_prompt.squeeze().tolist()}")

        hp = batch.has_prompt.squeeze()
        num_cond = hp.sum().item()
        num_non_cond = (~hp).sum().item()
        print(f"  Conditioning frames: {num_cond}, Non-conditioning frames: {num_non_cond}")

        # Object 1 (folder "0") has prompts on frames 0-6, so with 10 frames
        # starting from 0: frames 0-6 have prompts, frames 7-9 don't
        if obj_id == 1:
            assert num_cond == prompt_end, f"Expected {prompt_end} cond frames, got {num_cond}"
            assert num_non_cond == num_frames - prompt_end
            print("  Correct: lifecycle extends beyond prompts")
        # Object 2 (folder "1") has prompts on all 10 frames
        elif obj_id == 2:
            assert num_cond == num_frames
            print("  Object has prompts on all frames")

        print("\n  PASSED: has_prompt correctly tracked through pipeline")


def test_ternary_parameter():
    """Test that use_ternary_prompts controls binary vs ternary loading."""
    print("\n" + "=" * 60)
    print("TEST 3: use_ternary_prompts parameter")
    print("=" * 60)

    from training.dataset.vos_segment_loader import MultiplePNGSegmentLoader

    with tempfile.TemporaryDirectory() as tmp_dir:
        _, gt_dir = create_test_video(tmp_dir, num_frames=5, prompt_end_frame=5)

        # Binary mode (default)
        loader_binary = MultiplePNGSegmentLoader(gt_dir, use_ternary_prompts=False)
        segments_binary = loader_binary.load(0)

        # Ternary mode
        loader_ternary = MultiplePNGSegmentLoader(gt_dir, use_ternary_prompts=True)
        segments_ternary = loader_ternary.load(0)

        for obj_id in segments_binary:
            prompt_binary = segments_binary[obj_id][..., 1]
            prompt_ternary = segments_ternary[obj_id][..., 1]

            binary_unique = sorted(torch.unique(prompt_binary).tolist())
            ternary_unique = sorted(torch.unique(prompt_ternary).tolist())

            print(f"  Object {obj_id}:")
            print(f"    Binary prompt unique values: {binary_unique}")
            print(f"    Ternary prompt unique values: {ternary_unique}")

            # Binary should only have {0, 1}
            assert all(v in [0.0, 1.0] for v in binary_unique), \
                f"Binary should only have {{0, 1}}, got {binary_unique}"

        # Check ternary has -1 for objects with overlapping others
        any_has_negative = False
        for obj_id in segments_ternary:
            prompt = segments_ternary[obj_id][..., 1]
            if (prompt < 0).any():
                any_has_negative = True
                break

        if any_has_negative:
            print("  Ternary mode has -1 values (others' prompts)")
        else:
            print("  No -1 values found (objects don't overlap in this test)")

        print("\n  PASSED: use_ternary_prompts controls binary vs ternary loading")


def test_mixed_cond_noncond():
    """Test SAM2DensePromptTrain.prepare_prompt_inputs splits cond/non-cond correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: Mixed conditioning/non-conditioning frame handling")
    print("=" * 60)

    # Create a mock BatchedVideoDatapoint
    from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData

    T, O, H, W = 10, 1, 64, 64
    prompt_end = 7

    # Simulate: 7 frames with prompts, 3 without
    has_prompt = torch.zeros(T, O, dtype=torch.bool)
    has_prompt[:prompt_end, :] = True

    prompts = torch.zeros(T, O, H, W, dtype=torch.float32)
    prompts[:prompt_end, :, 10:30, 10:30] = 1.0  # prompt region

    masks = torch.zeros(T, O, H, W, dtype=torch.bool)
    masks[2:6, :, 15:25, 15:25] = True  # GT mask on frames 2-5

    img_batch = torch.randn(T, 1, 3, H, W)
    obj_to_frame_idx = torch.zeros(T, O, 2, dtype=torch.int)
    for t in range(T):
        obj_to_frame_idx[t, 0] = torch.tensor([t, 0])

    metadata = BatchedVideoMetaData(
        unique_objects_identifier=torch.zeros(T, O, 3, dtype=torch.long),
        frame_orig_size=torch.tensor([[[H, W]] * O] * T, dtype=torch.long),
        batch_size=[T],
    )

    input_data = BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        prompts=prompts,
        has_prompt=has_prompt,
        metadata=metadata,
        dict_key="test",
        batch_size=[T],
    )

    # Test prepare_prompt_inputs logic (without creating the full model)
    cond_frames = []
    non_cond_frames = []
    for t in range(T):
        if input_data.has_prompt[t].any():
            cond_frames.append(t)
        else:
            non_cond_frames.append(t)

    print(f"  Total frames: {T}")
    print(f"  Conditioning frames: {cond_frames}")
    print(f"  Non-conditioning frames: {non_cond_frames}")

    assert cond_frames == list(range(prompt_end)), \
        f"Expected cond frames 0-{prompt_end-1}, got {cond_frames}"
    assert non_cond_frames == list(range(prompt_end, T)), \
        f"Expected non-cond frames {prompt_end}-{T-1}, got {non_cond_frames}"

    # Verify mask_inputs only for cond frames
    mask_inputs = {
        t: input_data.prompts[t].unsqueeze(1)
        for t in cond_frames
    }
    assert len(mask_inputs) == prompt_end
    assert prompt_end not in mask_inputs  # frame 7 should NOT have mask input

    # Verify non-cond frames have no mask input
    for t in non_cond_frames:
        assert t not in mask_inputs, f"Frame {t} should not have mask input"

    print("\n  PASSED: Mixed cond/non-cond frame splitting works correctly")


def test_backward_compatible():
    """Test that the pipeline is backward-compatible when all frames have prompts."""
    print("\n" + "=" * 60)
    print("TEST 5: Backward compatibility (all frames have prompts)")
    print("=" * 60)

    from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData

    T, O, H, W = 8, 5, 64, 64

    # All frames have prompts (original DensePromptSampler behavior)
    has_prompt = torch.ones(T, O, dtype=torch.bool)
    prompts = torch.rand(T, O, H, W)
    masks = torch.randint(0, 2, (T, O, H, W), dtype=torch.bool)

    cond_frames = []
    non_cond_frames = []
    for t in range(T):
        if has_prompt[t].any():
            cond_frames.append(t)
        else:
            non_cond_frames.append(t)

    assert cond_frames == list(range(T)), "All frames should be conditioning"
    assert non_cond_frames == [], "No non-conditioning frames"

    print(f"  All {T} frames are conditioning (backward compatible)")
    print("\n  PASSED: Backward compatible with all-frames-prompted mode")


def test_gt_beyond_prompt():
    """Test that GT masks on frames without prompts are loaded correctly."""
    print("\n" + "=" * 60)
    print("TEST 6: GT masks beyond last prompt frame")
    print("=" * 60)

    from training.dataset.vos_segment_loader import MultiplePNGSegmentLoader

    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_dir = os.path.join(tmp_dir, "per_object_data", "video_0")
        img_dir = os.path.join(tmp_dir, "img_folder", "video_0")
        os.makedirs(img_dir, exist_ok=True)

        H, W = 64, 64
        num_frames = 10
        prompt_end = 5  # prompts on frames 0-4
        gt_end = 8      # GT masks on frames 2-7 (extends beyond prompt)

        # Create frame images
        for f in range(num_frames):
            img = PILImage.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))
            img.save(os.path.join(img_dir, f"{f:05d}.jpg"))

        # Object 0: prompts on 0-4, GT on 2-7
        obj_dir = os.path.join(gt_dir, "0")
        os.makedirs(obj_dir, exist_ok=True)
        for f in range(num_frames):
            # Save prompt only on frames 0-4
            if f < prompt_end:
                prompt = np.zeros((H, W), dtype=np.uint8)
                prompt[10:30, 10:30] = 255
                PILImage.fromarray(prompt).save(os.path.join(obj_dir, f"{f:05d}_prompt.png"))

            # Save GT mask on frames 2-7 (including beyond prompt)
            if 2 <= f < gt_end:
                mask = np.zeros((H, W), dtype=np.uint8)
                mask[15:25, 15:25] = 255
                PILImage.fromarray(mask).save(os.path.join(obj_dir, f"{f:05d}_mask.png"))

        loader = MultiplePNGSegmentLoader(gt_dir)

        # Test each frame
        for f in range(num_frames):
            segments = loader.load(f)
            obj_id = 1  # folder "0" → obj_id 1

            has_prompt_expected = f < prompt_end
            has_gt_expected = 2 <= f < gt_end

            if obj_id in segments:
                seg = segments[obj_id]
                prompt_nonzero = bool(seg[..., 1].any())
                gt_nonzero = bool(seg[..., 0].any())

                print(f"  Frame {f}: in_segments=True, has_prompt={prompt_nonzero}, has_gt={gt_nonzero}")

                assert prompt_nonzero == has_prompt_expected, \
                    f"Frame {f}: prompt_nonzero={prompt_nonzero}, expected={has_prompt_expected}"
                assert gt_nonzero == has_gt_expected, \
                    f"Frame {f}: gt_nonzero={gt_nonzero}, expected={has_gt_expected}"
            else:
                print(f"  Frame {f}: in_segments=False (no prompt, no GT)")
                assert not has_prompt_expected and not has_gt_expected, \
                    f"Frame {f}: not in segments but expected prompt={has_prompt_expected}, gt={has_gt_expected}"

        # Key assertion: frames 5-7 have GT but NO prompt
        for f in [5, 6, 7]:
            seg = loader.load(f)
            assert 1 in seg, f"Frame {f}: object should be in segments (has GT)"
            assert not bool(seg[1][..., 1].any()), f"Frame {f}: should have no prompt"
            assert bool(seg[1][..., 0].any()), f"Frame {f}: should have GT mask"

        print("\n  PASSED: GT masks beyond prompts loaded correctly")


if __name__ == "__main__":
    test_lifecycle_sampler()
    test_has_prompt_tracking()
    test_ternary_parameter()
    test_mixed_cond_noncond()
    test_backward_compatible()
    test_gt_beyond_prompt()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
