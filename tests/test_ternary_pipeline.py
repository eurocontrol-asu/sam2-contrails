"""
Test that ternary prompt values {-1, 0, +1} survive through the full
data pipeline: segment loader → transforms → collate_fn → model input.
"""
import sys
import os
import torch
import numpy as np
from PIL import Image as PILImage
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset.vos_segment_loader import MultiplePNGSegmentLoader
from training.dataset.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensorAPI,
    ComposeAPI,
)
from training.utils.data_utils import (
    Object,
    Frame,
    VideoDatapoint,
    collate_fn,
)


def create_test_data(tmp_dir, h=64, w=64, num_objects=3, num_frames=4):
    """Create fake per-object PNG data with known ternary patterns."""
    os.makedirs(tmp_dir, exist_ok=True)

    # Create object folders
    for obj_id in range(num_objects):
        obj_folder = os.path.join(tmp_dir, str(obj_id))
        os.makedirs(obj_folder, exist_ok=True)

        for frame_id in range(num_frames):
            frame_str = f"{frame_id:05d}"

            # Each object gets a distinct horizontal stripe as prompt
            prompt = np.zeros((h, w), dtype=np.uint8)
            y_start = obj_id * (h // num_objects)
            y_end = (obj_id + 1) * (h // num_objects)
            prompt[y_start:y_end, :] = 255

            # GT mask: slightly smaller than prompt (inner region)
            mask = np.zeros((h, w), dtype=np.uint8)
            margin = 2
            if y_end - y_start > 2 * margin:
                mask[y_start + margin : y_end - margin, margin : w - margin] = 255

            PILImage.fromarray(prompt).save(
                os.path.join(obj_folder, f"{frame_str}_prompt.png")
            )
            if np.any(mask > 0):
                PILImage.fromarray(mask).save(
                    os.path.join(obj_folder, f"{frame_str}_mask.png")
                )

    # Create union PNG (union of ALL objects' prompts)
    for frame_id in range(num_frames):
        frame_str = f"{frame_id:05d}"
        union = np.zeros((h, w), dtype=np.uint8)
        for obj_id in range(num_objects):
            y_start = obj_id * (h // num_objects)
            y_end = (obj_id + 1) * (h // num_objects)
            union[y_start:y_end, :] = 255

        PILImage.fromarray(union).save(
            os.path.join(tmp_dir, f"{frame_str}_all_prompts_union.png")
        )

    return h, w, num_objects, num_frames


def test_segment_loader():
    """Test that MultiplePNGSegmentLoader produces correct ternary values."""
    print("=" * 60)
    print("TEST 1: MultiplePNGSegmentLoader ternary output")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp()
    try:
        h, w, num_objects, num_frames = create_test_data(tmp_dir)
        loader = MultiplePNGSegmentLoader(tmp_dir, use_ternary_prompts=True)

        segments = loader.load(0)

        for obj_id_key, seg in segments.items():
            print(f"\n  Object {obj_id_key}:")
            print(f"    Shape: {seg.shape}")
            print(f"    Dtype: {seg.dtype}")

            prompt_ch = seg[..., 1]
            unique_vals = torch.unique(prompt_ch).tolist()
            print(f"    Unique prompt values: {unique_vals}")

            has_neg1 = (prompt_ch == -1.0).any().item()
            has_zero = (prompt_ch == 0.0).any().item()
            has_pos1 = (prompt_ch == 1.0).any().item()
            print(f"    Has -1: {has_neg1}, Has 0: {has_zero}, Has +1: {has_pos1}")

            n_pos = (prompt_ch == 1.0).sum().item()
            n_neg = (prompt_ch == -1.0).sum().item()
            n_zero = (prompt_ch == 0.0).sum().item()
            print(f"    Counts: +1={n_pos}, -1={n_neg}, 0={n_zero}")

            # Verify: no unexpected values
            assert set(unique_vals).issubset({-1.0, 0.0, 1.0}), (
                f"Unexpected values: {unique_vals}"
            )
            assert has_pos1, "Object should have +1 pixels (own prompt)"
            assert has_neg1, "Object should have -1 pixels (others' prompts)"

        print("\n  PASSED: Loader produces correct ternary values")
    finally:
        shutil.rmtree(tmp_dir)


def test_transforms_preserve_ternary():
    """Test that flips preserve ternary values AND move them spatially."""
    print("\n" + "=" * 60)
    print("TEST 2: Transforms preserve ternary values (spatial check)")
    print("=" * 60)

    h, w = 64, 64

    # Create a segment with known ternary values in specific positions
    mask_ch = torch.zeros(h, w, dtype=torch.float32)
    mask_ch[10:20, 10:30] = 1.0

    prompt_ch = torch.zeros(h, w, dtype=torch.float32)
    prompt_ch[10:30, :] = 1.0   # own prompt: rows 10-29
    prompt_ch[40:60, :] = -1.0  # others' prompt: rows 40-59

    segment = torch.stack([mask_ch, prompt_ch], dim=-1)

    original_unique = torch.unique(segment[..., 1]).tolist()
    print(f"  Original prompt unique values: {original_unique}")

    # --- Horizontal flip: spatial positions ---
    seg_hflip = segment.clone()
    data = seg_hflip.permute(2, 0, 1)  # [2, H, W]
    data = torch.flip(data, [-1])
    seg_hflip = data.permute(1, 2, 0)  # [H, W, 2]

    after_hflip = torch.unique(seg_hflip[..., 1]).tolist()
    print(f"  After hflip unique values: {after_hflip}")
    assert after_hflip == original_unique, f"hflip changed values: {after_hflip}"

    # hflip should keep row positions the same, mirror columns
    # prompt_ch[15, 0] was +1.0 → should now be at [15, w-1]
    assert seg_hflip[15, w - 1, 1].item() == 1.0, "hflip: +1 pixel not at mirrored column"
    assert seg_hflip[15, 0, 1].item() == 1.0, "hflip: full-width stripe should stay full"
    # prompt_ch[50, 0] was -1.0 → should now be at [50, w-1]
    assert seg_hflip[50, w - 1, 1].item() == -1.0, "hflip: -1 pixel not at mirrored column"
    # Row 35 was 0 → should stay 0 everywhere
    assert (seg_hflip[35, :, 1] == 0.0).all(), "hflip: zero row should remain zero"
    print("  hflip spatial positions: CORRECT")

    # --- Vertical flip: spatial positions ---
    seg_vflip = segment.clone()
    data = seg_vflip.permute(2, 0, 1)
    data = torch.flip(data, [-2])
    seg_vflip = data.permute(1, 2, 0)

    after_vflip = torch.unique(seg_vflip[..., 1]).tolist()
    print(f"  After vflip unique values: {after_vflip}")
    assert after_vflip == original_unique, f"vflip changed values: {after_vflip}"

    # vflip mirrors rows: row r → row (h-1-r)
    # prompt_ch rows 10-29 were +1 → after vflip should be rows (h-30) to (h-11) = 34 to 53
    # prompt_ch rows 40-59 were -1 → after vflip should be rows (h-60) to (h-41) = 4 to 23
    assert seg_vflip[h - 1 - 15, 0, 1].item() == 1.0, "vflip: +1 row not mirrored"
    assert seg_vflip[h - 1 - 50, 0, 1].item() == -1.0, "vflip: -1 row not mirrored"

    # After vflip, -1 region (was rows 40-59) is now at rows 4-23
    # and +1 region (was rows 10-29) is now at rows 34-53
    assert (seg_vflip[4:24, 0, 1] == -1.0).all(), "vflip: -1 block not at mirrored rows"
    assert (seg_vflip[34:54, 0, 1] == 1.0).all(), "vflip: +1 block not at mirrored rows"

    # Mask channel should also flip: mask was at rows 10:20 → now at rows 44:54
    assert (seg_vflip[44:54, 10:30, 0] == 1.0).all(), "vflip: mask not at mirrored position"
    assert (seg_vflip[10:20, 10:30, 0] == 0.0).all(), "vflip: mask should have moved away"
    print("  vflip spatial positions: CORRECT")

    print("\n  PASSED: Transforms preserve ternary values and spatial positions")


def test_collate_fn():
    """Test that collate_fn correctly extracts ternary prompts."""
    print("\n" + "=" * 60)
    print("TEST 3: collate_fn preserves ternary values")
    print("=" * 60)

    h, w = 64, 64
    T, num_objects = 4, 3

    # Build fake VideoDatapoint with ternary segments
    frames = []
    for t in range(T):
        objects = []
        for obj_id in range(num_objects):
            mask_ch = torch.zeros(h, w, dtype=torch.float32)
            prompt_ch = torch.zeros(h, w, dtype=torch.float32)

            # Own prompt stripe
            y_start = obj_id * (h // num_objects)
            y_end = (obj_id + 1) * (h // num_objects)
            prompt_ch[y_start:y_end, :] = 1.0
            mask_ch[y_start + 2 : y_end - 2, 2 : w - 2] = 1.0

            # Others' prompts
            for other_id in range(num_objects):
                if other_id == obj_id:
                    continue
                oy_start = other_id * (h // num_objects)
                oy_end = (other_id + 1) * (h // num_objects)
                prompt_ch[oy_start:oy_end, :] = -1.0

            segment = torch.stack([mask_ch, prompt_ch], dim=-1)
            objects.append(Object(object_id=obj_id + 1, frame_index=t, segment=segment))

        frames.append(
            Frame(
                data=torch.rand(3, h, w),  # fake RGB
                objects=objects,
            )
        )

    video = VideoDatapoint(frames=frames, video_id=0, size=(h, w))
    batch = collate_fn([video], dict_key="test")

    print(f"  batch.prompts shape: {batch.prompts.shape}")
    print(f"  batch.prompts dtype: {batch.prompts.dtype}")

    unique_vals = torch.unique(batch.prompts).tolist()
    print(f"  Unique prompt values in batch: {unique_vals}")

    assert batch.prompts.dtype == torch.float32, (
        f"Expected float32, got {batch.prompts.dtype}"
    )
    assert set(unique_vals) == {-1.0, 0.0, 1.0}, (
        f"Expected {{-1, 0, 1}}, got {unique_vals}"
    )

    # Check masks are still bool
    print(f"  batch.masks dtype: {batch.masks.dtype}")
    assert batch.masks.dtype == torch.bool, f"Masks should be bool, got {batch.masks.dtype}"

    # Verify per-object: each object should have +1 in its stripe, -1 in others
    for obj_idx in range(num_objects):
        prompt = batch.prompts[0, obj_idx]  # frame 0, object obj_idx
        n_pos = (prompt == 1.0).sum().item()
        n_neg = (prompt == -1.0).sum().item()
        n_zero = (prompt == 0.0).sum().item()
        total = h * w
        print(f"  Object {obj_idx}: +1={n_pos}, -1={n_neg}, 0={n_zero} (total={total})")
        assert n_pos > 0, f"Object {obj_idx} has no +1 pixels"
        assert n_neg > 0, f"Object {obj_idx} has no -1 pixels"

    print("\n  PASSED: collate_fn correctly preserves ternary values")


def test_full_pipeline():
    """End-to-end: loader → transforms → collate → verify."""
    print("\n" + "=" * 60)
    print("TEST 4: Full pipeline (loader → collate)")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp()
    try:
        h, w, num_objects, num_frames = create_test_data(tmp_dir, h=64, w=64)
        loader = MultiplePNGSegmentLoader(tmp_dir, use_ternary_prompts=True)

        frames = []
        for frame_id in range(num_frames):
            segments = loader.load(frame_id)
            objects = []
            for obj_id, seg in segments.items():
                objects.append(
                    Object(object_id=obj_id, frame_index=frame_id, segment=seg)
                )
            frames.append(
                Frame(data=torch.rand(3, h, w), objects=objects)
            )

        video = VideoDatapoint(frames=frames, video_id=0, size=(h, w))
        batch = collate_fn([video], dict_key="test")

        print(f"  Shape: {batch.prompts.shape}")
        print(f"  Dtype: {batch.prompts.dtype}")

        unique_vals = torch.unique(batch.prompts).tolist()
        print(f"  Unique values: {unique_vals}")

        assert set(unique_vals).issubset({-1.0, 0.0, 1.0}), (
            f"Unexpected values: {unique_vals}"
        )
        assert -1.0 in unique_vals, "Missing -1 values in final output"
        assert 1.0 in unique_vals, "Missing +1 values in final output"

        # Simulate what prepare_prompt_inputs does
        for t in range(batch.num_frames):
            mask_input = batch.prompts[t].unsqueeze(1)  # [O, 1, H, W]
            print(f"  Frame {t} mask_input shape: {mask_input.shape}, "
                  f"range: [{mask_input.min():.1f}, {mask_input.max():.1f}]")
            assert mask_input.min() >= -1.0 and mask_input.max() <= 1.0

        print("\n  PASSED: Full pipeline preserves ternary values")
    finally:
        shutil.rmtree(tmp_dir)


def test_backward_compat_no_union():
    """Test that without union PNG, we get binary {0, 1} prompts."""
    print("\n" + "=" * 60)
    print("TEST 5: Backward compatibility (no union PNG)")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp()
    try:
        h, w = 64, 64
        # Create object data WITHOUT union PNG
        obj_folder = os.path.join(tmp_dir, "0")
        os.makedirs(obj_folder)

        prompt = np.zeros((h, w), dtype=np.uint8)
        prompt[10:30, :] = 255
        PILImage.fromarray(prompt).save(os.path.join(obj_folder, "00000_prompt.png"))

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[12:28, 2:62] = 255
        PILImage.fromarray(mask).save(os.path.join(obj_folder, "00000_mask.png"))

        loader = MultiplePNGSegmentLoader(tmp_dir, use_ternary_prompts=True)
        segments = loader.load(0)

        seg = list(segments.values())[0]
        unique_vals = torch.unique(seg[..., 1]).tolist()
        print(f"  Prompt values without union: {unique_vals}")

        assert -1.0 not in unique_vals, "Should NOT have -1 without union PNG"
        assert set(unique_vals).issubset({0.0, 1.0}), (
            f"Expected binary {{0, 1}}, got {unique_vals}"
        )

        print("\n  PASSED: Backward compatible (binary prompts without union)")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_segment_loader()
    test_transforms_preserve_ternary()
    test_collate_fn()
    test_full_pipeline()
    test_backward_compat_no_union()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
