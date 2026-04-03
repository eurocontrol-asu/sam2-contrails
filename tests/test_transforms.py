"""
Tests for RandomAffine transform with [H,W,2] segments.

Verifies the fix that permutes [H,W,2] → [2,H,W] before F.affine
and permutes back after, matching the pattern used by hflip/vflip.
"""

import unittest

import torch
from PIL import Image as PILImage

from training.utils.data_utils import Object, Frame, VideoDatapoint
from training.dataset.transforms import RandomAffine


def _make_datapoint(h=64, w=64, num_frames=2):
    """Create a VideoDatapoint with [H,W,2] segments and a known mask+prompt pattern."""
    frames = []
    for frame_idx in range(num_frames):
        # RGB image
        img = PILImage.new("RGB", (w, h), color=(128, 128, 128))
        # Segment: [H,W,2] — channel 0 = GT mask, channel 1 = prompt
        segment = torch.zeros(h, w, 2)
        # Put a block in the center for both channels
        segment[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, 0] = 1.0  # GT mask
        segment[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, 1] = 1.0  # prompt
        obj = Object(object_id=0, frame_index=frame_idx, segment=segment)
        frames.append(Frame(data=img, objects=[obj]))
    return VideoDatapoint(frames=frames, video_id=0, size=(h, w))


class TestRandomAffineSegmentShape(unittest.TestCase):
    """Verify RandomAffine preserves [H,W,2] segment shape."""

    def test_output_shape_preserved(self):
        """Output segments must be [H,W,2] after transform."""
        dp = _make_datapoint(h=64, w=64)
        transform = RandomAffine(
            degrees=30, shear=15, consistent_transform=True, image_interpolation="bilinear"
        )
        result = transform(dp)
        # Result could be None if the affine zeroed out the mask — retry with identity
        if result is None:
            # Use degrees=0 to guarantee non-None
            transform_id = RandomAffine(
                degrees=0, consistent_transform=True, image_interpolation="bilinear"
            )
            result = transform_id(_make_datapoint(h=64, w=64))
        for frame in result.frames:
            for obj in frame.objects:
                self.assertEqual(obj.segment.shape, (64, 64, 2),
                                 f"Expected [H,W,2] but got {obj.segment.shape}")

    def test_both_channels_transformed_identically(self):
        """Both segment channels must undergo the same spatial warp."""
        dp = _make_datapoint(h=64, w=64)
        # Identity rotation to guarantee non-None result, but apply shear
        transform = RandomAffine(
            degrees=0, shear=15, consistent_transform=True, image_interpolation="bilinear"
        )
        result = transform(dp)
        self.assertIsNotNone(result, "Transform returned None with degrees=0")
        for frame in result.frames:
            for obj in frame.objects:
                gt_mask = obj.segment[:, :, 0]
                prompt = obj.segment[:, :, 1]
                # Both channels started identical, so after identical spatial warp
                # they must remain identical
                self.assertTrue(
                    torch.equal(gt_mask, prompt),
                    "GT mask and prompt channel diverged after affine transform"
                )

    def test_identity_transform_preserves_content(self):
        """An identity affine (degrees=0, no shear/scale/translate) preserves segment values."""
        dp = _make_datapoint(h=32, w=32)
        original_segment = dp.frames[0].objects[0].segment.clone()
        transform = RandomAffine(
            degrees=0, consistent_transform=True, image_interpolation="bilinear"
        )
        result = transform(dp)
        self.assertIsNotNone(result)
        self.assertTrue(
            torch.equal(result.frames[0].objects[0].segment, original_segment),
            "Identity transform modified the segment"
        )


class TestRandomAffineVisibilityCheck(unittest.TestCase):
    """Verify the first-frame visibility check uses channel 0 (GT mask) only."""

    def test_visibility_check_uses_gt_channel_only(self):
        """
        If GT mask (channel 0) is non-zero but prompt (channel 1) is zero,
        the transform should NOT return None.
        """
        dp = _make_datapoint(h=64, w=64, num_frames=2)
        # Set channel 0 (GT) to have content, channel 1 (prompt) to zero
        for frame in dp.frames:
            for obj in frame.objects:
                obj.segment[:, :, 1] = 0.0  # zero out prompt channel
        transform = RandomAffine(
            degrees=0, consistent_transform=True, image_interpolation="bilinear"
        )
        result = transform(dp)
        self.assertIsNotNone(result,
                             "Transform incorrectly returned None when GT mask was non-zero")

    def test_skips_transform_when_gt_mask_empty_in_first_frame(self):
        """
        If GT mask (channel 0) is all-zero in the first frame,
        transform_datapoint should return None (triggering fallback to
        untransformed datapoint in __call__).
        """
        dp = _make_datapoint(h=64, w=64, num_frames=2)
        # Zero out GT mask in first frame only, keep prompt
        dp.frames[0].objects[0].segment[:, :, 0] = 0.0
        transform = RandomAffine(
            degrees=0, consistent_transform=True, image_interpolation="bilinear",
            log_warning=False,
        )
        # transform_datapoint returns None for empty GT mask
        result = transform.transform_datapoint(dp)
        self.assertIsNone(result,
                          "transform_datapoint should return None when GT mask is empty in first frame")


class TestRandomAffineNoneSegments(unittest.TestCase):
    """Verify RandomAffine handles None segments gracefully."""

    def test_none_segment_passthrough(self):
        """Objects with segment=None should pass through without error."""
        dp = _make_datapoint(h=32, w=32, num_frames=1)
        # Add a second object with None segment
        dp.frames[0].objects.append(
            Object(object_id=1, frame_index=0, segment=None)
        )
        transform = RandomAffine(
            degrees=0, consistent_transform=True, image_interpolation="bilinear"
        )
        result = transform(dp)
        self.assertIsNotNone(result)
        self.assertIsNone(result.frames[0].objects[1].segment)
        # First object should still have valid [H,W,2] segment
        self.assertEqual(result.frames[0].objects[0].segment.shape, (32, 32, 2))


class TestRandomAffineInconsistentTransform(unittest.TestCase):
    """Verify non-consistent mode also works with [H,W,2] segments."""

    def test_inconsistent_mode_shape_preserved(self):
        """Output segments must be [H,W,2] even in non-consistent mode."""
        dp = _make_datapoint(h=32, w=32, num_frames=3)
        transform = RandomAffine(
            degrees=0, shear=10, consistent_transform=False, image_interpolation="bilinear"
        )
        result = transform(dp)
        self.assertIsNotNone(result)
        for frame in result.frames:
            for obj in frame.objects:
                self.assertEqual(obj.segment.shape, (32, 32, 2))


if __name__ == "__main__":
    unittest.main()
