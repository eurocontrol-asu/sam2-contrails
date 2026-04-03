"""Tests for contrailtrack.data — video frame loading and prompt reading.

Unit tests use synthetic data (no real dataset needed).
Integration tests use real GVCCS data and are skipped when data is absent.
"""

import os
import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


# ─── helpers ──────────────────────────────────────────────────────────────────

# Integration tests use real GVCCS data. Set GVCCS_TEST_DIR to point at your
# local copy of the GVCCS test split (e.g. /path/to/GVCCS/test/).
_gvccs_test_dir = os.environ.get("GVCCS_TEST_DIR", "")
_GVCCS_TEST = Path(_gvccs_test_dir) if _gvccs_test_dir else Path("__nonexistent__")
REAL_PROMPTS = _GVCCS_TEST / "per_object_data_age_5"
REAL_FRAMES = _GVCCS_TEST / "img_folder"
REAL_VIDEO_ID = "00001"

real_data_available = _gvccs_test_dir != "" and REAL_PROMPTS.exists() and REAL_FRAMES.exists()


def _make_prompt_dir(tmp: Path, video_id: str = "00001") -> Path:
    """Create a minimal per_object_data/ structure for testing."""
    vid_dir = tmp / video_id
    # Two objects (folders 0 and 1), two frames each
    for obj_folder, frames in [("0", ["00000", "00001"]), ("1", ["00000"])]:
        obj_dir = vid_dir / obj_folder
        obj_dir.mkdir(parents=True)
        for frame in frames:
            arr = np.zeros((64, 64), dtype=np.uint8)
            arr[10:30, 10:30] = 200  # non-zero prompt
            Image.fromarray(arr).save(obj_dir / f"{frame}_prompt.png")

    # Union masks for ternary
    for frame in ["00000", "00001"]:
        union = np.zeros((64, 64), dtype=np.uint8)
        union[5:35, 5:35] = 180
        Image.fromarray(union).save(vid_dir / f"{frame}_all_prompts_union.png")

    return tmp


# ─── prompt reader ────────────────────────────────────────────────────────────

class TestPromptReader(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _make_prompt_dir(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_objects_returns_sorted_folder_names(self):
        from contrailtrack.data.prompt_reader import list_objects
        objs = list_objects(self.tmp, "00001")
        self.assertEqual(objs, ["0", "1"])

    def test_list_objects_missing_video_returns_empty(self):
        from contrailtrack.data.prompt_reader import list_objects
        objs = list_objects(self.tmp, "99999")
        self.assertEqual(objs, [])

    def test_read_prompts_binary(self):
        from contrailtrack.data.prompt_reader import read_prompts
        prompts = read_prompts(self.tmp, "00001", encoding="binary")
        # folder "0" has frames 00000 and 00001; folder "1" has only 00000
        self.assertIn("0", prompts)
        self.assertIn("1", prompts)
        self.assertIn("00000", prompts["0"])
        self.assertIn("00001", prompts["0"])
        arr = prompts["0"]["00000"]
        self.assertEqual(arr.dtype, np.float32)
        self.assertGreater(arr.max(), 0)

    def test_read_prompts_age_weighted_same_as_binary_for_uniform_png(self):
        from contrailtrack.data.prompt_reader import read_prompts
        prompts_bin = read_prompts(self.tmp, "00001", encoding="binary")
        prompts_age = read_prompts(self.tmp, "00001", encoding="age_weighted")
        # Both should decode the same PNG values; no formula difference
        np.testing.assert_array_equal(
            prompts_bin["0"]["00000"], prompts_age["0"]["00000"]
        )

    def test_read_prompts_ternary_has_negative_region(self):
        from contrailtrack.data.prompt_reader import read_prompts
        prompts = read_prompts(self.tmp, "00001", encoding="ternary")
        arr = prompts["0"]["00000"]
        # Where own > 0 → positive; elsewhere union is negative
        self.assertGreater(arr.max(), 0)
        self.assertLess(arr.min(), 0)

    def test_read_prompts_missing_video_raises(self):
        from contrailtrack.data.prompt_reader import read_prompts
        with self.assertRaises(FileNotFoundError):
            read_prompts(self.tmp, "99999")

    def test_read_prompts_empty_prompt_skipped(self):
        """Prompt files where all pixels are zero should be excluded."""
        from contrailtrack.data.prompt_reader import read_prompts
        # Add an all-zero prompt file
        obj_dir = self.tmp / "00001" / "0"
        arr = np.zeros((64, 64), dtype=np.uint8)
        Image.fromarray(arr).save(obj_dir / "00002_prompt.png")

        prompts = read_prompts(self.tmp, "00001")
        self.assertNotIn("00002", prompts.get("0", {}))


class TestPromptReaderIntegration(unittest.TestCase):

    @unittest.skipUnless(real_data_available, "Real GVCCS data not available")
    def test_read_real_prompts_ternary(self):
        from contrailtrack.data.prompt_reader import read_prompts, list_objects
        objs = list_objects(REAL_PROMPTS, REAL_VIDEO_ID)
        self.assertGreater(len(objs), 0, "No objects found")

        prompts = read_prompts(REAL_PROMPTS, REAL_VIDEO_ID, encoding="ternary")
        obj_id = objs[0]
        self.assertIn(obj_id, prompts)

        frames = list(prompts[obj_id].keys())
        arr = prompts[obj_id][frames[0]]
        self.assertEqual(arr.shape, (1024, 1024))
        self.assertEqual(arr.dtype, np.float32)
        # Ternary: positive and negative components
        self.assertGreater(arr.max(), 0)
        self.assertLess(arr.min(), 0)

    @unittest.skipUnless(real_data_available, "Real GVCCS data not available")
    def test_read_real_prompts_binary_no_negative(self):
        from contrailtrack.data.prompt_reader import read_prompts
        prompts = read_prompts(REAL_PROMPTS, REAL_VIDEO_ID, encoding="binary")
        for obj_id, obj_prompts in prompts.items():
            for frame, arr in obj_prompts.items():
                self.assertGreaterEqual(arr.min(), 0.0, f"Negative value in binary prompt obj={obj_id} frame={frame}")
                self.assertLessEqual(arr.max(), 1.0)


# ─── video loader ────────────────────────────────────────────────────────────

class TestLoadFrames(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Create 3 synthetic JPEG frames named as integers
        for i in range(3):
            arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(arr).save(self.tmp / f"{i:05d}.jpg")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_four_tuple(self):
        from contrailtrack.data.video import load_frames
        result = load_frames(self.tmp, image_size=64)
        self.assertEqual(len(result), 4)

    def test_frame_tensor_shape(self):
        from contrailtrack.data.video import load_frames
        frames, frame_names, H, W = load_frames(self.tmp, image_size=64)
        self.assertEqual(frames.shape, (3, 3, 64, 64))

    def test_frame_names_sorted(self):
        from contrailtrack.data.video import load_frames
        _, frame_names, _, _ = load_frames(self.tmp, image_size=64)
        self.assertEqual(frame_names, ["00000", "00001", "00002"])

    def test_no_imagenet_normalization(self):
        """Values must stay in [0, 1] — no ImageNet shift applied."""
        from contrailtrack.data.video import load_frames
        frames, _, _, _ = load_frames(self.tmp, image_size=64)
        self.assertGreaterEqual(float(frames.min()), 0.0)
        self.assertLessEqual(float(frames.max()), 1.0)

    def test_original_dims_reported(self):
        from contrailtrack.data.video import load_frames
        _, _, H, W = load_frames(self.tmp, image_size=64)
        self.assertEqual(H, 128)
        self.assertEqual(W, 128)


class TestLoadFramesIntegration(unittest.TestCase):

    @unittest.skipUnless(real_data_available, "Real GVCCS data not available")
    def test_real_video_shape(self):
        from contrailtrack.data.video import load_frames
        video_dir = REAL_FRAMES / REAL_VIDEO_ID
        frames, frame_names, H, W = load_frames(video_dir, image_size=1024)
        self.assertEqual(frames.shape[1:], (3, 1024, 1024))
        self.assertGreater(len(frame_names), 0)
        self.assertEqual(H, 1024)
        self.assertEqual(W, 1024)


if __name__ == "__main__":
    unittest.main()
