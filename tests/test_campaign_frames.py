"""Tests for contrailtrack.campaign.frames — image symlink creation."""

import datetime as dt
import os
import tempfile
import unittest
from pathlib import Path

IMAGES_DIR = Path("/data/contrailnet/images/camera/visible/proj/2025/12/31")


@unittest.skipUnless(IMAGES_DIR.exists(), "Image directory not available")
class TestCreateFrameSymlinks(unittest.TestCase):

    def test_creates_symlinks(self):
        from contrailtrack.campaign.frames import create_frame_symlinks

        with tempfile.TemporaryDirectory() as tmpdir:
            count = create_frame_symlinks(
                images_dir=IMAGES_DIR,
                output_dir=Path(tmpdir),
                video_id="20251231100000_20251231120000",
                start=dt.datetime(2025, 12, 31, 10, 0, 0),
                stop=dt.datetime(2025, 12, 31, 12, 0, 0),
            )

            self.assertGreater(count, 0)

            video_dir = Path(tmpdir) / "20251231100000_20251231120000"
            self.assertTrue(video_dir.exists())

            # Symlinks should exist and point to real files
            links = sorted(video_dir.glob("*.jpg"))
            self.assertEqual(len(links), count)
            for link in links:
                self.assertTrue(link.is_symlink())
                self.assertTrue(link.resolve().exists())

    def test_prefix_stripped(self):
        """Symlinks should be named YYYYMMDDHHMMSS.jpg (no image_ prefix)."""
        from contrailtrack.campaign.frames import create_frame_symlinks

        with tempfile.TemporaryDirectory() as tmpdir:
            create_frame_symlinks(
                images_dir=IMAGES_DIR,
                output_dir=Path(tmpdir),
                video_id="test_vid",
                start=dt.datetime(2025, 12, 31, 12, 0, 0),
                stop=dt.datetime(2025, 12, 31, 12, 1, 0),
            )

            video_dir = Path(tmpdir) / "test_vid"
            for link in video_dir.glob("*.jpg"):
                # Should NOT start with "image_"
                self.assertFalse(link.name.startswith("image_"))
                # Should be a 14-digit timestamp
                stem = link.stem
                self.assertEqual(len(stem), 14)
                self.assertTrue(stem.isdigit())

    def test_time_filtering(self):
        """Only images within [start, stop) should be symlinked."""
        from contrailtrack.campaign.frames import create_frame_symlinks

        with tempfile.TemporaryDirectory() as tmpdir:
            # 30-minute window → expect ~60 frames (30s intervals)
            count = create_frame_symlinks(
                images_dir=IMAGES_DIR,
                output_dir=Path(tmpdir),
                video_id="test_30min",
                start=dt.datetime(2025, 12, 31, 12, 0, 0),
                stop=dt.datetime(2025, 12, 31, 12, 30, 0),
            )

            # 30 min / 30s = 60 frames (±1)
            self.assertGreaterEqual(count, 58)
            self.assertLessEqual(count, 62)

    def test_two_hour_window(self):
        """2-hour window → expect ~240 frames."""
        from contrailtrack.campaign.frames import create_frame_symlinks

        with tempfile.TemporaryDirectory() as tmpdir:
            count = create_frame_symlinks(
                images_dir=IMAGES_DIR,
                output_dir=Path(tmpdir),
                video_id="test_2h",
                start=dt.datetime(2025, 12, 31, 10, 0, 0),
                stop=dt.datetime(2025, 12, 31, 12, 0, 0),
            )

            self.assertGreaterEqual(count, 235)
            self.assertLessEqual(count, 245)

    def test_idempotent(self):
        """Running twice should not fail or duplicate."""
        from contrailtrack.campaign.frames import create_frame_symlinks

        with tempfile.TemporaryDirectory() as tmpdir:
            c1 = create_frame_symlinks(
                images_dir=IMAGES_DIR,
                output_dir=Path(tmpdir),
                video_id="test_idem",
                start=dt.datetime(2025, 12, 31, 12, 0, 0),
                stop=dt.datetime(2025, 12, 31, 12, 5, 0),
            )
            c2 = create_frame_symlinks(
                images_dir=IMAGES_DIR,
                output_dir=Path(tmpdir),
                video_id="test_idem",
                start=dt.datetime(2025, 12, 31, 12, 0, 0),
                stop=dt.datetime(2025, 12, 31, 12, 5, 0),
            )

            self.assertEqual(c1, c2)


if __name__ == "__main__":
    unittest.main()
