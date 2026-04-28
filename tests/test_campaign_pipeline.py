"""Tests for contrailtrack.campaign.pipeline — orchestration."""

import datetime as dt
import tempfile
import unittest
from pathlib import Path

IMAGES_DIR = Path("/data/contrailnet/images/camera/visible/proj/2025/12/31")
OPENSKY_PATH = Path("/data/contrailnet/flights/opensky/2025/2025-12-31.parquet")
ERA5_DIR = Path("/data/contrailnet/weather/era5/2025")

ALL_DATA = IMAGES_DIR.exists() and OPENSKY_PATH.exists() and ERA5_DIR.exists()


@unittest.skipUnless(ALL_DATA, "Data not available for pipeline test")
class TestRunWindow(unittest.TestCase):
    """Integration test: run a single window without inference."""

    def test_prompts_only(self):
        """Run the full pipeline up to prompt generation (no SAM2 inference).

        Uses a 30-minute window to keep DryAdvection fast (~200 flights vs 968).
        """
        from contrailtrack.campaign.pipeline import run_window

        start = dt.datetime(2025, 12, 31, 12, 0, 0)
        stop = dt.datetime(2025, 12, 31, 12, 30, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir)
            stats = run_window(
                start, stop,
                images_dir=IMAGES_DIR,
                opensky_path=OPENSKY_PATH,
                era5_dir=ERA5_DIR,
                output_base=output_base,
                run_inference=False,
                skip_existing=False,
            )

            vid = "20251231120000_20251231123000"
            self.assertEqual(stats["video_id"], vid)
            self.assertGreater(stats["frames"], 50)
            self.assertGreater(stats["flights"], 0)

            # Fleet JSON exists
            fleet_path = output_base / "fleet" / f"{vid}.json"
            self.assertTrue(fleet_path.exists())

            # DryAdvection parquet exists
            da_path = output_base / "dry_advection" / f"{vid}.parquet"
            self.assertTrue(da_path.exists())

            # Prompt PNGs exist
            prompts_dir = output_base / "prompts" / vid
            prompt_pngs = list(prompts_dir.rglob("*_prompt.png"))
            self.assertGreater(len(prompt_pngs), 0)

            # Union PNGs exist
            union_pngs = list(prompts_dir.glob("*_all_prompts_union.png"))
            self.assertGreater(len(union_pngs), 0)

            # Frame symlinks exist
            frames = list((output_base / "frames" / vid).glob("*.jpg"))
            self.assertGreater(len(frames), 50)


@unittest.skipUnless(ALL_DATA, "Data not available for pipeline test")
class TestVideoIdFromWindow(unittest.TestCase):

    def test_format(self):
        from contrailtrack.campaign.pipeline import video_id_from_window

        vid = video_id_from_window(
            dt.datetime(2025, 12, 31, 10, 0, 0),
            dt.datetime(2025, 12, 31, 12, 0, 0),
        )
        self.assertEqual(vid, "20251231100000_20251231120000")


if __name__ == "__main__":
    unittest.main()
