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
class TestRunDay(unittest.TestCase):
    """Integration test: run a single day without inference."""

    def test_day_prompts_only(self):
        """Run the full day-level pipeline (no SAM2 inference).

        Uses a single 30-minute window to keep DryAdvection fast.
        Verifies that day-level advection + per-window filtering works.
        """
        from contrailtrack.campaign.pipeline import run_day

        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir)

            # Monkey-patch daylight_windows to return one short window
            import contrailtrack.campaign.pipeline as mod
            original_fn = None

            def _fake_windows(date, lat, lon, window_hours=2.0):
                return [
                    (dt.datetime(2025, 12, 31, 12, 0, 0),
                     dt.datetime(2025, 12, 31, 12, 30, 0)),
                ]

            import contrailtrack.campaign.solar as solar_mod
            original_fn = solar_mod.daylight_windows
            solar_mod.daylight_windows = _fake_windows

            try:
                results = run_day(
                    dt.date(2025, 12, 31),
                    images_base=Path("/data/contrailnet/images/camera/visible/proj"),
                    flights_base=Path("/data/contrailnet/flights/opensky"),
                    era5_base=Path("/data/contrailnet/weather/era5"),
                    output_base=output_base,
                    run_inference=False,
                    skip_existing=False,
                )
            finally:
                solar_mod.daylight_windows = original_fn

            self.assertEqual(len(results), 1)
            stats = results[0]
            vid = "20251231120000_20251231123000"
            self.assertEqual(stats["video_id"], vid)
            self.assertGreater(stats["frames"], 50)

            # Day-level fleet JSON exists
            fleet_path = output_base / "fleet" / "20251231.json"
            self.assertTrue(fleet_path.exists())

            # Day-level DryAdvection parquet exists
            da_path = output_base / "dry_advection" / "20251231.parquet"
            self.assertTrue(da_path.exists())

            # Prompt PNGs exist for the window
            prompts_dir = output_base / "prompts" / vid
            prompt_pngs = list(prompts_dir.rglob("*_prompt.png"))
            self.assertGreater(len(prompt_pngs), 0)

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
