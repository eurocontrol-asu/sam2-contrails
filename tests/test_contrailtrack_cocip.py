"""Integration test for contrailtrack.prompts.cocip — runs CoCiP end-to-end.

Uses the smallest available fleet JSON (2-hour window) from the GVCCS test set.
Downloads ERA5 model-level data on first run; subsequent runs use the disk cache.

Run with:
    python -m pytest tests/test_contrailtrack_cocip.py -v -s
"""

import os
import unittest
from pathlib import Path

# Integration tests require real CoCiP fleet data. Set these env vars to point
# at your local copy, e.g.:
#   export COCIP_FLEET_JSON=/path/to/fleet/20230930055430_20230930075430.json
#   export COCIP_MET_CACHE=/path/to/met_cache
#   export COCIP_KNOWN_OUTPUT=/path/to/reference.parquet
_fleet_json = os.environ.get("COCIP_FLEET_JSON", "")
_met_cache = os.environ.get("COCIP_MET_CACHE", "")
_known_output = os.environ.get("COCIP_KNOWN_OUTPUT", "")
FLEET_JSON   = Path(_fleet_json) if _fleet_json else Path("__nonexistent__")
MET_CACHE    = Path(_met_cache) if _met_cache else Path("__nonexistent__")
KNOWN_OUTPUT = Path(_known_output) if _known_output else Path("__nonexistent__")

cocip_available = _fleet_json != "" and FLEET_JSON.exists()

try:
    from contrailtrack.prompts.cocip import run_cocip
    import pycontrails  # noqa
    pycontrails_available = True
except ImportError:
    pycontrails_available = False


@unittest.skipUnless(cocip_available and pycontrails_available,
                     "fleet JSON or pycontrails not available")
class TestRunCocip(unittest.TestCase):

    @classmethod
    def _tmp_dir(cls):
        """Return a temp directory for test output."""
        import tempfile
        return tempfile.mkdtemp()

    def test_run_cocip_produces_parquet(self):
        """run_cocip on a 2-hour fleet produces a non-empty contrail parquet."""
        import shutil, pandas as pd

        tmp = self._tmp_dir()
        try:
            out = run_cocip(
                fleet_json=FLEET_JSON,
                output_dir=Path(tmp),
                cache_dir=MET_CACHE,
            )
            self.assertIsNotNone(out, "Expected contrails but got None")
            self.assertTrue(out.exists())

            df = pd.read_parquet(out)
            self.assertGreater(len(df), 0)
            for col in ("flight_id", "longitude", "latitude", "formation_time"):
                self.assertIn(col, df.columns, f"Missing column: {col}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_cocip_output_matches_existing(self):
        """run_cocip output flight_ids should match the pre-computed reference."""
        import shutil, pandas as pd

        if not KNOWN_OUTPUT.exists():
            self.skipTest("Reference parquet not available for comparison")

        tmp = self._tmp_dir()
        try:
            out = run_cocip(
                fleet_json=FLEET_JSON,
                output_dir=Path(tmp),
                cache_dir=MET_CACHE,
            )
            if out is None:
                self.skipTest("CoCiP produced no contrails (possible met data difference)")

            df_new = pd.read_parquet(out)
            df_ref = pd.read_parquet(KNOWN_OUTPUT)

            new_flights = set(df_new["flight_id"].unique())
            ref_flights = set(df_ref["flight_id"].unique())

            # At least 80% of reference flights should appear in new output
            overlap = len(new_flights & ref_flights) / max(len(ref_flights), 1)
            self.assertGreater(overlap, 0.8,
                f"Only {overlap:.0%} flight overlap with reference "
                f"(new={len(new_flights)}, ref={len(ref_flights)})")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_cocip_skip_existing(self):
        """skip_existing=True returns the cached path without re-running CoCiP."""
        import shutil, time

        tmp = self._tmp_dir()
        try:
            out_dir = Path(tmp)
            # First run — uses cached ERA5
            out1 = run_cocip(
                fleet_json=FLEET_JSON,
                output_dir=out_dir,
                cache_dir=MET_CACHE,
            )
            self.assertIsNotNone(out1)

            # Second run with skip_existing — should return immediately
            from contrailtrack.prompts.cocip import run_cocip_batch
            fleet_tmp = out_dir / "fleet"
            fleet_tmp.mkdir()
            shutil.copy(FLEET_JSON, fleet_tmp / FLEET_JSON.name)

            t0 = time.time()
            results = run_cocip_batch(
                fleet_dir=fleet_tmp,
                output_dir=out_dir,
                cache_dir=MET_CACHE,
                skip_existing=True,
            )
            elapsed = time.time() - t0
            self.assertLess(elapsed, 5.0, "skip_existing should return instantly")
            self.assertEqual(len(results), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
