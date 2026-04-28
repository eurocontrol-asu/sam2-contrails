"""Tests for contrailtrack.campaign.era5_local — local ERA5 → MetDataset."""

import datetime as dt
import unittest
from pathlib import Path

ERA5_DIR = Path("/data/contrailnet/weather/era5/2025")
ERA5_PL = ERA5_DIR / "2025_12_31.pl.nc"


@unittest.skipUnless(ERA5_PL.exists(), "Local ERA5 files not available")
class TestLoadLocalERA5(unittest.TestCase):

    def test_load_single_day(self):
        from contrailtrack.campaign.era5_local import load_local_era5

        met = load_local_era5(ERA5_DIR, [dt.date(2025, 12, 31)])

        self.assertIsNotNone(met)
        # Should have the 4 wind variables (renamed to standard names)
        ds = met.data
        for var in ("air_temperature", "eastward_wind", "northward_wind", "lagrangian_tendency_of_air_pressure"):
            self.assertIn(var, ds.data_vars)

    def test_coordinate_names(self):
        """Coordinates must be named 'time' and 'level' for pycontrails."""
        from contrailtrack.campaign.era5_local import load_local_era5

        met = load_local_era5(ERA5_DIR, [dt.date(2025, 12, 31)])
        ds = met.data

        self.assertIn("time", ds.dims)
        self.assertIn("level", ds.dims)
        self.assertIn("latitude", ds.dims)
        self.assertIn("longitude", ds.dims)

    def test_no_extra_dims(self):
        """number and expver should be squeezed/dropped."""
        from contrailtrack.campaign.era5_local import load_local_era5

        met = load_local_era5(ERA5_DIR, [dt.date(2025, 12, 31)])
        ds = met.data

        self.assertNotIn("number", ds.dims)
        self.assertNotIn("expver", ds.dims)
        # 4D: time × level × lat × lon
        for var in ds.data_vars:
            self.assertEqual(len(ds[var].dims), 4)

    def test_pressure_levels(self):
        """Should have 8 pressure levels (150-400 hPa)."""
        from contrailtrack.campaign.era5_local import load_local_era5

        met = load_local_era5(ERA5_DIR, [dt.date(2025, 12, 31)])
        levels = met.data["level"].values

        self.assertEqual(len(levels), 8)
        self.assertIn(150, levels)
        self.assertIn(400, levels)

    def test_variable_selection(self):
        """Should only include requested variables (renamed to standard names)."""
        from contrailtrack.campaign.era5_local import load_local_era5

        met = load_local_era5(ERA5_DIR, [dt.date(2025, 12, 31)], variables=("t", "u"))
        ds = met.data

        self.assertIn("air_temperature", ds.data_vars)
        self.assertIn("eastward_wind", ds.data_vars)
        self.assertNotIn("northward_wind", ds.data_vars)
        self.assertNotIn("specific_cloud_ice_water_content", ds.data_vars)


if __name__ == "__main__":
    unittest.main()
