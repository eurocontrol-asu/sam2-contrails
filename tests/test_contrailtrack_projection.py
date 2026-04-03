"""Tests for contrailtrack.prompts.projection — camera projection pipeline.

Unit tests cover MiniProjector coordinate math and geometry_to_mask rasterisation.
Integration tests use real CoCiP geometry parquets and are skipped when unavailable.

All projection tests require geographiclib, rasterio, geopandas, and pycontrails.
Tests are skipped automatically when these are not installed.
"""

import os
import unittest
import tempfile
from pathlib import Path

import numpy as np

# Integration tests use real CoCiP data. Set COCIP_DATA_DIR to point at your
# local copy (e.g. /path/to/cocip/contrails/data_model_levels/test/).
_cocip_data_dir = os.environ.get("COCIP_DATA_DIR", "")
REAL_COCIP_DIR = Path(_cocip_data_dir) if _cocip_data_dir else Path("__nonexistent__")

real_data_available = _cocip_data_dir != "" and REAL_COCIP_DIR.exists()

try:
    import geographiclib  # noqa
    import rasterio       # noqa
    import geopandas      # noqa
    geo_deps_available = True
except ImportError:
    geo_deps_available = False

try:
    from contrailtrack.prompts.projection import MiniProjector, geometry_to_mask, contrail_geometry
    from contrailtrack.prompts.projection import project_cocip_to_pixels
    projection_available = geo_deps_available
except ImportError:
    projection_available = False


# ─── MiniProjector unit tests ─────────────────────────────────────────────────

@unittest.skipUnless(projection_available, "contrailtrack.prompts.projection not importable in this env")
class TestMiniProjector(unittest.TestCase):

    def setUp(self):
        self.proj = MiniProjector()

    def test_camera_location_maps_to_image_centre(self):
        """Nadir (directly above camera) should land near the image centre."""
        # A point directly above GVCCS at flight altitude
        px, py = self.proj.lonlat_to_pixels(
            lon_deg=np.array([MiniProjector.GVCCS_LONGITUDE]),
            lat_deg=np.array([MiniProjector.GVCCS_LATITUDE]),
            alt_ft=np.array([35_000.0]),
        )
        # Should be within ±5% of image centre (512 ± 26 px)
        self.assertAlmostEqual(float(px[0]), 512, delta=30)
        self.assertAlmostEqual(float(py[0]), 512, delta=30)

    def test_lonlat_to_pixels_returns_arrays(self):
        lon = np.array([2.34, 2.35, 2.36])
        lat = np.array([48.60, 48.61, 48.62])
        alt = np.array([35_000.0] * 3)
        px, py = self.proj.lonlat_to_pixels(lon, lat, alt)
        self.assertEqual(px.shape, (3,))
        self.assertEqual(py.shape, (3,))

    def test_roundtrip_pixels_to_lonlat(self):
        """pixels_to_lonlat(lonlat_to_pixels(lon, lat, alt)) ≈ (lon, lat)."""
        lon_in = np.array([2.30, 2.35, 2.40])
        lat_in = np.array([48.58, 48.60, 48.62])
        alt_ft = np.array([35_000.0] * 3)

        px, py = self.proj.lonlat_to_pixels(lon_in, lat_in, alt_ft)
        lon_out, lat_out = self.proj.pixels_to_lonlat(px, py, alt_ft)

        np.testing.assert_allclose(lon_out, lon_in, atol=0.01)
        np.testing.assert_allclose(lat_out, lat_in, atol=0.01)

    def test_distant_point_outside_image(self):
        """A point far from the camera should project outside [0, 1024]."""
        # The camera covers a 75km square (±37.5km). A point ~60km north
        # (lat ~49.15) is clearly outside this region.
        px, py = self.proj.lonlat_to_pixels(
            lon_deg=np.array([2.35]),
            lat_deg=np.array([49.15]),
            alt_ft=np.array([35_000.0]),
        )
        # Either px or py should be well outside [0, 1024]
        outside = (px[0] < -50 or px[0] > 1074 or py[0] < -50 or py[0] > 1074)
        self.assertTrue(outside, f"Expected out-of-bounds pixel, got ({px[0]:.1f}, {py[0]:.1f})")

    def test_altitude_affects_pixel_position(self):
        """Higher altitude → different pixel position (larger zenith angle flattens)."""
        lon = np.array([2.36])
        lat = np.array([48.62])
        px_low,  py_low  = self.proj.lonlat_to_pixels(lon, lat, np.array([30_000.0]))
        px_high, py_high = self.proj.lonlat_to_pixels(lon, lat, np.array([40_000.0]))
        # Higher altitude should produce a different projection
        diff = abs(float(px_high[0]) - float(px_low[0])) + abs(float(py_high[0]) - float(py_low[0]))
        self.assertGreater(diff, 1.0)

    def test_custom_camera_location(self):
        """Custom camera position changes the projection result."""
        proj2 = MiniProjector(camera_lat=48.7, camera_lon=2.4)
        lon = np.array([2.35])
        lat = np.array([48.60])
        alt = np.array([35_000.0])
        px1, py1 = self.proj.lonlat_to_pixels(lon, lat, alt)
        px2, py2 = proj2.lonlat_to_pixels(lon, lat, alt)
        self.assertFalse(
            np.allclose([px1, py1], [px2, py2], atol=1.0),
            "Different cameras should give different pixel coordinates"
        )


# ─── geometry_to_mask unit tests ──────────────────────────────────────────────

@unittest.skipUnless(projection_available, "contrailtrack.prompts.projection not importable in this env")
class TestGeometryToMask(unittest.TestCase):

    def test_simple_polygon_rasterised(self):
        from shapely.geometry import Polygon
        poly = Polygon([(10, 10), (50, 10), (50, 30), (10, 30)])
        mask = geometry_to_mask(poly, shape=(64, 64))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(int(mask.sum()), 0)

    def test_empty_geometry_returns_zeros(self):
        from shapely.geometry import Polygon
        mask = geometry_to_mask(Polygon(), shape=(64, 64))
        self.assertTrue(np.all(mask == 0))

    def test_output_shape_matches_requested(self):
        from shapely.geometry import Polygon
        poly = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
        for h, w in [(32, 32), (64, 128), (128, 64)]:
            mask = geometry_to_mask(poly, shape=(h, w))
            self.assertEqual(mask.shape, (h, w))

    def test_polygon_outside_image_gives_zeros(self):
        from shapely.geometry import Polygon
        poly = Polygon([(200, 200), (300, 200), (300, 250), (200, 250)])
        mask = geometry_to_mask(poly, shape=(64, 64))
        self.assertTrue(np.all(mask == 0))

    def test_multipolygon_rasterised(self):
        from shapely.geometry import MultiPolygon, Polygon
        mp = MultiPolygon([
            Polygon([(5, 5), (15, 5), (15, 15), (5, 15)]),
            Polygon([(30, 30), (40, 30), (40, 40), (30, 40)]),
        ])
        mask = geometry_to_mask(mp, shape=(64, 64))
        self.assertGreater(int(mask.sum()), 0)


# ─── contrail_geometry unit tests ─────────────────────────────────────────────

@unittest.skipUnless(projection_available, "contrailtrack.prompts.projection not importable in this env")
class TestContrailGeometry(unittest.TestCase):
    """Test contrail_geometry with a minimal synthetic CoCiP-style DataFrame."""

    def _make_cocip_df(self, n: int = 4):
        """Build a minimal synthetic CoCiP contrail DataFrame."""
        import pandas as pd

        # Two waypoints for two flights at two times
        lon = np.array([2.30, 2.32, 2.34, 2.36])[:n]
        lat = np.array([48.58, 48.59, 2.60, 48.61])[:n]
        lat = np.array([48.58, 48.59, 48.60, 48.61])[:n]
        t0 = pd.Timestamp("2023-01-01 12:00:00")

        return pd.DataFrame({
            "flight_id": ["fl_a", "fl_a", "fl_b", "fl_b"][:n],
            "time": [t0, t0, t0, t0][:n],
            "waypoint": [0, 1, 0, 1][:n],
            "longitude": lon,
            "latitude": lat,
            "level": [250.0] * n,           # pressure level in hPa → ~35k ft
            "width": [500.0] * n,            # contrail width in metres
            "sin_a": [0.0] * n,
            "cos_a": [1.0] * n,
            "segment_length": [2000.0] * n,
            "formation_time": [t0] * n,
        })

    def test_returns_geodataframe(self):
        import geopandas as gpd
        df = self._make_cocip_df()
        proj = MiniProjector()
        gdf = contrail_geometry(df, proj)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)

    def test_geometry_column_is_polygon(self):
        from shapely.geometry import Polygon
        df = self._make_cocip_df()
        proj = MiniProjector()
        gdf = contrail_geometry(df, proj)
        non_empty = gdf[~gdf.geometry.is_empty]
        self.assertGreater(len(non_empty), 0)
        for geom in non_empty.geometry:
            self.assertIsInstance(geom, Polygon)

    def test_required_columns_present(self):
        df = self._make_cocip_df()
        proj = MiniProjector()
        gdf = contrail_geometry(df, proj)
        for col in ("flight_id", "time", "formation_time"):
            self.assertIn(col, gdf.columns, f"Missing column: {col}")

    def test_formation_time_inferred_from_age(self):
        """If formation_time is absent but age is present, it should be inferred."""
        import pandas as pd
        df = self._make_cocip_df()
        t0 = df["time"].iloc[0]
        df["age"] = pd.Timedelta(minutes=2)
        del df["formation_time"]
        proj = MiniProjector()
        gdf = contrail_geometry(df, proj)
        self.assertIn("formation_time", gdf.columns)
        expected = t0 - pd.Timedelta(minutes=2)
        self.assertEqual(gdf["formation_time"].iloc[0], expected)


# ─── project_cocip_to_pixels unit tests ───────────────────────────────────────

@unittest.skipUnless(projection_available, "contrailtrack.prompts.projection not importable in this env")
class TestProjectCocipToPixels(unittest.TestCase):

    def _write_synthetic_cocip_parquet(self, path: Path):
        import pandas as pd
        t0 = pd.Timestamp("2023-01-01 12:00:00")
        df = pd.DataFrame({
            "flight_id": ["fl_a", "fl_a"],
            "time": [t0, t0],
            "waypoint": [0, 1],
            "longitude": [2.30, 2.32],
            "latitude": [48.58, 48.59],
            "level": [250.0, 250.0],
            "width": [500.0, 500.0],
            "sin_a": [0.0, 0.0],
            "cos_a": [1.0, 1.0],
            "segment_length": [2000.0, 0.0],
            "formation_time": [t0, t0],
        })
        df.to_parquet(path)

    def test_creates_output_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "input"
            src.mkdir()
            self._write_synthetic_cocip_parquet(src / "20230101120000_20230101130000.parquet")
            out = tmp / "output"
            results = project_cocip_to_pixels(src, out, skip_existing=False)
            self.assertEqual(len(results), 1)
            self.assertIsNotNone(results[0])
            self.assertTrue(results[0].exists())

    def test_skip_existing_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "input"
            src.mkdir()
            self._write_synthetic_cocip_parquet(src / "20230101120000_20230101130000.parquet")
            out = tmp / "output"
            project_cocip_to_pixels(src, out, skip_existing=False)
            results = project_cocip_to_pixels(src, out, skip_existing=True)
            # Should return the existing path without reprocessing
            self.assertEqual(len(results), 1)
            self.assertIsNotNone(results[0])

    def test_output_is_readable_geodataframe(self):
        import geopandas as gpd
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "input"
            src.mkdir()
            self._write_synthetic_cocip_parquet(src / "20230101120000_20230101130000.parquet")
            out = tmp / "output"
            results = project_cocip_to_pixels(src, out, skip_existing=False)
            gdf = gpd.read_parquet(results[0])
            self.assertIn("geometry", gdf.columns)
            self.assertIn("flight_id", gdf.columns)
            self.assertIn("formation_time", gdf.columns)

    def test_processes_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "input"
            (src / "train").mkdir(parents=True)
            (src / "test").mkdir(parents=True)
            self._write_synthetic_cocip_parquet(src / "train" / "video_a.parquet")
            self._write_synthetic_cocip_parquet(src / "test" / "video_b.parquet")
            out = tmp / "output"
            results = project_cocip_to_pixels(src, out, skip_existing=False)
            self.assertEqual(len(results), 2)
            self.assertTrue((out / "train" / "video_a.parquet").exists())
            self.assertTrue((out / "test" / "video_b.parquet").exists())


# ─── integration tests ────────────────────────────────────────────────────────

@unittest.skipUnless(projection_available and real_data_available,
                     "projection deps or real CoCiP data not available")
class TestProjectionIntegration(unittest.TestCase):

    def test_real_cocip_projects_to_geodataframe(self):
        import pandas as pd
        import geopandas as gpd

        parquets = list(REAL_COCIP_DIR.glob("*.parquet"))
        self.assertGreater(len(parquets), 0, "No parquet files found in real CoCiP dir")

        df = pd.read_parquet(parquets[0])
        proj = MiniProjector()
        gdf = contrail_geometry(df, proj)

        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertIn("geometry", gdf.columns)
        self.assertIn("flight_id", gdf.columns)
        self.assertGreater(len(gdf), 0)

    @unittest.skip("slow: processes all 24 real videos (~3h) — run manually only")
    def test_real_projection_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "geom"
            results = project_cocip_to_pixels(REAL_COCIP_DIR, out, skip_existing=False)
            processed = [r for r in results if r is not None]
            self.assertGreater(len(processed), 0)
            for p in processed:
                self.assertTrue(p.exists())


if __name__ == "__main__":
    unittest.main()
