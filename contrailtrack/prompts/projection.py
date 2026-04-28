"""Camera projection utilities: lat/lon → pixel-space conversions.

Core components
---------------
MiniProjector
    Converts geodetic (lat, lon, alt) coordinates to camera pixel (u, v) coordinates
    using a pinhole model with the GVCCS camera constants as defaults.

contrail_geometry(df, projector)
    Converts a raw CoCiP or DryAdvection DataFrame (lat/lon space) to a
    pixel-space GeoDataFrame ready for use by ``generate_prompts`` and
    ``generate_prompts_video``.  This is the primary projection entry-point.

project_cocip_to_pixels(cocip_dir, output_dir, projector)
    Batch helper that projects an entire directory of CoCiP parquets to
    geometry parquets on disk.  Retained for offline pre-computation; in the
    standard pipeline the in-memory path via ``contrail_geometry`` is preferred.

geometry_to_mask
    Rasterise a Shapely geometry to a binary NumPy mask.

MiniProjector and geometry_to_mask are adapted from the trailvision library.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MiniProjector — geometric camera projection (no pre-computed azimuth grid)
# Adapted from trailvision.transform.projection_analytical.MiniProjector
# ---------------------------------------------------------------------------

class MiniProjector:
    """Convert aircraft positions (lon, lat, alt_ft) to image pixel coordinates.

    Uses a simple geometric projection based on azimuth/zenith angles computed
    from the WGS84 geodesic. No look-up table required — calibration constants
    are embedded for the GVCCS all-sky camera at EUROCONTROL Brétigny.

    Default constants match the GVCCS camera; pass keyword args to override for
    other camera sites.
    """

    GVCCS_LONGITUDE = 2.3467954996250784
    GVCCS_LATITUDE  = 48.600518087374105
    GVCCS_HEIGHT_M  = 90.0  # camera height above ground (m)

    def __init__(
        self,
        camera_lat: float = GVCCS_LATITUDE,
        camera_lon: float = GVCCS_LONGITUDE,
        camera_alt_m: float = GVCCS_HEIGHT_M,
        cloud_height_m: float = 10_000.0,
        square_size_m: float = 75_000.0,
        resolution: int = 1024,
    ):
        """
        Args:
            camera_lat: Camera latitude (degrees).
            camera_lon: Camera longitude (degrees).
            camera_alt_m: Camera altitude above sea level (metres).
            cloud_height_m: Reference projection altitude (metres). Contrails are
                            projected to this plane for the zenith/tangent calc.
            square_size_m: Side length of the square coverage area in metres.
                           The image covers ±square_size/2 around the camera.
            resolution: Image resolution in pixels (H = W = resolution).
        """
        from geographiclib.geodesic import Geodesic

        self.camera_lat = camera_lat
        self.camera_lon = camera_lon
        self.camera_alt_m = camera_alt_m
        self.cloud_height_m = cloud_height_m
        self.square_size_m = square_size_m
        self.resolution = resolution

        self.half = square_size_m / 2.0
        self.step = square_size_m / (resolution - 1)
        self.geod = Geodesic.WGS84

    # ------------------------------------------------------------------
    # Low-level coordinate conversions
    # ------------------------------------------------------------------

    def _ft_to_m(self, ft: NDArray) -> NDArray:
        return np.asarray(ft) * 0.3048

    _R_EARTH = 6_371_000.0  # mean Earth radius (m) — <1 px error at 75 km

    def _azimuth_zenith(
        self,
        lat: NDArray,
        lon: NDArray,
        alt_ft: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Return azimuth (deg, geographic) and zenith (deg) from camera to target.

        Uses a flat-earth approximation (vectorised numpy) instead of per-point
        geodesic calls.  Error is < 0.1 pixel at the 75 km camera range.
        """
        alt_m = self._ft_to_m(np.asarray(alt_ft))
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))

        dlat = np.radians(lat - self.camera_lat)
        dlon = np.radians(lon - self.camera_lon)
        cos_cam = np.cos(np.radians(self.camera_lat))

        dy = dlat * self._R_EARTH              # north–south  (m)
        dx = dlon * cos_cam * self._R_EARTH     # east–west    (m)
        s_ground = np.sqrt(dx ** 2 + dy ** 2)

        # azimuth: clockwise from North (same convention as Geodesic.Inverse)
        azi1 = np.degrees(np.arctan2(dx, dy))

        dz = alt_m - self.camera_alt_m
        straight = np.sqrt(s_ground ** 2 + dz ** 2)
        elevation = np.degrees(np.arcsin(dz / straight))
        zenith = 90.0 - elevation
        return azi1, zenith

    def _azzen_to_xy(
        self,
        azimuth_deg: NDArray,
        zenith_deg: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Azimuth/zenith → flat projection plane (metres)."""
        az = np.radians(azimuth_deg)
        ze = np.radians(zenith_deg)
        r = self.cloud_height_m * np.tan(ze)
        x = r * np.cos(az)
        y = r * np.sin(az)
        # Map so that North is up and East is right, centred at camera
        return y + self.half, self.half - x

    def _xy_to_azzen(
        self,
        x: NDArray,
        y: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Flat projection plane (metres) → azimuth/zenith (deg)."""
        x_c = self.half - y
        y_c = x - self.half
        r = np.sqrt(x_c ** 2 + y_c ** 2)
        az = np.degrees(np.arctan2(y_c, x_c))
        ze = np.degrees(np.arctan(r / self.cloud_height_m))
        return az, ze

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lonlat_to_pixels(
        self,
        lon_deg: NDArray,
        lat_deg: NDArray,
        alt_ft: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Convert (longitude, latitude, altitude_ft) → pixel (x, y).

        Pixels outside the camera field of view may be outside [0, resolution].
        """
        az, ze = self._azimuth_zenith(
            np.asarray(lat_deg),
            np.asarray(lon_deg),
            np.asarray(alt_ft),
        )
        x_m, y_m = self._azzen_to_xy(az, ze)
        px = x_m / self.step
        py = y_m / self.step
        return np.asarray(px), np.asarray(py)

    def pixels_to_lonlat(
        self,
        px: NDArray,
        py: NDArray,
        alt_ft: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Convert pixel (x, y, altitude_ft) → (longitude, latitude)."""
        x_m = np.asarray(px) * self.step
        y_m = np.asarray(py) * self.step
        az, ze = self._xy_to_azzen(x_m, y_m)
        alt_m = self._ft_to_m(np.asarray(alt_ft))

        elevation_rad = np.radians(90.0 - ze)
        dz = alt_m - self.camera_alt_m
        dist = dz / np.tan(elevation_rad)

        direct = [
            self.geod.Direct(self.camera_lat, self.camera_lon, a, d)
            for a, d in zip(np.atleast_1d(az), np.atleast_1d(dist))
        ]
        lon2 = np.array([r["lon2"] for r in direct])
        lat2 = np.array([r["lat2"] for r in direct])
        return lon2, lat2


# ---------------------------------------------------------------------------
# geometry_to_mask — rasterise a shapely geometry into a boolean mask
# Adapted from trailvision.transform.mask.geometry_to_mask
# ---------------------------------------------------------------------------

def geometry_to_mask(
    geometry,
    shape: Tuple[int, int],
    transform=None,
) -> NDArray:
    """Rasterise a Shapely geometry into a uint8 mask.

    Args:
        geometry: Shapely geometry in pixel coordinates.
        shape: (H, W) output shape.
        transform: Affine transform. Defaults to identity (pixel coords = image coords).

    Returns:
        uint8 ndarray of shape (H, W), 1 inside the geometry, 0 outside.
    """
    from rasterio import features
    from rasterio.transform import IDENTITY

    if transform is None:
        transform = IDENTITY

    if geometry.is_empty:
        return np.zeros(shape, dtype=np.uint8)

    if hasattr(geometry, "geoms"):
        shapes = [(g, 1) for g in geometry.geoms if not g.is_empty]
    else:
        shapes = [(geometry, 1)]

    if not shapes:
        return np.zeros(shape, dtype=np.uint8)

    return features.rasterize(
        shapes=shapes,
        out_shape=shape,
        fill=0,
        dtype=np.uint8,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# contrail_geometry — project a CoCiP parquet DF to pixel-space GeoDataFrame
# Adapted from trailvision.data.cocip.contrail_geometry
# ---------------------------------------------------------------------------

_GDF_COLUMNS = ["flight_id", "time", "altitude_ft", "waypoint", "width", "formation_time"]


def contrail_geometry(df, projector: MiniProjector):
    """Convert a CoCiP contrail DataFrame to a pixel-space GeoDataFrame.

    Args:
        df: DataFrame with CoCiP contrail output. Required columns:
            longitude, latitude, level (pressure hPa), flight_id, time, waypoint,
            width (metres), formation_time (or age).
        projector: Camera projector (e.g. MiniProjector).

    Returns:
        GeoDataFrame with columns [flight_id, time, altitude_ft, waypoint, width,
        formation_time, geometry] where geometry is a Shapely Polygon in pixel
        coordinates (CRS=None).
    """
    import geopandas as gpd
    from shapely.geometry import Polygon
    from pycontrails.physics import units, geo
    from pycontrails.models.cocip.contrail_properties import contrail_vertices

    df = df.copy()

    # altitude_ft from pressure level
    df["altitude_ft"] = units.pl_to_ft(df["level"])

    # clean and sort
    df.dropna(subset=["longitude", "latitude", "altitude_ft"], inplace=True)
    df.sort_values(["flight_id", "time", "waypoint"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ensure formation_time
    if "formation_time" not in df.columns:
        if "age" in df.columns:
            df["formation_time"] = df["time"] - df["age"]
        else:
            df["formation_time"] = df["time"]

    last_indices = df.groupby(["flight_id", "time"]).waypoint.idxmax()

    # segment geometry
    if "segment_length" not in df.columns:
        df["segment_length"] = geo.segment_length(
            df["longitude"].values,
            df["latitude"].values,
            units.ft_to_m(df["altitude_ft"]).values,
        )
        df.loc[last_indices.values, "segment_length"] = 0

    if "sin_a" not in df.columns or "cos_a" not in df.columns:
        df["sin_a"], df["cos_a"] = geo.segment_angle(
            df["longitude"].values,
            df["latitude"].values,
        )
        df.loc[last_indices.values, ["sin_a", "cos_a"]] = 0

    # compute 4 polygon corners in lon/lat
    x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = contrail_vertices(
        lon=df["longitude"].values,
        lat=df["latitude"].values,
        sin_a=df["sin_a"].values,
        cos_a=df["cos_a"].values,
        width=df["width"].values,
        segment_length=df["segment_length"].values,
    )

    alt = df["altitude_ft"].values

    # project corners to pixel space
    px1, py1 = projector.lonlat_to_pixels(x_1, y_1, alt)
    px2, py2 = projector.lonlat_to_pixels(x_2, y_2, alt)
    px3, py3 = projector.lonlat_to_pixels(x_3, y_3, alt)
    px4, py4 = projector.lonlat_to_pixels(x_4, y_4, alt)

    coords = np.stack(
        [
            np.column_stack((px1, py1)),
            np.column_stack((px2, py2)),
            np.column_stack((px3, py3)),
            np.column_stack((px4, py4)),
        ],
        axis=1,
    )

    geometry = [
        Polygon(c) if not np.isnan(c).any() else Polygon()
        for c in coords
    ]

    keep = [c for c in _GDF_COLUMNS if c in df.columns]
    return gpd.GeoDataFrame(df[keep], geometry=geometry)


# ---------------------------------------------------------------------------
# project_cocip_to_pixels — batch processing
# ---------------------------------------------------------------------------

def project_cocip_to_pixels(
    cocip_dir: str | Path,
    output_dir: str | Path,
    projector: MiniProjector = None,
    glob_pattern: str = "*.parquet",
    skip_existing: bool = True,
) -> list:
    """Convert a directory of CoCiP lat/lon parquets to pixel-space geometry GDFs.

    This is the step between ``run_cocip`` and ``generate_prompts``:
      run_cocip → [this function] → generate_prompts

    Args:
        cocip_dir: Directory of CoCiP output parquets (lat/lon space).
                   May contain subdirectories (e.g. train/, test/) — these are
                   processed recursively one level deep.
        output_dir: Root output directory. Subdirectory structure mirrors cocip_dir.
        projector: Camera projector. Defaults to MiniProjector() with GVCCS constants.
        glob_pattern: Glob pattern for input files (default: ``*.parquet``).
        skip_existing: Skip files whose output parquet already exists.

    Returns:
        List of output parquet Paths (None for files that failed or produced empty GDFs).
    """
    import pandas as pd

    if projector is None:
        projector = MiniProjector()

    cocip_dir = Path(cocip_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files — support one level of subdirectory (train/test splits)
    files = list(cocip_dir.glob(glob_pattern))
    subdirs = [d for d in cocip_dir.iterdir() if d.is_dir()]

    tasks = []
    for f in files:
        tasks.append((f, output_dir / f.name))
    for sub in subdirs:
        sub_out = output_dir / sub.name
        sub_out.mkdir(parents=True, exist_ok=True)
        for f in sub.glob(glob_pattern):
            tasks.append((f, sub_out / f.name))

    results = []
    for src, dst in tqdm(tasks, desc="Projecting contrails"):
        if skip_existing and dst.exists():
            results.append(dst)
            continue
        try:
            df = pd.read_parquet(src)
            if df.empty:
                results.append(None)
                continue
            gdf = contrail_geometry(df, projector)
            gdf.to_parquet(dst)
            results.append(dst)
        except Exception as e:
            log.warning("%s: %s", src.name, e)
            results.append(None)

    return results
