"""Load local ERA5 pressure-level NetCDF files as a pycontrails MetDataset."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import structlog

log = structlog.get_logger()


def load_local_era5(
    era5_dir: Path,
    dates: list[dt.date],
    variables: tuple[str, ...] = ("t", "u", "v", "w"),
) -> "MetDataset":
    """Load local ERA5 pressure-level files and return a pycontrails MetDataset.

    The local files use CDS API v2 naming (``valid_time``, ``pressure_level``,
    ``number``, ``expver``). This function renames coordinates to the
    pycontrails convention and selects only the requested variables.
    """
    import xarray as xr
    from pycontrails import MetDataset

    era5_dir = Path(era5_dir)
    paths = []
    for d in dates:
        p = era5_dir / f"{d:%Y_%m_%d}.pl.nc"
        if not p.exists():
            raise FileNotFoundError(f"ERA5 file not found: {p}")
        paths.append(p)

    log.info("loading_local_era5", files=[p.name for p in paths])

    ds = xr.open_mfdataset(paths, combine="by_coords", engine="netcdf4")

    # Squeeze singleton dimensions
    if "number" in ds.dims:
        ds = ds.isel(number=0, drop=True)
    elif "number" in ds.coords:
        ds = ds.drop_vars("number")

    if "expver" in ds.dims:
        ds = ds.isel(expver=0, drop=True)
    elif "expver" in ds.coords:
        ds = ds.drop_vars("expver")

    # Rename to pycontrails convention
    rename = {}
    if "valid_time" in ds.dims:
        rename["valid_time"] = "time"
    if "pressure_level" in ds.dims:
        rename["pressure_level"] = "level"
    if rename:
        ds = ds.rename(rename)

    # Select requested variables
    ds = ds[list(variables)]

    # Rename short names to standard names expected by pycontrails
    standard_names = {
        "t": "air_temperature",
        "u": "eastward_wind",
        "v": "northward_wind",
        "w": "lagrangian_tendency_of_air_pressure",
        "q": "specific_humidity",
        "z": "geopotential",
        "ciwc": "specific_cloud_ice_water_content",
    }
    rename_vars = {k: standard_names[k] for k in ds.data_vars if k in standard_names}
    if rename_vars:
        ds = ds.rename(rename_vars)

    met = MetDataset(ds)
    log.info(
        "era5_loaded",
        variables=list(ds.data_vars),
        levels=list(ds["level"].values),
        time_steps=len(ds["time"]),
    )
    return met
