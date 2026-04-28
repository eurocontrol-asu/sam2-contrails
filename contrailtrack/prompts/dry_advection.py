"""Run DryAdvection on ADS-B fleet data to generate contrail trajectories.

DryAdvection is a lighter alternative to CoCiP for generating contrail prompts:

* **Faster**: requires only ERA5 wind fields (T, u, v, w) — no radiation data,
  no humidity scaling, no aircraft performance model.
* **Simpler**: no persistence prediction; every flight produces output.
* **Trade-off**: no thermodynamic filtering, no EF estimate. Use CoCiP when
  accurate persistence/lifetime prediction matters.

The output parquet format is identical to ``run_cocip``, so downstream steps
(``generate_prompts``, ``generate_prompts_video``) accept both without changes.

References
----------
pycontrails DryAdvection:
    https://py.contrails.org/api/pycontrails.models.dry_advection.html
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ERA5 model-level range (same as CoCiP for consistency)
_MODEL_LEVELS  = range(70, 91)
_PRESSURE_LEVELS = np.arange(170, 400, 10)

# Default initial contrail width in metres.
# DryAdvection will evolve this with wind shear over time.
_DEFAULT_INITIAL_WIDTH_M = 400.0


def run_dry_advection(
    fleet_json: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    grid: float = 1.0,
    initial_width_m: float = _DEFAULT_INITIAL_WIDTH_M,
    extra_params: dict | None = None,
    met: "MetDataset | None" = None,
    max_age: "pd.Timedelta | None" = None,
) -> Path | None:
    """Run DryAdvection on a single fleet JSON file.

    Downloads ERA5 wind data as needed (cached to ``cache_dir``), advects each
    flight waypoint forward, and saves the resulting trajectory DataFrame as a
    parquet file with the same schema as CoCiP output.

    Args:
        fleet_json: Path to the fleet JSON file.  When ``max_age`` is not
            provided, the file stem must follow the
            ``YYYYMMDDHHMMSS_YYYYMMDDHHMMSS`` naming convention so that the
            simulation duration can be inferred.
        output_dir: Directory to write the output parquet.
        cache_dir: ERA5 disk-cache directory.  Defaults to
            ``output_dir/../met_cache``.
        grid: ERA5 horizontal resolution in degrees (``1.0`` = coarse,
            ``0.25`` = fine).  Finer grids improve trajectory accuracy but
            increase download time and memory use.
        initial_width_m: Initial contrail plume width in metres.  This is
            evolved by the wind-shear model at each time step.
        extra_params: Additional ``DryAdvectionParams`` overrides (merged into
            the defaults).
        met: Pre-loaded MetDataset.  When provided, skips ERA5 download.
        max_age: Maximum simulation age.  When not provided, inferred from
            the fleet JSON filename (stop − start).

    Returns:
        Path to the output parquet, or ``None`` if DryAdvection produced no
        output (e.g. all waypoints out of the met domain).
    """
    from pycontrails import Fleet, Flight
    from pycontrails.models.dry_advection import DryAdvection

    fleet_json = Path(fleet_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = output_dir.parent / "met_cache"

    if max_age is None:
        # Parse simulation time window from the filename stem
        stem = fleet_json.stem
        start_str, stop_str = stem.split("_")
        start  = datetime.strptime(start_str, "%Y%m%d%H%M%S")
        stop   = datetime.strptime(stop_str,  "%Y%m%d%H%M%S")
        max_age = pd.Timedelta(stop - start)

    output_path = output_dir / fleet_json.with_suffix(".parquet").name

    # Load fleet
    with open(fleet_json) as fh:
        data = json.load(fh)
    flights = [Flight.from_dict(d) for d in data]
    fleet = Fleet.from_seq(flights)

    # Tag each waypoint with its formation time (preserved through advection)
    # and compute segment azimuth to enable wind-shear effects.
    fleet["formation_time"] = fleet["time"]
    fleet["azimuth"] = fleet.segment_azimuth()

    if met is None:
        from pycontrails import DiskCacheStore
        from pycontrails.datalib.ecmwf import ERA5ModelLevel

        # ERA5 time window: first waypoint → last waypoint + max simulation age
        t_min = pd.to_datetime(fleet["time"].min()).floor("h")
        t_max = pd.to_datetime(fleet["time"].max()).ceil("h") + max_age
        time  = (t_min, t_max)

        # ERA5 model-level met — wind fields only (no radiation, no CIWC)
        cache  = DiskCacheStore(cache_dir=str(cache_dir), allow_clear=True)
        era5ml = ERA5ModelLevel(
            time=time,
            variables=("t", "u", "v", "w"),
            grid=grid,
            model_levels=_MODEL_LEVELS,
            pressure_levels=_PRESSURE_LEVELS,
            cachestore=cache,
        )
        met = era5ml.open_metdataset()

    params: dict = {
        "dt_integration": pd.Timedelta(seconds=30),
        "max_age": max_age,
        "width": initial_width_m,
        "downselect_met": True,
    }
    if extra_params:
        params.update(extra_params)

    model  = DryAdvection(met=met, params=params)
    result = model.eval(source=fleet)

    if result is None or len(result) == 0:
        return None

    df = result.dataframe if hasattr(result, "dataframe") else pd.DataFrame(result.data)

    if df.empty:
        return None

    # Ensure formation_time is present (DryAdvection preserves custom columns)
    if "formation_time" not in df.columns:
        df["formation_time"] = df["time"]

    df.to_parquet(output_path)
    return output_path


def run_dry_advection_batch(
    fleet_dir: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    grid: float = 1.0,
    initial_width_m: float = _DEFAULT_INITIAL_WIDTH_M,
    extra_params: dict | None = None,
    skip_existing: bool = True,
) -> list:
    """Run DryAdvection on all fleet JSON files in a directory.

    Args:
        fleet_dir: Directory containing fleet JSON files.
        output_dir: Directory to write output parquets.
        cache_dir: ERA5 cache directory.
        grid: ERA5 horizontal resolution in degrees.
        initial_width_m: Initial plume width in metres.
        extra_params: Additional ``DryAdvectionParams`` overrides.
        skip_existing: Skip files where the output parquet already exists.

    Returns:
        List of output parquet paths (``None`` entries where DryAdvection
        produced no contrails).
    """
    fleet_dir  = Path(fleet_dir)
    output_dir = Path(output_dir)
    fleet_files = sorted(fleet_dir.glob("*.json"))

    results = []
    for fleet_json in tqdm(fleet_files, desc="Running DryAdvection"):
        out_path = output_dir / fleet_json.with_suffix(".parquet").name
        if skip_existing and out_path.exists():
            results.append(out_path)
            continue
        result = run_dry_advection(
            fleet_json=fleet_json,
            output_dir=output_dir,
            cache_dir=cache_dir,
            grid=grid,
            initial_width_m=initial_width_m,
            extra_params=extra_params,
        )
        results.append(result)

    return results
