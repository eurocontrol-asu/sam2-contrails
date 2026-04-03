"""Run CoCiP (Contrail Cirrus Prediction) on ADS-B fleet data.

This module wraps the pycontrails CoCiP model with the settings validated
for the GVCCS contrail detection project:
  - ERA5 model-level meteorology (not pressure levels)
  - HistogramMatching humidity scaling
  - PSFlight aircraft performance model
  - SAC and persistence filters disabled (to capture forming contrails)

The output contrail parquets are in lat/lon space. Camera projection to
pixel coordinates is handled in-memory by generate_prompts() and
generate_prompts_video() in prompts/writer.py — no separate projection
step is required.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# Default CoCiP integration params
_DEFAULT_PARAMS = {
    "dt_integration": pd.Timedelta(seconds=30),
    "filter_sac": False,
    "filter_initially_persistent": False,
    "downselect_met": True,
    "humidity_scaling": None,  # overridden in run_cocip with HistogramMatching
    "aircraft_performance": None,  # overridden with PSFlight
}

# ERA5 model level range used for the met dataset
_MODEL_LEVELS = range(70, 91)
_PRESSURE_LEVELS = np.arange(170, 400, 10)


def run_cocip(
    fleet_json: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path = None,
    grid: float = 1.0,
    extra_params: dict = None,
) -> Path:
    """Run CoCiP on a single fleet JSON file.

    Downloads ERA5 met/rad data as needed (cached to cache_dir), runs CoCiP,
    and saves the contrail GeoDataFrame as a parquet file.

    Args:
        fleet_json: Path to fleet JSON file containing a list of Flight dicts.
                    File stem must be of the form YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.
        output_dir: Directory to save the output contrail parquet.
        cache_dir: Directory for ERA5 disk cache. Defaults to output_dir/../met_cache/.
        grid: ERA5 horizontal resolution in degrees (1.0 = coarse, 0.25 = fine).
        extra_params: Override any default CoCiP params (merged into defaults).

    Returns:
        Path to the output contrail parquet file, or None if CoCiP produced no contrails.
    """
    from pycontrails import Fleet, Flight
    from pycontrails import DiskCacheStore
    from pycontrails.datalib.ecmwf import ERA5, ERA5ModelLevel
    from pycontrails.models.cocip import Cocip
    from pycontrails.models.humidity_scaling import HistogramMatching
    from pycontrails.models.ps_model import PSFlight

    fleet_json = Path(fleet_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = output_dir.parent / "met_cache"

    # Parse video time window from filename
    stem = fleet_json.stem
    start_str, stop_str = stem.split("_")
    start = datetime.strptime(start_str, "%Y%m%d%H%M%S")
    stop = datetime.strptime(stop_str, "%Y%m%d%H%M%S")
    max_age = pd.Timedelta(stop - start)

    output_path = output_dir / fleet_json.with_suffix(".parquet").name
    source_path = output_dir / f"{fleet_json.stem}_source.parquet"

    # Load fleet
    with open(fleet_json) as fh:
        data = json.load(fh)
    fleet = Fleet.from_seq([Flight.from_dict(d) for d in data])

    # ERA5 time window: from first waypoint to last waypoint + 10h (contrail persistence)
    t_min = pd.to_datetime(fleet["time"].min()).floor("h")
    t_max = pd.to_datetime(fleet["time"].max()).ceil("h") + pd.Timedelta("10h")
    time = (t_min, t_max)

    cache = DiskCacheStore(cache_dir=str(cache_dir), allow_clear=True)

    # ERA5 model-level met
    era5ml = ERA5ModelLevel(
        time=time,
        variables=("t", "q", "u", "v", "w", "ciwc"),
        grid=grid,
        model_levels=_MODEL_LEVELS,
        pressure_levels=_PRESSURE_LEVELS,
        cachestore=cache,
    )

    # ERA5 surface radiation
    era5sl = ERA5(
        time=time,
        variables=Cocip.rad_variables,
        cachestore=cache,
    )

    met = era5ml.open_metdataset()
    rad = era5sl.open_metdataset()

    params = dict(_DEFAULT_PARAMS)
    params["humidity_scaling"] = HistogramMatching()
    params["aircraft_performance"] = PSFlight()
    params["persistent_buffer"] = max_age
    params["max_age"] = max_age
    if extra_params:
        params.update(extra_params)

    cocip = Cocip(met=met, rad=rad, params=params)
    cocip.eval(source=fleet)

    contrails = cocip.contrail
    if contrails is None or contrails.empty:
        return None

    # cocip.contrail is a plain DataFrame in pycontrails >= 0.50
    df = contrails.dataframe if hasattr(contrails, "dataframe") else contrails
    df.to_parquet(output_path)

    source = cocip.source
    if source is not None:
        source.dataframe.to_parquet(source_path)

    return output_path


def run_cocip_batch(
    fleet_dir: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path = None,
    grid: float = 1.0,
    extra_params: dict = None,
    skip_existing: bool = True,
) -> list:
    """Run CoCiP on all fleet JSON files in a directory.

    Args:
        fleet_dir: Directory containing fleet JSON files.
        output_dir: Directory to save output contrail parquets.
        cache_dir: ERA5 cache directory.
        grid: ERA5 horizontal resolution.
        extra_params: Additional CoCiP params.
        skip_existing: Skip files where the output parquet already exists.

    Returns:
        List of output parquet paths (None entries for videos with no contrails).
    """
    fleet_dir = Path(fleet_dir)
    output_dir = Path(output_dir)
    fleet_files = sorted(fleet_dir.glob("*.json"))

    results = []
    for fleet_json in tqdm(fleet_files, desc="Running CoCiP"):
        out_path = output_dir / fleet_json.with_suffix(".parquet").name
        if skip_existing and out_path.exists():
            results.append(out_path)
            continue
        result = run_cocip(
            fleet_json=fleet_json,
            output_dir=output_dir,
            cache_dir=cache_dir,
            grid=grid,
            extra_params=extra_params,
        )
        results.append(result)

    return results


def load_fleet_json(path: str | Path) -> "Fleet":
    """Load a fleet JSON file into a pycontrails Fleet object."""
    from pycontrails import Fleet, Flight

    path = Path(path)
    with open(path) as fh:
        data = json.load(fh)
    return Fleet.from_seq([Flight.from_dict(d) for d in data])
