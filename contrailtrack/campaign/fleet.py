"""Convert OpenSky ADS-B parquet to pycontrails Fleet JSON."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger()

OPENSKY_COLUMN_MAP: dict[str, str] = {
    "timestamp": "time",
    "typecode": "aircraft_type",
    "altitude": "altitude_ft",
}


def opensky_to_fleet(
    parquet_path: Path,
    start: datetime,
    stop: datetime,
    *,
    resample_interval: pd.Timedelta = pd.Timedelta(seconds=10),
    level_min_hpa: float = 150.0,
    level_max_hpa: float = 400.0,
) -> list[dict]:
    """Convert OpenSky ADS-B parquet to a list of pycontrails Flight dicts.

    Filters by time window and cruise altitude, resamples to a uniform grid.
    """
    from pycontrails import Flight
    from pycontrails.physics import units

    df = pd.read_parquet(parquet_path)
    df = df.rename(columns=OPENSKY_COLUMN_MAP)

    # Normalise timestamps to tz-naive datetime64[ns] (pycontrails requirement).
    # OpenSky parquets use PyArrow-backed tz-aware timestamps.
    import numpy as np
    df["time"] = pd.to_datetime(np.array(df["time"], dtype="datetime64[ns]"))

    # Filter to time window
    df = df.loc[(df["time"] >= pd.Timestamp(start)) & (df["time"] <= pd.Timestamp(stop))]
    if df.empty:
        return []

    df = df.sort_values(["flight_id", "time"]).reset_index(drop=True)

    flights: list[dict] = []
    for fid, group in df.groupby("flight_id"):
        if "altitude_ft" not in group.columns or group["altitude_ft"].isna().all():
            continue

        attrs = {
            "flight_id": fid,
            "callsign": group["callsign"].iloc[0] if "callsign" in group.columns else "",
            "registration": group["registration"].iloc[0] if "registration" in group.columns else "",
            "aircraft_type": group["aircraft_type"].iloc[0] if "aircraft_type" in group.columns else "",
        }

        data = {
            "longitude": group["longitude"].values,
            "latitude": group["latitude"].values,
            "altitude_ft": group["altitude_ft"].values,
            "time": group["time"].values,
        }

        try:
            flight = Flight(data=data, attrs=attrs)
        except Exception:
            continue

        level = units.ft_to_pl(flight["altitude_ft"])
        flight = flight.filter((level < level_max_hpa) & (level >= level_min_hpa))
        if flight.size == 0:
            continue

        flight = flight.resample_and_fill(resample_interval)
        if flight.size == 0:
            continue

        flights.append(flight.to_dict())

    log.info("fleet_converted", n_flights=len(flights), window=f"{start}–{stop}")
    return flights


def save_fleet_json(flights: list[dict], output_path: Path) -> Path:
    """Save fleet dicts as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(flights, indent=2))
    return output_path
