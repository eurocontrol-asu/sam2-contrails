"""Convert a GVCCS ADS-B traffic parquet to a pycontrails Fleet JSON.

Reads raw ADS-B waypoints from the GVCCS parquet, filters by aircraft type
and cruise altitude, resamples to a uniform grid, and writes one JSON per
flight suitable for CoCiP or DryAdvection.

Usage::

    # Convert the first video's traffic data
    uv run python examples/02_fleet_from_traffic.py data/parquet/20230930055430_20230930075430.parquet

    # Custom output path
    uv run python examples/02_fleet_from_traffic.py data/parquet/20230930055430_20230930075430.parquet -o data/fleet/vid1.json

    # Keep all aircraft types (not just those supported by PSFlight)
    uv run python examples/02_fleet_from_traffic.py data/parquet/20230930055430_20230930075430.parquet --no-filter-ps-types
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import structlog
import typer

log = structlog.get_logger()

app = typer.Typer(add_completion=False)

# GVCCS parquet column names → contrailtrack conventions
COLUMN_MAP: dict[str, str] = {
    "FLIGHT_ID": "flight_id",
    "CALL_SIGN": "callsign",
    "REGISTRATION": "registration",
    "ICAO_TYPE": "aircraft_type",
    "TIMESTAMP_S": "time",
    "LONGITUDE": "longitude",
    "LATITUDE": "latitude",
    "ALTI_FT": "altitude_ft",
}


def parquet_to_fleet(
    parquet_path: Path,
    *,
    resample_interval: pd.Timedelta = pd.Timedelta(seconds=10),
    filter_ps_types: bool = True,
    level_min_hpa: float = 150.0,
    level_max_hpa: float = 400.0,
) -> list[dict]:
    """Convert a GVCCS ADS-B parquet to a list of pycontrails Flight dicts."""
    from pycontrails import Flight
    from pycontrails.physics import units

    df = pd.read_parquet(parquet_path)
    df = df.rename(columns=COLUMN_MAP)
    df = df.sort_values(["flight_id", "time"]).reset_index(drop=True)

    if filter_ps_types:
        from pycontrails.models.ps_model import PSFlight

        ps_types = set(PSFlight().synonym_dict.keys())
    else:
        ps_types = None

    flights: list[dict] = []
    for fid, group in df.groupby("flight_id"):
        attrs = {
            "flight_id": fid,
            "callsign": group["callsign"].iloc[0],
            "registration": group["registration"].iloc[0],
            "aircraft_type": group["aircraft_type"].iloc[0],
        }

        if ps_types is not None and attrs["aircraft_type"] not in ps_types:
            continue

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

    return flights


@app.command()
def main(
    parquet: Annotated[
        Path, typer.Argument(help="Path to a GVCCS ADS-B traffic parquet.")
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output fleet JSON. Default: data/fleet/<stem>.json"),
    ] = None,
    filter_ps_types: Annotated[
        bool,
        typer.Option(help="Keep only aircraft types supported by PSFlight."),
    ] = True,
    level_min_hpa: Annotated[
        float, typer.Option(help="Lower pressure bound in hPa (higher altitude).")
    ] = 150.0,
    level_max_hpa: Annotated[
        float, typer.Option(help="Upper pressure bound in hPa (lower altitude).")
    ] = 400.0,
    resample_s: Annotated[
        float, typer.Option(help="Resampling interval in seconds.")
    ] = 10.0,
) -> None:
    """Convert a GVCCS ADS-B parquet to a pycontrails Fleet JSON."""
    if not parquet.exists():
        log.error("parquet_not_found", path=str(parquet))
        raise typer.Exit(code=1)

    out = output or Path(f"data/fleet/{parquet.stem}.json")

    log.info("reading_parquet", path=str(parquet))
    flights = parquet_to_fleet(
        parquet,
        resample_interval=pd.Timedelta(seconds=resample_s),
        filter_ps_types=filter_ps_types,
        level_min_hpa=level_min_hpa,
        level_max_hpa=level_max_hpa,
    )

    if not flights:
        log.warning("no_flights_passed_filters")
        raise typer.Exit(code=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(flights, indent=2))

    log.info("fleet_written", n_flights=len(flights), path=str(out))


if __name__ == "__main__":
    app()
