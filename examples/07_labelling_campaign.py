"""Run the labelling-campaign pipeline for one or more dates.

Computes sunrise/sunset for the camera location (Brétigny, France), splits each
day into 2-hour video windows, and for each window:

1. Creates symlinked frame directories (stripping ``image_`` prefix)
2. Converts OpenSky ADS-B traffic → pycontrails Fleet JSON
3. Runs DryAdvection with local ERA5 → contrail trajectories
4. Generates 5-minute age-weighted prompt PNGs
5. Runs SAM2 ternary inference → COCO RLE JSON predictions

Usage::

    # Single day
    uv run python examples/07_labelling_campaign.py 2025-12-31

    # Date range
    uv run python examples/07_labelling_campaign.py 2025-12-29:2025-12-31

    # Without inference (prompts only)
    uv run python examples/07_labelling_campaign.py 2025-12-31 --no-inference

    # Single continuous video (no 2h window splits, no boundary discontinuities)
    uv run python examples/07_labelling_campaign.py 2025-12-31 --single-video

    # Custom output directory
    uv run python examples/07_labelling_campaign.py 2025-12-31 --output /data/contrailnet/sam2
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import structlog
import typer

log = structlog.get_logger()

app = typer.Typer(add_completion=False)

BRETIGNY_LAT = 48.600518
BRETIGNY_LON = 2.346795
BRETIGNY_ALT_M = 90.0


def _parse_dates(date_str: str) -> list[dt.date]:
    """Parse a date or date range (YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD)."""
    if ":" in date_str:
        start_s, end_s = date_str.split(":")
        start = dt.date.fromisoformat(start_s)
        end = dt.date.fromisoformat(end_s)
        dates = []
        d = start
        while d <= end:
            dates.append(d)
            d += dt.timedelta(days=1)
        return dates
    return [dt.date.fromisoformat(date_str)]


@app.command()
def main(
    date: Annotated[
        str,
        typer.Argument(help="Date (YYYY-MM-DD) or range (YYYY-MM-DD:YYYY-MM-DD)."),
    ],
    output: Annotated[
        Path, typer.Option(help="Root output directory."),
    ] = Path("/data/contrailnet/sam2"),
    images_base: Annotated[
        Path, typer.Option(help="Base path for camera images."),
    ] = Path("/data/contrailnet/images/camera/visible/proj"),
    flights_base: Annotated[
        Path, typer.Option(help="Base path for OpenSky flight parquets."),
    ] = Path("/data/contrailnet/flights/opensky"),
    era5_base: Annotated[
        Path, typer.Option(help="Base path for ERA5 NetCDF files."),
    ] = Path("/data/contrailnet/weather/era5"),
    window_hours: Annotated[
        float, typer.Option(help="Video window duration in hours."),
    ] = 2.0,
    max_age_min: Annotated[
        float, typer.Option(help="Prompt age window in minutes."),
    ] = 5.0,
    max_propagation_frames: Annotated[
        int, typer.Option(help="Max frames to continue prediction after last prompt."),
    ] = 100,
    single_video: Annotated[
        bool, typer.Option(help="Process each day as a single continuous video (no 2h window splits)."),
    ] = False,
    no_inference: Annotated[
        bool, typer.Option(help="Skip SAM2 inference (prompts only)."),
    ] = False,
    skip_existing: Annotated[
        bool, typer.Option(help="Skip steps whose output already exists."),
    ] = True,
    device: Annotated[
        str, typer.Option(help="Compute device for inference."),
    ] = "cuda",
) -> None:
    """Run the labelling-campaign pipeline."""
    from contrailtrack.campaign.pipeline import run_day

    dates = _parse_dates(date)
    log.info("campaign_start", dates=[str(d) for d in dates], output=str(output))

    all_results = []
    for d in dates:
        try:
            results = run_day(
                d,
                images_base=images_base,
                flights_base=flights_base,
                era5_base=era5_base,
                output_base=output,
                window_hours=window_hours,
                max_age_min=max_age_min,
                max_propagation_frames=max_propagation_frames,
                run_inference=not no_inference,
                skip_existing=skip_existing,
                device=device,
                single_video=single_video,
            )
            all_results.extend(results)
        except FileNotFoundError as e:
            log.error("missing_data", date=str(d), error=str(e))
            continue

    # Summary
    total_windows = len(all_results)
    total_frames = sum(r.get("frames", 0) for r in all_results)
    total_flights = sum(r.get("flights", 0) for r in all_results)
    total_prompts = sum(r.get("prompt_files", 0) for r in all_results)
    total_preds = sum(r.get("predictions", 0) for r in all_results)

    log.info(
        "campaign_done",
        dates=len(dates),
        windows=total_windows,
        frames=total_frames,
        flights=total_flights,
        prompt_files=total_prompts,
        predictions=total_preds,
    )


if __name__ == "__main__":
    app()
