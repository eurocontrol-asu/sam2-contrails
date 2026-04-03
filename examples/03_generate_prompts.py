"""Generate contrail prompt masks from a fleet JSON (unannotated / custom data).

Runs a contrail model (CoCiP or DryAdvection), projects the output to pixel
coordinates, and writes per-object prompt PNG masks ready for inference.

This example is intended for **unannotated** videos (e.g. your own camera data,
not the GVCCS benchmark).  For GVCCS, use ``contrailtrack generate-prompts
--annotations /path/to/annotations.json`` instead, which resolves frame
timestamps from the COCO metadata and creates consistent sequential names.

Two frame-matching modes:

- **No --frames** (default): prompts are named sequentially (``00000_prompt.png``,
  ``00001_prompt.png``, …).  Ensure your video frames are also sequentially named
  in chronological order so indices align at inference time.

- **With --frames**: pass a directory of JPEG frames named by capture timestamp
  (``YYYYMMDDHHMMSS.jpg``).  Prompts are named with matching timestamp stems.

Usage::

    # Generate prompts (sequential naming, no frames needed)
    uv run python examples/03_generate_prompts.py data/fleet/20230930055430_20230930075430.json

    # Generate prompts using timestamp-named frames for exact matching
    uv run python examples/03_generate_prompts.py data/fleet/20230930055430_20230930075430.json \\
        --frames data/frames/20230930055430_20230930075430

    # Use DryAdvection instead (faster, no radiation data needed)
    uv run python examples/03_generate_prompts.py data/fleet/20230930055430_20230930075430.json \\
        --model dry-advection

    # 10-minute age window
    uv run python examples/03_generate_prompts.py data/fleet/20230930055430_20230930075430.json \\
        --max-age-min 10
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import structlog
import typer

log = structlog.get_logger()

app = typer.Typer(add_completion=False)


class ContrailModel(str, Enum):
    cocip = "cocip"
    dry_advection = "dry-advection"


# GVCCS camera site: EUROCONTROL Bretigny, France
BRETIGNY_LAT = 48.600518
BRETIGNY_LON = 2.346795
BRETIGNY_ALT_M = 90.0


@app.command()
def main(
    fleet_json: Annotated[
        Path, typer.Argument(help="Fleet JSON (output of example 02).")
    ],
    model: Annotated[
        ContrailModel, typer.Option(help="Contrail model to run.")
    ] = ContrailModel.cocip,
    max_age_min: Annotated[
        float, typer.Option(help="Maximum contrail age in minutes.")
    ] = 5.0,
    buffer_px: Annotated[
        float, typer.Option(help="Buffer around contrail segments in pixels.")
    ] = 5.0,
    image_size: Annotated[
        int, typer.Option(help="Prompt mask resolution (square).")
    ] = 1024,
    era5_grid: Annotated[
        float, typer.Option(help="ERA5 grid resolution in degrees.")
    ] = 1.0,
    camera_lat: Annotated[
        float, typer.Option(help="Camera latitude.")
    ] = BRETIGNY_LAT,
    camera_lon: Annotated[
        float, typer.Option(help="Camera longitude.")
    ] = BRETIGNY_LON,
    camera_alt_m: Annotated[
        float, typer.Option(help="Camera altitude in metres.")
    ] = BRETIGNY_ALT_M,
    frames: Annotated[
        Path | None,
        typer.Option(
            "--frames",
            help=(
                "Optional directory of JPEG frames named by capture timestamp "
                "(YYYYMMDDHHMMSS.jpg). When provided, prompts are named with matching "
                "timestamp stems for exact frame matching. When omitted, prompts are "
                "named sequentially from contrail model times."
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Prompt output directory."),
    ] = None,
) -> None:
    """Generate per-object prompt masks from a fleet JSON."""
    from contrailtrack.prompts.projection import MiniProjector
    from contrailtrack.prompts.writer import generate_prompts_video

    if not fleet_json.exists():
        log.error("fleet_json_not_found", path=str(fleet_json))
        raise typer.Exit(code=1)

    model_name = model.value.replace("-", "_")
    contrail_dir = Path(f"data/{model_name}/")
    prompts_dir = output or Path(f"data/prompts_{model_name}_age{max_age_min:.0f}/")

    # Step 1: Run contrail model (saves to disk for caching; ERA5 download is expensive)
    if model == ContrailModel.cocip:
        from contrailtrack.prompts.cocip import run_cocip

        log.info("running_cocip", fleet=fleet_json.name)
        contrail_path = run_cocip(
            fleet_json=fleet_json, output_dir=contrail_dir, grid=era5_grid,
        )
    else:
        from contrailtrack.prompts.dry_advection import run_dry_advection

        log.info("running_dry_advection", fleet=fleet_json.name)
        contrail_path = run_dry_advection(
            fleet_json=fleet_json, output_dir=contrail_dir, grid=era5_grid,
        )

    if contrail_path is None:
        log.warning("no_contrails_predicted")
        raise typer.Exit(code=1)

    log.info("contrail_model_done", output=str(contrail_path))

    # Step 2: Write prompt masks (projection to pixel space happens in-memory)
    import pandas as pd

    stem = fleet_json.stem

    if frames is not None and not frames.exists():
        log.error("frames_dir_not_found", path=str(frames))
        raise typer.Exit(code=1)

    projector = MiniProjector(
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        camera_alt_m=camera_alt_m,
        resolution=image_size,
    )

    contrail_df = pd.read_parquet(contrail_path)
    log.info("writing_prompts", max_age_min=max_age_min, frames_dir=str(frames) if frames else "none")
    stats = generate_prompts_video(
        contrail_df=contrail_df,
        output_dir=prompts_dir,
        video_id=stem,
        images_dir=frames,
        projector=projector,
        max_age_min=max_age_min,
        buffer_px=buffer_px,
        image_size=image_size,
    )

    log.info(
        "prompts_written",
        objects=stats["total_objects"],
        frames=stats["total_frames_with_prompts"],
        files=stats["total_prompt_files_written"],
        output_dir=str(prompts_dir),
    )


if __name__ == "__main__":
    app()
