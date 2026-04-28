"""Orchestration: run the full prompt-generation + inference pipeline for a video window."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import structlog

log = structlog.get_logger()

BRETIGNY_LAT = 48.600518
BRETIGNY_LON = 2.346795
BRETIGNY_ALT_M = 90.0


def video_id_from_window(start: dt.datetime, stop: dt.datetime) -> str:
    return f"{start:%Y%m%d%H%M%S}_{stop:%Y%m%d%H%M%S}"


def run_window(
    start: dt.datetime,
    stop: dt.datetime,
    *,
    images_dir: Path,
    opensky_path: Path,
    era5_dir: Path,
    output_base: Path,
    camera_lat: float = BRETIGNY_LAT,
    camera_lon: float = BRETIGNY_LON,
    camera_alt_m: float = BRETIGNY_ALT_M,
    max_age_min: float = 5.0,
    buffer_px: float = 5.0,
    encoding: str = "ternary",
    score_threshold: float = 0.5,
    max_propagation_frames: int = 100,
    run_inference: bool = True,
    skip_existing: bool = True,
    device: str = "cuda",
) -> dict:
    """Run the full pipeline for a single 2-hour video window.

    Steps:
        1. Create frame symlinks (strip ``image_`` prefix)
        2. Convert OpenSky traffic → fleet JSON
        3. Load local ERA5 met data
        4. Run DryAdvection → contrail parquet
        5. Generate age-weighted prompt PNGs
        6. (Optional) Run SAM2 inference → COCO RLE JSON

    Returns a stats dict summarising what was produced.
    """
    import pandas as pd

    vid = video_id_from_window(start, stop)
    log.info("run_window_start", video_id=vid, start=str(start), stop=str(stop))

    frames_dir = output_base / "frames"
    fleet_dir = output_base / "fleet"
    da_dir = output_base / "dry_advection"
    prompts_dir = output_base / "prompts"
    preds_dir = output_base / "predictions"

    stats = {"video_id": vid, "start": str(start), "stop": str(stop)}

    # ── 1. Frame symlinks ────────────────────────────────────────────────────
    from contrailtrack.campaign.frames import create_frame_symlinks

    frame_count = create_frame_symlinks(
        images_dir=images_dir,
        output_dir=frames_dir,
        video_id=vid,
        start=start,
        stop=stop,
    )
    stats["frames"] = frame_count
    if frame_count == 0:
        log.warning("no_frames_in_window", video_id=vid)
        return stats

    # ── 2. Fleet JSON ────────────────────────────────────────────────────────
    fleet_path = fleet_dir / f"{vid}.json"
    if skip_existing and fleet_path.exists():
        with open(fleet_path) as f:
            fleet_data = json.load(f)
        log.info("fleet_cached", path=str(fleet_path), n_flights=len(fleet_data))
    else:
        from contrailtrack.campaign.fleet import opensky_to_fleet, save_fleet_json

        fleet_data = opensky_to_fleet(opensky_path, start=start, stop=stop)
        if not fleet_data:
            log.warning("no_flights_in_window", video_id=vid)
            stats["flights"] = 0
            return stats
        save_fleet_json(fleet_data, fleet_path)

    stats["flights"] = len(fleet_data)
    log.info("fleet_ready", n_flights=len(fleet_data))

    # ── 3. DryAdvection ──────────────────────────────────────────────────────
    da_path = da_dir / f"{vid}.parquet"
    if skip_existing and da_path.exists():
        log.info("dry_advection_cached", path=str(da_path))
    else:
        from contrailtrack.campaign.era5_local import load_local_era5
        from contrailtrack.prompts.dry_advection import run_dry_advection

        # Determine which daily ERA5 files are needed
        dates_needed = sorted({start.date(), stop.date()})
        met = load_local_era5(era5_dir, dates_needed)

        da_result = run_dry_advection(
            fleet_json=fleet_path,
            output_dir=da_dir,
            met=met,
        )
        if da_result is None:
            log.warning("no_contrails_advected", video_id=vid)
            stats["contrails"] = 0
            return stats
        da_path = da_result

    contrail_df = pd.read_parquet(da_path)
    stats["contrail_waypoints_raw"] = len(contrail_df)

    # DryAdvection sets formation_time=time; derive the correct value from the age column.
    if "age" in contrail_df.columns and pd.api.types.is_timedelta64_dtype(contrail_df["age"]):
        contrail_df["formation_time"] = contrail_df["time"] - contrail_df["age"]

    # Age pre-filter: drop waypoints older than the prompt window (huge reduction).
    age_seconds = (contrail_df["time"] - contrail_df["formation_time"]).dt.total_seconds()
    contrail_df = contrail_df.loc[age_seconds <= max_age_min * 60].copy()

    # Geographic pre-filter: keep only waypoints within a box around the camera.
    # The camera FOV is ~1° × 0.7° at cruise altitude; 0.75° margin covers drift.
    margin = 0.75  # degrees
    contrail_df = contrail_df.loc[
        (contrail_df["longitude"] >= camera_lon - margin)
        & (contrail_df["longitude"] <= camera_lon + margin)
        & (contrail_df["latitude"] >= camera_lat - margin)
        & (contrail_df["latitude"] <= camera_lat + margin)
    ].copy()
    stats["contrail_waypoints"] = len(contrail_df)
    log.info("dry_advection_ready", raw=stats["contrail_waypoints_raw"], filtered=len(contrail_df))

    if contrail_df.empty:
        log.warning("no_contrails_near_camera", video_id=vid)
        return stats

    # ── 4. Prompt generation ─────────────────────────────────────────────────
    prompt_video_dir = prompts_dir / vid
    if skip_existing and prompt_video_dir.exists() and any(prompt_video_dir.rglob("*_prompt.png")):
        log.info("prompts_cached", path=str(prompt_video_dir))
        stats["prompt_files"] = len(list(prompt_video_dir.rglob("*_prompt.png")))
    else:
        from contrailtrack.prompts.projection import MiniProjector
        from contrailtrack.prompts.writer import generate_prompts_video

        projector = MiniProjector(
            camera_lat=camera_lat,
            camera_lon=camera_lon,
            camera_alt_m=camera_alt_m,
            resolution=1024,
        )
        prompt_stats = generate_prompts_video(
            contrail_df=contrail_df,
            output_dir=prompts_dir,
            video_id=vid,
            images_dir=frames_dir / vid,
            projector=projector,
            max_age_min=max_age_min,
            buffer_px=buffer_px,
            image_size=1024,
        )
        stats["prompt_objects"] = prompt_stats["total_objects"]
        stats["prompt_files"] = prompt_stats["total_prompt_files_written"]
        log.info("prompts_generated", **prompt_stats)

    # ── 5. Inference ─────────────────────────────────────────────────────────
    if not run_inference:
        stats["inference"] = "skipped"
        return stats

    pred_path = preds_dir / f"{vid}.json"
    if skip_existing and pred_path.exists():
        log.info("predictions_cached", path=str(pred_path))
        stats["inference"] = "cached"
        return stats

    import contrailtrack as ct

    log.info("loading_model", device=device)
    model = ct.load_model(config="ternary", device=device)

    log.info("loading_frames", path=str(frames_dir / vid))
    frame_data, frame_names, h, w = ct.load_frames(frames_dir / vid, image_size=model.image_size)
    log.info("frames_loaded", n_frames=len(frame_names))

    log.info("reading_prompts", encoding=encoding)
    prompt_data = ct.read_prompts(prompts_dir, vid, encoding=encoding)
    log.info("prompts_loaded", n_objects=len(prompt_data))

    if not prompt_data:
        log.warning("no_prompts_for_inference", video_id=vid)
        stats["inference"] = "no_prompts"
        return stats

    n_obj = len(prompt_data)
    batch_size = 20 if n_obj > 30 else None
    log.info("running_inference", n_objects=n_obj, n_frames=len(frame_names), batch_size=batch_size)
    predictions = ct.run_video(
        model=model,
        frames=frame_data,
        frame_names=frame_names,
        prompts=prompt_data,
        original_height=h,
        original_width=w,
        score_threshold=score_threshold,
        max_propagation_frames=max_propagation_frames,
        object_batch_size=batch_size,
        device=device,
    )

    preds_dir.mkdir(parents=True, exist_ok=True)
    ct.export_coco_json(
        predictions=predictions,
        video_name=vid,
        frame_names=frame_names,
        height=h,
        width=w,
        output_path=pred_path,
        metadata={"config": "ternary", "encoding": encoding, "max_propagation_frames": max_propagation_frames},
    )

    n_masks = sum(len(v) for v in predictions.values()) if isinstance(predictions, dict) else 0
    stats["predictions"] = n_masks
    stats["inference"] = "done"
    log.info("inference_done", masks=n_masks, output=str(pred_path))

    return stats


def run_day(
    date: dt.date,
    *,
    images_base: Path,
    flights_base: Path,
    era5_base: Path,
    output_base: Path,
    camera_lat: float = BRETIGNY_LAT,
    camera_lon: float = BRETIGNY_LON,
    camera_alt_m: float = BRETIGNY_ALT_M,
    window_hours: float = 2.0,
    max_age_min: float = 5.0,
    max_propagation_frames: int = 100,
    run_inference: bool = True,
    skip_existing: bool = True,
    device: str = "cuda",
) -> list[dict]:
    """Run the pipeline for all daylight windows of a single day."""
    from contrailtrack.campaign.solar import daylight_windows

    images_dir = images_base / f"{date:%Y}" / f"{date:%m}" / f"{date:%d}"
    opensky_path = flights_base / f"{date:%Y}" / f"{date:%Y-%m-%d}.parquet"
    era5_dir = era5_base / f"{date:%Y}"

    if not images_dir.exists():
        raise FileNotFoundError(f"No images for {date}: {images_dir}")
    if not opensky_path.exists():
        raise FileNotFoundError(f"No traffic data for {date}: {opensky_path}")

    windows = daylight_windows(date, camera_lat, camera_lon, window_hours=window_hours)
    log.info("day_start", date=str(date), n_windows=len(windows))

    results = []
    for start, stop in windows:
        stats = run_window(
            start, stop,
            images_dir=images_dir,
            opensky_path=opensky_path,
            era5_dir=era5_dir,
            output_base=output_base,
            camera_lat=camera_lat,
            camera_lon=camera_lon,
            camera_alt_m=camera_alt_m,
            max_age_min=max_age_min,
            max_propagation_frames=max_propagation_frames,
            run_inference=run_inference,
            skip_existing=skip_existing,
            device=device,
        )
        results.append(stats)

    log.info("day_done", date=str(date), windows_processed=len(results))
    return results
