"""Orchestration: run the full prompt-generation + inference pipeline.

Day-level flow:
    1. Convert all flights for the day → single Fleet JSON
    2. Load ERA5 met for the day
    3. Run DryAdvection once for the full day → single parquet
    4. For each 2-hour video window:
       a. Create frame symlinks
       b. Filter advected waypoints by ``time`` within the window
       c. Generate age-weighted prompt PNGs
       d. (Optional) Run SAM2 inference → COCO RLE JSON

Running DryAdvection at day level ensures contrails emitted near window
boundaries are not lost — a flight at 09:58 produces waypoints that drift
into the 10:00–12:00 window and get picked up correctly.
"""

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
    contrail_df: "pd.DataFrame",
    images_dir: Path,
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
    """Run prompts + inference for a single video window.

    Expects pre-advected contrail waypoints (from day-level DryAdvection).
    Filters them by ``time`` to the window, then generates prompts and
    optionally runs SAM2 inference.

    Args:
        start: Window start (UTC).
        stop: Window end (UTC).
        contrail_df: Day-level advected contrail DataFrame (already has
            corrected ``formation_time`` and age/geographic pre-filters).
        images_dir: Directory containing camera images for this day.
        output_base: Root output directory.
    """
    import pandas as pd

    vid = video_id_from_window(start, stop)
    log.info("run_window_start", video_id=vid, start=str(start), stop=str(stop))

    frames_dir = output_base / "frames"
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

    # ── 2. Filter advected waypoints to this window ──────────────────────────
    window_df = contrail_df.loc[
        (contrail_df["time"] >= pd.Timestamp(start))
        & (contrail_df["time"] < pd.Timestamp(stop))
    ].copy()
    stats["contrail_waypoints"] = len(window_df)
    log.info("window_waypoints", video_id=vid, n_waypoints=len(window_df))

    if window_df.empty:
        log.warning("no_contrails_in_window", video_id=vid)
        return stats

    # ── 3. Prompt generation ─────────────────────────────────────────────────
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
            contrail_df=window_df,
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

    # ── 3b. Export prompts as COCO RLE JSON ──────────────────────────────────
    prompt_json_path = prompts_dir / f"{vid}.json"
    if not (skip_existing and prompt_json_path.exists()):
        from contrailtrack.output.coco import export_prompts_coco_json

        frame_paths = sorted((frames_dir / vid).glob("*.jpg"))
        frame_names_list = [p.stem for p in frame_paths]
        export_prompts_coco_json(
            prompts_dir=prompts_dir,
            video_id=vid,
            frame_names=frame_names_list,
            height=1024,
            width=1024,
            output_path=prompt_json_path,
            metadata={"encoding": encoding, "max_age_min": max_age_min},
        )
        log.info("prompts_json_exported", path=str(prompt_json_path))

    # ── 4. Inference ─────────────────────────────────────────────────────────
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


def _advect_day(
    date: dt.date,
    *,
    opensky_path: Path,
    era5_dir: Path,
    output_base: Path,
    daylight_start: dt.datetime,
    daylight_stop: dt.datetime,
    camera_lat: float = BRETIGNY_LAT,
    camera_lon: float = BRETIGNY_LON,
    max_age_min: float = 5.0,
    skip_existing: bool = True,
) -> "pd.DataFrame | None":
    """Run fleet conversion + DryAdvection for the full day.

    Returns the filtered contrail DataFrame (age + geographic filters applied),
    or None if no contrails survive filtering.
    """
    import pandas as pd

    fleet_dir = output_base / "fleet"
    da_dir = output_base / "dry_advection"

    day_id = f"{date:%Y%m%d}"

    # ── Fleet JSON (full day) ────────────────────────────────────────────────
    fleet_path = fleet_dir / f"{day_id}.json"
    if skip_existing and fleet_path.exists():
        with open(fleet_path) as f:
            fleet_data = json.load(f)
        log.info("fleet_cached", path=str(fleet_path), n_flights=len(fleet_data))
    else:
        from contrailtrack.campaign.fleet import opensky_to_fleet, save_fleet_json

        fleet_data = opensky_to_fleet(opensky_path, start=daylight_start, stop=daylight_stop)
        if not fleet_data:
            log.warning("no_flights_for_day", date=str(date))
            return None
        save_fleet_json(fleet_data, fleet_path)

    log.info("fleet_ready", n_flights=len(fleet_data))

    # ── DryAdvection (full day) ──────────────────────────────────────────────
    da_path = da_dir / f"{day_id}.parquet"
    if skip_existing and da_path.exists():
        log.info("dry_advection_cached", path=str(da_path))
    else:
        from contrailtrack.campaign.era5_local import load_local_era5
        from contrailtrack.prompts.dry_advection import run_dry_advection

        dates_needed = sorted({daylight_start.date(), daylight_stop.date()})
        met = load_local_era5(era5_dir, dates_needed)

        da_result = run_dry_advection(
            fleet_json=fleet_path,
            output_dir=da_dir,
            met=met,
            max_age=pd.Timedelta(minutes=max_age_min),
        )
        if da_result is None:
            log.warning("no_contrails_advected", date=str(date))
            return None
        da_path = da_result

    contrail_df = pd.read_parquet(da_path)
    n_raw = len(contrail_df)

    # DryAdvection sets formation_time=time; derive the correct value from age.
    if "age" in contrail_df.columns and pd.api.types.is_timedelta64_dtype(contrail_df["age"]):
        contrail_df["formation_time"] = contrail_df["time"] - contrail_df["age"]

    # Age filter: drop waypoints older than the prompt window.
    age_seconds = (contrail_df["time"] - contrail_df["formation_time"]).dt.total_seconds()
    contrail_df = contrail_df.loc[age_seconds <= max_age_min * 60].copy()

    # Geographic filter: keep waypoints within camera FOV + margin.
    margin = 0.75  # degrees
    contrail_df = contrail_df.loc[
        (contrail_df["longitude"] >= camera_lon - margin)
        & (contrail_df["longitude"] <= camera_lon + margin)
        & (contrail_df["latitude"] >= camera_lat - margin)
        & (contrail_df["latitude"] <= camera_lat + margin)
    ].copy()

    log.info("dry_advection_ready", raw=n_raw, filtered=len(contrail_df))

    if contrail_df.empty:
        return None

    return contrail_df


def _run_day_single_video(
    date: dt.date,
    *,
    images_dir: Path,
    opensky_path: Path,
    era5_dir: Path,
    output_base: Path,
    daylight_start: dt.datetime,
    daylight_stop: dt.datetime,
    camera_lat: float = BRETIGNY_LAT,
    camera_lon: float = BRETIGNY_LON,
    camera_alt_m: float = BRETIGNY_ALT_M,
    max_age_min: float = 5.0,
    encoding: str = "ternary",
    score_threshold: float = 0.5,
    max_propagation_frames: int = 100,
    run_inference: bool = True,
    skip_existing: bool = True,
    device: str = "cuda",
) -> dict:
    """Run the full day as a single continuous video.

    Eliminates window-boundary discontinuities: SAM2 memory propagates
    across the entire daylight period without resets.
    """
    import pandas as pd

    vid = video_id_from_window(daylight_start, daylight_stop)
    log.info("single_video_start", video_id=vid, date=str(date))

    frames_dir = output_base / "frames"
    prompts_dir = output_base / "prompts"
    preds_dir = output_base / "predictions"

    stats = {"video_id": vid, "start": str(daylight_start), "stop": str(daylight_stop), "mode": "single_video"}

    # ── 1. Frame symlinks (full day) ─────────────────────────────────────────
    from contrailtrack.campaign.frames import create_frame_symlinks

    frame_count = create_frame_symlinks(
        images_dir=images_dir,
        output_dir=frames_dir,
        video_id=vid,
        start=daylight_start,
        stop=daylight_stop,
    )
    stats["frames"] = frame_count
    if frame_count == 0:
        log.warning("no_frames", video_id=vid)
        return stats

    # ── 2. Day-level DryAdvection ────────────────────────────────────────────
    contrail_df = _advect_day(
        date,
        opensky_path=opensky_path,
        era5_dir=era5_dir,
        output_base=output_base,
        daylight_start=daylight_start,
        daylight_stop=daylight_stop,
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        max_age_min=max_age_min,
        skip_existing=skip_existing,
    )
    if contrail_df is None:
        log.warning("no_contrails", video_id=vid)
        stats["contrail_waypoints"] = 0
        return stats

    stats["contrail_waypoints"] = len(contrail_df)

    # ── 3. Prompt generation (full day) ──────────────────────────────────────
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
            buffer_px=5.0,
            image_size=1024,
        )
        stats["prompt_objects"] = prompt_stats["total_objects"]
        stats["prompt_files"] = prompt_stats["total_prompt_files_written"]
        log.info("prompts_generated", **prompt_stats)

    # ── 3b. Export prompts as COCO RLE JSON ──────────────────────────────────
    prompt_json_path = prompts_dir / f"{vid}.json"
    if not (skip_existing and prompt_json_path.exists()):
        from contrailtrack.output.coco import export_prompts_coco_json

        frame_paths = sorted((frames_dir / vid).glob("*.jpg"))
        frame_names_list = [p.stem for p in frame_paths]
        export_prompts_coco_json(
            prompts_dir=prompts_dir,
            video_id=vid,
            frame_names=frame_names_list,
            height=1024,
            width=1024,
            output_path=prompt_json_path,
            metadata={"encoding": encoding, "max_age_min": max_age_min},
        )
        log.info("prompts_json_exported", path=str(prompt_json_path))

    # ── 4. Inference ─────────────────────────────────────────────────────────
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
    batch_size = min(5, n_obj) if n_obj > 5 else None
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
        metadata={"config": "ternary", "encoding": encoding, "max_propagation_frames": max_propagation_frames,
                  "mode": "single_video"},
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
    single_video: bool = False,
) -> list[dict]:
    """Run the pipeline for all daylight windows of a single day.

    Args:
        single_video: If True, process the entire day as a single
            continuous video (no window splits, no boundary
            discontinuities).  Default False uses 2-hour windows.

    DryAdvection runs once for the full day; each window filters the
    advected waypoints by ``time``, ensuring no contrails are lost at
    window boundaries.
    """
    from contrailtrack.campaign.solar import daylight_windows

    images_dir = images_base / f"{date:%Y}" / f"{date:%m}" / f"{date:%d}"
    opensky_path = flights_base / f"{date:%Y}" / f"{date:%Y-%m-%d}.parquet"
    era5_dir = era5_base / f"{date:%Y}"

    if not images_dir.exists():
        raise FileNotFoundError(f"No images for {date}: {images_dir}")
    if not opensky_path.exists():
        raise FileNotFoundError(f"No traffic data for {date}: {opensky_path}")

    windows = daylight_windows(date, camera_lat, camera_lon, window_hours=window_hours)
    log.info("day_start", date=str(date), n_windows=len(windows), single_video=single_video)

    if not windows:
        return []

    daylight_start = windows[0][0]
    daylight_stop = windows[-1][1]

    if single_video:
        stats = _run_day_single_video(
            date,
            images_dir=images_dir,
            opensky_path=opensky_path,
            era5_dir=era5_dir,
            output_base=output_base,
            daylight_start=daylight_start,
            daylight_stop=daylight_stop,
            camera_lat=camera_lat,
            camera_lon=camera_lon,
            camera_alt_m=camera_alt_m,
            max_age_min=max_age_min,
            max_propagation_frames=max_propagation_frames,
            run_inference=run_inference,
            skip_existing=skip_existing,
            device=device,
        )
        return [stats]

    # ── Day-level: fleet + DryAdvection ──────────────────────────────────────
    contrail_df = _advect_day(
        date,
        opensky_path=opensky_path,
        era5_dir=era5_dir,
        output_base=output_base,
        daylight_start=daylight_start,
        daylight_stop=daylight_stop,
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        max_age_min=max_age_min,
        skip_existing=skip_existing,
    )

    if contrail_df is None:
        log.warning("no_contrails_for_day", date=str(date))
        return [{"video_id": video_id_from_window(s, e), "contrail_waypoints": 0} for s, e in windows]

    log.info("day_advection_done", date=str(date), total_waypoints=len(contrail_df))

    # ── Per-window: prompts + inference ──────────────────────────────────────
    results = []
    for start, stop in windows:
        stats = run_window(
            start, stop,
            contrail_df=contrail_df,
            images_dir=images_dir,
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
