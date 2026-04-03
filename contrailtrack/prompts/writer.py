"""Generate per_object_data/ prompt PNGs from raw CoCiP or DryAdvection output.

Expected input format (output of run_cocip or run_dry_advection):
    Columns: flight_id, time, formation_time, longitude, latitude, level,
             waypoint, width (plus optional sin_a, cos_a, segment_length)
    Coordinates: lat/lon — projection to pixel space is done in-memory.
"""

from __future__ import annotations

import logging
from datetime import datetime as dt_cls
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

log = logging.getLogger(__name__)


def _lazy_imports():
    """Import geo dependencies lazily — they require rasterio, geopandas, shapely."""
    from shapely import box
    from contrailtrack.prompts.projection import geometry_to_mask, contrail_geometry, MiniProjector
    from contrailtrack.data.coco import COCOWithVideo
    return box, geometry_to_mask, contrail_geometry, MiniProjector, COCOWithVideo


def generate_prompts(
    contrail_dir: str | Path,
    output_dir: str | Path,
    coco_annotations: str | Path = None,
    images_source_dir: str | Path = None,
    projector=None,
    max_age_min: float = 5.0,
    buffer_px: float = 5.0,
    image_size: int = 1024,
    write_gt_masks: bool = True,
    write_symlinks: bool = True,
    require_gt_in_video: bool = False,
    video_ids: set | None = None,
) -> dict:
    """Generate per_object_data/ prompt PNGs for all contrail parquets in a directory.

    Reads raw CoCiP or DryAdvection output parquets (lat/lon space), projects them
    to pixel coordinates in-memory, converts each contrail segment to an age-weighted
    mask, and writes:
      - {output_dir}/{video_id}/{flight_id}/{frame}_prompt.png
      - {output_dir}/{video_id}/{frame}_all_prompts_union.png
      - {output_dir}/{video_id}/{flight_id}/{frame}_mask.png  (if write_gt_masks and coco provided)

    Args:
        contrail_dir: Directory containing {video_name}.parquet files (raw CoCiP or
                      DryAdvection output in lat/lon space).
                      Video names are formatted as YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.
        output_dir: Root output directory (e.g., "per_object_data_age_5/").
        coco_annotations: Optional path to COCO annotations.json (with video metadata
                          and image times). When provided, video/frame metadata comes
                          from COCO and GT masks can be written. When None, videos are
                          enumerated from the parquet filenames and frame timestamps are
                          read from the contrail parquet's ``time`` column.
        images_source_dir: Directory containing source JPEG images to symlink.
                           If None, symlinks are skipped even if write_symlinks=True.
        projector: Camera projector for lat/lon → pixel conversion. Defaults to
                   MiniProjector() with GVCCS camera constants.
        max_age_min: Age window in minutes. Contrail segments older than this are ignored.
                     Age weighting: newest=1.0, at max_age=0.0.
        buffer_px: Buffer around contrail geometries in pixels.
        image_size: Image dimensions (H=W). Used for camera bounding box clipping and,
                    when coco_annotations is None, for mask dimensions.
        write_gt_masks: Whether to write GT mask PNGs alongside prompt PNGs.
                        Ignored when coco_annotations is None.
        write_symlinks: Whether to create symlinks to JPEG frames in img_folder/.
        require_gt_in_video: If True, skip flights that have no GT annotation anywhere
                             in the video. Only applies when coco_annotations is provided.
        video_ids: Optional set of video folder names (zero-padded, e.g. {"00001", "00003"})
                   to process. When None, all videos are processed.

    Returns:
        dict with keys: videos_processed, videos_skipped, total_frames_with_prompts,
                        total_objects, total_prompt_files_written.
    """
    import pandas as pd
    box, geometry_to_mask, contrail_geometry, MiniProjector, COCOWithVideo = _lazy_imports()

    if projector is None:
        projector = MiniProjector()

    contrail_dir = Path(contrail_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_view = box(0, 0, image_size, image_size)

    stats = {
        "videos_processed": 0,
        "videos_skipped": 0,
        "total_frames_with_prompts": 0,
        "total_objects": 0,
        "total_prompt_files_written": 0,
    }

    # ── Build video list ──────────────────────────────────────────────────────
    if coco_annotations is not None:
        coco = COCOWithVideo(str(coco_annotations))
        coco_videos = coco.loadVideos(ids=coco.getVideoIds())

        def _iter_videos():
            for v in coco_videos:
                folder_name = str(v["id"]).zfill(5)
                if video_ids is not None and folder_name not in video_ids:
                    continue
                start = dt_cls.fromisoformat(v["start"])
                stop  = dt_cls.fromisoformat(v["stop"])
                video_name = start.strftime("%Y%m%d%H%M%S") + "_" + stop.strftime("%Y%m%d%H%M%S")
                parquet_matches = list(contrail_dir.glob(f"{video_name}.parquet"))
                if parquet_matches:
                    yield v["id"], video_name, parquet_matches[0]
                else:
                    stats["videos_skipped"] += 1
    else:
        coco = None

        def _iter_videos():
            for idx, p in enumerate(sorted(contrail_dir.glob("*.parquet")), start=1):
                folder_name = str(idx).zfill(5)
                if video_ids is not None and folder_name not in video_ids:
                    continue
                yield idx, p.stem, p

    # ── Process each video ────────────────────────────────────────────────────
    for video_id, video_name, parquet_path in tqdm(list(_iter_videos()), desc="Videos"):
        try:
            df = pd.read_parquet(parquet_path)
            geometries = contrail_geometry(df, projector)
        except Exception as e:
            log.warning("Could not read %s: %s", parquet_path, e)
            stats["videos_skipped"] += 1
            continue

        # Compute age in minutes
        geometries["age"] = (
            geometries["time"] - geometries["formation_time"]
        ).dt.total_seconds() / 60.0

        # Filter by age window
        geometries = geometries.loc[geometries["age"] <= max_age_min].copy()
        if geometries.empty:
            stats["videos_skipped"] += 1
            continue

        # Normalise age: newest=1.0, oldest (at max_age)=0.0
        geometries["age"] = 1.0 - geometries["age"] / max_age_min

        # Buffer and clip to camera view
        geometries["geometry"] = geometries.geometry.buffer(buffer_px, cap_style="flat")
        geometries = geometries.clip(camera_view)
        geometries = geometries[~geometries.geometry.is_empty]
        if geometries.empty:
            stats["videos_skipped"] += 1
            continue

        # Determine active flights
        active_flight_ids = set(geometries.flight_id)

        if coco is not None:
            img_ids = coco.getImgIds(videoIds=[video_id])
            anns_video = coco.loadAnns(coco.getAnnIds(imgIds=img_ids))
            if require_gt_in_video:
                gt_ids = {a.get("flight_id") for a in anns_video if a.get("flight_id")}
                active_flight_ids = active_flight_ids & gt_ids

        if not active_flight_ids:
            stats["videos_skipped"] += 1
            continue

        v_out_dir = output_dir / str(video_id).zfill(5)
        v_out_dir.mkdir(parents=True, exist_ok=True)

        # ── Build frame list ──────────────────────────────────────────────────
        if coco is not None:
            images = sorted(coco.loadImgs(ids=img_ids), key=lambda x: x["id"])

            def _iter_frames():
                for frame_idx, img in enumerate(images):
                    yield frame_idx, img["height"], img["width"], dt_cls.fromisoformat(img["time"]), img
        else:
            frame_times = sorted(geometries["time"].unique())

            def _iter_frames():
                for frame_idx, ft in enumerate(frame_times):
                    yield frame_idx, image_size, image_size, ft.to_pydatetime(), None

        for frame_idx, h, w, img_time, img in _iter_frames():
            frame_str = str(frame_idx).zfill(5)

            # Symlink source image if requested (COCO mode only)
            if write_symlinks and images_source_dir is not None and img is not None:
                src_path = Path(images_source_dir) / img.get("file_name")
                dst_path = (
                    output_dir.parent / "img_folder" / str(video_id).zfill(5) / frame_str
                ).with_suffix(src_path.suffix)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if not dst_path.exists():
                    dst_path.symlink_to(src_path)

            # Get geometries for this frame
            img_geoms = geometries.loc[geometries.time == img_time].copy()

            # Build per-flight age-weighted masks
            frame_prompts = {}
            for fid, group in img_geoms.groupby("flight_id"):
                if fid not in active_flight_ids:
                    continue
                combined = np.zeros((h, w), dtype=np.float32)
                for _, row in group.iterrows():
                    poly_mask = geometry_to_mask(row["geometry"], shape=(h, w))
                    combined = np.maximum(combined, poly_mask * row["age"])
                if combined.any():
                    frame_prompts[fid] = combined

            # Write union mask (needed for ternary encoding at inference time)
            if frame_prompts:
                union = np.zeros((h, w), dtype=np.float32)
                for m in frame_prompts.values():
                    union = np.maximum(union, m)
                PILImage.fromarray((union * 255).astype(np.uint8)).save(
                    v_out_dir / f"{frame_str}_all_prompts_union.png"
                )
                stats["total_frames_with_prompts"] += 1

            # Write per-object prompt PNGs (and optionally GT mask PNGs)
            frame_anns = coco.loadAnns(coco.getAnnIds(imgIds=img["id"])) if (coco and img) else []
            for fid in active_flight_ids:
                obj_folder = v_out_dir / fid

                # GT mask (only when COCO annotations available)
                if write_gt_masks and frame_anns:
                    gt = np.zeros((h, w), dtype=np.uint8)
                    for ann in (a for a in frame_anns if a.get("flight_id") == fid):
                        gt[coco.annToMask(ann) == 1] = 255
                    if gt.any():
                        obj_folder.mkdir(parents=True, exist_ok=True)
                        PILImage.fromarray(gt).save(obj_folder / f"{frame_str}_mask.png")

                # Prompt mask
                if fid in frame_prompts:
                    obj_folder.mkdir(parents=True, exist_ok=True)
                    p_map = (frame_prompts[fid] * 255).astype(np.uint8)
                    PILImage.fromarray(p_map).save(obj_folder / f"{frame_str}_prompt.png")
                    stats["total_prompt_files_written"] += 1

        stats["videos_processed"] += 1
        # Count only flights that had at least one prompt PNG written
        stats["total_objects"] += sum(
            1 for fid in active_flight_ids if any((v_out_dir / fid).glob("*_prompt.png"))
        )

    return stats


def _parse_frame_timestamp(stem: str) -> "pd.Timestamp":
    """Parse a timestamp from a frame filename stem.

    Supports ``YYYYMMDDHHMMSS`` (e.g. ``20230930055430``) format.
    """
    import pandas as pd
    from datetime import datetime

    try:
        dt = datetime.strptime(stem, "%Y%m%d%H%M%S")
        return pd.Timestamp(dt)
    except ValueError:
        return None


def generate_prompts_video(
    contrail_df,
    output_dir: str | Path,
    video_id: str,
    images_dir: str | Path | None = None,
    projector=None,
    max_age_min: float = 5.0,
    buffer_px: float = 5.0,
    image_size: int = 1024,
) -> dict:
    """Generate per_object_data/ prompt PNGs for a single unannotated video.

    This is the annotation-free version of ``generate_prompts`` — no COCO
    annotations required. Use this when running inference on newly recorded
    videos that have not been labelled yet.

    Two frame-matching modes are supported:

    **No images directory** (``images_dir=None``, default):
        Frames are derived from the unique timestamps in ``contrail_df``.
        Prompts are named sequentially: ``00000_prompt.png``, ``00001_prompt.png``, …
        Ensure your video frames are also sequentially named in chronological order
        so that the indices align.

    **With images directory** (``images_dir`` provided):
        Frame filenames must encode the capture timestamp as
        ``YYYYMMDDHHMMSS.jpg`` (e.g. ``20230930055430.jpg``). Prompts are named
        using the same timestamp stem so they match at inference time.

    Args:
        contrail_df: Raw CoCiP or DryAdvection output DataFrame (lat/lon space).
        output_dir: Root output directory. Writes to
                    ``{output_dir}/{video_id}/{flight_id}/{frame}_prompt.png``.
        video_id: Identifier string used as the output subfolder name.
        images_dir: Optional folder of JPEG frames named as timestamps. When
                    provided, prompts are named to match the frame stems. When
                    None, prompts are named sequentially from contrail model times.
        projector: Camera projector for lat/lon → pixel conversion. Defaults to
                   MiniProjector() with GVCCS camera constants.
        max_age_min: Age window in minutes. Contrail segments older than this are ignored.
        buffer_px: Pixel buffer around contrail geometry edges.
        image_size: Image dimensions (H = W). Used for bounding-box clipping and,
                    when ``images_dir`` is None, for mask dimensions.

    Returns:
        dict with keys: total_frames_with_prompts, total_objects, total_prompt_files_written.
    """
    box, geometry_to_mask, contrail_geometry, MiniProjector, _ = _lazy_imports()

    if projector is None:
        projector = MiniProjector()

    output_dir = Path(output_dir)
    camera_view = box(0, 0, image_size, image_size)

    # Project raw contrail data to pixel-space geometry
    geometries = contrail_geometry(contrail_df, projector)
    geometries["age"] = (
        geometries["time"] - geometries["formation_time"]
    ).dt.total_seconds() / 60.0
    geometries = geometries.loc[geometries["age"] <= max_age_min].copy()
    if geometries.empty:
        return {"total_frames_with_prompts": 0, "total_objects": 0, "total_prompt_files_written": 0}

    geometries["age"] = 1.0 - geometries["age"] / max_age_min
    geometries["geometry"] = geometries.geometry.buffer(buffer_px, cap_style="flat")
    geometries = geometries.clip(camera_view)
    geometries = geometries[~geometries.geometry.is_empty]
    if geometries.empty:
        return {"total_frames_with_prompts": 0, "total_objects": 0, "total_prompt_files_written": 0}

    active_flight_ids = sorted(set(geometries.flight_id))

    v_out_dir = output_dir / str(video_id)
    v_out_dir.mkdir(parents=True, exist_ok=True)

    # Build frame list: either from image files or from contrail model times
    if images_dir is not None:
        images_dir = Path(images_dir)
        frame_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg"))
        if not frame_paths:
            raise FileNotFoundError(f"No JPEG frames found in {images_dir}")
        with PILImage.open(frame_paths[0]) as im:
            w, h = im.size

        def _iter_frames():
            for frame_path in frame_paths:
                frame_time = _parse_frame_timestamp(frame_path.stem)
                if frame_time is not None:
                    yield frame_path.stem, frame_time, h, w
    else:
        h = w = image_size
        frame_times_unique = sorted(geometries["time"].unique())

        def _iter_frames():
            for idx, ft in enumerate(frame_times_unique):
                yield str(idx).zfill(5), ft, h, w

    written_flight_ids = set()
    stats = {"total_frames_with_prompts": 0, "total_objects": 0, "total_prompt_files_written": 0}

    for frame_str, frame_time, h, w in tqdm(list(_iter_frames()), desc=f"Frames [{video_id}]"):
        img_geoms = geometries.loc[geometries.time == frame_time].copy()

        prompts = {}
        for fid, group in img_geoms.groupby("flight_id"):
            if fid not in active_flight_ids:
                continue
            combined = np.zeros((h, w), dtype=np.float32)
            for _, row in group.iterrows():
                poly_mask = geometry_to_mask(row["geometry"], shape=(h, w))
                combined = np.maximum(combined, poly_mask * row["age"])
            if combined.any():
                prompts[fid] = combined

        if prompts:
            union = np.zeros((h, w), dtype=np.float32)
            for m in prompts.values():
                union = np.maximum(union, m)
            PILImage.fromarray((union * 255).astype(np.uint8)).save(
                v_out_dir / f"{frame_str}_all_prompts_union.png"
            )
            stats["total_frames_with_prompts"] += 1

        for fid, p_map_f32 in prompts.items():
            fid_dir = v_out_dir / fid
            fid_dir.mkdir(parents=True, exist_ok=True)
            p_map = (p_map_f32 * 255).astype(np.uint8)
            PILImage.fromarray(p_map).save(fid_dir / f"{frame_str}_prompt.png")
            stats["total_prompt_files_written"] += 1
            written_flight_ids.add(fid)

    stats["total_objects"] = len(written_flight_ids)
    return stats
