"""contrailtrack CLI — contrail detection, tracking, and attribution."""

from pathlib import Path
from typing import List, Optional

import structlog
import typer

log = structlog.get_logger()

app = typer.Typer(
    name="contrailtrack",
    help="SAM2-based contrail detection, tracking, and attribution.",
    no_args_is_help=True,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _resolve_video_name(labels: Path, video_id: str) -> Optional[str]:
    """Map a zero-padded video_id (e.g. '00001') to its COCO timestamp name.

    Returns None if the video is not found or if video_id is already a timestamp name.
    """
    import json

    try:
        coco_id = int(video_id)
    except ValueError:
        return None  # already a timestamp name

    with open(labels) as f:
        data = json.load(f)
    for v in data.get("videos", []):
        if v["id"] == coco_id:
            start = v["start"].replace("-", "").replace(":", "").replace("T", "")
            stop  = v["stop"].replace("-", "").replace(":", "").replace("T", "")
            return f"{start}_{stop}"
    return None


def _video_set(videos: Optional[List[str]]) -> Optional[set]:
    """Normalise a list of video IDs to a set of zero-padded strings.

    Examples:
        ["1", "3"]    → {"00001", "00003"}
        None / []     → None  (= all videos)
    """
    if not videos:
        return None
    return {v.strip().zfill(5) for v in videos if v.strip()}


def _resolve_fleet_files(
    fleet_dir: Path,
    annotations: Optional[Path],
    selected: Optional[set],
) -> list:
    """Return the fleet JSON files to process.

    When *annotations* and *selected* are both provided, resolves video IDs to
    fleet JSON stems via COCO metadata.  Otherwise returns all (or filtered-by-
    stem) fleet JSONs.
    """
    all_files = sorted(fleet_dir.glob("*.json"))

    if annotations is not None and selected is not None:
        from datetime import datetime as dt_cls
        from contrailtrack.data.coco import COCOWithVideo

        coco = COCOWithVideo(str(annotations))
        stems = set()
        for v in coco.loadVideos(ids=coco.getVideoIds()):
            if str(v["id"]).zfill(5) not in selected:
                continue
            start = dt_cls.fromisoformat(v["start"])
            stop  = dt_cls.fromisoformat(v["stop"])
            stems.add(start.strftime("%Y%m%d%H%M%S") + "_" + stop.strftime("%Y%m%d%H%M%S"))
        return [f for f in all_files if f.stem in stems]

    if selected is not None:
        # No annotations: filter by stem substring
        return [f for f in all_files if any(v in f.stem for v in selected)]

    return all_files


def _run_single(
    images: Path,
    prompts: Path,
    out: Path,
    config: str,
    checkpoint: Optional[Path],
    encoding: str,
    device: str,
    score_threshold: float,
    max_propagation_frames: int,
    object_batch_size: Optional[int],
    labels: Optional[Path],
    eval_out: Optional[Path],
):
    from contrailtrack.model.loader import load_model
    from contrailtrack.data.video import load_frames
    from contrailtrack.data.prompt_reader import read_prompts
    from contrailtrack.inference.predictor import run_video
    from contrailtrack.output.coco import export_coco_json

    video_id = images.name

    log.info("loading_model", config=config, device=device)
    model = load_model(checkpoint=checkpoint, config=config, device=device)

    log.info("loading_frames", path=str(images))
    frames, frame_names, orig_h, orig_w = load_frames(images, image_size=model.image_size)
    log.info("frames_loaded", n_frames=len(frame_names), resolution=f"{orig_w}x{orig_h}")

    log.info("reading_prompts", encoding=encoding)
    prompt_data = read_prompts(prompts, video_id, encoding=encoding)
    log.info("prompts_loaded", n_objects=len(prompt_data))

    log.info("running_inference")
    predictions = run_video(
        model=model,
        frames=frames,
        frame_names=frame_names,
        prompts=prompt_data,
        original_height=orig_h,
        original_width=orig_w,
        score_threshold=score_threshold,
        max_propagation_frames=max_propagation_frames,
        object_batch_size=object_batch_size,
        device=device,
    )

    out_path = export_coco_json(
        predictions=predictions,
        video_name=video_id,
        frame_names=frame_names,
        height=orig_h,
        width=orig_w,
        output_path=out,
        metadata={"config": config, "encoding": encoding},
    )
    log.info("predictions_saved", output=str(out_path))

    if labels is not None:
        from contrailtrack.eval.metrics import evaluate as _evaluate
        eval_output = eval_out or out.parent / "evaluation"
        results = _evaluate(
            predictions_path=out_path,
            gt_annotations=labels,
            video_name=video_id,
            output_dir=eval_output,
        )
        seg   = results.get("segmentation", {})
        track = results.get("tracking", {}).get("metrics", {})
        attr  = results.get("attribution", {}).get("metrics", {})
        log.info(
            "evaluation_done",
            mAP=round(seg.get("mAP", 0), 4),
            detection_rate=round(track.get("detection_rate", 0), 4),
            completeness=round(track.get("mean_completeness", 0), 4),
            tiou=round(track.get("mean_tiou", 0), 4),
            attr_precision=round(attr.get("attribution_precision", 0), 4),
        )


def _evaluate_single(
    predictions: Path,
    labels: Path,
    video_name: Optional[str],
    out: Path,
    iou_thresholds: list,
    iou_threshold: float,
    skip_segmentation: bool,
):
    from contrailtrack.eval.metrics import evaluate as _evaluate

    # Resolve zero-padded video ID (e.g. "00001") → COCO timestamp name
    resolved = video_name or _resolve_video_name(labels, predictions.stem)
    log.info("evaluating", predictions=str(predictions), video=resolved or predictions.stem)
    results = _evaluate(
        predictions_path=predictions,
        gt_annotations=labels,
        video_name=resolved,
        iou_thresholds=iou_thresholds,
        iou_threshold=iou_threshold,
        output_dir=out,
        skip_segmentation=skip_segmentation,
    )
    seg   = results.get("segmentation", {})
    track = results.get("tracking", {}).get("metrics", {})
    attr  = results.get("attribution", {}).get("metrics", {})
    log.info(
        "results",
        mAP=round(seg.get("mAP", 0), 4) if seg else None,
        detection_rate=round(track.get("detection_rate", 0), 4),
        completeness=round(track.get("mean_completeness", 0), 4),
        tiou=round(track.get("mean_tiou", 0), 4),
        attr_precision=round(attr.get("attribution_precision", 0), 4),
    )


# ── commands ───────────────────────────────────────────────────────────────────

_VIDEOS_HELP = (
    "Video IDs to process — repeat the flag for each ID "
    "(e.g. --videos 1 --videos 3). Leading zeros are optional. Default: all."
)


@app.command("run-cocip")
def run_cocip_cmd(
    fleet_dir: Path = typer.Option(..., help="Directory of fleet JSON files."),
    out: Path = typer.Option(..., help="Output directory for contrail parquets."),
    annotations: Optional[Path] = typer.Option(
        None,
        help="COCO annotations.json — required to filter by video ID. "
             "When omitted, --videos matches against fleet JSON stems.",
    ),
    videos: Optional[List[str]] = typer.Option(None, help=_VIDEOS_HELP),
    cache_dir: Optional[Path] = typer.Option(None, help="ERA5 disk cache directory."),
    grid: float = typer.Option(1.0, help="ERA5 horizontal resolution in degrees."),
    skip_existing: bool = typer.Option(True, help="Skip files where the output parquet exists."),
):
    """Run CoCiP on a directory of fleet JSON files.

    All videos:

        contrailtrack run-cocip --fleet-dir data/fleet/ --out data/cocip/ --annotations annotations.json

    Subset by video ID (requires --annotations):

        contrailtrack run-cocip --fleet-dir data/fleet/ --out data/cocip/ \\
            --annotations annotations.json --videos 00001 --videos 00003
    """
    from contrailtrack.prompts.cocip import run_cocip

    selected = _video_set(videos)
    fleet_files = _resolve_fleet_files(fleet_dir, annotations, selected)
    if not fleet_files:
        log.warning("no_fleet_files_found", fleet_dir=str(fleet_dir), filter=videos)
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)
    log.info("run_cocip_batch", n_files=len(fleet_files))

    results = []
    for fleet_json in fleet_files:
        out_path = out / fleet_json.with_suffix(".parquet").name
        if skip_existing and out_path.exists():
            log.info("skipping_existing", file=fleet_json.name)
            results.append(out_path)
            continue
        log.info("running_cocip", file=fleet_json.name)
        result = run_cocip(fleet_json=fleet_json, output_dir=out, cache_dir=cache_dir, grid=grid)
        results.append(result)

    n_ok = sum(1 for r in results if r is not None)
    log.info("done", processed=len(results), with_contrails=n_ok)


@app.command("run-dry-advection")
def run_dry_advection_cmd(
    fleet_dir: Path = typer.Option(..., help="Directory of fleet JSON files."),
    out: Path = typer.Option(..., help="Output directory for contrail parquets."),
    annotations: Optional[Path] = typer.Option(
        None,
        help="COCO annotations.json — required to filter by video ID. "
             "When omitted, --videos matches against fleet JSON stems.",
    ),
    videos: Optional[List[str]] = typer.Option(None, help=_VIDEOS_HELP),
    cache_dir: Optional[Path] = typer.Option(None, help="ERA5 disk cache directory."),
    grid: float = typer.Option(1.0, help="ERA5 horizontal resolution in degrees."),
    initial_width_m: float = typer.Option(400.0, help="Initial contrail plume width in metres."),
    skip_existing: bool = typer.Option(True, help="Skip files where the output parquet exists."),
):
    """Run DryAdvection on a directory of fleet JSON files.

    All videos:

        contrailtrack run-dry-advection --fleet-dir data/fleet/ --out data/dry_advection/ \\
            --annotations annotations.json

    Subset by video ID:

        contrailtrack run-dry-advection --fleet-dir data/fleet/ --out data/dry_advection/ \\
            --annotations annotations.json --videos 00001 --videos 00003
    """
    from contrailtrack.prompts.dry_advection import run_dry_advection

    selected = _video_set(videos)
    fleet_files = _resolve_fleet_files(fleet_dir, annotations, selected)
    if not fleet_files:
        log.warning("no_fleet_files_found", fleet_dir=str(fleet_dir), filter=videos)
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)
    log.info("run_dry_advection_batch", n_files=len(fleet_files))

    results = []
    for fleet_json in fleet_files:
        out_path = out / fleet_json.with_suffix(".parquet").name
        if skip_existing and out_path.exists():
            log.info("skipping_existing", file=fleet_json.name)
            results.append(out_path)
            continue
        log.info("running_dry_advection", file=fleet_json.name)
        result = run_dry_advection(
            fleet_json=fleet_json,
            output_dir=out,
            cache_dir=cache_dir,
            grid=grid,
            initial_width_m=initial_width_m,
        )
        results.append(result)

    n_ok = sum(1 for r in results if r is not None)
    log.info("done", processed=len(results), with_contrails=n_ok)


@app.command("generate-prompts")
def generate_prompts_cmd(
    contrail_dir: Path = typer.Option(
        ..., help="Directory of raw CoCiP or DryAdvection output parquets (lat/lon space)."
    ),
    out: Path = typer.Option(..., help="Output prompt directory."),
    annotations: Optional[Path] = typer.Option(
        None,
        help="COCO annotations.json. When provided, video folders are named by video ID "
             "and GT masks are written. When omitted, runs annotation-free (no masks).",
    ),
    videos: Optional[List[str]] = typer.Option(None, help=_VIDEOS_HELP),
    images_source: Optional[Path] = typer.Option(
        None, help="Source JPEG image directory for symlinking frames (requires --annotations)."
    ),
    max_age: float = typer.Option(5.0, help="Contrail age window in minutes."),
    buffer: float = typer.Option(5.0, help="Buffer around contrail geometries in pixels."),
    image_size: int = typer.Option(1024, help="Camera image size (H=W) in pixels."),
    camera_lat: float = typer.Option(48.600518, help="Camera latitude (degrees)."),
    camera_lon: float = typer.Option(2.346795, help="Camera longitude (degrees)."),
    camera_alt: float = typer.Option(90.0, help="Camera altitude above ground (metres)."),
    no_gt_masks: bool = typer.Option(False, help="Skip writing GT mask PNGs."),
    require_gt: bool = typer.Option(
        False, help="Skip flights with no GT annotation in the video (requires --annotations)."
    ),
):
    """Generate per-flight prompt PNGs from CoCiP or DryAdvection output parquets.

    With annotations (COCO dataset — writes GT masks, video ID folders):

        contrailtrack generate-prompts --contrail-dir data/cocip/ \\
            --annotations annotations.json --out prompts/

    Without annotations (unannotated traffic — no GT masks, parquet stem as folder):

        contrailtrack generate-prompts --contrail-dir data/cocip/ --out prompts/

    Subset of videos:

        contrailtrack generate-prompts --contrail-dir data/cocip/ \\
            --annotations annotations.json --out prompts/ --videos 00001 --videos 00003
    """
    from contrailtrack.prompts.writer import generate_prompts as _generate
    from contrailtrack.prompts.projection import MiniProjector

    projector = MiniProjector(
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        camera_alt_m=camera_alt,
        resolution=image_size,
    )

    selected = _video_set(videos)
    log.info(
        "generating_prompts",
        contrail_dir=str(contrail_dir),
        output=str(out),
        annotations=str(annotations) if annotations else "none",
        max_age=max_age,
        videos=sorted(selected) if selected else "all",
    )

    stats = _generate(
        contrail_dir=contrail_dir,
        coco_annotations=annotations,
        output_dir=out,
        images_source_dir=images_source,
        projector=projector,
        max_age_min=max_age,
        buffer_px=buffer,
        image_size=image_size,
        write_gt_masks=not no_gt_masks,
        write_symlinks=images_source is not None,
        require_gt_in_video=require_gt,
        video_ids=selected,
    )

    log.info(
        "done",
        videos_processed=stats["videos_processed"],
        videos_skipped=stats["videos_skipped"],
        total_objects=stats["total_objects"],
        prompt_files=stats["total_prompt_files_written"],
    )


@app.command()
def run(
    images: Optional[Path] = typer.Option(
        None, help="Single video frames directory (mutually exclusive with --images-dir)."
    ),
    images_dir: Optional[Path] = typer.Option(
        None, help="Root directory of per-video frame folders (batch mode)."
    ),
    videos: Optional[List[str]] = typer.Option(None, help=_VIDEOS_HELP),
    prompts: Path = typer.Option(..., help="Prompt masks root directory."),
    out: Path = typer.Option(
        ..., help="Output path: JSON file (single) or directory (batch)."
    ),
    config: str = typer.Option(
        "ternary",
        help='Model config: "ternary" (recommended) or "original", or full YAML path.',
    ),
    checkpoint: Optional[Path] = typer.Option(
        None, help="Local .pt checkpoint. Omit to auto-download from HF Hub.",
    ),
    encoding: str = typer.Option(
        "ternary",
        help='Prompt encoding: "ternary", "age_weighted", or "binary".',
    ),
    device: str = typer.Option("cuda", help="Compute device: cuda or cpu."),
    score_threshold: float = typer.Option(0.5, help="Object score threshold."),
    max_propagation_frames: int = typer.Option(
        50, help="Max frames to propagate after last prompt (0 = unlimited)."
    ),
    object_batch_size: Optional[int] = typer.Option(None, help="Objects per batch."),
    labels: Optional[Path] = typer.Option(
        None, help="GT annotations.json — triggers evaluation if provided."
    ),
    eval_out: Optional[Path] = typer.Option(
        None, help="Directory to write evaluation results."
    ),
):
    """Run SAM2 contrail inference on one video or an entire directory.

    Single video:

        contrailtrack run --images data/frames/00001 --prompts data/prompts/ --out results/00001.json

    All videos:

        contrailtrack run --images-dir data/frames/ --prompts data/prompts/ --out results/

    Subset:

        contrailtrack run --images-dir data/frames/ --prompts data/prompts/ --out results/ \\
            --videos 00001 --videos 00003
    """
    if images is not None and images_dir is not None:
        log.error("specify_either_images_or_images_dir")
        raise typer.Exit(code=1)
    if images is None and images_dir is None:
        log.error("one_of_images_or_images_dir_required")
        raise typer.Exit(code=1)

    selected = _video_set(videos)

    if images is not None:
        _run_single(
            images=images, prompts=prompts, out=out, config=config,
            checkpoint=checkpoint, encoding=encoding, device=device,
            score_threshold=score_threshold,
            max_propagation_frames=max_propagation_frames,
            object_batch_size=object_batch_size,
            labels=labels, eval_out=eval_out,
        )
    else:
        out.mkdir(parents=True, exist_ok=True)
        video_dirs = sorted(d for d in images_dir.iterdir() if d.is_dir())
        if selected is not None:
            video_dirs = [d for d in video_dirs if d.name.zfill(5) in selected]
        if not video_dirs:
            log.warning("no_videos_found", images_dir=str(images_dir), filter=videos)
            raise typer.Exit(code=1)
        log.info("batch_run", n_videos=len(video_dirs))
        for video_dir in video_dirs:
            log.info("processing_video", video_id=video_dir.name)
            _run_single(
                images=video_dir,
                prompts=prompts,
                out=out / f"{video_dir.name}.json",
                config=config,
                checkpoint=checkpoint,
                encoding=encoding,
                device=device,
                score_threshold=score_threshold,
                max_propagation_frames=max_propagation_frames,
                object_batch_size=object_batch_size,
                labels=labels,
                eval_out=(eval_out / video_dir.name) if eval_out else None,
            )


@app.command()
def evaluate(
    predictions: Optional[Path] = typer.Argument(
        None, help="Single COCO JSON prediction file."
    ),
    predictions_dir: Optional[Path] = typer.Option(
        None, help="Directory of COCO JSON files (batch mode)."
    ),
    videos: Optional[List[str]] = typer.Option(None, help=_VIDEOS_HELP),
    labels: Path = typer.Option(..., help="GT annotations.json."),
    video_name: Optional[str] = typer.Option(
        None, help="Video name override (single mode only; inferred from filename by default).",
    ),
    out: Path = typer.Option(Path("evaluation"), help="Output directory for results."),
    iou_range: str = typer.Option(
        "0.25:0.75", help="IoU range for mAP as 'low:high' (step 0.05)."
    ),
    iou_threshold: float = typer.Option(
        0.25, help="IoU threshold for tracking/attribution matching."
    ),
    skip_segmentation: bool = typer.Option(False, help="Skip slow COCO mAP computation."),
):
    """Evaluate contrail predictions against ground-truth annotations.

    Single file:

        contrailtrack evaluate results/00001.json --labels annotations.json --out eval/

    All predictions in a directory:

        contrailtrack evaluate --predictions-dir results/ --labels annotations.json --out eval/

    Subset:

        contrailtrack evaluate --predictions-dir results/ --labels annotations.json --out eval/ \\
            --videos 00001 --videos 00003
    """
    import numpy as np

    if predictions is not None and predictions_dir is not None:
        log.error("specify_either_predictions_or_predictions_dir")
        raise typer.Exit(code=1)
    if predictions is None and predictions_dir is None:
        log.error("one_of_predictions_or_predictions_dir_required")
        raise typer.Exit(code=1)

    lo, hi = [float(x) for x in iou_range.split(":")]
    iou_thresholds = list(np.arange(lo, hi + 0.01, 0.05))
    selected = _video_set(videos)

    if predictions is not None:
        _evaluate_single(
            predictions=predictions, labels=labels, video_name=video_name,
            out=out, iou_thresholds=iou_thresholds, iou_threshold=iou_threshold,
            skip_segmentation=skip_segmentation,
        )
    else:
        pred_files = sorted(predictions_dir.glob("*.json"))
        if selected is not None:
            pred_files = [p for p in pred_files if p.stem.zfill(5) in selected]
        if not pred_files:
            log.warning("no_prediction_files_found", predictions_dir=str(predictions_dir), filter=videos)
            raise typer.Exit(code=1)
        log.info("batch_evaluate", n_videos=len(pred_files))
        for pred_file in pred_files:
            log.info("evaluating_video", video=pred_file.stem)
            _evaluate_single(
                predictions=pred_file, labels=labels, video_name=None,
                out=out / pred_file.stem,
                iou_thresholds=iou_thresholds, iou_threshold=iou_threshold,
                skip_segmentation=skip_segmentation,
            )


@app.command()
def train(
    config: Path = typer.Argument(..., help="Path to training config YAML."),
    num_gpus: int = typer.Option(1, help="Number of GPUs to use."),
):
    """Fine-tune SAM2 on GVCCS contrail data.

        contrailtrack train sam2/configs/sam2.1_training/sam2.1_hiera_b+_GVCCS_finetune_age.yaml --num-gpus 2
    """
    import subprocess
    import sys

    repo_root = Path(__file__).parent.parent
    train_script = repo_root / "training" / "train.py"
    cmd = [sys.executable, str(train_script), "-c", str(config), "--use-cluster", "0", "--num-gpus", str(num_gpus)]

    log.info("starting_training", config=str(config), num_gpus=num_gpus)
    subprocess.run(cmd, check=True, cwd=repo_root)


@app.command("evaluate-dataset")
def evaluate_dataset_cmd(
    predictions_dir: Path = typer.Option(..., help="Directory of per-video COCO JSON files."),
    labels: Path = typer.Option(..., help="GT annotations.json."),
    out: Path = typer.Option(..., help="Output directory (per-video subfolders + dataset summary)."),
    videos: Optional[List[str]] = typer.Option(None, help=_VIDEOS_HELP),
    iou_range: str = typer.Option(
        "0.25:0.75", help="IoU range for mAP as 'low:high' (step 0.05)."
    ),
    iou_threshold: float = typer.Option(
        0.25, help="IoU threshold for tracking/attribution matching."
    ),
    skip_segmentation: bool = typer.Option(False, help="Skip slow COCO mAP computation."),
):
    """Evaluate all videos with true dataset-wide metrics.

    All prediction files are pooled and evaluated in a single pass — segmentation
    mAP, tracking, and attribution are computed over the full dataset, not averaged
    across videos.

    Writes:

    - ``{out}/dataset_results.json``  — dataset-level metrics
    - ``{out}/tracking_per_flight.csv`` — one row per flight
    - ``{out}/attribution_details.csv`` — one row per prediction

    All videos:

        contrailtrack evaluate-dataset --predictions-dir results/ \\
            --labels annotations.json --out evaluation/

    Subset:

        contrailtrack evaluate-dataset --predictions-dir results/ \\
            --labels annotations.json --out evaluation/ --videos 1 --videos 3
    """
    import json
    import numpy as np
    import pandas as pd
    from contrailtrack.eval.metrics import (
        _load_gt,
        _predictions_to_coco,
        _evaluate_segmentation,
        _evaluate_tracking,
        _evaluate_attribution,
        _to_serialisable,
    )

    lo, hi = [float(x) for x in iou_range.split(":")]
    iou_thresholds = list(np.arange(lo, hi + 0.01, 0.05))
    selected = _video_set(videos)

    pred_files = sorted(predictions_dir.glob("*.json"))
    if selected is not None:
        pred_files = [p for p in pred_files if p.stem.zfill(5) in selected]
    if not pred_files:
        log.warning("no_prediction_files_found", predictions_dir=str(predictions_dir))
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)

    # Load the full dataset GT once (no video filter)
    coco_gt = _load_gt(labels, video_name=None)
    log.info("evaluate_dataset", n_videos=len(pred_files))

    # Pool all predictions across all videos into one list
    all_coco_results = []
    n_loaded = 0
    for pred_file in pred_files:
        video_id = pred_file.stem
        resolved_name = _resolve_video_name(labels, video_id)
        if resolved_name is None:
            log.warning("video_not_found_in_annotations", video=video_id)
            continue
        try:
            results = _predictions_to_coco(pred_file, coco_gt, resolved_name)
            all_coco_results.extend(results)
            n_loaded += 1
        except Exception as e:
            log.warning("video_load_failed", video=video_id, error=str(e))

    if n_loaded == 0:
        log.error("no_videos_loaded")
        raise typer.Exit(code=1)

    log.info("predictions_pooled", n_videos=n_loaded, n_predictions=len(all_coco_results))

    # ── Single-pass dataset-wide evaluation ───────────────────────────────────
    dataset_results = {}

    if not skip_segmentation:
        dataset_results["segmentation"] = _evaluate_segmentation(
            coco_gt, all_coco_results, iou_thresholds
        )

    dataset_results["tracking"] = _evaluate_tracking(coco_gt, all_coco_results, iou_threshold)
    dataset_results["attribution"] = _evaluate_attribution(coco_gt, all_coco_results, iou_threshold)
    dataset_results["n_videos"] = n_loaded

    # ── Save outputs ───────────────────────────────────────────────────────────
    with open(out / "dataset_results.json", "w") as f:
        json.dump(_to_serialisable(dataset_results), f, indent=2)

    per_flight = dataset_results["tracking"].get("per_flight", [])
    if per_flight:
        pd.DataFrame(per_flight).to_csv(out / "tracking_per_flight.csv", index=False)

    per_pred = dataset_results["attribution"].get("per_prediction", [])
    if per_pred:
        pd.DataFrame(per_pred).to_csv(out / "attribution_details.csv", index=False)

    seg   = dataset_results.get("segmentation", {})
    track = dataset_results["tracking"]["metrics"]
    attr  = dataset_results["attribution"]["metrics"]
    log.info(
        "dataset_evaluation_done",
        n_videos=n_loaded,
        mAP=round(seg.get("mAP", 0), 4) if seg else None,
        detection_rate=round(track.get("detection_rate", 0), 4),
        mean_completeness=round(track.get("mean_completeness", 0), 4),
        mean_tiou=round(track.get("mean_tiou", 0), 4),
        attribution_precision=round(attr.get("attribution_precision", 0), 4),
        output=str(out),
    )


if __name__ == "__main__":
    app()
