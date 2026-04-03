"""Evaluation metrics for contrail detection, tracking, and attribution."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _load_gt(annotations_path: Path, video_name: str | None = None) -> COCO:
    """Load GT COCO annotations, optionally filtering to a single video."""
    with redirect_stdout(io.StringIO()):
        coco = COCO(str(annotations_path))

    if video_name is None:
        return coco

    # Build video_name → video_id mapping
    name_to_vid = {}
    for v in coco.dataset.get("videos", []):
        start = v["start"].replace("-", "").replace(":", "").replace("T", "")
        stop = v["stop"].replace("-", "").replace(":", "").replace("T", "")
        name_to_vid[f"{start}_{stop}"] = v["id"]

    vid_id = name_to_vid.get(video_name)
    if vid_id is None:
        raise ValueError(
            f"Video {video_name!r} not found in annotations. "
            f"Available: {list(name_to_vid.keys())}"
        )

    # Filter to this video's images and annotations
    img_ids = {img["id"] for img in coco.dataset["images"] if img.get("video_id") == vid_id}
    filtered = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": [img for img in coco.dataset["images"] if img["id"] in img_ids],
        "videos": [v for v in coco.dataset["videos"] if v["id"] == vid_id],
        "categories": coco.dataset["categories"],
        "annotations": [a for a in coco.dataset["annotations"] if a["image_id"] in img_ids],
    }

    filtered_coco = COCO()
    with redirect_stdout(io.StringIO()):
        filtered_coco.dataset = filtered
        filtered_coco.createIndex()
    return filtered_coco


def _build_image_lookup(coco_gt: COCO) -> dict:
    """Build {video_id: {frame_idx: image_id}} from COCO GT images."""
    from collections import defaultdict
    imgs_by_video = defaultdict(list)
    for img in coco_gt.dataset["images"]:
        vid = img.get("video_id")
        if vid is not None:
            imgs_by_video[vid].append(img)

    lookup = {}
    for vid, imgs in imgs_by_video.items():
        lookup[vid] = {
            idx: img["id"]
            for idx, img in enumerate(sorted(imgs, key=lambda x: x["id"]))
        }
    return lookup


def _predictions_to_coco(
    predictions_path: Path,
    coco_gt: COCO,
    video_name: str,
) -> list[dict]:
    """Convert a contrailtrack prediction JSON to COCO evaluation format.

    Predictions use flight_id directly (no mapping file needed).
    """
    with open(predictions_path) as f:
        pred_data = json.load(f)

    # Find the COCO video_id for this video
    name_to_vid = {}
    for v in coco_gt.dataset.get("videos", []):
        start = v["start"].replace("-", "").replace(":", "").replace("T", "")
        stop = v["stop"].replace("-", "").replace(":", "").replace("T", "")
        name_to_vid[f"{start}_{stop}"] = v["id"]

    vid_id = name_to_vid.get(video_name)
    if vid_id is None:
        return []

    image_lookup = _build_image_lookup(coco_gt)
    vid_images = image_lookup.get(vid_id, {})

    results = []
    for ann in pred_data.get("annotations", []):
        frame_idx = ann.get("frame_idx")
        flight_id = ann.get("flight_id")
        segmentation = ann.get("segmentation")
        score = ann.get("score", 1.0)

        if frame_idx is None or flight_id is None or segmentation is None:
            continue

        image_id = vid_images.get(frame_idx)
        if image_id is None:
            continue

        if isinstance(segmentation.get("counts"), str):
            segmentation["counts"] = segmentation["counts"].encode("utf-8")

        area = float(mask_utils.area(segmentation))
        bbox = mask_utils.toBbox(segmentation).tolist()

        results.append({
            "image_id": image_id,
            "category_id": 1,
            "segmentation": segmentation,
            "score": score,
            "bbox": bbox,
            "area": area,
            "_flight_id": flight_id,
            "_score": score,
        })

    return results


def _evaluate_segmentation(coco_gt: COCO, coco_results: list[dict], iou_thresholds: list) -> dict:
    """Run COCO-style segmentation mAP."""
    if not coco_results:
        return {"mAP": 0.0}

    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.iouThrs = np.array(iou_thresholds)
    coco_eval.evaluate()
    coco_eval.accumulate()

    buf = io.StringIO()
    with redirect_stdout(buf):
        coco_eval.summarize()

    return {
        "mAP": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]) if len(coco_eval.stats) > 1 else None,
        "summary": buf.getvalue(),
    }


def _evaluate_tracking(coco_gt: COCO, coco_results: list[dict], iou_threshold: float) -> dict:
    """Evaluate tracking: detection rate, completeness, temporal IoU per flight."""
    if not coco_results:
        return {"metrics": {}, "per_flight": []}

    # Group GT annotations by flight_id
    gt_by_flight = {}
    for ann in coco_gt.dataset["annotations"]:
        fid = ann.get("flight_id")
        if fid:
            gt_by_flight.setdefault(fid, []).append(ann)

    # Group predictions by flight_id
    pred_by_flight = {}
    for r in coco_results:
        fid = r.get("_flight_id")
        if fid:
            pred_by_flight.setdefault(fid, []).append(r)

    per_flight = []
    for fid, gt_anns in gt_by_flight.items():
        gt_img_ids = {a["image_id"] for a in gt_anns}
        pred_anns = pred_by_flight.get(fid, [])
        pred_img_ids = {r["image_id"] for r in pred_anns}

        # Check IoU overlap on matched frames
        matched_frames = 0
        for img_id in gt_img_ids & pred_img_ids:
            gt_masks = [a for a in gt_anns if a["image_id"] == img_id]
            pr_masks = [r for r in pred_anns if r["image_id"] == img_id]
            if gt_masks and pr_masks:
                gt_rle = gt_masks[0]["segmentation"]
                pr_rle = pr_masks[0]["segmentation"]
                if isinstance(gt_rle, list):
                    gt_rle = mask_utils.merge(mask_utils.frPyObjects(gt_rle, 1024, 1024))
                if isinstance(gt_rle.get("counts"), str):
                    gt_rle["counts"] = gt_rle["counts"].encode("utf-8")
                iou = mask_utils.iou([pr_rle], [gt_rle], [0])[0][0]
                if iou >= iou_threshold:
                    matched_frames += 1

        completeness = matched_frames / len(gt_img_ids) if gt_img_ids else 0.0
        detected = matched_frames > 0

        # Temporal IoU
        if gt_img_ids and pred_img_ids:
            intersection = len(gt_img_ids & pred_img_ids)
            union = len(gt_img_ids | pred_img_ids)
            tiou = intersection / union
        else:
            tiou = 0.0

        per_flight.append({
            "flight_id": fid,
            "gt_frames": len(gt_img_ids),
            "pred_frames": len(pred_img_ids),
            "matched_frames": matched_frames,
            "completeness": completeness,
            "tiou": tiou,
            "detected": detected,
        })

    n = len(per_flight)
    metrics = {
        "n_flights": n,
        "detection_rate": sum(f["detected"] for f in per_flight) / n if n else 0.0,
        "mean_completeness": np.mean([f["completeness"] for f in per_flight]) if n else 0.0,
        "mean_tiou": np.mean([f["tiou"] for f in per_flight]) if n else 0.0,
    }

    return {"metrics": metrics, "per_flight": per_flight}


def _evaluate_attribution(coco_gt: COCO, coco_results: list[dict], iou_threshold: float) -> dict:
    """Evaluate attribution: is each prediction assigned to the correct flight?"""
    if not coco_results:
        return {"metrics": {}, "per_prediction": []}

    # Build GT lookup: image_id → {flight_id: rle_mask}
    gt_lookup = {}
    for ann in coco_gt.dataset["annotations"]:
        fid = ann.get("flight_id")
        if not fid:
            continue
        img_id = ann["image_id"]
        seg = ann["segmentation"]
        if isinstance(seg, list):
            seg = mask_utils.merge(mask_utils.frPyObjects(seg, 1024, 1024))
        if isinstance(seg.get("counts"), str):
            seg["counts"] = seg["counts"].encode("utf-8")
        gt_lookup.setdefault(img_id, {})[fid] = seg

    per_prediction = []
    for r in coco_results:
        img_id = r["image_id"]
        pred_fid = r.get("_flight_id")
        pred_rle = r["segmentation"]

        gt_at_frame = gt_lookup.get(img_id, {})
        if not gt_at_frame:
            per_prediction.append({"flight_id": pred_fid, "image_id": img_id, "correct": None, "best_iou": 0.0})
            continue

        # Find best matching GT flight
        best_iou = 0.0
        best_fid = None
        for gt_fid, gt_rle in gt_at_frame.items():
            iou = mask_utils.iou([pred_rle], [gt_rle], [0])[0][0]
            if iou > best_iou:
                best_iou = iou
                best_fid = gt_fid

        correct = (best_fid == pred_fid) if best_iou >= iou_threshold else None

        per_prediction.append({
            "flight_id": pred_fid,
            "image_id": img_id,
            "correct": correct,
            "best_iou": float(best_iou),
            "matched_gt_flight": best_fid,
        })

    assessed = [p for p in per_prediction if p["correct"] is not None]
    n_correct = sum(1 for p in assessed if p["correct"])
    metrics = {
        "n_predictions": len(per_prediction),
        "n_assessed": len(assessed),
        "n_correct": n_correct,
        "attribution_precision": n_correct / len(assessed) if assessed else 0.0,
    }

    return {"metrics": metrics, "per_prediction": per_prediction}


def evaluate(
    predictions_path: str | Path,
    gt_annotations: str | Path,
    video_name: str | None = None,
    iou_thresholds: list | None = None,
    iou_threshold: float = 0.25,
    output_dir: str | Path | None = None,
    skip_segmentation: bool = False,
) -> dict:
    """Run full evaluation: segmentation mAP + tracking + attribution.

    Args:
        predictions_path: COCO JSON prediction file.
        gt_annotations: Path to GT COCO annotations.json.
        video_name: Video name (e.g. "20230930055430_20230930075430"). If None,
                    inferred from the predictions filename.
        iou_thresholds: Custom IoU thresholds for mAP (default: 0.25-0.75 step 0.05).
        iou_threshold: IoU threshold for tracking/attribution matching (default: 0.25).
        output_dir: If set, write result CSVs and JSON here.
        skip_segmentation: Skip slow COCO mAP computation.

    Returns:
        Dict with keys: "segmentation", "tracking", "attribution".
    """
    predictions_path = Path(predictions_path)
    gt_annotations = Path(gt_annotations)

    if iou_thresholds is None:
        iou_thresholds = list(np.arange(0.25, 0.76, 0.05))

    if video_name is None:
        video_name = predictions_path.stem

    coco_gt = _load_gt(gt_annotations, video_name=video_name)
    coco_results = _predictions_to_coco(predictions_path, coco_gt, video_name)

    results = {}

    if not skip_segmentation and coco_results:
        results["segmentation"] = _evaluate_segmentation(coco_gt, coco_results, iou_thresholds)

    results["tracking"] = _evaluate_tracking(coco_gt, coco_results, iou_threshold)
    results["attribution"] = _evaluate_attribution(coco_gt, coco_results, iou_threshold)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(_to_serialisable(results), f, indent=2)

        per_flight = results["tracking"].get("per_flight", [])
        if per_flight:
            pd.DataFrame(per_flight).to_csv(output_dir / "tracking_per_flight.csv", index=False)

        per_pred = results["attribution"].get("per_prediction", [])
        if per_pred:
            pd.DataFrame(per_pred).to_csv(output_dir / "attribution_details.csv", index=False)

    return results


def evaluate_segmentation(
    predictions_path: str | Path,
    gt_annotations: str | Path,
    video_name: str | None = None,
    iou_thresholds: list | None = None,
) -> dict:
    """Run segmentation mAP evaluation only."""
    return evaluate(
        predictions_path, gt_annotations, video_name=video_name,
        iou_thresholds=iou_thresholds, skip_segmentation=False,
    ).get("segmentation", {})


def evaluate_tracking(
    predictions_path: str | Path,
    gt_annotations: str | Path,
    video_name: str | None = None,
    iou_threshold: float = 0.25,
) -> dict:
    """Run tracking evaluation only."""
    return evaluate(
        predictions_path, gt_annotations, video_name=video_name,
        iou_threshold=iou_threshold, skip_segmentation=True,
    ).get("tracking", {})


def evaluate_attribution(
    predictions_path: str | Path,
    gt_annotations: str | Path,
    video_name: str | None = None,
    iou_threshold: float = 0.25,
) -> dict:
    """Run attribution evaluation only."""
    return evaluate(
        predictions_path, gt_annotations, video_name=video_name,
        iou_threshold=iou_threshold, skip_segmentation=True,
    ).get("attribution", {})


def _to_serialisable(obj):
    """Recursively convert numpy types to Python native for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
