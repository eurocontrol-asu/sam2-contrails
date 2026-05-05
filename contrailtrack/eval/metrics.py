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


def _rle_from_ann(ann: dict, coco_gt: COCO) -> dict:
    """Return a bytes-counts RLE for a GT annotation (handles polygon or RLE input)."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        img = coco_gt.imgs[ann["image_id"]]
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, img["height"], img["width"]))
    else:
        rle = dict(seg)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def _rle_bytes(rle: dict) -> dict:
    """Ensure RLE counts field is bytes (required by mask_utils.iou)."""
    if isinstance(rle.get("counts"), str):
        rle = dict(rle)
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def _evaluate_tracking(coco_gt: COCO, coco_results: list[dict], iou_threshold: float) -> dict:
    """Evaluate tracking with spatial IoU matching (GT-centric, decoupled from attribution).

    For each GT flight, find the best-matching prediction at each frame purely by
    spatial IoU (any prediction regardless of its flight_id label). This separates
    tracking quality from attribution quality.

    Per-flight fields match the legacy evaluate_tracking_unified output:
    ``num_frames``, ``detected_frames``, ``completeness``, ``temporal_iou``
    (mean spatial IoU over detected frames), ``mean_score``, ``fragmentation``
    (number of distinct prediction IDs used), ``pred_frames_outside_gt``,
    ``max_detection_gap``, ``pred_ids_used``.
    """
    if not coco_results:
        return {"metrics": {}, "per_flight": []}

    # GT: flight_id -> {image_id: [annotations]}
    gt_by_flight: dict[str, dict[int, list]] = {}
    for ann in coco_gt.dataset["annotations"]:
        fid = ann.get("flight_id")
        if fid:
            gt_by_flight.setdefault(fid, {}).setdefault(ann["image_id"], []).append(ann)

    # Predictions indexed by image_id for per-frame spatial search
    pred_by_image: dict[int, list] = {}
    for r in coco_results:
        pred_by_image.setdefault(r["image_id"], []).append(r)

    # Prediction image_ids per flight_id (for pred_frames_outside_gt)
    pred_img_ids_by_flight: dict[str, set] = {}
    for r in coco_results:
        fid = r.get("_flight_id")
        if fid:
            pred_img_ids_by_flight.setdefault(fid, set()).add(r["image_id"])

    per_flight = []
    for flight_id, frame_anns in gt_by_flight.items():
        gt_image_ids = set(frame_anns.keys())
        num_frames = len(gt_image_ids)

        detected_frames = 0
        total_iou = 0.0
        total_score = 0.0
        pred_ids_used: set[str] = set()
        best_matches = []

        for image_id in sorted(frame_anns.keys()):
            anns = frame_anns[image_id]

            # Merge all GT masks for this flight at this frame into one RLE
            if len(anns) == 1:
                gt_rle = _rle_from_ann(anns[0], coco_gt)
            else:
                gt_rle = mask_utils.merge([_rle_from_ann(a, coco_gt) for a in anns])

            # Find best-matching prediction by spatial IoU (any prediction)
            preds_at_frame = pred_by_image.get(image_id, [])
            best_iou = 0.0
            best_pred_fid = None
            best_score = None

            if preds_at_frame:
                pr_rles = [_rle_bytes(r["segmentation"]) for r in preds_at_frame]
                ious = mask_utils.iou(pr_rles, [gt_rle], [0])  # shape (n_pred, 1)
                best_idx = int(np.argmax(ious[:, 0]))
                best_iou = float(ious[best_idx, 0])
                best_pred_fid = preds_at_frame[best_idx].get("_flight_id")
                best_score = preds_at_frame[best_idx].get("_score")

            if best_iou >= iou_threshold:
                detected_frames += 1
                total_iou += best_iou
                if best_score is not None:
                    total_score += best_score
                if best_pred_fid is not None:
                    pred_ids_used.add(best_pred_fid)
                best_matches.append({"image_id": image_id, "detected": True, "iou": best_iou})
            else:
                best_matches.append({"image_id": image_id, "detected": False, "iou": best_iou})

        completeness = detected_frames / num_frames if num_frames > 0 else 0.0
        # temporal_iou: mean spatial IoU over detected frames (not frame-set overlap)
        temporal_iou = total_iou / detected_frames if detected_frames > 0 else 0.0
        mean_score = total_score / detected_frames if detected_frames > 0 else 0.0
        fragmentation = len(pred_ids_used)

        pred_frames_outside_gt = len(
            pred_img_ids_by_flight.get(flight_id, set()) - gt_image_ids
        )

        max_gap = 0
        current_gap = 0
        for m in best_matches:
            if not m["detected"]:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0

        per_flight.append({
            "flight_id": flight_id,
            "num_frames": num_frames,
            "detected_frames": detected_frames,
            "completeness": completeness,
            "temporal_iou": temporal_iou,
            "mean_score": mean_score,
            "fragmentation": fragmentation,
            "pred_frames_outside_gt": pred_frames_outside_gt,
            "max_detection_gap": max_gap,
            "pred_ids_used": sorted(pred_ids_used),
        })

    n = len(per_flight)
    detected = [f for f in per_flight if f["detected_frames"] > 0]
    metrics = {
        "n_flights": n,
        "detection_rate": len(detected) / n if n else 0.0,
        "mean_completeness": float(np.mean([f["completeness"] for f in per_flight])) if n else 0.0,
        "mean_tiou": float(np.mean([f["temporal_iou"] for f in per_flight])) if n else 0.0,
        "mean_fragmentation": float(np.mean([f["fragmentation"] for f in per_flight])) if n else 0.0,
        "iou_threshold": iou_threshold,
    }

    return {"metrics": metrics, "per_flight": per_flight}


def _evaluate_attribution(coco_gt: COCO, coco_results: list[dict], iou_threshold: float) -> dict:
    """Evaluate attribution using greedy per-frame IoU matching.

    At each frame, greedily matches predictions to GT annotations by highest IoU
    (above threshold), then classifies each match by outcome:

    - ``correct_attribution``: pred and GT share the same flight_id
    - ``wrong_attribution``: pred flight_id != GT flight_id (both are new contrails)
    - ``false_attribution``: pred has a flight_id but GT is an old contrail (no flight_id)
    - ``missed_attribution``: GT is a new contrail but pred has no flight_id
    - ``correct_omission``: GT is an old contrail and pred has no flight_id

    Metrics reported are the per-prediction attribution precision and recall
    (matching the legacy evaluate_attribution_unified methodology).
    """
    if not coco_results:
        return {"metrics": {}, "per_prediction": [], "first_frame_attribution": {}}

    from collections import defaultdict

    # Index GT and predictions by image_id
    gt_by_image: dict[int, list] = defaultdict(list)
    for ann in coco_gt.dataset["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)

    pred_by_image: dict[int, list] = defaultdict(list)
    for r in coco_results:
        pred_by_image[r["image_id"]].append(r)

    all_image_ids = set(pred_by_image) | set(gt_by_image)

    per_prediction = []
    counters = dict(
        correct_attribution=0, wrong_attribution=0,
        false_attribution=0, missed_attribution=0,
        correct_omission=0, unmatched_predictions=0,
        unmatched_gt_new=0, unmatched_gt_old=0,
    )

    for image_id in all_image_ids:
        preds = pred_by_image.get(image_id, [])
        gt_anns = gt_by_image.get(image_id, [])

        if not preds or not gt_anns:
            for pred in preds:
                counters["unmatched_predictions"] += 1
                per_prediction.append({
                    "image_id": image_id,
                    "pred_flight_id": pred.get("_flight_id"),
                    "score": pred.get("_score"),
                    "outcome": "unmatched_prediction",
                    "matched_gt_flight": None,
                    "iou": 0.0,
                })
            for ann in gt_anns:
                if ann.get("flight_id"):
                    counters["unmatched_gt_new"] += 1
                else:
                    counters["unmatched_gt_old"] += 1
            continue

        # Build RLE arrays
        pr_rles = [_rle_bytes(r["segmentation"]) for r in preds]
        gt_rles = [_rle_from_ann(a, coco_gt) for a in gt_anns]

        iou_matrix = mask_utils.iou(pr_rles, gt_rles, [0] * len(gt_rles))

        matched_preds: set[int] = set()
        matched_gts: set[int] = set()

        # Greedy matching: highest IoU first
        while True:
            best_val = 0.0
            best_pair = None
            for i in range(len(preds)):
                if i in matched_preds:
                    continue
                for j in range(len(gt_anns)):
                    if j in matched_gts:
                        continue
                    if iou_matrix[i, j] > best_val:
                        best_val = iou_matrix[i, j]
                        best_pair = (i, j)
            if best_pair is None or best_val < iou_threshold:
                break

            i, j = best_pair
            matched_preds.add(i)
            matched_gts.add(j)

            pred_fid = preds[i].get("_flight_id")
            gt_fid = gt_anns[j].get("flight_id")

            if pred_fid:
                if gt_fid:
                    outcome = "correct_attribution" if pred_fid == gt_fid else "wrong_attribution"
                else:
                    outcome = "false_attribution"
            else:
                outcome = "missed_attribution" if gt_fid else "correct_omission"

            counters[outcome] += 1
            per_prediction.append({
                "image_id": image_id,
                "pred_flight_id": pred_fid,
                "score": preds[i].get("_score"),
                "outcome": outcome,
                "matched_gt_flight": gt_fid,
                "iou": float(best_val),
            })

        for i in range(len(preds)):
            if i not in matched_preds:
                counters["unmatched_predictions"] += 1
                per_prediction.append({
                    "image_id": image_id,
                    "pred_flight_id": preds[i].get("_flight_id"),
                    "score": preds[i].get("_score"),
                    "outcome": "unmatched_prediction",
                    "matched_gt_flight": None,
                    "iou": 0.0,
                })
        for j in range(len(gt_anns)):
            if j not in matched_gts:
                if gt_anns[j].get("flight_id"):
                    counters["unmatched_gt_new"] += 1
                else:
                    counters["unmatched_gt_old"] += 1

    ca = counters["correct_attribution"]
    wa = counters["wrong_attribution"]
    fa = counters["false_attribution"]
    ma = counters["missed_attribution"]
    co = counters["correct_omission"]
    un_new = counters["unmatched_gt_new"]
    un_old = counters["unmatched_gt_old"]

    total_attributed = ca + wa + fa
    total_new_gt = ca + wa + ma + un_new
    total_old_gt = fa + co + un_old

    metrics = {
        "attribution_precision": ca / total_attributed if total_attributed > 0 else 0.0,
        "attribution_recall": ca / total_new_gt if total_new_gt > 0 else 0.0,
        "wrong_attribution_rate": wa / total_attributed if total_attributed > 0 else 0.0,
        "false_attribution_rate": fa / total_attributed if total_attributed > 0 else 0.0,
        "missed_attribution_rate": (ma + un_new) / total_new_gt if total_new_gt > 0 else 0.0,
        "correct_omission_rate": (co + un_old) / total_old_gt if total_old_gt > 0 else 0.0,
        "iou_threshold": iou_threshold,
        **{k: v for k, v in counters.items()},
        "total_attributed_predictions": total_attributed,
        "total_new_gt": total_new_gt,
        "total_old_gt": total_old_gt,
    }

    # First-frame attribution: check attribution on the first frame each GT flight is matched
    total_gt_flights = len(
        {a.get("flight_id") for a in coco_gt.dataset["annotations"] if a.get("flight_id")}
    )
    matched_rows = [r for r in per_prediction if r.get("matched_gt_flight") is not None]
    first_by_flight: dict[str, dict] = {}
    for row in matched_rows:
        fid = row["matched_gt_flight"]
        if fid is None:
            continue
        if fid not in first_by_flight or row["image_id"] < first_by_flight[fid]["image_id"]:
            first_by_flight[fid] = row

    ff_rows = list(first_by_flight.values())
    ff_correct = sum(1 for r in ff_rows if r["outcome"] == "correct_attribution")
    ff_wrong = sum(1 for r in ff_rows if r["outcome"] == "wrong_attribution")
    ff_false = sum(1 for r in ff_rows if r["outcome"] == "false_attribution")
    ff_attributed = ff_correct + ff_wrong + ff_false

    first_frame_attribution = {
        "total_gt_flights": total_gt_flights,
        "flights_matched_at_least_once": len(first_by_flight),
        "flights_never_matched": total_gt_flights - len(first_by_flight),
        "metrics": {
            "attribution_precision": ff_correct / ff_attributed if ff_attributed > 0 else 0.0,
            "attribution_recall": ff_correct / total_gt_flights if total_gt_flights > 0 else 0.0,
            "detection_rate": len(first_by_flight) / total_gt_flights if total_gt_flights > 0 else 0.0,
        },
    }

    return {
        "metrics": metrics,
        "per_prediction": per_prediction,
        "first_frame_attribution": first_frame_attribution,
    }


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
