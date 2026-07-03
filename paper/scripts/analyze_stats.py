#!/usr/bin/env python3
"""
Statistical analyses for the paper:

1. Video-level cluster bootstrap 95% CIs for detection rate, track
   completeness, temporal IoU, attribution precision and attribution recall
   (first-detection), for all five prompt variants.
2. Paired video-level bootstrap for the binary vs best-variant differences.
3. ROC analysis of prompt rejection (per-object max score vs has-GT),
   with AUC, and FPR/FNR at the operational 0.5 threshold.

Outputs paper/scripts/stats_results.json used by tables and figures.

Usage (from a directory containing evaluation/ and outputs/):
    python paper/scripts/analyze_stats.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

VARIANTS = ["base", "age_5", "age_10", "ternary_5", "ternary_10"]
N_BOOT = 10_000
SEED = 20260702
IOU_DET = 0.25  # detection threshold used throughout the paper


def load_flight_video_map(mappings_dir: Path):
    """(video_id, flight_id) pairs; flight ids may repeat across videos."""
    fv = {}
    for vdir in sorted(mappings_dir.iterdir()):
        fm = vdir / "flight_mapping.json"
        if not fm.exists():
            continue
        m = json.load(open(fm))
        vid = m["video_id"]
        for fid in m["flight_to_object"]:
            fv.setdefault(fid, set()).add(vid)
    return fv


def image_video_map(annotations: Path):
    coco = json.load(open(annotations))
    out = {}
    for im in coco["images"]:
        vid = im.get("video_id")
        if vid is None:
            # fall back to file_name prefix "<video>/<frame>.jpg"
            vid = int(str(im["file_name"]).split("/")[0])
        out[im["id"]] = vid
    return out


def boot_ci(values_per_video: dict, stat_fn, rng, n_boot=N_BOOT):
    """Cluster bootstrap over videos: resample videos with replacement,
    pool their records, compute stat_fn(pooled_records)."""
    vids = sorted(values_per_video)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(len(vids), size=len(vids), replace=True)
        pooled = []
        for i in sample:
            pooled.extend(values_per_video[vids[i]])
        stats[b] = stat_fn(pooled)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def tracking_records_by_video(eval_dir: Path, fvmap):
    j = json.load(open(eval_dir / "evaluation_results.json"))
    per_flight = j["tracking_unified"]["per_flight"]
    by_video = defaultdict(list)
    ambiguous = 0
    for rec in per_flight:
        vids = fvmap.get(rec["flight_id"], set())
        if len(vids) == 1:
            by_video[next(iter(vids))].append(rec)
        else:
            ambiguous += 1
    return by_video, ambiguous, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", type=Path, default=Path("evaluation"))
    ap.add_argument("--outputs_root", type=Path, default=Path("outputs"))
    ap.add_argument("--mappings", type=Path, default=Path("inputs/mappings"))
    ap.add_argument("--annotations", type=Path, default=Path("annotations.json"))
    ap.add_argument(
        "--pod_age5", type=Path, default=Path("per_object_data_age_5")
    )
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "stats_results.json"
    )
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    fvmap = load_flight_video_map(args.mappings)
    imvid = image_video_map(args.annotations)

    results = {"n_boot": N_BOOT, "seed": SEED, "variants": {}}

    paired_store = {}

    for var in VARIANTS:
        eval_dir = args.eval_root / var
        by_video, ambiguous, j = tracking_records_by_video(eval_dir, fvmap)
        n_flights = sum(len(v) for v in by_video.values())
        print(f"[{var}] {n_flights} flights across {len(by_video)} videos "
              f"({ambiguous} ambiguous flight ids dropped)")

        det = lambda recs: np.mean([r["detected_frames"] > 0 for r in recs])
        comp = lambda recs: np.mean([r["completeness"] for r in recs])
        tiou = lambda recs: np.mean(
            [r["temporal_iou"] for r in recs if r["detected_frames"] > 0]
        )
        # temporal IoU over ALL flights (undetected contribute 0) — this is
        # the aggregation reported in the main results table
        tiou_all = lambda recs: np.mean([r["temporal_iou"] for r in recs])

        # point estimates on the FULL per-flight list (matches paper tables)
        allrec = [r for recs in by_video.values() for r in recs]
        point = {
            "detection_rate": float(det(allrec)),
            "completeness": float(comp(allrec)),
            "temporal_iou_detected": float(tiou(allrec)),
            "temporal_iou": float(tiou_all(allrec)),
        }

        cis = {
            "detection_rate": boot_ci(by_video, det, rng),
            "completeness": boot_ci(by_video, comp, rng),
            "temporal_iou_detected": boot_ci(by_video, tiou, rng),
            "temporal_iou": boot_ci(by_video, tiou_all, rng),
        }

        # attribution: per-prediction records grouped by video via image_id
        pp = j["attribution_unified"]["per_prediction"]
        attr_by_video = defaultdict(list)
        for rec in pp:
            attr_by_video[imvid.get(rec["image_id"], -1)].append(rec)

        def prec(recs):
            c = sum(r["outcome"] == "correct_attribution" for r in recs)
            w = sum(r["outcome"] == "wrong_attribution" for r in recs)
            return c / (c + w) if (c + w) else np.nan

        point["attribution_precision"] = float(prec(pp))
        cis["attribution_precision"] = boot_ci(attr_by_video, prec, rng)

        results["variants"][var] = {"point": point, "ci95": cis}
        paired_store[var] = (by_video, attr_by_video)

    # ── paired differences on the same resampled videos ──────────────────
    def paired_diff(var_a, var_b):
        (bv_a, av_a) = paired_store[var_a]
        (bv_b, av_b) = paired_store[var_b]
        vids = sorted(set(bv_a) & set(bv_b))
        diffs = {"completeness": [], "detection_rate": [],
                 "temporal_iou": [], "attribution_precision": []}
        for b in range(N_BOOT):
            sample = rng.choice(len(vids), size=len(vids), replace=True)
            ra, rb, aa, ab = [], [], [], []
            for i in sample:
                ra.extend(bv_a[vids[i]]); rb.extend(bv_b[vids[i]])
                aa.extend(av_a.get(vids[i], [])); ab.extend(av_b.get(vids[i], []))
            diffs["completeness"].append(
                np.mean([r["completeness"] for r in ra])
                - np.mean([r["completeness"] for r in rb])
            )
            diffs["detection_rate"].append(
                np.mean([r["detected_frames"] > 0 for r in ra])
                - np.mean([r["detected_frames"] > 0 for r in rb])
            )
            diffs["temporal_iou"].append(
                np.mean([r["temporal_iou"] for r in ra])
                - np.mean([r["temporal_iou"] for r in rb])
            )
            pa = [r for r in aa if r["outcome"] in ("correct_attribution", "wrong_attribution")]
            pb = [r for r in ab if r["outcome"] in ("correct_attribution", "wrong_attribution")]
            diffs["attribution_precision"].append(
                np.mean([r["outcome"] == "correct_attribution" for r in pa])
                - np.mean([r["outcome"] == "correct_attribution" for r in pb])
            )
        return {
            k: {
                "mean": float(np.mean(v)),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "p_leq_0": float(np.mean(np.array(v) <= 0)),
            }
            for k, v in diffs.items()
        }

    results["paired_ternary5_minus_base"] = paired_diff("ternary_5", "base")
    results["paired_ternary5_minus_age5"] = paired_diff("ternary_5", "age_5")
    results["paired_age5_minus_age10"] = paired_diff("age_5", "age_10")

    # ── ROC of prompt rejection (best variant) ───────────────────────────
    roc = compute_roc(args.outputs_root / "ternary_5", args.pod_age5)
    if roc is not None:
        results["prompt_rejection_roc"] = roc

    # ── prompt-as-prediction baseline (raw age-5 prompts vs GT) ──────────
    base = prompt_as_prediction(args.pod_age5)
    if base is not None:
        results["prompt_as_prediction"] = base

    args.out.write_text(json.dumps(results, indent=2))
    print(f"→ {args.out}")


def compute_roc(pred_dir: Path, pod_dir: Path):
    """Per-object max score vs has-GT → ROC/AUC, rates at 0.5 threshold.

    Mirrors plot_discrimination.compute_score_distributions: an object is a
    prompted (video, flight); it is positive when its per_object_data folder
    contains at least one *_mask.png (ground truth), and its score is the max
    prediction score over the video (0 when the model never fired).
    """
    import os

    if not pred_dir.exists() or not pod_dir.exists():
        print("outputs/per-object data not available; skipping ROC")
        return None
    scores, labels, videos = [], [], []
    for vid_dir in sorted(os.listdir(pod_dir)):
        pred_file = pred_dir / f"{vid_dir}.json"
        if not pred_file.exists():
            print(f"missing {pred_file}; skipping ROC")
            return None
        d = json.load(open(pred_file))
        by_obj = defaultdict(list)
        for ann in d["annotations"]:
            by_obj[int(ann["object_id"])].append(float(ann.get("score", 0)))
        for obj_folder in sorted(os.listdir(pod_dir / vid_dir)):
            obj_path = pod_dir / vid_dir / obj_folder
            if not obj_path.is_dir() or not obj_folder.isdigit():
                continue
            files = os.listdir(obj_path)
            if not any(f.endswith("_prompt.png") for f in files):
                continue
            has_gt = any(f.endswith("_mask.png") for f in files)
            scores.append(max(by_obj.get(int(obj_folder), [0.0])))
            labels.append(int(has_gt))
            videos.append(vid_dir)

    scores = np.array(scores); labels = np.array(labels); videos = np.array(videos)
    order = np.argsort(-scores)
    s, l = scores[order], labels[order]
    P, N = l.sum(), (1 - l).sum()
    tpr = np.cumsum(l) / P
    fpr = np.cumsum(1 - l) / N
    auc = float(np.trapezoid(tpr, fpr))
    thr = 0.5
    tp = ((scores >= thr) & (labels == 1)).sum()
    fn = ((scores < thr) & (labels == 1)).sum()
    fp = ((scores >= thr) & (labels == 0)).sum()
    tn = ((scores < thr) & (labels == 0)).sum()

    # video-level cluster bootstrap for AUC
    rng = np.random.default_rng(SEED + 1)
    vids = np.unique(videos)
    aucs = []
    for b in range(2000):
        sel = rng.choice(vids, size=len(vids), replace=True)
        idx = np.concatenate([np.where(videos == v)[0] for v in sel])
        ss, ll = scores[idx], labels[idx]
        o = np.argsort(-ss)
        ll = ll[o]
        Pb, Nb = ll.sum(), (1 - ll).sum()
        if Pb == 0 or Nb == 0:
            continue
        aucs.append(np.trapezoid(np.cumsum(ll) / Pb, np.cumsum(1 - ll) / Nb))
    return {
        "n_objects": int(len(scores)),
        "n_with_gt": int(P),
        "n_without_gt": int(N),
        "auc": auc,
        "auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "threshold": thr,
        "tpr_at_thr": float(tp / (tp + fn)),
        "fnr_at_thr": float(fn / (tp + fn)),
        "fpr_at_thr": float(fp / (fp + tn)),
        "tnr_at_thr": float(tn / (fp + tn)),
        "roc_fpr": fpr[:: max(1, len(fpr) // 500)].tolist(),
        "roc_tpr": tpr[:: max(1, len(tpr) // 500)].tolist(),
        "mean_score_gt": float(scores[labels == 1].mean()),
        "mean_score_nogt": float(scores[labels == 0].mean()),
    }


def _prompt_iou_one(obj_path_str):
    """IoU between binarized prompt and GT mask for every GT frame of one
    object. Returns list of per-frame IoUs (only frames that have GT)."""
    import os

    from PIL import Image

    obj_path = Path(obj_path_str)
    ious = []
    for f in os.listdir(obj_path):
        if not f.endswith("_mask.png"):
            continue
        frame = f[: -len("_mask.png")]
        pf = obj_path / f"{frame}_prompt.png"
        gt = np.array(Image.open(obj_path / f).convert("L")) > 127
        if not gt.any():
            continue
        if pf.exists():
            pr = np.array(Image.open(pf).convert("L")) > 0
        else:
            pr = np.zeros_like(gt)
        inter = np.logical_and(pr, gt).sum()
        union = np.logical_or(pr, gt).sum()
        ious.append(float(inter) / float(union) if union else 0.0)
    return ious


def prompt_as_prediction(pod_dir: Path):
    """Treat the rasterized (age-5) prompt itself as the predicted mask and
    score it against ground truth on every GT frame. Quantifies how much the
    visual model adds over the physics prior alone."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    if not pod_dir.exists():
        print("per_object_data_age_5 not available; skipping prompt baseline")
        return None
    jobs = []
    for vid_dir in sorted(os.listdir(pod_dir)):
        vpath = pod_dir / vid_dir
        if not vpath.is_dir():
            continue
        for obj_folder in sorted(os.listdir(vpath)):
            opath = vpath / obj_folder
            if opath.is_dir() and obj_folder.isdigit():
                if any(f.endswith("_mask.png") for f in os.listdir(opath)):
                    jobs.append(str(opath))
    print(f"prompt-as-prediction: {len(jobs)} objects with GT")
    per_flight_iou, per_frame_iou = [], []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for ious in ex.map(_prompt_iou_one, jobs, chunksize=8):
            if ious:
                per_frame_iou.extend(ious)
                per_flight_iou.append(float(np.mean(ious)))
    per_frame = np.array(per_frame_iou)
    per_flight = np.array(per_flight_iou)
    return {
        "n_flights": int(len(per_flight)),
        "n_gt_frames": int(len(per_frame)),
        "mean_frame_iou": float(per_frame.mean()),
        "median_frame_iou": float(np.median(per_frame)),
        "frac_frames_iou_ge_25": float((per_frame >= 0.25).mean()),
        "frac_frames_iou_ge_50": float((per_frame >= 0.50).mean()),
        "mean_flight_iou": float(per_flight.mean()),
        "flights_detected_at_25": float((per_flight >= 0.25).mean()),
    }


if __name__ == "__main__":
    main()
