"""Analyse which glitches were fixed by negative-only prompts and which persist.

For each baseline glitch event, classifies it as:
  - fixed:     present in baseline, absent in negprompt run
  - persisting: present in both
  - new:       present in negprompt only (regression)

Key hypothesis:
  Fixed glitches occur AFTER the object's last positive prompt (during propagation).
  Persisting glitches occur DURING the positive prompt window (memory overwhelms prompt).

Generates per-case visualisation panels saved to docs/teleportation_diagnosis/glitch_fix/.
"""

import sys
sys.path.insert(0, "/data/common/dataiku/config/projects/TRAILVISION/lib/python/sam2")

import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# ── Paths ──────────────────────────────────────────────────────────────────────
BASELINE_DIR   = Path("outputs/ternary_5_rerun_remapped")
NEGPROMPT_DIR  = Path("outputs/ternary_5_negprompt")
FRAMES_DIR     = Path("/data/common/TRAILVISION/GVCCS_V/test/img_folder")
PROMPTS_DIR    = Path("/data/common/TRAILVISION/GVCCS_V/test/per_object_data_age_5")
OUT_DIR        = Path("docs/teleportation_diagnosis/glitch_fix")

JUMP_THRESHOLD = 150   # px
MAX_CASES      = 6     # visualise top N fixed + top N persisting


# ── Helpers ────────────────────────────────────────────────────────────────────

def bbox_centroid(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def decode_mask(seg, h, w):
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    # polygon fallback
    from pycocotools import mask as mu
    rle = mu.frPyObjects(seg, h, w)
    return mu.decode(mu.merge(rle)).astype(bool)


def load_flight_mapping(video_id):
    """Returns {flight_id: object_folder_int}."""
    p = PROMPTS_DIR / video_id / "flight_mapping.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    return d.get("flight_to_object", {})


def last_positive_prompt_frame(video_id, flight_id, flight_mapping):
    """Return the last frame name (str) that has a positive prompt for this object."""
    obj_folder = str(flight_mapping.get(flight_id, ""))
    obj_dir = PROMPTS_DIR / video_id / obj_folder
    if not obj_dir.exists():
        return None
    prompt_files = sorted(obj_dir.glob("*_prompt.png"))
    if not prompt_files:
        return None
    # Check which ones are non-zero
    last = None
    for pf in prompt_files:
        arr = np.array(Image.open(pf))
        if arr.any():
            frame_name = pf.stem.replace("_prompt", "")
            last = frame_name
    return last


def find_glitches(pred_dir):
    """Returns {video_id: {flight_id: [(frame_idx_before, frame_idx_after, dist), ...]}}"""
    glitches = {}
    for jf in sorted(pred_dir.glob("*.json")):
        video_id = jf.stem
        data = json.load(open(jf))
        by_obj = defaultdict(list)
        for ann in data["annotations"]:
            by_obj[ann["flight_id"]].append(ann)

        video_glitches = {}
        for flight_id, anns in by_obj.items():
            anns_sorted = sorted(anns, key=lambda a: a["frame_idx"])
            prev_cx, prev_cy, prev_idx = None, None, None
            obj_glitches = []
            for ann in anns_sorted:
                cx, cy = bbox_centroid(ann["bbox"])
                fi = ann["frame_idx"]
                if prev_cx is not None:
                    dist = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    if dist > JUMP_THRESHOLD:
                        obj_glitches.append((prev_idx, fi, dist))
                prev_cx, prev_cy, prev_idx = cx, cy, fi
            if obj_glitches:
                video_glitches[flight_id] = obj_glitches
        if video_glitches:
            glitches[video_id] = video_glitches
    return glitches


def classify_glitches(baseline_glitches, negprompt_glitches):
    """Returns lists of (video_id, flight_id, glitch_list, category)."""
    fixed, persisting, new_cases = [], [], []

    all_videos = set(baseline_glitches) | set(negprompt_glitches)
    for vid in sorted(all_videos):
        base_objs = baseline_glitches.get(vid, {})
        neg_objs  = negprompt_glitches.get(vid, {})

        for fid, glist in base_objs.items():
            if fid in neg_objs:
                persisting.append((vid, fid, glist, "persisting"))
            else:
                fixed.append((vid, fid, glist, "fixed"))

        for fid, glist in neg_objs.items():
            if fid not in base_objs:
                new_cases.append((vid, fid, glist, "new"))

    # Sort by largest total jump distance first
    key = lambda x: sum(g[2] for g in x[2])
    fixed.sort(key=key, reverse=True)
    persisting.sort(key=key, reverse=True)
    new_cases.sort(key=key, reverse=True)
    return fixed, persisting, new_cases


# ── Visualisation ──────────────────────────────────────────────────────────────

def load_frame_rgb(video_id, frame_idx):
    frame_dir = FRAMES_DIR / video_id
    files = sorted(frame_dir.glob("*.jpg"))
    if frame_idx < len(files):
        return np.array(Image.open(files[frame_idx]).convert("RGB")), files[frame_idx].stem
    return np.zeros((1024, 1024, 3), dtype=np.uint8), f"{frame_idx:05d}"


def get_pred_at_frame(pred_dir, video_id, flight_id, frame_idx):
    """Return (mask bool [H,W], score, bbox) or None."""
    jf = pred_dir / f"{video_id}.json"
    if not jf.exists():
        return None
    data = json.load(open(jf))
    h = data["video"].get("height", 1024)
    w = data["video"].get("width", 1024)
    for ann in data["annotations"]:
        if ann["flight_id"] == flight_id and ann["frame_idx"] == frame_idx:
            mask = decode_mask(ann["segmentation"], h, w)
            return mask, ann["score"], ann["bbox"]
    return None


def load_prompt_ternary(video_id, flight_id, frame_name, flight_mapping):
    """Return float32 [H,W] ternary array: >0 positive (own), <0 negative (union), or None."""
    obj_folder = str(flight_mapping.get(flight_id, ""))
    own_path = PROMPTS_DIR / video_id / obj_folder / f"{frame_name}_prompt.png" if obj_folder else None
    union_path = PROMPTS_DIR / video_id / f"{frame_name}_all_prompts_union.png"

    own = np.array(Image.open(own_path)).astype(np.float32) / 255.0 if (own_path and own_path.exists()) else None
    union = np.array(Image.open(union_path)).astype(np.float32) / 255.0 if union_path.exists() else None

    if own is None and union is None:
        return None
    if own is None:
        return -union
    if union is None:
        return own
    return np.where(own > 0, own, -union)


def blend_ternary(rgb, ternary, alpha=0.55):
    """Blue overlay for positive prompt values, orange for negative."""
    out = rgb.copy().astype(float)
    pos = np.clip(ternary, 0, 1)[:, :, np.newaxis]
    neg = np.clip(-ternary, 0, 1)[:, :, np.newaxis]
    out = out * (1 - alpha * pos) + np.array([30, 100, 255]) * alpha * pos
    out = out * (1 - alpha * neg) + np.array([230, 100, 0]) * alpha * neg
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_preds(rgb, base_pred, neg_pred, alpha=0.5):
    """Red overlay for baseline prediction, blue for negprompt prediction."""
    out = rgb.copy().astype(float)
    if base_pred is not None:
        m = base_pred[0].astype(float)[:, :, np.newaxis]
        out = out * (1 - alpha * m) + np.array([255, 60, 60]) * alpha * m
    if neg_pred is not None:
        m = neg_pred[0].astype(float)[:, :, np.newaxis]
        out = out * (1 - alpha * m) + np.array([60, 120, 255]) * alpha * m
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_bbox_rect(ax, bbox, color, lw=2):
    x, y, w, h = bbox
    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=lw,
                                    edgecolor=color, facecolor="none"))


def visualise_case(video_id, flight_id, base_glitches, neg_glitches, category,
                   last_pos_frame_name, out_path):
    """3-row × 2-col figure using FULL images (no crop):
      Row 0: raw image
      Row 1: ternary prompt overlay on full image (blue=pos, orange=neg)
      Row 2: prediction overlay on full image (red=baseline, blue=negprompt)
              with bbox outlines so positions are clear
      Col 0: frame BEFORE glitch
      Col 1: glitch frame
    """
    all_jumps = base_glitches if base_glitches else neg_glitches
    frame_before = all_jumps[0][0]
    frame_glitch  = all_jumps[0][1]
    jump_dist     = all_jumps[0][2]

    flight_mapping = load_flight_mapping(video_id)

    rgb_before, fname_before = load_frame_rgb(video_id, frame_before)
    rgb_glitch,  fname_glitch  = load_frame_rgb(video_id, frame_glitch)

    # ── Prompts ──────────────────────────────────────────────────────────────
    tern_b = load_prompt_ternary(video_id, flight_id, fname_before, flight_mapping)
    tern_g = load_prompt_ternary(video_id, flight_id, fname_glitch,  flight_mapping)
    prompt_b = blend_ternary(rgb_before, tern_b) if tern_b is not None else rgb_before.copy()
    prompt_g = blend_ternary(rgb_glitch,  tern_g) if tern_g is not None else rgb_glitch.copy()

    # ── Predictions ──────────────────────────────────────────────────────────
    base_pred_b = get_pred_at_frame(BASELINE_DIR,  video_id, flight_id, frame_before)
    neg_pred_b  = get_pred_at_frame(NEGPROMPT_DIR, video_id, flight_id, frame_before)
    base_pred_g = get_pred_at_frame(BASELINE_DIR,  video_id, flight_id, frame_glitch)
    neg_pred_g  = get_pred_at_frame(NEGPROMPT_DIR, video_id, flight_id, frame_glitch)

    pred_b = blend_preds(rgb_before, base_pred_b, neg_pred_b)
    pred_g = blend_preds(rgb_glitch,  base_pred_g, neg_pred_g)

    def _score(pred):
        return f"{pred[1]:.2f}" if pred else "—"

    # ── Figure ───────────────────────────────────────────────────────────────
    cat_color = {"fixed": "#1a9641", "persisting": "#d7191c", "new": "#f07030"}[category]
    cat_label = {"fixed": "FIXED ✓", "persisting": "PERSISTING ✗", "new": "NEW ⚠"}[category]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14),
                             gridspec_kw={"hspace": 0.15, "wspace": 0.05})

    panels = [
        [rgb_before, rgb_glitch],
        [prompt_b,   prompt_g],
        [pred_b,     pred_g],
    ]
    col_titles = [
        [f"frame {frame_before}  (before)",
         f"frame {frame_glitch}  (glitch  Δ={jump_dist:.0f} px)"],
        ["prompt  (blue=pos, orange=neg)", "prompt"],
        [f"base={_score(base_pred_b)}  neg={_score(neg_pred_b)}",
         f"base={_score(base_pred_g)}  neg={_score(neg_pred_g)}"],
    ]
    row_labels = ["Raw", "Prompt", "Pred\n(red=base\nblue=neg)"]

    for r in range(3):
        for c in range(2):
            ax = axes[r, c]
            ax.imshow(panels[r][c])
            ax.set_title(col_titles[r][c], fontsize=8, pad=3)
            ax.axis("off")
        axes[r, 0].set_ylabel(row_labels[r], fontsize=9, labelpad=4)

    # Add bbox outlines on prediction row so positions are unambiguous
    for c, (bp, np_) in enumerate([(base_pred_b, neg_pred_b), (base_pred_g, neg_pred_g)]):
        ax = axes[2, c]
        if bp is not None:
            _add_bbox_rect(ax, bp[2], "#ff3c3c", lw=2)
        if np_ is not None:
            _add_bbox_rect(ax, np_[2], "#3c78ff", lw=2)

    fig.suptitle(
        f"{cat_label}  ·  {video_id} / {flight_id}  ·  last +prompt = {last_pos_frame_name}",
        fontsize=11, fontweight="bold", color=cat_color, y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Finding glitches in baseline...")
    baseline_glitches = find_glitches(BASELINE_DIR)
    print("Finding glitches in negprompt...")
    negprompt_glitches = find_glitches(NEGPROMPT_DIR)

    fixed, persisting, new_cases = classify_glitches(baseline_glitches, negprompt_glitches)

    n_base   = sum(len(v) for v in baseline_glitches.values())
    n_neg    = sum(len(v) for v in negprompt_glitches.values())
    n_fixed  = len(fixed)
    n_pers   = len(persisting)
    n_new    = len(new_cases)

    print(f"\n{'='*60}")
    print(f"Baseline glitches: {n_base} objects  |  Negprompt: {n_neg} objects")
    print(f"Fixed: {n_fixed}  |  Persisting: {n_pers}  |  New: {n_new}")
    print(f"{'='*60}\n")

    # ── Per-case timing analysis ─────────────────────────────────────────────
    print("Timing analysis (glitch frame vs last positive prompt):")
    print(f"{'Category':>12}  {'Video':>6}  {'Object':>12}  {'Last+prompt':>11}  {'Glitch@':>8}  {'PostEvidence?':>13}")
    print("-" * 75)

    def analyse_timing(cases, label):
        post_count = 0
        during_count = 0
        for vid, fid, glist, _ in cases:
            fm = load_flight_mapping(vid)
            lp = last_positive_prompt_frame(vid, fid, fm)
            glitch_frame = glist[0][1]
            lp_int = int(lp) if lp else -1
            post = glitch_frame > lp_int if lp else None
            flag = "POST ✓" if post else "DURING ✗" if post is not None else "unknown"
            if post:
                post_count += 1
            elif post is not None:
                during_count += 1
            print(f"{label:>12}  {vid:>6}  {fid:>12}  {str(lp):>11}  {glitch_frame:>8}  {flag:>13}")
        return post_count, during_count

    fp, fd = analyse_timing(fixed, "FIXED")
    pp, pd = analyse_timing(persisting, "PERSISTING")

    print(f"\nFixed:      {fp} post-evidence, {fd} during-evidence")
    print(f"Persisting: {pp} post-evidence, {pd} during-evidence")

    # ── Visualise top cases ──────────────────────────────────────────────────
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    print(f"\nGenerating visualisations → {OUT_DIR}/")

    for i, (vid, fid, glist, cat) in enumerate(fixed[:MAX_CASES]):
        fm = load_flight_mapping(vid)
        lp = last_positive_prompt_frame(vid, fid, fm)
        neg_glist = negprompt_glitches.get(vid, {}).get(fid, [])
        out = OUT_DIR / f"fixed_{i+1:02d}_{vid}_{fid[:8]}.png"
        print(f"  fixed {i+1}: {vid}/{fid} last+={lp} glitch@{glist[0][1]}")
        visualise_case(vid, fid, glist, neg_glist, "fixed", lp, out)

    for i, (vid, fid, glist, cat) in enumerate(persisting[:MAX_CASES]):
        fm = load_flight_mapping(vid)
        lp = last_positive_prompt_frame(vid, fid, fm)
        neg_glist = negprompt_glitches.get(vid, {}).get(fid, [])
        out = OUT_DIR / f"persisting_{i+1:02d}_{vid}_{fid[:8]}.png"
        print(f"  persisting {i+1}: {vid}/{fid} last+={lp} glitch@{glist[0][1]}")
        visualise_case(vid, fid, glist, neg_glist, "persisting", lp, out)

    for i, (vid, fid, glist, cat) in enumerate(new_cases[:MAX_CASES]):
        fm = load_flight_mapping(vid)
        lp = last_positive_prompt_frame(vid, fid, fm)
        base_glist = []
        out = OUT_DIR / f"new_{i+1:02d}_{vid}_{fid[:8]}.png"
        print(f"  new {i+1}: {vid}/{fid} last+={lp} glitch@{glist[0][1]}")
        visualise_case(vid, fid, base_glist, glist, "new", lp, out)

    print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
