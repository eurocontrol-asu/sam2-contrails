#!/usr/bin/env python3
"""
Illustrative example figures for the SAM2 contrail tracking paper.

Each figure shows a sequence of frames for a single GT contrail, with four
panels per frame: the raw satellite image, the trajectory prompt overlay,
the ground-truth mask overlay, and the SAM2 prediction overlay.

Usage:
    # Auto-select the best example (most frames with all four data types)
    python paper/scripts/plot_examples.py

    # Specify video and object explicitly
    python paper/scripts/plot_examples.py --video_id 10 --obj_id 0 --n_frames 5

    # Use a different predictions directory
    python paper/scripts/plot_examples.py --predictions_dir outputs/test --out_dir paper/figures
"""

import argparse
import json
import os
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
import paper_style as ps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


# ── Style ──────────────────────────────────────────────────────────────────────

# Overlay colours from shared paper_style (consistent paper-wide)
OVERLAY_GT         = tuple(ps.GT_COLOR   / 255.0)   # green     #009E73
OVERLAY_PREDICTION = tuple(ps.PRED_COLOR / 255.0)   # vermilion #D55E00


def _setup():
    pass  # publication.mplstyle already loaded by paper_style import


def save(fig, name, out_dir):
    ps.save_figure(fig, name, out_dir)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_image(img_folder, video_name, frame_name):
    """Load image as brightened grayscale → RGB array (consistent with other figures)."""
    p = Path(img_folder) / video_name / f"{frame_name}.jpg"
    gray = ps.load_gray(p)
    return np.stack([gray, gray, gray], axis=-1)


def load_binary_mask(path):
    """Load a grayscale PNG as a boolean H×W mask."""
    return np.array(Image.open(path).convert("L")) > 127


def decode_rle(ann):
    """Decode a COCO RLE annotation to a boolean H×W mask."""
    return mask_utils.decode(ann["segmentation"]).astype(bool)


def _blend(img_rgb, mask_bool, color_rgb, alpha):
    """Return a copy of img_rgb with mask pixels blended with color_rgb."""
    out = img_rgb.astype(float) / 255.0
    r, g, b = color_rgb
    out[mask_bool, 0] = (1 - alpha) * out[mask_bool, 0] + alpha * r
    out[mask_bool, 1] = (1 - alpha) * out[mask_bool, 1] + alpha * g
    out[mask_bool, 2] = (1 - alpha) * out[mask_bool, 2] + alpha * b
    return (out * 255).clip(0, 255).astype(np.uint8)


def _mask_bbox(mask_bool):
    """Return (r0, r1, c0, c1) tight bounding box of a boolean mask."""
    rows = np.where(mask_bool.any(axis=1))[0]
    cols = np.where(mask_bool.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return rows[0], rows[-1], cols[0], cols[-1]


def _crop_roi(img_rgb, masks, pad=80):
    """
    Compute a tight ROI around the union of all boolean masks, expand by
    `pad` pixels on each side, and crop both the image and every mask.

    Returns (img_crop, mask_crops, (r0, c0)) where (r0, c0) is the offset.
    If no mask has any positive pixels, returns the full image unchanged.
    """
    H, W = img_rgb.shape[:2]
    union = np.zeros((H, W), dtype=bool)
    for m in masks:
        if m is not None:
            union |= m

    bb = _mask_bbox(union)
    if bb is None:
        return img_rgb, masks, (0, 0)

    r0, r1, c0, c1 = bb
    r0 = max(0,     r0 - pad)
    r1 = min(H - 1, r1 + pad)
    c0 = max(0,     c0 - pad)
    c1 = min(W - 1, c1 + pad)

    img_crop   = img_rgb[r0:r1+1, c0:c1+1]
    mask_crops = [m[r0:r1+1, c0:c1+1] if m is not None else None for m in masks]
    return img_crop, mask_crops, (r0, c0)


# ── Discovery helpers ──────────────────────────────────────────────────────────

def _video_name(video_id):
    return f"{int(video_id):05d}"


def find_best_example(gvccs_dir, predictions_dir, id_offset):
    """
    Scan all videos and return the (video_name, obj_folder, json_id, frames)
    combination that has the most frames where image + prompt + GT + prediction
    are all available.

    id_offset: 1 for old (pre-fix) JSONs where json_object_id = folder_int + 1;
               0 for new JSONs where json_object_id = folder_int.
    """
    gvccs_dir = Path(gvccs_dir)
    pred_dir  = Path(predictions_dir)
    pod_base  = gvccs_dir / "per_object_data_age_5"
    img_base  = gvccs_dir / "img_folder"

    best = None

    for vid_dir in sorted(os.listdir(pod_base)):
        pred_file = pred_dir / f"{vid_dir}.json"
        if not pred_file.exists():
            continue

        with open(pred_file) as f:
            d = json.load(f)

        # Build: json_object_id -> {frame_name -> ann}
        by_obj_frame = {}
        for ann in d["annotations"]:
            oid = ann["object_id"]
            fn  = ann["frame_name"]
            by_obj_frame.setdefault(oid, {})[fn] = ann

        for obj_folder in sorted(os.listdir(pod_base / vid_dir)):
            obj_path = pod_base / vid_dir / obj_folder
            if not obj_path.is_dir() or not obj_folder.isdigit():
                continue

            files = os.listdir(obj_path)
            mask_frames   = {f.replace("_mask.png",   "") for f in files if f.endswith("_mask.png")}
            prompt_frames = {f.replace("_prompt.png", "") for f in files if f.endswith("_prompt.png")}
            if not mask_frames:
                continue

            json_id    = int(obj_folder) + id_offset
            pred_frames = set(by_obj_frame.get(json_id, {}).keys())

            common = sorted(mask_frames & prompt_frames & pred_frames)
            if not common:
                continue

            # Verify images actually exist
            common_with_img = [fn for fn in common
                               if (img_base / vid_dir / f"{fn}.jpg").exists()]
            if not common_with_img:
                continue

            if best is None or len(common_with_img) > len(best["frames"]):
                best = {
                    "video": vid_dir,
                    "folder": obj_folder,
                    "json_id": json_id,
                    "frames": common_with_img,
                    "by_obj_frame": by_obj_frame,
                }

    return best


def load_example_data(best, gvccs_dir, frame_names, id_offset):
    """
    Load image, ternary prompt, GT mask, and prediction for each requested frame.
    Returns a list of dicts, one per frame.
    """
    gvccs_dir = Path(gvccs_dir)
    vid   = best["video"]
    folder = best["folder"]
    json_id = best["json_id"]
    pod_base = gvccs_dir / "per_object_data_age_5"
    img_base = gvccs_dir / "img_folder"

    rows = []
    for fn in frame_names:
        img      = load_image(img_base, vid, fn)
        mask_path = pod_base / vid / folder / f"{fn}_mask.png"
        ternary  = ps.load_ternary_prompt(pod_base, vid, int(folder), fn)
        gt       = load_binary_mask(mask_path)
        pred_ann = best["by_obj_frame"][json_id][fn]
        pred     = decode_rle(pred_ann)
        score    = pred_ann.get("score", None)
        rows.append({"frame": fn, "img": img, "ternary": ternary, "gt": gt,
                     "pred": pred, "score": score})
    return rows


# ── Figure ────────────────────────────────────────────────────────────────────

def _draw_roi_box(ax, img_full, roi_offset, roi_shape):
    """
    Draw the full grayscale image on ax with a coloured rectangle showing the ROI.
    roi_offset: (r0, c0), roi_shape: (crop_H, crop_W)
    """
    gray = img_full[:, :, 0]  # already grayscale from load_image
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    r0, c0 = roi_offset
    h, w   = roi_shape
    rect = plt.Rectangle((c0, r0), w, h,
                          linewidth=1.2, edgecolor="#E69F00",
                          facecolor="none")
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_example_sequence(rows, video_id, obj_id, out_dir, name="fig_examples"):
    """
    Figure layout — consistent with Figs 6 and 7 (N frames × 3 rows):

      Row 0  — full image + GT overlay (crimson)
      Row 1  — full image + ternary prompt (blue=target, orange=competing)
      Row 2  — full image + faint GT + prediction (vermilion) + score badge

    Shows full images without cropping: the ternary prompt's negative signal
    spans most of the image, making a crop context unnecessary.
    """
    n_frames = len(rows)

    cell_w = ps.FIG_W_FULL / n_frames
    cell_h = cell_w  # square cells

    fig, axes = plt.subplots(
        3, n_frames,
        figsize=(ps.FIG_W_FULL, 3 * cell_h),
        constrained_layout=False,
        gridspec_kw={"hspace": 0.05, "wspace": 0.04})
    fig.subplots_adjust(left=0.14)
    if n_frames == 1:
        axes = axes[:, np.newaxis]
    axes = np.asarray(axes)

    row_labels = ["Image + GT", "Prompt\n(+ neg. signal)", "Prediction"]

    for col, rd in enumerate(rows):
        img   = rd["img"]       # full 1024×1024 RGB
        gt    = rd["gt"]
        ternary = rd["ternary"]
        pred  = rd["pred"]
        score = rd.get("score")

        # Row 0: image + GT overlay
        vis0 = ps.blend_mask(img, gt, ps.GT_COLOR, ps.ALPHA_GT)
        axes[0, col].imshow(vis0, interpolation="bilinear")

        # Row 1: ternary prompt (blue positive + orange negative)
        vis1 = ps.blend_ternary_prompt(img, ternary)
        axes[1, col].imshow(vis1, interpolation="bilinear")

        # Row 2: faint GT underlay + prediction
        vis2 = ps.blend_mask(img, gt, ps.GT_COLOR, ps.ALPHA_GT_UNDER)
        if pred is not None and pred.any():
            vis2 = ps.blend_mask(vis2, pred, ps.PRED_COLOR, ps.ALPHA_PRED)
        axes[2, col].imshow(vis2, interpolation="bilinear")

        if score is not None:
            badge_kw = {**ps.SCORE_BADGE_KW,
                        "fontsize": 8,
                        "bbox": {**ps.SCORE_BADGE_KW["bbox"], "pad": 1.5}}
            axes[2, col].text(0.97, 0.05, f"{score:.2f}",
                              transform=axes[2, col].transAxes, **badge_kw)

        # Column title: frame index + elapsed time since the first shown frame
        # (GVCCS cadence = 30 s/frame)
        f0 = int(rows[0]["frame"])
        dt = (int(rd["frame"]) - f0) * 0.5  # minutes
        title = f"frame {int(rd['frame'])}" if col == 0 else f"+{dt:.1f} min"
        axes[0, col].set_title(title, fontsize=ps.FONT_COL_TITLE - 2, pad=2.5)

        for r in range(3):
            ps.clean_ax(axes[r, col])
            if col == 0:
                axes[r, col].set_ylabel(row_labels[r],
                                        fontsize=ps.FONT_ROW_LABEL, labelpad=4)

    save(fig, name, out_dir)
    plt.close(fig)


# ── Multi-example overview ─────────────────────────────────────────────────────

def fig_multi_example(examples_data, out_dir, name="fig_examples_overview",
                      roi_pad=80):
    """
    Overview figure: one row per GT contrail.
    Columns: thumbnail (full image + ROI box) | GT crop | prediction crop.

    All image panels are cropped to the mask bounding box so that
    even thin contrails are clearly visible.
    """
    n_examples = len(examples_data)
    # 4 columns: thumbnail | raw crop | GT | prediction
    col_titles  = ["Context", "Crop", "GT mask", "Prediction"]
    col_colors  = [None,       None,   OVERLAY_GT, OVERLAY_PREDICTION]

    thumb_w = 1.1   # inches — narrow thumbnail
    crop_w  = 2.0   # inches — wide crop panels
    row_h   = 1.9

    col_widths = [thumb_w, crop_w, crop_w, crop_w]
    fig_w = sum(col_widths)          # no extra padding — wspace handles gaps
    fig_h = row_h * n_examples

    fig, axes = plt.subplots(
        n_examples, 4,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": col_widths,
                     "hspace": 0.03, "wspace": 0.03})
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for row_idx, ex in enumerate(examples_data):
        rd = ex["rows"][len(ex["rows"]) // 2]   # middle frame
        ternary_mask = np.abs(rd["ternary"]) > 1e-6
        all_masks = [ternary_mask, rd["gt"], rd["pred"]]
        img_c, mask_c, offset = _crop_roi(rd["img"], all_masks, pad=roi_pad)
        crop_shape = img_c.shape[:2]

        for c, (title, color) in enumerate(zip(col_titles, col_colors)):
            ax = axes[row_idx, c]

            if c == 0:
                # Thumbnail with ROI box
                _draw_roi_box(ax, rd["img"], offset, crop_shape)

            elif c == 1:
                ax.imshow(img_c)

            else:
                mk = mask_c[1] if c == 2 else mask_c[2]   # gt or pred
                ax.imshow(_blend(img_c, mk, color, ps.ALPHA_PRED))

            ax.set_xticks([])
            ax.set_yticks([])

            # Column header: text annotation inside top row panels (no set_title gap)
            if row_idx == 0:
                ax.text(0.5, 0.03, title, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7.5, color="black",
                        fontweight="bold",
                        bbox=dict(facecolor="white", alpha=0.88, pad=2.0,
                                  boxstyle="round,pad=0.3", edgecolor="none"),
                        clip_on=True)

        score = rd.get("score")
        frame_label = (f"V{int(ex['video']):02d}·O{ex['obj_id']}\n"
                       f"frame {rd['frame']}"
                       + (f"\ns={score:.2f}" if score is not None else ""))
        axes[row_idx, 0].set_ylabel(frame_label, fontsize=7, labelpad=4,
                                    rotation=90, va="center")

    save(fig, name, out_dir)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate illustrative example figures")
    parser.add_argument("--video_id",   type=int, default=14,
                        help="Video ID. Default: 14 (strong negative signal, 25-frame lifecycle).")
    parser.add_argument("--obj_id",     type=int, default=109,
                        help="Object folder ID. Default: 109 (flight 34ba8a2b, mean neg/pos=21x).")
    parser.add_argument("--n_frames",   type=int, default=4,
                        help="Number of frames to show in the sequence figure.")
    parser.add_argument("--n_examples", type=int, default=4,
                        help="Number of contrails shown in the multi-example overview figure.")
    parser.add_argument("--predictions_dir", type=Path, default=Path("outputs/ternary_5"),
                        help="Directory of per-video SAM2 prediction JSON files.")
    parser.add_argument("--gvccs_dir", type=Path,
                        default=Path("/data/common/TRAILVISION/GVCCS_V/test"),
                        help="GVCCS test directory (contains img_folder/ and per_object_data/).")
    parser.add_argument("--out_dir",   type=Path, default=Path("paper/figures"),
                        help="Output directory for figures.")
    parser.add_argument("--id_offset", type=int, default=0,
                        help="object_id offset in JSONs vs folder names. "
                             "Use 0 for post-fix JSONs (json_id = folder), "
                             "1 for pre-fix JSONs (json_id = folder + 1).")
    parser.add_argument("--roi_pad", type=int, default=100,
                        help="Padding in pixels around the mask bounding box "
                             "when cropping for zoomed panels.")
    parser.add_argument("--auto_select", action="store_true",
                        help="Auto-select the best object (ignored when --obj_id is not set, "
                             "since auto-selection is the default in that case).")
    args = parser.parse_args()

    _setup()

    pod_base = args.gvccs_dir / "per_object_data_age_5"
    img_base = args.gvccs_dir / "img_folder"

    # ── Sequence figure for one specific contrail ───────────────────────────
    # Logic:
    #   --video_id + --obj_id  → use exactly that object
    #   --video_id only        → auto-select best object within that video
    #   neither                → auto-select best object across all videos

    if args.video_id is not None and args.obj_id is not None:
        # User specified video + object explicitly
        vid = _video_name(args.video_id)
        folder = f"{args.obj_id}"
        json_id = args.obj_id + args.id_offset
        pred_file = args.predictions_dir / f"{vid}.json"
        with open(pred_file) as f:
            d = json.load(f)
        by_obj_frame = {}
        for ann in d["annotations"]:
            oid = ann["object_id"]
            fn  = ann["frame_name"]
            by_obj_frame.setdefault(oid, {})[fn] = ann

        files = os.listdir(pod_base / vid / folder)
        mask_frames   = sorted(f.replace("_mask.png",   "") for f in files if f.endswith("_mask.png"))
        prompt_frames = sorted(f.replace("_prompt.png", "") for f in files if f.endswith("_prompt.png"))
        pred_frames   = sorted(by_obj_frame.get(json_id, {}).keys())
        common = sorted(set(mask_frames) & set(prompt_frames) & set(pred_frames))

        if not common:
            print(f"\nERROR: No frames found with all three data types for "
                  f"video={vid}, folder={folder}, json_id={json_id}.")
            print(f"  Mask frames  ({len(mask_frames)}): {mask_frames[:5]}")
            print(f"  Prompt frames({len(prompt_frames)}): {prompt_frames[:5]}")
            print(f"  Pred frames  ({len(pred_frames)}): {pred_frames[:5]}")
            all_obj_ids = sorted(by_obj_frame.keys())
            print(f"  Object IDs with predictions in this video: {all_obj_ids[:20]}")
            print(f"\nHints:")
            print(f"  - If pred_frames is empty, the JSON uses a different id_offset."
                  f" Try --id_offset 1 (pre-fix JSONs where json_id = folder + 1).")
            print(f"  - Object folder {folder!r} → json_id {json_id}. "
                  f"With --id_offset 1 that would be json_id {int(folder)+1}.")
            print(f"  - Run without --video_id/--obj_id to auto-select a working example.")
            import sys; sys.exit(1)

        best = {"video": vid, "folder": folder, "json_id": json_id,
                "frames": common, "by_obj_frame": by_obj_frame}
    elif args.video_id is not None:
        # Auto-select best object within the requested video
        vid = _video_name(args.video_id)
        print(f"Auto-selecting best object in video {vid}...")
        candidates = _collect_top_examples(args.gvccs_dir, args.predictions_dir,
                                           args.id_offset, n_top=999)
        candidates = [c for c in candidates if c["video"] == vid]
        if not candidates:
            print(f"ERROR: No objects with mask + prompt + prediction found in video {vid}.")
            print("  Try a different video or omit --video_id for global auto-selection.")
            import sys; sys.exit(1)
        best = candidates[0]   # already sorted by n overlapping frames
        print(f"  Selected object folder={best['folder']} "
              f"({len(best['frames'])} overlapping frames)")
    else:
        print("Auto-selecting best example...")
        best = find_best_example(args.gvccs_dir, args.predictions_dir, args.id_offset)
        if best is None:
            raise RuntimeError("No example found with image + prompt + GT + prediction "
                               "in the same frame. Check --id_offset and --predictions_dir.")

    print(f"  Video={best['video']}, folder={best['folder']}, "
          f"json_id={best['json_id']}, frames available={len(best['frames'])}")

    # Select n_frames evenly spaced
    all_frames = best["frames"]
    n = min(args.n_frames, len(all_frames))
    if n == 0:
        print("ERROR: 0 frames selected — nothing to plot.")
        import sys; sys.exit(1)
    if n == len(all_frames):
        selected = all_frames
    else:
        idxs = np.round(np.linspace(0, len(all_frames) - 1, n)).astype(int)
        selected = [all_frames[i] for i in idxs]

    print(f"  Selected frames: {selected}")
    rows = load_example_data(best, args.gvccs_dir, selected, args.id_offset)
    fig_example_sequence(rows, best["video"], best["folder"], args.out_dir)

    # ── Multi-example overview figure ───────────────────────────────────────

    print("Finding diverse examples for overview figure...")
    # Collect top N examples sorted by number of overlapping frames
    candidates = _collect_top_examples(args.gvccs_dir, args.predictions_dir,
                                       args.id_offset, n_top=args.n_examples * 5)

    # Deduplicate by video so we show diverse videos
    seen_videos = set()
    diverse = []
    for c in candidates:
        if c["video"] not in seen_videos:
            seen_videos.add(c["video"])
            diverse.append(c)
        if len(diverse) == args.n_examples:
            break

    print(f"  Selected {len(diverse)} examples for overview")
    examples_data = []
    for c in diverse:
        # Pick one representative frame (middle of the sequence)
        mid_frame = [c["frames"][len(c["frames"]) // 2]]
        rows_ex = load_example_data(c, args.gvccs_dir, mid_frame, args.id_offset)
        examples_data.append({
            "video": c["video"],
            "obj_id": c["folder"],
            "rows": rows_ex,
        })

    fig_multi_example(examples_data, args.out_dir, roi_pad=args.roi_pad)

    print("\nDone.")


def _collect_top_examples(gvccs_dir, predictions_dir, id_offset, n_top=20):
    """Return up to n_top examples sorted by number of overlapping frames (best first)."""
    gvccs_dir    = Path(gvccs_dir)
    pred_dir     = Path(predictions_dir)
    pod_base     = gvccs_dir / "per_object_data_age_5"
    img_base     = gvccs_dir / "img_folder"

    results = []

    for vid_dir in sorted(os.listdir(pod_base)):
        pred_file = pred_dir / f"{vid_dir}.json"
        if not pred_file.exists():
            continue
        with open(pred_file) as f:
            d = json.load(f)

        by_obj_frame = {}
        for ann in d["annotations"]:
            oid = ann["object_id"]
            fn  = ann["frame_name"]
            by_obj_frame.setdefault(oid, {})[fn] = ann

        for obj_folder in sorted(os.listdir(pod_base / vid_dir)):
            obj_path = pod_base / vid_dir / obj_folder
            if not obj_path.is_dir() or not obj_folder.isdigit():
                continue
            files = os.listdir(obj_path)
            mask_frames   = {f.replace("_mask.png",   "") for f in files if f.endswith("_mask.png")}
            prompt_frames = {f.replace("_prompt.png", "") for f in files if f.endswith("_prompt.png")}
            if not mask_frames:
                continue
            json_id = int(obj_folder) + id_offset
            pred_frames = set(by_obj_frame.get(json_id, {}).keys())
            common = sorted(mask_frames & prompt_frames & pred_frames)
            common = [fn for fn in common if (img_base / vid_dir / f"{fn}.jpg").exists()]
            if common:
                results.append({
                    "video": vid_dir, "folder": obj_folder,
                    "json_id": json_id, "frames": common,
                    "by_obj_frame": by_obj_frame,
                })

    results.sort(key=lambda x: -len(x["frames"]))
    return results[:n_top]


if __name__ == "__main__":
    main()
