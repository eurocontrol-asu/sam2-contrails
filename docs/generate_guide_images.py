"""Generate example images for the SAM2 labelling guide.

All examples use large, clearly visible contrails so figures are readable
even when printed or viewed at reduced size in a DOCX.

Colour convention (matches Encord):
  Prompt  = ORANGE polygons
  Contrail = RED multi-polygons
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pycocotools import mask as mask_utils

PRED_DIR = Path("/data/contrailnet/sam2/predictions")
PROMPT_DIR = Path("/data/contrailnet/sam2/prompts")
FRAMES_DIR = Path("/data/contrailnet/sam2/frames")
OUT_DIR = Path(__file__).parent / "images"
OUT_DIR.mkdir(exist_ok=True)

BLUE = np.array([0, 114, 178]) / 255
ORANGE = np.array([230, 159, 0]) / 255
RED = np.array([213, 94, 0]) / 255
GREEN = np.array([0, 158, 115]) / 255
YELLOW = np.array([240, 228, 66]) / 255
CYAN = np.array([86, 180, 233]) / 255
PINK = np.array([204, 121, 167]) / 255
GRAY = np.array([0.6, 0.6, 0.6])

PROMPT_COLOR = np.array([255, 165, 0]) / 255    # bright orange — distinct from red on blue sky
CONTRAIL_COLOR = np.array([200, 30, 30]) / 255   # crimson red — clearly different from orange


def decode_rle(rle):
    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("utf-8")
    return mask_utils.decode(r).astype(bool)


def load_frame(video_id, frame_name):
    path = FRAMES_DIR / video_id / f"{frame_name}.jpg"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def blend_mask(img, mask, color, alpha=0.45):
    out = img.astype(float) / 255.0
    for c in range(3):
        out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha) + color[c] * alpha, out[:, :, c])
    return (out * 255).astype(np.uint8)


def draw_contours(img, mask, color, thickness=2):
    color_rgb = tuple(int(c * 255) for c in color)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    cv2.drawContours(out, contours, -1, color_rgb, thickness)
    return out


def load_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    by_flight = defaultdict(list)
    for ann in data["annotations"]:
        by_flight[ann["flight_id"]].append(ann)
    return by_flight


def save_fig(fig, name, dpi=250):
    path = OUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {path}")


def fmt_time(frame_name):
    return f"{frame_name[8:10]}:{frame_name[10:12]}"


def crop_to_mask(img, mask, pad=80):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img, (0, 0, img.shape[1], img.shape[0])
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    h, w = y1 - y0, x1 - x0
    if h > w:
        extra = (h - w) // 2
        x0 = max(0, x0 - extra)
        x1 = min(img.shape[1], x1 + extra)
    elif w > h:
        extra = (w - h) // 2
        y0 = max(0, y0 - extra)
        y1 = min(img.shape[0], y1 + extra)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


# ── Figure 1: Dataset overview ───────────────────────────────────────────────

def generate_dataset_overview():
    print("\n[1] Dataset overview")
    vid = "20250215065930_20250215171000"
    times = ["20250215080000", "20250215120000", "20250215160000"]
    labels = ["Morning (08:00 UTC)", "Midday (12:00 UTC)", "Afternoon (16:00 UTC)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Ground camera images at different times of day (15 February 2025)",
                 fontsize=14, fontweight="bold", y=1.02)

    for i, (frame, label) in enumerate(zip(times, labels)):
        img = load_frame(vid, frame)
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=12, pad=8)
        axes[i].axis("off")

    fig.tight_layout()
    save_fig(fig, "01_dataset_overview.png")


# ── Figure 2: Ontology diagram ──────────────────────────────────────────────

def generate_ontology_diagram():
    print("\n[2] Ontology diagram")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Prompt box — orange (matches Encord)
    prompt_box = plt.Rectangle((0.3, 1.0), 3.8, 3.5, fill=True, facecolor="#FFF3E0",
                                edgecolor=tuple(PROMPT_COLOR), linewidth=2.5, zorder=2, joinstyle="round")
    ax.add_patch(prompt_box)
    ax.text(2.2, 3.9, "Prompt", fontsize=18, fontweight="bold", ha="center", va="center",
            color=tuple(PROMPT_COLOR))
    ax.text(2.2, 3.0, "Orange polygon", fontsize=11, ha="center", va="center",
            color="gray", style="italic")
    ax.text(2.2, 2.2, "flight_id (text)", fontsize=12, ha="center", va="center")
    ax.text(2.2, 1.5, 'e.g. "SWR15X_1161"', fontsize=10, ha="center", va="center", color="gray")

    # Contrail box — red (matches Encord)
    contrail_box = plt.Rectangle((5.9, 1.0), 3.8, 3.5, fill=True, facecolor="#FFEBEE",
                                  edgecolor=tuple(CONTRAIL_COLOR), linewidth=2.5, zorder=2, joinstyle="round")
    ax.add_patch(contrail_box)
    ax.text(7.8, 3.9, "Contrail", fontsize=18, fontweight="bold", ha="center", va="center",
            color=tuple(CONTRAIL_COLOR))
    ax.text(7.8, 3.0, "Red multi-polygon", fontsize=11, ha="center", va="center",
            color="gray", style="italic")
    ax.text(7.8, 2.2, "flight #relation", fontsize=12, ha="center", va="center")
    ax.text(7.8, 1.5, "score (confidence)", fontsize=10, ha="center", va="center", color="gray")

    ax.annotate("", xy=(5.9, 2.75), xytext=(4.1, 2.75),
                arrowprops=dict(arrowstyle="<->", lw=2.5, color="black"))
    ax.text(5.0, 3.3, "flight #relation", fontsize=11, ha="center", va="center",
            fontweight="bold", color="black")

    save_fig(fig, "02_ontology.png")


# ── Figure 3: Model overview ────────────────────────────────────────────────

def generate_model_overview():
    print("\n[3] Model overview")
    vid = "20250215065930_20250215171000"
    frame = "20250215102000"

    preds = load_annotations(PRED_DIR / f"{vid}.json")
    prompts = load_annotations(PROMPT_DIR / f"{vid}.json")
    img = load_frame(vid, frame)

    prompt_overlay = img.copy()
    pred_overlay = img.copy()
    colors = [BLUE, ORANGE, GREEN, RED, CYAN, PINK, YELLOW, GRAY,
              np.array([0.5, 0.2, 0.8]), np.array([0.2, 0.7, 0.4])]

    ci = 0
    for flight_id, anns in preds.items():
        pred_ann = next((a for a in anns if a["frame_name"] == frame), None)
        if pred_ann is None:
            continue
        color = colors[ci % len(colors)]
        ci += 1
        pred_mask = decode_rle(pred_ann["segmentation"])
        pred_overlay = blend_mask(pred_overlay, pred_mask, color, alpha=0.5)
        pred_overlay = draw_contours(pred_overlay, pred_mask, color, thickness=2)

        prompt_ann_list = prompts.get(flight_id, [])
        prompt_ann = next((a for a in prompt_ann_list if a["frame_name"] == frame), None)
        if prompt_ann:
            prompt_mask = decode_rle(prompt_ann["segmentation"])
            prompt_overlay = blend_mask(prompt_overlay, prompt_mask, color, alpha=0.5)
            prompt_overlay = draw_contours(prompt_overlay, prompt_mask, color, thickness=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [
        "Raw camera image",
        "Prompts\n(projected flight paths, wind-advected)",
        "Predictions\n(SAM2 contrail detections)",
    ]
    images = [img, prompt_overlay, pred_overlay]

    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")

    fig.suptitle(f"What the model produces  —  {frame[6:8]}/{frame[4:6]}/{frame[:4]} "
                 f"{fmt_time(frame)} UTC",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "03_model_overview.png")


# ── Figure 4: Good prediction lifecycle ──────────────────────────────────────

def generate_good_prediction():
    """Contrail tracked over time: Prompt=ORANGE (top row), Contrail=RED (bottom row)."""
    print("\n[4] Good prediction lifecycle")
    vid = "20250215065930_20250215171000"
    flight = "AAL93_2595"

    preds = load_annotations(PRED_DIR / f"{vid}.json")
    prompts = load_annotations(PROMPT_DIR / f"{vid}.json")

    pred_anns = sorted(preds[flight], key=lambda a: a["frame_name"])
    prompt_anns = sorted(prompts.get(flight, []), key=lambda a: a["frame_name"])

    indices = np.linspace(0, len(pred_anns) - 1, 4, dtype=int)
    selected = [pred_anns[i] for i in indices]

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for col, ann in enumerate(selected):
        frame = ann["frame_name"]
        img = load_frame(vid, frame)
        pred_mask = decode_rle(ann["segmentation"])

        prompt_ann = next((a for a in prompt_anns if a["frame_name"] == frame), None)
        if prompt_ann:
            prompt_mask = decode_rle(prompt_ann["segmentation"])
            top = blend_mask(img, prompt_mask, PROMPT_COLOR, alpha=0.55)
            top = draw_contours(top, prompt_mask, PROMPT_COLOR, thickness=3)
        else:
            top = img.copy()
            prompt_mask = np.zeros_like(pred_mask)

        bot = blend_mask(img, pred_mask, CONTRAIL_COLOR, alpha=0.5)
        bot = draw_contours(bot, pred_mask, CONTRAIL_COLOR, thickness=3)

        union_mask = prompt_mask | pred_mask
        top_crop, _ = crop_to_mask(top, union_mask, pad=100)
        bot_crop, _ = crop_to_mask(bot, union_mask, pad=100)

        axes[0, col].imshow(top_crop)
        axes[1, col].imshow(bot_crop)

        time_str = fmt_time(frame)
        axes[0, col].set_title(f"{time_str} UTC", fontsize=12, fontweight="bold", pad=6)

        score_str = f"score = {ann.get('score', 0):.2f}"
        axes[1, col].set_title(score_str, fontsize=11, color=tuple(CONTRAIL_COLOR), pad=6)

        for row in range(2):
            axes[row, col].axis("off")

    axes[0, 0].set_ylabel("Prompt\n(last 5 min of flight)", fontsize=13, fontweight="bold",
                           color=tuple(PROMPT_COLOR), labelpad=15)
    axes[1, 0].set_ylabel("Contrail\n(model prediction)", fontsize=13, fontweight="bold",
                           color=tuple(CONTRAIL_COLOR), labelpad=15)

    fig.suptitle(f"Contrail lifecycle  —  flight {flight}  ({len(pred_anns)} frames tracked)",
                 fontsize=15, fontweight="bold", y=1.0)

    fig.legend(
        handles=[
            Patch(facecolor=PROMPT_COLOR, edgecolor=PROMPT_COLOR, label="Prompt (orange in Encord)"),
            Patch(facecolor=CONTRAIL_COLOR, edgecolor=CONTRAIL_COLOR, label="Contrail (red in Encord)"),
        ],
        loc="lower center", ncol=2, fontsize=12, frameon=True, fancybox=True,
        edgecolor="lightgray", borderpad=0.8
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_fig(fig, "04_good_prediction.png")


# ── Figure 5: Sun artifact false positive ────────────────────────────────────

def generate_sun_artifact():
    print("\n[5] Sun artifact false positive")
    vid = "20250215065930_20250215171000"
    flight_artifact = "MAC223F_028"
    frame_artifact = "20250215091330"
    flight_real = "AAL93_2595"
    frame_real = "20250215110000"

    preds = load_annotations(PRED_DIR / f"{vid}.json")

    img_art = load_frame(vid, frame_artifact)
    ann_art = next(a for a in preds[flight_artifact] if a["frame_name"] == frame_artifact)
    mask_art = decode_rle(ann_art["segmentation"])

    img_real = load_frame(vid, frame_real)
    ann_real = next(a for a in preds[flight_real] if a["frame_name"] == frame_real)
    mask_real = decode_rle(ann_real["segmentation"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    # Left: false positive — highlight in magenta to clearly mark as wrong
    MAGENTA = np.array([0.8, 0.0, 0.4])
    overlay_art = blend_mask(img_art, mask_art, MAGENTA, alpha=0.55)
    overlay_art = draw_contours(overlay_art, mask_art, MAGENTA, thickness=3)
    axes[0].imshow(overlay_art)
    axes[0].set_title(
        f"FALSE POSITIVE  —  Sun artifact\n"
        f"{fmt_time(frame_artifact)} UTC  |  score = {ann_art.get('score', 0):.2f}  |  DELETE this",
        fontsize=12, fontweight="bold", color=tuple(MAGENTA), pad=10
    )
    axes[0].axis("off")

    # Right: real contrail — shown in red (Contrail colour)
    overlay_real = blend_mask(img_real, mask_real, GREEN, alpha=0.45)
    overlay_real = draw_contours(overlay_real, mask_real, GREEN, thickness=3)
    axes[1].imshow(overlay_real)
    axes[1].set_title(
        f"REAL CONTRAIL  —  Correct prediction\n"
        f"{fmt_time(frame_real)} UTC  |  score = {ann_real.get('score', 0):.2f}  |  KEEP this",
        fontsize=12, fontweight="bold", color=tuple(GREEN), pad=10
    )
    axes[1].axis("off")

    fig.suptitle("How to tell a sun artifact from a real contrail",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "05_sun_artifact.png")


# ── Figure 6: Sun artifact zoomed gallery ────────────────────────────────────

def generate_sun_artifacts_gallery():
    print("\n[6] Sun artifacts gallery")
    MAGENTA = np.array([0.8, 0.0, 0.4])
    cases = [
        ("20250215065930_20250215171000", "MAC223F_028", "20250215091330"),
        ("20250215065930_20250215171000", "AWC445_1062", "20250215084400"),
        ("20250529035630_20250529194100", "EXS1MK_1376", "20250529065000"),
        ("20250219065230_20250219171630", "OCN3_692", "20250219080000"),
    ]

    fig, axes = plt.subplots(1, len(cases), figsize=(18, 5))
    fig.suptitle("Sun artifact examples  —  all are false positives that should be DELETED",
                 fontsize=14, fontweight="bold", color=tuple(MAGENTA), y=1.02)

    for i, (vid, flight, frame) in enumerate(cases):
        try:
            img = load_frame(vid, frame)
            preds_local = load_annotations(PRED_DIR / f"{vid}.json")
            ann = next(a for a in preds_local[flight] if a["frame_name"] == frame)
            mask = decode_rle(ann["segmentation"])
            overlay = blend_mask(img, mask, MAGENTA, alpha=0.55)
            overlay = draw_contours(overlay, mask, MAGENTA, thickness=3)
            cropped, _ = crop_to_mask(overlay, mask, pad=120)
            axes[i].imshow(cropped)
            date_str = f"{frame[6:8]}/{frame[4:6]}/{frame[:4]}"
            axes[i].set_title(
                f"{date_str}  {fmt_time(frame)} UTC\n"
                f"score = {ann.get('score', 0):.2f}",
                fontsize=11, pad=8
            )
        except Exception as e:
            axes[i].text(0.5, 0.5, str(e), ha="center", transform=axes[i].transAxes, fontsize=8)
        axes[i].axis("off")

    fig.tight_layout()
    save_fig(fig, "06_sun_artifacts_gallery.png")


# ── Figure 7: Annotation example (Encord-style: Prompt=orange, Contrail=red) ─

def generate_annotation_example():
    """Three-panel view: Prompt only, Contrail only, both overlaid — zoomed crop."""
    print("\n[7] Annotation example — what you see in Encord")
    vid = "20250215065930_20250215171000"
    flight = "AAL93_2595"
    frame = "20250215102230"  # ratio ~0.9 — Prompt and Contrail similarly sized

    preds = load_annotations(PRED_DIR / f"{vid}.json")
    prompts = load_annotations(PROMPT_DIR / f"{vid}.json")

    img = load_frame(vid, frame)
    pred_ann = next(a for a in preds[flight] if a["frame_name"] == frame)
    prompt_ann = next((a for a in prompts[flight] if a["frame_name"] == frame), None)

    pred_mask = decode_rle(pred_ann["segmentation"])
    prompt_mask = decode_rle(prompt_ann["segmentation"]) if prompt_ann else np.zeros_like(pred_mask)
    union_mask = prompt_mask | pred_mask

    # Panel 1: Prompt only (orange)
    img_prompt = blend_mask(img, prompt_mask, PROMPT_COLOR, alpha=0.5)
    img_prompt = draw_contours(img_prompt, prompt_mask, PROMPT_COLOR, thickness=3)

    # Panel 2: Contrail only (red)
    img_contrail = blend_mask(img, pred_mask, CONTRAIL_COLOR, alpha=0.45)
    img_contrail = draw_contours(img_contrail, pred_mask, CONTRAIL_COLOR, thickness=3)

    # Panel 3: both overlaid — Contrail first, Prompt on top
    img_both = blend_mask(img, pred_mask, CONTRAIL_COLOR, alpha=0.35)
    img_both = draw_contours(img_both, pred_mask, CONTRAIL_COLOR, thickness=3)
    img_both = blend_mask(img_both, prompt_mask, PROMPT_COLOR, alpha=0.4)
    img_both = draw_contours(img_both, prompt_mask, PROMPT_COLOR, thickness=3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    panels = [
        (img_prompt, f"Prompt only (orange)\nflight_id = {flight}"),
        (img_contrail, f"Contrail only (red)\nscore = {pred_ann.get('score', 0):.2f}"),
        (img_both, "Both overlaid\nPrompt + Contrail"),
    ]
    for ax, (panel_img, title) in zip(axes, panels):
        crop, _ = crop_to_mask(panel_img, union_mask, pad=100)
        ax.imshow(crop)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle(f"What you see in Encord  —  {fmt_time(frame)} UTC",
                 fontsize=15, fontweight="bold", y=1.01)

    fig.legend(
        handles=[
            Patch(facecolor=PROMPT_COLOR, edgecolor=PROMPT_COLOR, label="Prompt (orange polygon)"),
            Patch(facecolor=CONTRAIL_COLOR, edgecolor=CONTRAIL_COLOR, label="Contrail (red multi-polygon)"),
        ],
        loc="lower center", ncol=2, fontsize=12, frameon=True, fancybox=True,
        edgecolor="lightgray", borderpad=0.8
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_fig(fig, "07_annotation_example.png")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_dataset_overview()
    generate_ontology_diagram()
    generate_model_overview()
    generate_good_prediction()
    generate_sun_artifact()
    generate_sun_artifacts_gallery()
    generate_annotation_example()
    print(f"\nAll images saved to {OUT_DIR}")
