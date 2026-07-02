"""
Shared style constants for all paper figures.

Variant naming (use these strings everywhere, in scripts and LaTeX):
    Binary               — binary prompt, 1-min window  (code: base)
    Age-weighted (5 min) — age-weighted, 5-min           (code: age_5)
    Age-weighted (10 min)— age-weighted, 10-min          (code: age_10)
    + neg. signal (5 min)— ternary prompt, 5-min         (code: ternary_5)  ← best
    + neg. signal (10 min)— ternary prompt, 10-min       (code: ternary_10)

Color rules:
  · Image overlays   → GT always green (#1a9641), predictions always vermilion (#D55E00),
                        positive prompt always blue (#0072B2), negative signal always
                        orange (#E69F00).  Do NOT change these by variant.
  · Chart elements   → each variant gets its own fixed Okabe-Ito color (VARIANT_COLOR).
  · Detection status → green border / green text = detected; red = missed.

Prompt rendering rules (consistent across ALL figures):
  · Binary prompt      → solid blue tint (PROMPT_BLUE, alpha=ALPHA_PROMPT)
  · Age-weighted prompt→ gradient blue tint (intensity ∝ age weight)
  · Ternary prompt     → blue where positive, ORANGE where negative signal
                         Use blend_ternary_prompt() which handles both channels.
  · The negative signal is reconstructed from all_prompts_union.png:
        ternary = where(pos>0, pos, -union)   — values in [-1, 1]

All figures must:
  · Use publication.mplstyle (loaded at import time here).
  · Carry a compact frameless legend INSIDE the figure whenever more than one
    series/overlay is shown — a figure must be interpretable without the
    caption. The caption repeats the mapping for accessibility.
  · Use panel letters (A, B, …) for multi-panel figures.
  · Size figures at FINAL print width (3.5 in single / 7.16 in double column)
    so fonts render at true point size — never rely on LaTeX downscaling.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Load publication style from skill ────────────────────────────────────────
import os as _os

_SKILL_CANDIDATES = [
    Path(_os.environ.get("SCIVIZ_SKILL_DIR", "")),
    Path("/data/dataiku/.claude/skills/scientific-visualization"),
    Path(__file__).parent / "scientific-visualization",
]
for _cand in _SKILL_CANDIDATES:
    if _cand and (_cand / "assets" / "publication.mplstyle").exists():
        _SKILL = _cand
        break
else:
    raise FileNotFoundError(
        "scientific-visualization skill not found; set SCIVIZ_SKILL_DIR"
    )
plt.style.use(str(_SKILL / "assets" / "publication.mplstyle"))

# ── Variant catalogue ─────────────────────────────────────────────────────────
# Maps evaluation-directory code → display name (used in captions / tables)
VARIANT_LABELS = {
    "base": "Binary",
    "age_5": "Age-weighted (5 min)",
    "age_10": "Age-weighted (10 min)",
    "ternary_5": "Age-wtd. + neg. signal (5 min)",
    "ternary_10": "Age-wtd. + neg. signal (10 min)",
}

# Short labels for in-figure legends (keep full names for captions/tables)
VARIANT_SHORT = {
    "base": "Binary (1 min)",
    "age_5": "Age-wtd. (5 min)",
    "age_10": "Age-wtd. (10 min)",
    "ternary_5": "Age-wtd. + neg. (5 min)",
    "ternary_10": "Age-wtd. + neg. (10 min)",
}

# Line styles / markers for redundant (colorblind-safe) encoding in line charts:
# solid = 5-min window, dashed = 10-min window, dotted = binary baseline
VARIANT_LS = {
    "base": (0, (1, 1.2)),
    "age_5": "-",
    "age_10": (0, (4, 1.6)),
    "ternary_5": "-",
    "ternary_10": (0, (4, 1.6)),
}
VARIANT_MARKER = {
    "base": "o",
    "age_5": "s",
    "age_10": "s",
    "ternary_5": "D",
    "ternary_10": "D",
}

# Ablation-chart colors — one fixed Okabe-Ito color per variant
# (for line / bar charts that compare multiple variants side by side)
# Color families: gray=binary, warm orange=age variants, blue/purple=ternary
# Color encodes the prompt family (gray=binary, orange=age, blue=ternary);
# within a family the 5-min variant is saturated and the 10-min variant is a
# lighter tint. Line style and marker provide redundant, grayscale-safe coding.
VARIANT_COLOR = {
    "base": "#5A5A5A",  # dark gray    — baseline
    "age_5": "#E69F00",  # orange       — age 5-min
    "age_10": "#F2C56E",  # light orange — age 10-min
    "ternary_5": "#0072B2",  # blue         — best variant
    "ternary_10": "#56B4E9",  # sky blue     — ternary 10-min
}

# Ablation order: baseline first, then 2×2 block (encoding × window)
VARIANT_ORDER = ["base", "age_5", "age_10", "ternary_5", "ternary_10"]

# ── Image-overlay colors (RGB uint8) ─────────────────────────────────────────
# NEVER change these by variant — consistent across the whole paper.
GT_COLOR = np.array([26, 150, 65], dtype=np.uint8)   # green    #1a9641
PRED_COLOR = np.array([213, 94, 0], dtype=np.uint8)  # vermilion #D55E00
PROMPT_COLOR = np.array([240, 228, 66], dtype=np.uint8)  # yellow   #F0E442

GT_HEX = "#1a9641"
PRED_HEX = "#D55E00"
PROMPT_HEX = "#F0E442"
NEG_HEX = "#E69F00"  # orange — ternary negative signal

# Negative-signal color as uint8 and normalised float arrays
NEG_COLOR = np.array([230, 159, 0], dtype=np.uint8)   # orange #E69F00
NEG_ORANGE = np.array([0.90, 0.62, 0.0])              # normalised for blending

# Detection-status colors (borders, markers)
DETECT_HEX = "#1a9641"  # dark green
MISS_HEX = "#cc2222"  # red

# ── IoU detection threshold ───────────────────────────────────────────────────
IOU_THR = 0.30

# ── Figure dimensions (inches) ───────────────────────────────────────────────
# Elsevier elsarticle [review] text width: 345 pt = 4.78 in. Figures are
# designed at FINAL print width so fonts render at true point size.
FIG_W_FULL = 4.78
FIG_W_HALF = 3.5
FIG_W_TWO3 = 4.0


# ── Helper: save figure ───────────────────────────────────────────────────────
def save_figure(fig, name, out_dir="paper/figures"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  → {name}.png")


# ── Grayscale display constants ───────────────────────────────────────────────
# Stretch display range so sky images appear lighter.
# vmin < 0 means the darkest raw pixels map to a lighter gray;
# vmax < 255 means midtones are boosted relative to the white point.
GRAY_VMIN = -30
GRAY_VMAX = 200


def load_gray(path):
    """Load image as a brightened uint8 grayscale array ready for RGB blending.
    Applies the same stretch as GRAY_VMIN/GRAY_VMAX so blended overlays are
    consistent with direct imshow(cmap='gray', vmin=GRAY_VMIN, vmax=GRAY_VMAX).
    """
    from PIL import Image as _Image

    arr = np.array(_Image.open(path).convert("L")).astype(np.float32)
    # Linear stretch: map [GRAY_VMIN, GRAY_VMAX] → [0, 255]
    arr = (arr - GRAY_VMIN) / (GRAY_VMAX - GRAY_VMIN) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


# ── Helper: image rendering ───────────────────────────────────────────────────
def desaturate(rgb, s=0.30):
    g = np.mean(rgb, axis=2, keepdims=True)
    return np.clip(rgb * (1 - s) + g * s, 0, 255).astype(np.uint8)


def blend_mask(rgb, mask, color, alpha):
    if mask is None or not mask.any():
        return rgb.copy()
    out = rgb.astype(np.float32)
    for c in range(3):
        out[:, :, c] = np.where(
            mask, out[:, :, c] * (1 - alpha) + color[c] * alpha, out[:, :, c]
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def panel_letter(ax, letter, x=-0.08, y=1.05):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


# ── Canonical style constants (used by ALL figure scripts) ──────────────────
# Overlay alphas
ALPHA_PROMPT = 0.75  # blue age-weighted prompt tint
ALPHA_PRED = 0.85  # vermilion prediction overlay
ALPHA_GT = 0.65  # green GT overlay (primary)
ALPHA_GT_UNDER = 0.25  # green GT underlay (faint, behind predictions)

# Font sizes (at figure scale — not at print scale)
FONT_SCORE_BADGE = 16  # score badges: bold white on vermilion
FONT_COL_TITLE = 11  # column/panel titles above axes
FONT_ROW_LABEL = 10  # row labels (left side)
FONT_PANEL_LETTER = 12  # panel letters (A, B, ...)

# Score badge style
SCORE_BADGE_KW = dict(
    fontsize=FONT_SCORE_BADGE,
    fontweight="bold",
    color="white",
    ha="right",
    va="bottom",
    bbox=dict(fc=PRED_HEX, alpha=0.92, pad=3.0, boxstyle="round,pad=0.3", ec="none"),
)

# Prompt overlay colour (blue, for age-weighted prompt tint on images)
PROMPT_BLUE = np.array([0.13, 0.47, 0.71])


def score_badge(ax, score):
    """Draw a score badge at bottom-right of an axes."""
    ax.text(0.97, 0.05, f"{score:.2f}", transform=ax.transAxes, **SCORE_BADGE_KW)


def col_title(ax, text):
    """Draw a column title centred above an axes."""
    ax.text(
        0.5,
        1.02,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=FONT_COL_TITLE,
        fontweight="bold",
    )


def clean_ax(ax):
    """Remove ticks and spines from an image axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def blend_prompt_blue(rgb, prompt_float, alpha=ALPHA_PROMPT):
    """Blend blue age-weighted prompt onto RGB image.
    prompt_float: float [0,1] array (age-normalised prompt intensity).
    """
    vis = rgb.astype(float)
    nz = prompt_float > 1e-6
    if nz.any():
        for c_i in range(3):
            vis[:, :, c_i] = np.where(
                nz,
                vis[:, :, c_i] * (1 - prompt_float * alpha)
                + PROMPT_BLUE[c_i] * 255 * prompt_float * alpha,
                vis[:, :, c_i],
            )
    return np.clip(vis, 0, 255).astype(np.uint8)


def load_ternary_prompt(pod_dir, video, obj_id, frame):
    """Reconstruct ternary prompt array in [-1, 1].

    Positive values  → target flight's age-weighted footprint (blue).
    Negative values  → competing flights' union footprint (orange).

    Formula (mirrors training/dataset/vos_segment_loader.py):
        ternary = where(pos > 0, pos, -union)

    Args:
        pod_dir: path to per_object_data_age_5 or per_object_data_age_10 dir
        video:   video id string, e.g. "00005"
        obj_id:  integer object id
        frame:   zero-padded frame string, e.g. "00047"
    Returns:
        float32 array shape (H, W) in [-1, 1], or zeros if files missing.
    """
    from PIL import Image as _Image

    pod = Path(pod_dir) / video
    pos_path   = pod / str(obj_id) / f"{frame}_prompt.png"
    union_path = pod / f"{frame}_all_prompts_union.png"

    def _load(p):
        if not p.exists():
            return np.zeros((1024, 1024), np.float32)
        arr = np.array(_Image.open(p).convert("L")).astype(np.float32)
        mx = arr.max()
        return arr / mx if mx > 0 else arr

    pos   = _load(pos_path)
    union = _load(union_path)
    return np.where(pos > 0, pos, -union)   # values in [-1, 1]


def blend_ternary_prompt(rgb, ternary, alpha=ALPHA_PROMPT):
    """Blend full ternary prompt onto RGB image.

    Positive regions → blue  tint (PROMPT_BLUE), intensity ∝ pos value.
    Negative regions → orange tint (NEG_ORANGE),  intensity ∝ neg value.

    Args:
        rgb:     uint8 (H, W, 3) image
        ternary: float32 (H, W) in [-1, 1]  (from load_ternary_prompt)
        alpha:   blending strength
    Returns:
        uint8 (H, W, 3) image with prompt overlaid.
    """
    vis = rgb.astype(np.float32)
    pos = np.clip(ternary, 0, 1)
    neg = np.clip(-ternary, 0, 1)

    pos_mask = pos > 1e-6
    neg_mask = neg > 1e-6

    if pos_mask.any():
        for c in range(3):
            vis[:, :, c] = np.where(
                pos_mask,
                vis[:, :, c] * (1 - pos * alpha) + PROMPT_BLUE[c] * 255 * pos * alpha,
                vis[:, :, c],
            )
    if neg_mask.any():
        for c in range(3):
            vis[:, :, c] = np.where(
                neg_mask,
                vis[:, :, c] * (1 - neg * alpha) + NEG_ORANGE[c] * 255 * neg * alpha,
                vis[:, :, c],
            )
    return np.clip(vis, 0, 255).astype(np.uint8)
