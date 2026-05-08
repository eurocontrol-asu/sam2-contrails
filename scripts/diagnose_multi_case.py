"""Multi-case teleportation diagnosis with visualizations.

Runs the tensor diagnostic on multiple teleportation cases, collecting
memory-conditioned feature norms, prompt embeddings, and generating
spatial heatmap visualizations.
"""

import sys
sys.path.insert(0, "/data/common/dataiku/config/projects/TRAILVISION/lib/python/sam2")

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils

from contrailtrack.model.loader import load_model
from contrailtrack.data.video import load_frames
from contrailtrack.data.prompt_reader import read_prompts


PRED_DIR = Path("/data/common/dataiku/config/projects/TRAILVISION/lib/python/sam2/outputs/ternary_5")
IMAGES_DIR = Path("/data/common/TRAILVISION/GVCCS_V/test/img_folder")
PROMPTS_DIR = Path("/data/common/TRAILVISION/GVCCS_V/test/per_object_data_age_5")
CHECKPOINT = "/data/common/TRAILVISION/SAM2/log_ternary_5/checkpoints/checkpoint.pt"
OUT_DIR = Path("/data/common/dataiku/config/projects/TRAILVISION/lib/python/sam2/docs/teleportation_diagnosis")


def find_teleportations(video_id, min_jump=150):
    """Find teleportation events from predictions."""
    pred_file = PRED_DIR / f"{video_id}.json"
    with open(pred_file) as f:
        data = json.load(f)
    preds = data["annotations"] if isinstance(data, dict) else data

    by_obj = defaultdict(list)
    for ann in preds:
        obj_id = str(ann.get("object_id", ann.get("category_id", "")))
        frame = ann.get("frame_name", ann.get("image_id", ""))
        rle = ann["segmentation"]
        m = mask_utils.decode(rle)
        coords = np.argwhere(m > 0)
        if len(coords) == 0:
            continue
        cy, cx = coords.mean(axis=0)
        by_obj[obj_id].append({
            "frame": frame, "cx": cx, "cy": cy,
            "area": len(coords), "score": ann.get("score", 0),
        })

    events = []
    for obj_id, frames in by_obj.items():
        frames.sort(key=lambda x: x["frame"])
        for i in range(1, len(frames)):
            prev, curr = frames[i - 1], frames[i]
            dist = math.sqrt((curr["cx"] - prev["cx"])**2 + (curr["cy"] - prev["cy"])**2)
            if dist > min_jump:
                events.append({
                    "video": video_id, "object": obj_id,
                    "frame_before": prev["frame"], "frame_after": curr["frame"],
                    "cx_before": prev["cx"], "cy_before": prev["cy"],
                    "cx_after": curr["cx"], "cy_after": curr["cy"],
                    "jump": dist,
                    "score_before": prev["score"], "score_after": curr["score"],
                    "all_frames": frames,
                })
    return events


def pixel_to_feat(px, py, orig_h, orig_w, feat_h, feat_w):
    fx = int(px / orig_w * feat_w)
    fy = int(py / orig_h * feat_h)
    return min(fx, feat_w - 1), min(fy, feat_h - 1)


def extract_backbone(model, frames_tensor, frame_idx, device):
    img = frames_tensor[frame_idx:frame_idx + 1].to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        backbone_out = model.image_encoder(img)
    for key in backbone_out:
        if isinstance(backbone_out[key], torch.Tensor):
            backbone_out[key] = backbone_out[key].float()
        elif isinstance(backbone_out[key], list):
            backbone_out[key] = [t.float() if isinstance(t, torch.Tensor) else t for t in backbone_out[key]]
    if model.use_high_res_features_in_sam:
        backbone_out["backbone_fpn"][0] = model.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = model.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
    _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
    return vision_feats, vision_pos_embeds, feat_sizes


@torch.inference_mode()
def analyze_case(model, case, device="cuda"):
    """Run full tensor diagnostic for a teleportation case.

    Returns a dict with all collected data for the document.
    """
    video_id = case["video"]
    obj_id = case["object"]

    print(f"\n{'='*70}")
    print(f"ANALYZING: video={video_id} object={obj_id}")
    print(f"  Jump: {case['frame_before']} -> {case['frame_after']} ({case['jump']:.0f}px)")
    print(f"{'='*70}")

    frames, frame_names, orig_h, orig_w = load_frames(str(IMAGES_DIR / video_id))
    name_to_idx = {name: idx for idx, name in enumerate(frame_names)}
    T = len(frames)

    all_prompts = read_prompts(str(PROMPTS_DIR), video_id, encoding="ternary")
    if obj_id not in all_prompts:
        print(f"  Object {obj_id} not found in prompts. Available: {list(all_prompts.keys())[:10]}")
        return None
    obj_prompts = all_prompts[obj_id]
    prompt_frames = sorted(obj_prompts.keys())

    # Find the teleportation frame and the one before it
    tp_frame_after = case["frame_after"]
    tp_frame_before = case["frame_before"]

    # Make sure both frames are in our prompt set
    if tp_frame_after not in obj_prompts or tp_frame_before not in obj_prompts:
        print(f"  Teleportation frames not in prompt range. Prompt frames: {prompt_frames[0]}-{prompt_frames[-1]}")
        return None

    # Key locations
    correct_loc = (case["cx_before"], case["cy_before"])
    teleport_loc = (case["cx_after"], case["cy_after"])

    # Build up memory by running all frames up to and including the teleportation frame
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    frame_data = []

    tp_after_idx = name_to_idx.get(tp_frame_after)
    tp_before_idx = name_to_idx.get(tp_frame_before)
    if tp_after_idx is None or tp_before_idx is None:
        print(f"  Frame names not found in video")
        return None

    H_feat, W_feat = None, None
    prev_cx, prev_cy = None, None

    for frame_name in prompt_frames:
        frame_idx = name_to_idx[frame_name]

        prompt_mask = obj_prompts[frame_name]
        mask_tensor = torch.from_numpy(prompt_mask).float().to(device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        if mask_tensor.shape[-2:] != (model.image_size, model.image_size):
            mask_tensor = F.interpolate(
                mask_tensor, size=(model.image_size, model.image_size),
                mode="bilinear", align_corners=False,
            )

        vision_feats, vision_pos_embeds, feat_sizes = extract_backbone(
            model, frames, frame_idx, device
        )
        H, W = feat_sizes[-1]
        H_feat, W_feat = H, W
        C = vision_feats[-1].size(2)
        B = vision_feats[-1].size(1)

        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        # Memory-conditioned features
        pix_feat = model._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats[-1:],
            current_vision_pos_embeds=vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=output_dict,
            num_frames=T,
            track_in_reverse=False,
        )

        # SAM heads
        multimask = model._use_multimask(is_init_cond_frame=True, point_inputs=None)
        sam_out = model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=mask_tensor,
            high_res_features=high_res_features,
            multimask_output=multimask,
        )
        _, _, _, low_res, high_res, obj_ptr, obj_score = sam_out

        current_out = {
            "point_inputs": None, "mask_inputs": mask_tensor,
            "pred_masks": low_res, "pred_masks_high_res": high_res,
            "obj_ptr": obj_ptr,
        }
        if not model.training:
            current_out["object_score_logits"] = obj_score

        model._encode_memory_in_output(
            vision_feats, feat_sizes, None, True,
            high_res, obj_score, current_out,
        )
        output_dict["cond_frame_outputs"][frame_idx] = current_out

        # Compute centroid and collect data
        pred = high_res[0, 0]
        if pred.shape != (orig_h, orig_w):
            pred = F.interpolate(
                pred.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w),
                mode="bilinear", align_corners=False,
            )[0, 0]
        mask_bool = pred > 0.0
        coords = torch.nonzero(mask_bool)
        if len(coords) > 0:
            cy = coords[:, 0].float().mean().item()
            cx = coords[:, 1].float().mean().item()
            area = len(coords)
        else:
            cx, cy, area = None, None, 0
        score = obj_score[0].sigmoid().item()

        # Detect jump
        jump = 0
        if prev_cx is not None and cx is not None:
            jump = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        if cx is not None:
            prev_cx, prev_cy = cx, cy

        # Raw backbone norms
        raw_feat_2d = vision_feats[-1].squeeze(1).view(H, W, C)

        # Memory delta
        raw_4d = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        delta = pix_feat - raw_4d

        # Dense prompt embeddings
        sam_point_coords = torch.zeros(1, 1, 2, device=device)
        sam_point_labels = -torch.ones(1, 1, dtype=torch.int32, device=device)
        if mask_tensor.shape[-2:] != model.sam_prompt_encoder.mask_input_size:
            sam_mask_prompt = F.interpolate(
                mask_tensor.float(),
                size=model.sam_prompt_encoder.mask_input_size,
                align_corners=False, mode="bilinear", antialias=True,
            )
        else:
            sam_mask_prompt = mask_tensor
        _, dense_emb = model.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None, masks=sam_mask_prompt,
        )

        # Prompt statistics
        prompt_pos = (prompt_mask > 0).sum()
        prompt_neg = (prompt_mask < 0).sum()
        prompt_zero = (prompt_mask == 0).sum()

        # Collect per-frame data
        fd = {
            "frame_name": frame_name,
            "frame_idx": frame_idx,
            "cx": cx, "cy": cy, "area": area,
            "score": score, "jump": jump,
            "prompt_pos_pixels": int(prompt_pos),
            "prompt_neg_pixels": int(prompt_neg),
            "prompt_zero_pixels": int(prompt_zero),
        }

        # For key frames (before and after teleportation), collect detailed tensor data
        is_key = frame_name in (tp_frame_before, tp_frame_after)
        is_near_key = False
        # Also collect for 2 frames before the teleportation
        for pf in prompt_frames:
            pf_idx = name_to_idx.get(pf)
            if pf_idx is not None and tp_before_idx - 3 <= pf_idx <= tp_after_idx:
                if frame_name == pf:
                    is_near_key = True

        if is_key or is_near_key:
            pf_norms = pix_feat[0].norm(dim=0).cpu().numpy()  # [H, W]
            delta_norms = delta[0].norm(dim=0).cpu().numpy()  # [H, W]
            de_norms = dense_emb[0].norm(dim=0).cpu().numpy()  # [H, W]
            fused = pix_feat + dense_emb
            fused_norms = fused[0].norm(dim=0).cpu().numpy()  # [H, W]

            fd["pix_feat_norms"] = pf_norms
            fd["delta_norms"] = delta_norms
            fd["dense_emb_norms"] = de_norms
            fd["fused_norms"] = fused_norms

            # Values at key locations
            if correct_loc[0] is not None:
                fx_c, fy_c = pixel_to_feat(correct_loc[0], correct_loc[1], orig_h, orig_w, W_feat, H_feat)
                fd["pf_correct"] = pf_norms[fy_c, fx_c]
                fd["delta_correct"] = delta_norms[fy_c, fx_c]
                fd["de_correct"] = de_norms[fy_c, fx_c]
                fd["fused_correct"] = fused_norms[fy_c, fx_c]
            if teleport_loc[0] is not None:
                fx_t, fy_t = pixel_to_feat(teleport_loc[0], teleport_loc[1], orig_h, orig_w, W_feat, H_feat)
                fd["pf_teleport"] = pf_norms[fy_t, fx_t]
                fd["delta_teleport"] = delta_norms[fy_t, fx_t]
                fd["de_teleport"] = de_norms[fy_t, fx_t]
                fd["fused_teleport"] = fused_norms[fy_t, fx_t]

        frame_data.append(fd)

    return {
        "video": video_id,
        "object": obj_id,
        "tp_frame_before": tp_frame_before,
        "tp_frame_after": tp_frame_after,
        "correct_loc": correct_loc,
        "teleport_loc": teleport_loc,
        "jump": case["jump"],
        "orig_h": orig_h, "orig_w": orig_w,
        "feat_h": H_feat, "feat_w": W_feat,
        "frame_data": frame_data,
    }


def generate_visualizations(result, out_dir):
    """Generate spatial heatmap visualizations for a case."""
    os.makedirs(out_dir, exist_ok=True)

    video_id = result["video"]
    obj_id = result["object"]
    orig_h, orig_w = result["orig_h"], result["orig_w"]

    # Find the key frames with detailed data
    key_frames = [fd for fd in result["frame_data"] if "pix_feat_norms" in fd]
    if not key_frames:
        return

    for fd in key_frames:
        frame_name = fd["frame_name"]
        is_teleport = frame_name == result["tp_frame_after"]

        # Load the actual image for overlay
        img_path = IMAGES_DIR / video_id / f"{frame_name}.jpg"
        if img_path.exists():
            img = np.array(Image.open(img_path))
        else:
            img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        # Load the prompt for this frame
        prompt_dir = PROMPTS_DIR / video_id / obj_id
        prompt_path = prompt_dir / f"{frame_name}_prompt.png"
        if prompt_path.exists():
            raw_prompt = np.array(Image.open(prompt_path)).astype(np.float32) / 255.0
        else:
            raw_prompt = np.zeros((orig_h, orig_w), dtype=np.float32)

        # Also load union
        union_path = PROMPTS_DIR / video_id / f"{frame_name}_all_prompts_union.png"
        if union_path.exists():
            union = np.array(Image.open(union_path)).astype(np.float32) / 255.0
            ternary = np.where(raw_prompt > 0, raw_prompt, -union)
        else:
            ternary = raw_prompt

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

        correct = result["correct_loc"]
        teleport = result["teleport_loc"]

        # Row 1: Image, Ternary Prompt, Memory Features, Memory Delta
        # 1. Image with markers
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        if correct[0] is not None:
            ax1.plot(correct[0], correct[1], 'g+', markersize=15, markeredgewidth=3, label="Correct")
        if teleport[0] is not None:
            ax1.plot(teleport[0], teleport[1], 'rx', markersize=15, markeredgewidth=3, label="Teleport")
        ax1.set_title(f"Frame {frame_name}" + (" [TELEPORT]" if is_teleport else ""), fontsize=11)
        ax1.legend(fontsize=8)
        ax1.axis("off")

        # 2. Ternary prompt
        ax2 = fig.add_subplot(gs[0, 1])
        cmap = plt.cm.RdBu
        vmax = max(abs(ternary.min()), abs(ternary.max()), 0.01)
        ax2.imshow(ternary, cmap=cmap, vmin=-vmax, vmax=vmax)
        if correct[0] is not None:
            ax2.plot(correct[0], correct[1], 'g+', markersize=15, markeredgewidth=3)
        if teleport[0] is not None:
            ax2.plot(teleport[0], teleport[1], 'rx', markersize=15, markeredgewidth=3)
        ax2.set_title(f"Ternary prompt\npos={fd['prompt_pos_pixels']} neg={fd['prompt_neg_pixels']}", fontsize=10)
        ax2.axis("off")

        # 3. Memory-conditioned feature norms (spatial heatmap)
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(fd["pix_feat_norms"], cmap="hot", interpolation="nearest")
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        # Mark locations in feature space
        fh, fw = fd["pix_feat_norms"].shape
        if correct[0] is not None:
            fx_c, fy_c = pixel_to_feat(correct[0], correct[1], orig_h, orig_w, fw, fh)
            ax3.plot(fx_c, fy_c, 'g+', markersize=12, markeredgewidth=2)
        if teleport[0] is not None:
            fx_t, fy_t = pixel_to_feat(teleport[0], teleport[1], orig_h, orig_w, fw, fh)
            ax3.plot(fx_t, fy_t, 'cx', markersize=12, markeredgewidth=2)
        vals = ""
        if "pf_correct" in fd:
            vals += f"correct={fd['pf_correct']:.1f}"
        if "pf_teleport" in fd:
            vals += f" teleport={fd['pf_teleport']:.1f}"
        ax3.set_title(f"Memory-cond features (L2)\n{vals}", fontsize=9)

        # 4. Memory delta norms
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(fd["delta_norms"], cmap="hot", interpolation="nearest")
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        if correct[0] is not None:
            ax4.plot(fx_c, fy_c, 'g+', markersize=12, markeredgewidth=2)
        if teleport[0] is not None:
            ax4.plot(fx_t, fy_t, 'cx', markersize=12, markeredgewidth=2)
        vals = ""
        if "delta_correct" in fd:
            vals += f"correct={fd['delta_correct']:.1f}"
        if "delta_teleport" in fd:
            vals += f" teleport={fd['delta_teleport']:.1f}"
        ax4.set_title(f"Memory delta (L2)\n{vals}", fontsize=9)

        # Row 2: Dense prompt embeddings, Fused features, centroid trajectory, summary
        # 5. Dense prompt embedding norms
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(fd["dense_emb_norms"], cmap="viridis", interpolation="nearest")
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        if correct[0] is not None:
            ax5.plot(fx_c, fy_c, 'g+', markersize=12, markeredgewidth=2)
        if teleport[0] is not None:
            ax5.plot(fx_t, fy_t, 'rx', markersize=12, markeredgewidth=2)
        vals = ""
        if "de_correct" in fd:
            vals += f"correct={fd['de_correct']:.1f}"
        if "de_teleport" in fd:
            vals += f" teleport={fd['de_teleport']:.1f}"
        ax5.set_title(f"Dense prompt embed (L2)\n{vals}", fontsize=9)

        # 6. Fused features (src = pix_feat + dense_emb)
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(fd["fused_norms"], cmap="hot", interpolation="nearest")
        plt.colorbar(im6, ax=ax6, fraction=0.046)
        if correct[0] is not None:
            ax6.plot(fx_c, fy_c, 'g+', markersize=12, markeredgewidth=2)
        if teleport[0] is not None:
            ax6.plot(fx_t, fy_t, 'cx', markersize=12, markeredgewidth=2)
        vals = ""
        if "fused_correct" in fd:
            vals += f"correct={fd['fused_correct']:.1f}"
        if "fused_teleport" in fd:
            vals += f" teleport={fd['fused_teleport']:.1f}"
        ax6.set_title(f"Fused src (L2)\n{vals}", fontsize=9)

        # 7. Centroid trajectory
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(img, alpha=0.4)
        cxs = [fd2["cx"] for fd2 in result["frame_data"] if fd2["cx"] is not None]
        cys = [fd2["cy"] for fd2 in result["frame_data"] if fd2["cy"] is not None]
        fns = [fd2["frame_name"] for fd2 in result["frame_data"] if fd2["cx"] is not None]
        if cxs:
            colors = np.linspace(0, 1, len(cxs))
            scatter = ax7.scatter(cxs, cys, c=colors, cmap="coolwarm", s=30, zorder=5)
            ax7.plot(cxs, cys, 'k-', alpha=0.3, linewidth=1)
            for i, fn in enumerate(fns):
                if fn in (result["tp_frame_before"], result["tp_frame_after"]):
                    ax7.annotate(fn, (cxs[i], cys[i]), fontsize=7, fontweight="bold",
                                color="red" if fn == result["tp_frame_after"] else "green")
        ax7.set_title("Centroid trajectory\n(blue=early, red=late)", fontsize=10)
        ax7.set_xlim(0, orig_w)
        ax7.set_ylim(orig_h, 0)
        ax7.axis("off")

        # 8. Summary text
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis("off")
        summary_lines = [
            f"Video: {video_id}",
            f"Object: {obj_id}",
            f"Frame: {frame_name}",
            f"",
            f"Prediction centroid:",
            f"  cx={fd['cx']:.0f}" if fd["cx"] else "  no mask",
            f"  cy={fd['cy']:.0f}" if fd["cy"] else "",
            f"  area={fd['area']}",
            f"  score={fd['score']:.3f}",
            f"",
            f"Correct loc: ({correct[0]:.0f}, {correct[1]:.0f})" if correct[0] else "",
            f"Teleport loc: ({teleport[0]:.0f}, {teleport[1]:.0f})" if teleport[0] else "",
            f"Jump: {fd['jump']:.0f}px" if fd['jump'] > 0 else "",
        ]
        if "pf_correct" in fd and "pf_teleport" in fd:
            summary_lines += [
                f"",
                f"Memory-cond L2:",
                f"  correct:  {fd['pf_correct']:.1f}",
                f"  teleport: {fd['pf_teleport']:.1f}",
                f"  ratio: {fd['pf_teleport']/fd['pf_correct']:.2f}x" if fd["pf_correct"] > 0 else "",
                f"",
                f"Prompt embed L2:",
                f"  correct:  {fd['de_correct']:.1f}",
                f"  teleport: {fd['de_teleport']:.1f}",
            ]
        ax8.text(0.05, 0.95, "\n".join(summary_lines), transform=ax8.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        tag = "TELEPORT" if is_teleport else "before"
        fig.suptitle(
            f"Teleportation Diagnosis: video={video_id} obj={obj_id} frame={frame_name} ({tag})",
            fontsize=13, fontweight="bold"
        )

        outpath = out_dir / f"{video_id}_obj{obj_id}_{frame_name}_{tag}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}")

    # Also generate a summary timeline figure
    fig2, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    fnames = [fd["frame_name"] for fd in result["frame_data"]]
    x = range(len(fnames))

    # Panel 1: Centroid position
    cxs = [fd["cx"] if fd["cx"] is not None else np.nan for fd in result["frame_data"]]
    cys = [fd["cy"] if fd["cy"] is not None else np.nan for fd in result["frame_data"]]
    axes[0].plot(x, cxs, 'b.-', label="cx")
    axes[0].plot(x, cys, 'r.-', label="cy")
    axes[0].set_ylabel("Pixel position")
    axes[0].legend()
    axes[0].set_title(f"video={video_id} obj={obj_id}: Centroid trajectory and feature norms")
    # Mark teleportation
    for i, fd in enumerate(result["frame_data"]):
        if fd["frame_name"] == result["tp_frame_after"]:
            axes[0].axvline(i, color='red', alpha=0.5, linestyle='--', label="teleport")

    # Panel 2: Score and area
    scores = [fd["score"] for fd in result["frame_data"]]
    areas = [fd["area"] for fd in result["frame_data"]]
    ax2a = axes[1]
    ax2b = ax2a.twinx()
    ax2a.plot(x, scores, 'g.-', label="score")
    ax2b.plot(x, areas, 'm.-', label="area", alpha=0.6)
    ax2a.set_ylabel("Score", color='g')
    ax2b.set_ylabel("Area (px)", color='m')
    ax2a.legend(loc="upper left")
    ax2b.legend(loc="upper right")
    for i, fd in enumerate(result["frame_data"]):
        if fd["frame_name"] == result["tp_frame_after"]:
            axes[1].axvline(i, color='red', alpha=0.5, linestyle='--')

    # Panel 3: Feature norms at key locations (where available)
    pf_c = [fd.get("pf_correct", np.nan) for fd in result["frame_data"]]
    pf_t = [fd.get("pf_teleport", np.nan) for fd in result["frame_data"]]
    de_c = [fd.get("de_correct", np.nan) for fd in result["frame_data"]]
    de_t = [fd.get("de_teleport", np.nan) for fd in result["frame_data"]]
    axes[2].plot(x, pf_c, 'g.-', label="mem_feat @ correct", linewidth=2)
    axes[2].plot(x, pf_t, 'r.-', label="mem_feat @ teleport", linewidth=2)
    axes[2].plot(x, de_c, 'g.--', label="prompt_emb @ correct", alpha=0.6)
    axes[2].plot(x, de_t, 'r.--', label="prompt_emb @ teleport", alpha=0.6)
    axes[2].set_ylabel("L2 norm")
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Frame")
    for i, fd in enumerate(result["frame_data"]):
        if fd["frame_name"] == result["tp_frame_after"]:
            axes[2].axvline(i, color='red', alpha=0.5, linestyle='--')

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(fnames, rotation=45, fontsize=7)

    fig2.tight_layout()
    outpath2 = out_dir / f"{video_id}_obj{obj_id}_timeline.png"
    fig2.savefig(outpath2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {outpath2}")


def collect_summary_table(results):
    """Print a summary comparison table across all cases."""
    print("\n" + "=" * 90)
    print("SUMMARY: Feature norms at teleportation frames")
    print("=" * 90)
    header = f"{'Case':<25s} | {'mem@correct':>11s} {'mem@teleport':>12s} {'ratio':>6s} | {'prompt@c':>8s} {'prompt@t':>8s} | {'fused@c':>8s} {'fused@t':>8s}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r is None:
            continue
        tp_frame = r["tp_frame_after"]
        fd = None
        for f in r["frame_data"]:
            if f["frame_name"] == tp_frame and "pf_correct" in f:
                fd = f
                break
        if fd is None:
            continue

        label = f"v{r['video']} o{r['object']} f{tp_frame}"
        pf_c = fd.get("pf_correct", 0)
        pf_t = fd.get("pf_teleport", 0)
        ratio = pf_t / pf_c if pf_c > 0 else float('inf')
        de_c = fd.get("de_correct", 0)
        de_t = fd.get("de_teleport", 0)
        fu_c = fd.get("fused_correct", 0)
        fu_t = fd.get("fused_teleport", 0)
        print(f"{label:<25s} | {pf_c:>11.1f} {pf_t:>12.1f} {ratio:>6.2f} | {de_c:>8.1f} {de_t:>8.1f} | {fu_c:>8.1f} {fu_t:>8.1f}")


@torch.inference_mode()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Define cases to analyze (from find_teleportations output)
    cases_to_find = [
        ("00006", "28"),    # Known case: serial teleportation frames 12-16
        ("00002", "107"),   # 516px jump, frames 53→54
        ("00017", "38"),    # 507px jump, frames 49→50
        ("00002", "140"),   # 769px jump, frames 182→188
    ]

    print("Finding teleportation events...")
    all_cases = []
    for video_id, target_obj in cases_to_find:
        events = find_teleportations(video_id, min_jump=100)
        obj_events = [e for e in events if e["object"] == target_obj]
        if obj_events:
            # Take the largest jump for this object
            best = max(obj_events, key=lambda e: e["jump"])
            all_cases.append(best)
            print(f"  Found: {video_id}/{target_obj} jump={best['jump']:.0f}px ({best['frame_before']}->{best['frame_after']})")
        else:
            print(f"  Not found: {video_id}/{target_obj} (no events > 100px)")

    if not all_cases:
        print("No cases found!")
        return

    print(f"\nLoading model...")
    device = "cuda"
    model = load_model(checkpoint=CHECKPOINT, config="ternary", device=device)

    results = []
    for case in all_cases:
        result = analyze_case(model, case, device=device)
        if result is not None:
            generate_visualizations(result, OUT_DIR)
            results.append(result)

    collect_summary_table(results)

    # Save raw data for the document
    summary = []
    for r in results:
        if r is None:
            continue
        s = {
            "video": r["video"], "object": r["object"],
            "tp_frame_before": r["tp_frame_before"],
            "tp_frame_after": r["tp_frame_after"],
            "correct_loc": [float(x) if x is not None else None for x in r["correct_loc"]],
            "teleport_loc": [float(x) if x is not None else None for x in r["teleport_loc"]],
            "jump": r["jump"],
            "frame_data": [
                {k: v for k, v in fd.items() if not isinstance(v, np.ndarray)}
                for fd in r["frame_data"]
            ],
        }
        summary.append(s)

    with open(OUT_DIR / "diagnosis_data.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved raw data to {OUT_DIR / 'diagnosis_data.json'}")


if __name__ == "__main__":
    main()
