"""Ablation experiment: diagnose teleportation glitches.

Runs inference on a single object (folder "28" in video 00006) with three modes
at each frame:
  1. Normal:    memory attention + prompt
  2. No-memory: bypass memory attention, prompt only
  3. No-prompt: memory attention only, no prompt

Compares mask centroids/areas/scores across modes to isolate the cause.
"""

import sys
sys.path.insert(0, "/data/common/dataiku/config/projects/TRAILVISION/lib/python/sam2")

import math
import numpy as np
import torch
import torch.nn.functional as F

from contrailtrack.model.loader import load_model
from contrailtrack.data.video import load_frames
from contrailtrack.data.prompt_reader import read_prompts


def compute_centroid(mask_bool):
    coords = torch.nonzero(mask_bool)
    if len(coords) == 0:
        return None, None, 0
    cy = coords[:, 0].float().mean().item()
    cx = coords[:, 1].float().mean().item()
    return cx, cy, len(coords)


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
def run_ablation():
    device = "cuda"
    checkpoint = "/data/common/TRAILVISION/SAM2/log_ternary_5/checkpoints/checkpoint.pt"

    print("Loading model...")
    model = load_model(checkpoint=checkpoint, config="ternary", device=device)

    print("Loading frames...")
    frames, frame_names, orig_h, orig_w = load_frames(
        "/data/common/TRAILVISION/GVCCS_V/test/img_folder/00006"
    )
    name_to_idx = {name: idx for idx, name in enumerate(frame_names)}

    print("Loading prompts...")
    all_prompts = read_prompts(
        "/data/common/TRAILVISION/GVCCS_V/test/per_object_data_age_5",
        "00006", encoding="ternary"
    )
    obj_prompts = all_prompts["28"]
    prompt_frames = sorted(obj_prompts.keys())
    print(f"Object 28: {len(prompt_frames)} prompt frames ({prompt_frames[0]}-{prompt_frames[-1]})")

    T = len(frames)

    # Normal mode builds up memory
    normal_output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

    print()
    header = f"{'frame':>5s} | {'mode':<10s} | {'cx':>6s} {'cy':>6s} {'area':>7s} | {'score':>5s} | {'note':<s}"
    print(header)
    print("-" * len(header))

    prev_normal_cx = None
    prev_normal_cy = None

    for frame_name in prompt_frames:
        frame_idx = name_to_idx[frame_name]

        # Prepare prompt tensor
        prompt_mask = obj_prompts[frame_name]
        mask_tensor = torch.from_numpy(prompt_mask).float().to(device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        if mask_tensor.shape[-2:] != (model.image_size, model.image_size):
            mask_tensor = F.interpolate(
                mask_tensor, size=(model.image_size, model.image_size),
                mode="bilinear", align_corners=False,
            )

        # Extract backbone features (shared across modes)
        vision_feats, vision_pos_embeds, feat_sizes = extract_backbone(
            model, frames, frame_idx, device
        )

        # High-res features for SAM head
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        results = {}

        # === MODE 1: NORMAL (memory + prompt) ===
        pix_feat_normal = model._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats[-1:],
            current_vision_pos_embeds=vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=normal_output_dict,
            num_frames=T,
            track_in_reverse=False,
        )
        multimask = model._use_multimask(is_init_cond_frame=True, point_inputs=None)
        sam_out_normal = model._forward_sam_heads(
            backbone_features=pix_feat_normal,
            point_inputs=None,
            mask_inputs=mask_tensor,
            high_res_features=high_res_features,
            multimask_output=multimask,
        )
        _, _, _, low_res_normal, high_res_normal, obj_ptr_normal, obj_score_normal = sam_out_normal
        # Store in memory for next frames
        current_out_normal = {
            "point_inputs": None, "mask_inputs": mask_tensor,
            "pred_masks": low_res_normal, "pred_masks_high_res": high_res_normal,
            "obj_ptr": obj_ptr_normal,
        }
        if not model.training:
            current_out_normal["object_score_logits"] = obj_score_normal
        model._encode_memory_in_output(
            vision_feats, feat_sizes, None, True,
            high_res_normal, obj_score_normal, current_out_normal,
        )
        normal_output_dict["cond_frame_outputs"][frame_idx] = current_out_normal

        # Decode normal mask
        pred_normal = high_res_normal[0, 0]
        if pred_normal.shape != (orig_h, orig_w):
            pred_normal = F.interpolate(
                pred_normal.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w), mode="bilinear", align_corners=False,
            )[0, 0]
        mask_normal = pred_normal > 0.0
        score_normal = obj_score_normal[0].sigmoid().item()
        cx_n, cy_n, area_n = compute_centroid(mask_normal)

        # Check for jump
        note = ""
        if prev_normal_cx is not None and cx_n is not None:
            dist = math.sqrt((cx_n - prev_normal_cx)**2 + (cy_n - prev_normal_cy)**2)
            if dist > 150:
                note = f"JUMP {dist:.0f}px"
        if cx_n is not None:
            prev_normal_cx, prev_normal_cy = cx_n, cy_n

        cx_str = f"{cx_n:.0f}" if cx_n else "none"
        cy_str = f"{cy_n:.0f}" if cy_n else "none"
        print(f"{frame_name:>5s} | {'normal':<10s} | {cx_str:>6s} {cy_str:>6s} {area_n:>7d} | {score_normal:>5.3f} | {note}")

        # === MODE 2: NO-MEMORY (bypass memory attention, prompt only) ===
        B = vision_feats[-1].size(1)
        C = model.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat_nomem = vision_feats[-1] + model.no_mem_embed
        pix_feat_nomem = pix_feat_nomem.permute(1, 2, 0).view(B, C, H, W)

        sam_out_nomem = model._forward_sam_heads(
            backbone_features=pix_feat_nomem,
            point_inputs=None,
            mask_inputs=mask_tensor,
            high_res_features=high_res_features,
            multimask_output=multimask,
        )
        _, _, _, _, high_res_nomem, _, obj_score_nomem = sam_out_nomem
        pred_nomem = high_res_nomem[0, 0]
        if pred_nomem.shape != (orig_h, orig_w):
            pred_nomem = F.interpolate(
                pred_nomem.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w), mode="bilinear", align_corners=False,
            )[0, 0]
        mask_nomem = pred_nomem > 0.0
        score_nomem = obj_score_nomem[0].sigmoid().item()
        cx_nm, cy_nm, area_nm = compute_centroid(mask_nomem)

        cx_str = f"{cx_nm:.0f}" if cx_nm else "none"
        cy_str = f"{cy_nm:.0f}" if cy_nm else "none"
        print(f"{'':>5s} | {'no-memory':<10s} | {cx_str:>6s} {cy_str:>6s} {area_nm:>7d} | {score_nomem:>5.3f} |")

        # === MODE 3: NO-PROMPT (memory only, no mask input) ===
        sam_out_noprompt = model._forward_sam_heads(
            backbone_features=pix_feat_normal,  # same memory-conditioned features
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=multimask,
        )
        _, _, _, _, high_res_noprompt, _, obj_score_noprompt = sam_out_noprompt
        pred_noprompt = high_res_noprompt[0, 0]
        if pred_noprompt.shape != (orig_h, orig_w):
            pred_noprompt = F.interpolate(
                pred_noprompt.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w), mode="bilinear", align_corners=False,
            )[0, 0]
        mask_noprompt = pred_noprompt > 0.0
        score_noprompt = obj_score_noprompt[0].sigmoid().item()
        cx_np, cy_np, area_np = compute_centroid(mask_noprompt)

        cx_str = f"{cx_np:.0f}" if cx_np else "none"
        cy_str = f"{cy_np:.0f}" if cy_np else "none"
        print(f"{'':>5s} | {'no-prompt':<10s} | {cx_str:>6s} {cy_str:>6s} {area_np:>7d} | {score_noprompt:>5.3f} |")
        print()


if __name__ == "__main__":
    run_ablation()
