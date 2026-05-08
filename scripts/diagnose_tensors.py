"""Deeper diagnostic: inspect intermediate tensor magnitudes at teleportation frame.

At frame 13 (the teleportation frame for object 28 in video 00006), extracts:
  1. Memory-conditioned features (pix_feat) at correct location vs teleport target
  2. Dense prompt embeddings at both locations
  3. src = pix_feat + dense_embeddings at both locations
  4. The mask_downsampler output for a zero-valued input region
"""

import sys
sys.path.insert(0, "/data/common/dataiku/config/projects/TRAILVISION/lib/python/sam2")

import numpy as np
import torch
import torch.nn.functional as F

from contrailtrack.model.loader import load_model
from contrailtrack.data.video import load_frames
from contrailtrack.data.prompt_reader import read_prompts


def pixel_to_feat(px, py, orig_h, orig_w, feat_h, feat_w):
    """Convert pixel coordinates to feature map coordinates."""
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
def run_tensor_diagnostic():
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

    T = len(frames)

    # Key locations (pixel coords from ablation results)
    correct_loc = (862, 427)    # where the prompt is at frame 13
    teleport_loc = (648, 623)   # where prediction jumps to at frame 13

    # Build up memory by running frames 7-12 normally
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    prompt_frames = sorted(obj_prompts.keys())

    print("\n=== Building memory (frames 7-12) ===")
    for frame_name in prompt_frames:
        frame_idx = name_to_idx[frame_name]
        if frame_idx > name_to_idx["00013"]:
            break

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

        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

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

        # Compute centroid for tracking
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
            cx, cy, area = 0, 0, 0
        score = obj_score[0].sigmoid().item()
        print(f"  {frame_name}: cx={cx:.0f} cy={cy:.0f} area={area} score={score:.3f}")

    # === NOW FRAME 13: the teleportation frame ===
    print("\n" + "=" * 70)
    print("=== FRAME 13 TENSOR ANALYSIS ===")
    print("=" * 70)

    frame_name = "00013"
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

    if len(vision_feats) > 1:
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
        ]
    else:
        high_res_features = None

    # Feature map size
    H, W = feat_sizes[-1]
    print(f"\nFeature map size: {H}x{W}")
    print(f"Correct location (pixel): cx={correct_loc[0]}, cy={correct_loc[1]}")
    print(f"Teleport location (pixel): cx={teleport_loc[0]}, cy={teleport_loc[1]}")

    fx_c, fy_c = pixel_to_feat(correct_loc[0], correct_loc[1], orig_h, orig_w, W, H)
    fx_t, fy_t = pixel_to_feat(teleport_loc[0], teleport_loc[1], orig_h, orig_w, W, H)
    print(f"Correct location (feat): fx={fx_c}, fy={fy_c}")
    print(f"Teleport location (feat): fx={fx_t}, fy={fy_t}")

    # --- 1. Raw backbone features (before memory attention) ---
    print("\n--- 1. RAW BACKBONE FEATURES (before memory attention) ---")
    raw_feat = vision_feats[-1]  # [H*W, 1, C]
    C = raw_feat.size(2)
    raw_feat_2d = raw_feat.squeeze(1).view(H, W, C)  # [H, W, C]

    raw_correct = raw_feat_2d[fy_c, fx_c]
    raw_teleport = raw_feat_2d[fy_t, fx_t]
    print(f"  Correct loc:  L2={raw_correct.norm():.3f}  mean={raw_correct.mean():.4f}  std={raw_correct.std():.4f}")
    print(f"  Teleport loc: L2={raw_teleport.norm():.3f}  mean={raw_teleport.mean():.4f}  std={raw_teleport.std():.4f}")
    cosine = F.cosine_similarity(raw_correct.unsqueeze(0), raw_teleport.unsqueeze(0)).item()
    print(f"  Cosine similarity: {cosine:.4f}")

    # --- 2. Memory-conditioned features ---
    print("\n--- 2. MEMORY-CONDITIONED FEATURES (pix_feat) ---")
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
    # pix_feat: [1, C, H, W]
    pf_correct = pix_feat[0, :, fy_c, fx_c]
    pf_teleport = pix_feat[0, :, fy_t, fx_t]
    print(f"  Correct loc:  L2={pf_correct.norm():.3f}  mean={pf_correct.mean():.4f}  std={pf_correct.std():.4f}")
    print(f"  Teleport loc: L2={pf_teleport.norm():.3f}  mean={pf_teleport.mean():.4f}  std={pf_teleport.std():.4f}")
    cosine = F.cosine_similarity(pf_correct.unsqueeze(0), pf_teleport.unsqueeze(0)).item()
    print(f"  Cosine similarity: {cosine:.4f}")

    # Spatial distribution of pix_feat L2 norms
    pf_norms = pix_feat[0].norm(dim=0)  # [H, W]
    print(f"  Global norm stats: min={pf_norms.min():.3f} max={pf_norms.max():.3f} mean={pf_norms.mean():.3f}")
    # Top-5 locations by L2 norm
    flat_norms = pf_norms.flatten()
    topk_vals, topk_idxs = flat_norms.topk(10)
    print(f"  Top-10 locations by L2 norm:")
    for val, idx in zip(topk_vals, topk_idxs):
        fy = idx // W
        fx = idx % W
        px = fx.item() / W * orig_w
        py = fy.item() / H * orig_h
        label = ""
        if abs(px - correct_loc[0]) < 30 and abs(py - correct_loc[1]) < 30:
            label = " <-- CORRECT"
        if abs(px - teleport_loc[0]) < 30 and abs(py - teleport_loc[1]) < 30:
            label = " <-- TELEPORT"
        print(f"    feat({fx.item()},{fy.item()}) pixel({px:.0f},{py:.0f}) norm={val:.3f}{label}")

    # --- 3. No-memory features (backbone + no_mem_embed) ---
    print("\n--- 3. NO-MEMORY FEATURES (backbone + no_mem_embed) ---")
    B = vision_feats[-1].size(1)
    nomem_feat = vision_feats[-1] + model.no_mem_embed
    nomem_feat_4d = nomem_feat.permute(1, 2, 0).view(B, C, H, W)
    nm_correct = nomem_feat_4d[0, :, fy_c, fx_c]
    nm_teleport = nomem_feat_4d[0, :, fy_t, fx_t]
    print(f"  Correct loc:  L2={nm_correct.norm():.3f}  mean={nm_correct.mean():.4f}  std={nm_correct.std():.4f}")
    print(f"  Teleport loc: L2={nm_teleport.norm():.3f}  mean={nm_teleport.mean():.4f}  std={nm_teleport.std():.4f}")

    # --- 4. Dense prompt embeddings ---
    print("\n--- 4. DENSE PROMPT EMBEDDINGS ---")
    # Run prompt through SAM prompt encoder
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
    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=(sam_point_coords, sam_point_labels),
        boxes=None,
        masks=sam_mask_prompt,
    )
    # dense_emb: [1, C, H_pe, W_pe]
    print(f"  Dense embedding shape: {dense_emb.shape}")
    de_correct = dense_emb[0, :, fy_c, fx_c]
    de_teleport = dense_emb[0, :, fy_t, fx_t]
    print(f"  Correct loc:  L2={de_correct.norm():.3f}  mean={de_correct.mean():.4f}  std={de_correct.std():.4f}")
    print(f"  Teleport loc: L2={de_teleport.norm():.3f}  mean={de_teleport.mean():.4f}  std={de_teleport.std():.4f}")
    cosine = F.cosine_similarity(de_correct.unsqueeze(0), de_teleport.unsqueeze(0)).item()
    print(f"  Cosine similarity (correct vs teleport): {cosine:.4f}")

    # Also check what no_mask_embed looks like
    no_mask = model.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
    no_mask_vec = no_mask[0, :, 0, 0]
    print(f"\n  no_mask_embed: L2={no_mask_vec.norm():.3f}  mean={no_mask_vec.mean():.4f}")
    cosine_nm_correct = F.cosine_similarity(no_mask_vec.unsqueeze(0), de_correct.unsqueeze(0)).item()
    cosine_nm_teleport = F.cosine_similarity(no_mask_vec.unsqueeze(0), de_teleport.unsqueeze(0)).item()
    print(f"  Cosine(no_mask_embed, correct_loc_embed): {cosine_nm_correct:.4f}")
    print(f"  Cosine(no_mask_embed, teleport_loc_embed): {cosine_nm_teleport:.4f}")

    # Check prompt mask values at both locations in the resized prompt
    print(f"\n  Prompt mask value at correct loc ({fx_c},{fy_c}): {sam_mask_prompt[0,0,fy_c*4,fx_c*4]:.4f}")
    print(f"  Prompt mask value at teleport loc ({fx_t},{fy_t}): {sam_mask_prompt[0,0,fy_t*4,fx_t*4]:.4f}")

    # --- 5. Fused features (src = pix_feat + dense_emb) ---
    print("\n--- 5. FUSED FEATURES (src = pix_feat + dense_embeddings) ---")
    src = pix_feat + dense_emb
    src_correct = src[0, :, fy_c, fx_c]
    src_teleport = src[0, :, fy_t, fx_t]
    print(f"  Correct loc:  L2={src_correct.norm():.3f}  mean={src_correct.mean():.4f}  std={src_correct.std():.4f}")
    print(f"  Teleport loc: L2={src_teleport.norm():.3f}  mean={src_teleport.mean():.4f}  std={src_teleport.std():.4f}")

    # Compare fused vs no-memory+prompt
    src_nomem = nomem_feat_4d + dense_emb
    snm_correct = src_nomem[0, :, fy_c, fx_c]
    snm_teleport = src_nomem[0, :, fy_t, fx_t]
    print(f"\n  No-memory fused:")
    print(f"  Correct loc:  L2={snm_correct.norm():.3f}")
    print(f"  Teleport loc: L2={snm_teleport.norm():.3f}")

    # --- 6. Delta from memory attention ---
    print("\n--- 6. MEMORY ATTENTION DELTA (pix_feat - raw_backbone) ---")
    raw_4d = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
    delta = pix_feat - raw_4d
    delta_correct = delta[0, :, fy_c, fx_c]
    delta_teleport = delta[0, :, fy_t, fx_t]
    print(f"  Correct loc:  L2={delta_correct.norm():.3f}  mean={delta_correct.mean():.4f}")
    print(f"  Teleport loc: L2={delta_teleport.norm():.3f}  mean={delta_teleport.mean():.4f}")

    # Spatial distribution of memory delta norms
    delta_norms = delta[0].norm(dim=0)
    print(f"  Delta norm stats: min={delta_norms.min():.3f} max={delta_norms.max():.3f} mean={delta_norms.mean():.3f}")
    topk_vals, topk_idxs = delta_norms.flatten().topk(10)
    print(f"  Top-10 memory attention hotspots:")
    for val, idx in zip(topk_vals, topk_idxs):
        fy = idx // W
        fx = idx % W
        px = fx.item() / W * orig_w
        py = fy.item() / H * orig_h
        label = ""
        if abs(px - correct_loc[0]) < 30 and abs(py - correct_loc[1]) < 30:
            label = " <-- CORRECT"
        if abs(px - teleport_loc[0]) < 30 and abs(py - teleport_loc[1]) < 30:
            label = " <-- TELEPORT"
        print(f"    feat({fx.item()},{fy.item()}) pixel({px:.0f},{py:.0f}) delta_norm={val:.3f}{label}")

    # --- 7. Compare frame 12 (last good frame) vs frame 13 ---
    print("\n--- 7. FRAME 12 vs FRAME 13 MEMORY COMPARISON ---")
    frame12_idx = name_to_idx["00012"]
    vision_feats_12, vision_pos_embeds_12, feat_sizes_12 = extract_backbone(
        model, frames, frame12_idx, device
    )
    pix_feat_12 = model._prepare_memory_conditioned_features(
        frame_idx=frame12_idx,
        is_init_cond_frame=True,
        current_vision_feats=vision_feats_12[-1:],
        current_vision_pos_embeds=vision_pos_embeds_12[-1:],
        feat_sizes=feat_sizes_12[-1:],
        output_dict=output_dict,
        num_frames=T,
        track_in_reverse=False,
    )
    pf12_correct = pix_feat_12[0, :, fy_c, fx_c]
    pf12_teleport = pix_feat_12[0, :, fy_t, fx_t]
    print(f"  Frame 12 correct loc:  L2={pf12_correct.norm():.3f}")
    print(f"  Frame 12 teleport loc: L2={pf12_teleport.norm():.3f}")
    print(f"  Frame 13 correct loc:  L2={pf_correct.norm():.3f}")
    print(f"  Frame 13 teleport loc: L2={pf_teleport.norm():.3f}")

    # Memory delta comparison
    raw12_4d = vision_feats_12[-1].permute(1, 2, 0).view(B, C, H, W)
    delta12 = pix_feat_12 - raw12_4d
    d12_correct = delta12[0, :, fy_c, fx_c]
    d12_teleport = delta12[0, :, fy_t, fx_t]
    print(f"\n  Frame 12 memory delta correct:  L2={d12_correct.norm():.3f}")
    print(f"  Frame 12 memory delta teleport: L2={d12_teleport.norm():.3f}")
    print(f"  Frame 13 memory delta correct:  L2={delta_correct.norm():.3f}")
    print(f"  Frame 13 memory delta teleport: L2={delta_teleport.norm():.3f}")

    # --- 8. Memory bank state ---
    print("\n--- 8. MEMORY BANK STATE ---")
    n_cond = len(output_dict["cond_frame_outputs"])
    n_noncond = len(output_dict["non_cond_frame_outputs"])
    print(f"  Conditioning frames in memory: {n_cond}")
    print(f"  Non-conditioning frames: {n_noncond}")
    for fidx in sorted(output_dict["cond_frame_outputs"].keys()):
        out = output_dict["cond_frame_outputs"][fidx]
        osl = out.get("object_score_logits")
        score = osl[0].sigmoid().item() if osl is not None else -1
        mask_area = (out["pred_masks_high_res"][0, 0] > 0).sum().item()
        print(f"    Frame {fidx} (name {frame_names[fidx]}): score={score:.3f} mask_area={mask_area}")


if __name__ == "__main__":
    run_tensor_diagnostic()
