"""Core SAM2 dense-prompt inference engine for contrail tracking.

Adapted from the SAM2 multi-object VOS inference pattern.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def _extract_backbone_features(model, video_frames, frame_idx, device):
    """Extract backbone features for a single frame on-the-fly.

    Moves only one frame to GPU, runs the image encoder, and returns
    the prepared vision features. This avoids caching all frames' features
    in GPU/CPU memory simultaneously.
    """
    img = video_frames[frame_idx : frame_idx + 1].to(device)  # [1, 3, H, W]

    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.amp.autocast(
        autocast_device, dtype=torch.float16, enabled=(autocast_device == "cuda")
    ):
        backbone_out = model.image_encoder(img)

    for key in backbone_out:
        if isinstance(backbone_out[key], torch.Tensor):
            backbone_out[key] = backbone_out[key].float()
        elif isinstance(backbone_out[key], list):
            backbone_out[key] = [
                t.float() if isinstance(t, torch.Tensor) else t
                for t in backbone_out[key]
            ]

    if model.use_high_res_features_in_sam:
        backbone_out["backbone_fpn"][0] = model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

    _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(
        backbone_out
    )

    return {
        "vision_feats": vision_feats,
        "vision_pos_embeds": vision_pos_embeds,
        "feat_sizes": feat_sizes,
    }


@torch.inference_mode()
def dense_prompt_inference_multi_object(
    model,
    video_frames,       # [T, 3, H, W] tensor (on CPU)
    prompts_per_obj,    # {obj_id: {frame_idx: numpy_mask}}
    original_height,
    original_width,
    device="cuda",
    score_threshold=0.0,
    max_propagation_frames=0,
):
    """Inference with per-object memory banks and memory-only propagation.

    Each object is processed on its prompted frames (conditioning) and then
    continues to propagate from memory alone (non-conditioning) after its last
    prompt. Propagation stops when either:
      - The object score drops below ``score_threshold``, or
      - ``max_propagation_frames`` frames have been processed beyond the last prompt, or
      - The video ends.

    When score_threshold <= 0 and max_propagation_frames <= 0, this behaves
    identically to prompt-only mode (no propagation beyond prompts).

    Backbone features are computed once per frame and shared across all objects
    active on that frame. Each object maintains its own output_dict (memory bank).

    Args:
        model: SAM2Base model.
        video_frames: Float32 tensor [T, 3, H, W] on CPU.
        prompts_per_obj: {obj_id: {frame_idx: np.ndarray}} prompt masks.
        original_height: Output mask height.
        original_width: Output mask width.
        device: "cuda" or "cpu".
        score_threshold: Stop propagation when object score drops below this (0 = disabled).
        max_propagation_frames: Max frames to propagate after last prompt (0 = disabled).

    Returns:
        {frame_idx: {obj_id: {"mask": np.ndarray bool [H, W], "score": float}}}
    """
    import numpy as np

    model.eval()
    T = len(video_frames)

    propagation_enabled = score_threshold > 0 or max_propagation_frames > 0

    prompt_frames_per_obj = {
        obj_id: sorted(obj_prompts.keys())
        for obj_id, obj_prompts in prompts_per_obj.items()
    }

    all_prompt_frames = set()
    for obj_prompts in prompts_per_obj.values():
        all_prompt_frames.update(obj_prompts.keys())

    if not all_prompt_frames:
        return {}

    first_frame = min(all_prompt_frames)
    last_frame = T - 1 if propagation_enabled else max(all_prompt_frames)

    output_dicts = {
        obj_id: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        for obj_id in prompts_per_obj.keys()
    }

    obj_started = {obj_id: False for obj_id in prompts_per_obj}
    obj_retired = {obj_id: False for obj_id in prompts_per_obj}
    obj_last_prompt_frame = {
        obj_id: max(frames) for obj_id, frames in prompt_frames_per_obj.items()
    }
    obj_propagation_count = {obj_id: 0 for obj_id in prompts_per_obj}

    predictions = defaultdict(dict)

    total_prompts = sum(len(p) for p in prompts_per_obj.values())
    log.info(
        "inference_start frames=%s-%s objects=%d prompts=%d propagation=%s",
        first_frame, last_frame, len(prompts_per_obj), total_prompts, propagation_enabled,
    )

    feats_cache = None
    feats_frame_idx = -1

    frames_to_visit = (
        range(first_frame, last_frame + 1)
        if propagation_enabled
        else sorted(all_prompt_frames)
    )

    for frame_idx in tqdm(frames_to_visit, desc="Tracking frames"):
        active_objects = []

        for obj_id in prompts_per_obj:
            if obj_retired[obj_id]:
                continue

            has_prompt = frame_idx in prompts_per_obj[obj_id]

            if has_prompt:
                obj_started[obj_id] = True
                active_objects.append((obj_id, True))
            elif propagation_enabled and obj_started[obj_id]:
                if frame_idx > obj_last_prompt_frame[obj_id]:
                    active_objects.append((obj_id, False))

        if not active_objects:
            continue

        if feats_frame_idx != frame_idx:
            feats_cache = _extract_backbone_features(
                model, video_frames, frame_idx, device
            )
            feats_frame_idx = frame_idx

        for obj_id, has_prompt in active_objects:
            if has_prompt:
                prompt_mask = prompts_per_obj[obj_id][frame_idx]
                mask_tensor = torch.from_numpy(prompt_mask).float().to(device)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

                if mask_tensor.shape[-2:] != (model.image_size, model.image_size):
                    mask_tensor = torch.nn.functional.interpolate(
                        mask_tensor,
                        size=(model.image_size, model.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                # Negative-only prompt: all values <= 0 (no positive region for this object).
                # Treated as a suppression signal — allow score-based retirement.
                is_negative_only = mask_tensor.max().item() <= 0.0

                current_out = model.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=True,
                    current_vision_feats=feats_cache["vision_feats"],
                    current_vision_pos_embeds=feats_cache["vision_pos_embeds"],
                    feat_sizes=feats_cache["feat_sizes"],
                    point_inputs=None,
                    mask_inputs=mask_tensor,
                    output_dict=output_dicts[obj_id],
                    num_frames=T,
                    track_in_reverse=False,
                    run_mem_encoder=True,
                )

                output_dicts[obj_id]["cond_frame_outputs"][frame_idx] = current_out
                if not is_negative_only:
                    obj_propagation_count[obj_id] = 0

            else:
                is_negative_only = False
                current_out = model.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=False,
                    current_vision_feats=feats_cache["vision_feats"],
                    current_vision_pos_embeds=feats_cache["vision_pos_embeds"],
                    feat_sizes=feats_cache["feat_sizes"],
                    point_inputs=None,
                    mask_inputs=None,
                    output_dict=output_dicts[obj_id],
                    num_frames=T,
                    track_in_reverse=False,
                    run_mem_encoder=True,
                )

                obj_propagation_count[obj_id] += 1
                output_dicts[obj_id]["non_cond_frame_outputs"][frame_idx] = current_out

            pred_mask = current_out["pred_masks_high_res"]
            if pred_mask.shape[-2:] != (original_height, original_width):
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask,
                    size=(original_height, original_width),
                    mode="bilinear",
                    align_corners=False,
                )

            mask = (pred_mask[0, 0] > 0.0).cpu().numpy()
            obj_score_prob = current_out["object_score_logits"][0].sigmoid().item()

            predictions[frame_idx][obj_id] = {"mask": mask, "score": obj_score_prob}

            if propagation_enabled:
                should_retire = False
                # Retire on score: fires on propagation frames and negative-only
                # conditioning frames. Never fires on positive conditioning frames
                # (object is explicitly asserted present).
                score_retirement_eligible = not has_prompt or is_negative_only
                if score_threshold > 0 and score_retirement_eligible and obj_score_prob < score_threshold:
                    should_retire = True
                if (
                    not has_prompt
                    and max_propagation_frames > 0
                    and obj_propagation_count[obj_id] >= max_propagation_frames
                ):
                    should_retire = True

                if should_retire:
                    obj_retired[obj_id] = True
                    del output_dicts[obj_id]

        if propagation_enabled and all(obj_retired.values()):
            log.info("all_objects_retired frame=%d", frame_idx)
            break

    if propagation_enabled:
        for obj_id in prompts_per_obj:
            n_prop = obj_propagation_count[obj_id]
            if n_prop > 0:
                status = "retired" if obj_retired[obj_id] else "end-of-video"
                log.debug("propagation_stats object=%s frames=%d status=%s", obj_id, n_prop, status)

    return predictions
