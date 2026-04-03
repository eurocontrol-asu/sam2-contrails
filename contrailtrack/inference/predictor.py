"""Run dense-prompt SAM2 inference on a contrail video."""

from collections import defaultdict

import torch

from contrailtrack.inference._engine import dense_prompt_inference_multi_object


def run_video(
    model,
    frames,
    frame_names: list,
    prompts: dict,
    original_height: int = None,
    original_width: int = None,
    score_threshold: float = 0.5,
    max_propagation_frames: int = 0,
    object_batch_size: int = None,
    device: str = "cuda",
) -> dict:
    """Run dense-prompt inference on a video.

    Args:
        model: SAM2Base model returned by load_model().
        frames: float32 tensor [T, 3, H, H] from load_frames().
        frame_names: list of frame name strings (no extension), same order as frames.
        prompts: {obj_id: {frame_name: np.ndarray}} from read_prompts().
        original_height: Original video height in pixels (from load_frames(); used to
                         resize output masks back to native resolution).
        original_width: Original video width in pixels (from load_frames()).
        score_threshold: Object score below which propagation stops (0 = disabled).
        max_propagation_frames: Max frames to propagate after last prompt (0 = disabled).
        object_batch_size: Process objects in batches to save GPU memory (None = all at once).
        device: "cuda" or "cpu".

    Returns:
        {frame_idx (int): {obj_id (int): {"mask": np.ndarray bool [H,W], "score": float}}}
        Masks have shape (original_height, original_width).
    """
    # Fall back to tensor size if original dims not provided
    if original_height is None or original_width is None:
        _, _, h, w = frames.shape
        original_height = original_height or h
        original_width = original_width or w

    # Build {obj_id: {frame_idx: mask}} required by dense_prompt_inference_multi_object
    name_to_idx = {name: idx for idx, name in enumerate(frame_names)}
    prompts_indexed = {}
    for obj_id, obj_prompts in prompts.items():
        indexed = {}
        for frame_name, mask in obj_prompts.items():
            if frame_name in name_to_idx:
                indexed[name_to_idx[frame_name]] = mask
        if indexed:
            prompts_indexed[obj_id] = indexed

    if not prompts_indexed:
        return {}

    if object_batch_size is None:
        return dict(
            dense_prompt_inference_multi_object(
                model=model,
                video_frames=frames,
                prompts_per_obj=prompts_indexed,
                original_height=original_height,
                original_width=original_width,
                device=device,
                score_threshold=score_threshold,
                max_propagation_frames=max_propagation_frames,
            )
        )

    # Batched processing — reduces peak GPU memory usage
    obj_ids = list(prompts_indexed.keys())
    all_predictions = defaultdict(dict)
    for batch_start in range(0, len(obj_ids), object_batch_size):
        batch = obj_ids[batch_start: batch_start + object_batch_size]
        batch_preds = dense_prompt_inference_multi_object(
            model=model,
            video_frames=frames,
            prompts_per_obj={oid: prompts_indexed[oid] for oid in batch},
            original_height=original_height,
            original_width=original_width,
            device=device,
            score_threshold=score_threshold,
            max_propagation_frames=max_propagation_frames,
        )
        for frame_idx, frame_preds in batch_preds.items():
            all_predictions[frame_idx].update(frame_preds)
        if device == "cuda":
            torch.cuda.empty_cache()

    return dict(all_predictions)
