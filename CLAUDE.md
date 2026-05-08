# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All development uses the `sam2` conda environment:
```bash
/data/dataiku/.conda/envs/sam2/bin/python
```

The repo is also installable via `uv` (see README). For Encord upload tasks, use the `mask2former` env instead (it has `pycocotools` + `encord`).

## Commands

**Run tests (from repo root):**
```bash
python -m unittest tests.<module> -v
# e.g.
python -m unittest tests.test_ternary_pipeline -v
python -m unittest tests.test_transforms -v
```

**Training:**
```bash
python training/train.py \
  -c sam2/configs/sam2.1_training/sam2.1_hiera_b+_GVCCS_finetune_ternary.yaml \
  --use-cluster 0 --num-gpus 2
# or via CLI:
uv run contrailtrack train sam2/configs/sam2.1_training/<config>.yaml --num-gpus 2
```

**Inference:**
```bash
uv run contrailtrack run \
  --images data/frames/00001 --prompts data/prompts/ --out results/00001.json

# Batch:
uv run contrailtrack run \
  --images-dir data/frames/ --prompts data/prompts/ --out results/

# Subset by video ID:
uv run contrailtrack run --images-dir data/frames/ --prompts data/prompts/ --out results/ \
  --videos 1 --videos 3
```

**Evaluation:**
```bash
uv run contrailtrack evaluate results/00001.json --labels annotations.json --out eval/

# Dataset-wide (single pass, not per-video average):
uv run contrailtrack evaluate-dataset \
  --predictions-dir results/ --labels annotations.json --out evaluation/

# GVCCS test set (with correct GT filtering and flight_id mapping):
uv run contrailtrack evaluate-dataset \
  --predictions-dir outputs/ternary_5_rerun_remapped \
  --labels /data/common/CAMERA/datasets/GVCCS/test/annotations.json \
  --out evaluation/ternary_5_rerun_final \
  --flight-mappings /data/common/TRAILVISION/GVCCS_V/test/per_object_data_age_5
```

**Labelling campaign upload (Encord):**
```bash
PYTHONPATH=/data/common/dataiku/lib/python \
  /data/dataiku/.conda/envs/mask2former/bin/python \
  /data/common/dataiku/lib/python/utils/encord/run_sam2_upload.py \
  2025-01-02 SAM2 user@eurocontrol.int
```

See `docs/labelling_campaign.md` for the full campaign workflow.

---

## Architecture Overview

This repo forks Meta's SAM2 and adds a contrail-specific tracking pipeline on top.

### What's upstream (Meta's SAM2, `sam2/`)
- Image encoder (Hiera backbone + FPN neck)
- Memory attention (cross-attention to memory bank)
- SAM prompt encoder + mask decoder
- Memory encoder (CXBlock fuser)
- Training scaffolding (`training/train.py`, `training/trainer.py`)

### What's custom (`contrailtrack/`, `training/`)

**`contrailtrack/`** — the public-facing package:
- `model/loader.py` — wraps `build_sam.py`, auto-downloads from HF Hub
- `data/video.py` — frame loading (no ImageNet norm — see Critical Gotchas)
- `data/prompt_reader.py` — reads prompt PNGs, supports `binary`/`age_weighted`/`ternary` encodings
- `inference/predictor.py` — per-object independent memory banks, object batching
- `output/coco.py` — RLE export for predictions and prompts
- `eval/metrics.py` — segmentation mAP, tracking (completeness, T-IoU), attribution precision
- `prompts/` — CoCiP/DryAdvection contrail models, camera projection, prompt PNG writer
- `campaign/pipeline.py` — orchestrates fleet→DryAdvection→prompts→inference for unannotated days
- `cli.py` — Typer CLI entry point (`contrailtrack` command)

**`training/`** — custom training extensions:
- `model/sam2.py::SAM2DensePromptTrain` — processes all frames sequentially (vs Meta's click-frame-first approach); critical for dense multi-frame prompts
- `dataset/vos_sampler.py::LifecycleSampler` — single-object training exposing full contrail lifecycle (prompt frames → GT-only frames → no-object frames)
- `dataset/vos_segment_loader.py::MultiplePNGSegmentLoader` — segments as `[H,W,2]` (ch0=GT mask, ch1=prompt); supports GT-without-prompt frames

### Three Model Config Variants

| Variant | `use_prompt_in_obj_ptr` | `use_prompt_gate` | Notes |
|---------|------------------------|-------------------|-------|
| `original` | false | false | Binary prompts, 1-min window; baseline |
| `prompt_in_memory` | true | false | Prompt spatial info added to object pointer |
| `new` / `ternary` | true | true | + MLP gate on prompt contribution; best results |

Config file pairs (training ↔ inference must stay in sync):
- `sam2/configs/sam2.1_training/sam2.1_hiera_b+_GVCCS_finetune_<variant>.yaml`
- `sam2/configs/sam2.1/sam2.1_hiera_b+_GVCCS_<variant>.yaml`

### Data Layout

**GVCCS test dataset:**
```
/data/common/CAMERA/datasets/GVCCS/test/
  img_folder/{video_id}/{frame}.jpg
  annotations.json                         # COCO-Video format

/data/common/TRAILVISION/GVCCS_V/test/
  per_object_data/{video_id}/{object_id}/{frame}_prompt.png   # binary 1-min
  per_object_data/{video_id}/{object_id}/{frame}_mask.png
  per_object_data/{video_id}/flight_mapping.json              # object_folder → flight_id
  per_object_data_age_5/   # age-weighted 5-min (each video also has flight_mapping.json)
  per_object_data_age_10/  # age-weighted 10-min (same)
```

**IMPORTANT**: Always use the `flight_mapping.json` from the **same prompts directory** used
for inference. `inputs/mappings/` contains a different (shifted) mapping and must NOT be used
for GVCCS test-set evaluation.

**Predictions output (COCO RLE JSON, one per video):**
```
outputs/{variant}/{video_id}.json
```

**Labelling campaign data:**
```
/data/contrailnet/sam2/
  frames/{video_id}/        # symlinks to camera images
  prompts/{video_id}.json   # COCO RLE of prompts
  predictions/{video_id}.json
```

---

## Critical Gotchas

### No ImageNet normalization — training AND inference must match
Training deliberately omits `NormalizeAPI`. Images enter the model as `[0,1]` tensors.  
`load_video_frames()` applies ImageNet norm by default — **always override**:
```python
load_video_frames(..., img_mean=(0,0,0), img_std=(1,1,1))
```
`contrailtrack.data.video.load_frames()` already does this correctly.

### Training ↔ inference config sync
`num_maskmem` controls an `nn.Parameter` shape — a mismatch **crashes** `load_state_dict`.  
`max_obj_ptrs_in_encoder` controls temporal PE normalization — a mismatch gives wrong results silently.  
Always edit the training/inference config pair together.

### Segments are `[H,W,2]`
Channel 0 = GT mask, channel 1 = prompt. Any spatial transform must permute to `[2,H,W]` first, apply, then permute back. `RandomAffine` was fixed for this; `RandomMosaic` is incompatible (hard assert on `[H,W]` shape).

### ColorJitter requires explicit `hue`
The custom `ColorJitter` class has no default for `hue`. Always set `hue: null` in YAML configs.

### Ternary prompt: negative channel is the union
`ternary = where(pos > 0, pos, -union)` where `union = per_object_data_age_5/{video}/{frame}_all_prompts_union.png`. See `vos_segment_loader.py` for the exact loading logic.

### `evaluate-dataset` vs `evaluate`
`evaluate-dataset` pools all predictions across all videos and evaluates in one pass (true dataset-wide metrics). `evaluate` runs per-video and would need manual aggregation. Always use `evaluate-dataset` for reporting numbers.
