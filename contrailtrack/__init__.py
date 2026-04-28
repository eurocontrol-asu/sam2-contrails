"""contrailtrack — SAM2-based contrail detection, tracking, and attribution."""

__version__ = "0.1.0"

from contrailtrack.model.loader import load_model, list_configs
from contrailtrack.data.video import load_frames
from contrailtrack.data.prompt_reader import read_prompts, list_objects
from contrailtrack.inference.predictor import run_video
from contrailtrack.output.coco import encode_rle, export_coco_json, export_prompts_coco_json
from contrailtrack.eval.metrics import evaluate, evaluate_segmentation, evaluate_tracking, evaluate_attribution  # noqa: F401
from contrailtrack.prompts.projection import MiniProjector, project_cocip_to_pixels
from contrailtrack.prompts.writer import generate_prompts, generate_prompts_video

__all__ = [
    "load_model",
    "list_configs",
    "load_frames",
    "read_prompts",
    "list_objects",
    "run_video",
    "encode_rle",
    "export_coco_json",
    "export_prompts_coco_json",
    "evaluate",
    "evaluate_segmentation",
    "evaluate_tracking",
    "evaluate_attribution",
    "MiniProjector",
    "project_cocip_to_pixels",
    "generate_prompts",
    "generate_prompts_video",
]
