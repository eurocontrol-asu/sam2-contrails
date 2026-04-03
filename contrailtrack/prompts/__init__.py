"""contrailtrack.prompts — contrail model and prompt generation utilities."""

from contrailtrack.prompts.cocip import run_cocip, run_cocip_batch, load_fleet_json
from contrailtrack.prompts.dry_advection import run_dry_advection, run_dry_advection_batch
from contrailtrack.prompts.projection import MiniProjector, project_cocip_to_pixels
from contrailtrack.prompts.writer import generate_prompts, generate_prompts_video

__all__ = [
    # CoCiP (full thermodynamic model)
    "run_cocip",
    "run_cocip_batch",
    "load_fleet_json",
    # DryAdvection (wind-field advection only, no radiation)
    "run_dry_advection",
    "run_dry_advection_batch",
    # Camera projection
    "MiniProjector",
    "project_cocip_to_pixels",
    # Prompt mask generation
    "generate_prompts",
    "generate_prompts_video",
]
