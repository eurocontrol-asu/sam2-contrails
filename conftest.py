"""pytest configuration — repo root on sys.path."""
import sys
from pathlib import Path

# contrailtrack is not pip-installed; add repo root so imports work
sys.path.insert(0, str(Path(__file__).parent))
