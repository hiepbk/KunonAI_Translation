"""Utility functions for path management"""
from pathlib import Path

# Get project root (where setup.py is located)
_PROJECT_ROOT = Path(__file__).parent

# Weights folder - automatically relative to project root
WEIGHTS_DIR = _PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

