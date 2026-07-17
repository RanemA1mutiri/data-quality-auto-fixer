"""Pytest root marker — ensures the repo root (and thus `src/`) is importable
when running `pytest` from any environment (local or CI)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
