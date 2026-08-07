"""FastAPI application package."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
repository_root_path = str(REPOSITORY_ROOT)

if repository_root_path not in sys.path:
    sys.path.insert(0, repository_root_path)
