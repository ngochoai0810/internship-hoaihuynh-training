from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_verification_config_uses_project_local_pytest_paths() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    pytest_options = pyproject["tool"]["pytest"]["ini_options"]

    assert pytest_options["cache_dir"] == "my-project/.pytest-cache"
    assert pytest_options["addopts"] == "--basetemp=my-project/.pytest-basetemp"


def test_generated_outputs_are_ignored() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text()

    assert "my-project/.pytest-basetemp/" in gitignore
    assert "my-project/.pytest-cache/" in gitignore
    assert "my-project/src/*.egg-info/" in gitignore


def test_requests_type_stubs_are_declared() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text()

    assert "types-requests==" in requirements
