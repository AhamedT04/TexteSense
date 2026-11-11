"""Métadonnées d'exécution : ce qu'il faut pour reproduire un résultat."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from app.config import BASE_DIR

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "scikit-learn",
    "numpy",
    "pandas",
    "onnxruntime",
    "accelerate",
)


def git_commit() -> str | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=BASE_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def run_metadata(**extra: Any) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "packages": package_versions(),
        **extra,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
