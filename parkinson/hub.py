"""Skore Hub integration helpers for Bobathon ESILV."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SKORE_FILE = REPO_ROOT / ".skore"


def load_skore_credentials(repo_root: Path | None = None) -> dict[str, Any]:
    """Load configuration from ``.skore`` and configure environment variables.

    Sets ``SKORE_HUB_API_KEY`` so that ``skore.login(mode='hub')`` uses the
    saved workspace API key rather than opening an interactive browser flow.
    """
    path = (repo_root or REPO_ROOT) / ".skore"
    if not path.is_file():
        raise FileNotFoundError(f".skore file not found at {path}")

    config = json.loads(path.read_text())
    if "api_key" in config:
        os.environ["SKORE_HUB_API_KEY"] = config["api_key"]
    if "hub_url" in config:
        os.environ["SKORE_HUB_URI"] = config["hub_url"]

    return config
