"""Load repo-root .env into os.environ. Secrets stay out of git and the browser."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_repo_env(root: Path | None = None) -> Path | None:
    path = (root or ROOT) / ".env"
    if not path.is_file():
        return None
    load_dotenv(path, override=False)
    return path


def wandb_api_key(root: Path | None = None) -> str | None:
    load_repo_env(root)
    key = (os.environ.get("WANDB_API_KEY") or "").strip()
    return key or None
