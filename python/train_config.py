"""Load a training recipe YAML. GoalToSense is not involved here."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_train_yaml(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "run" not in raw or "train" not in raw:
        raise ValueError(f"{path} must have top-level 'run' and 'train'")
    raw["_path"] = str(path.resolve())
    raw["_root"] = str(ROOT)
    for key in ("ckpt", "out"):
        if raw["run"].get(key):
            p = Path(raw["run"][key])
            raw["run"][key] = str(p if p.is_absolute() else ROOT / p)
    catalog = (raw.get("goal") or {}).get("catalog")
    if catalog:
        cp = Path(catalog)
        raw["goal"]["catalog"] = str(cp if cp.is_absolute() else ROOT / cp)
    return raw


def load_goal_catalog(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping of log_name → GoalSpec")
    return data


def spec_for_log(catalog: dict, log_name: str) -> dict | None:
    return catalog.get(log_name)
