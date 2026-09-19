#!/usr/bin/env python3
"""Train from a YAML recipe. Requires a live Minecraft server when minecraft.required is true."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_config import load_goal_catalog, load_train_yaml, spec_for_log
from sense.rarity import rarest_log


def minecraft_up(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", default=str(ROOT / "configs/train/rarest_minecraft.yaml"))
    args = p.parse_args()
    cfg = load_train_yaml(args.config)
    catalog = load_goal_catalog(cfg["goal"]["catalog"])
    mc = cfg["minecraft"]
    print(json.dumps({
        "run": cfg["run"]["name"],
        "ckpt": cfg["run"]["ckpt"],
        "selector": cfg["goal"]["selector"],
        "catalog_logs": list(catalog),
        "example_rarest": spec_for_log(catalog, rarest_log({"oak_log": 9, "cherry_log": 1}))["id"],
        "minecraft": f"{mc['host']}:{mc['port']}",
    }, indent=2))

    if mc.get("required", True) and not minecraft_up(mc["host"], mc["port"]):
        print(
            f"Minecraft is not running at {mc['host']}:{mc['port']}.\n"
            "Start Paper (./start-server.sh) then:\n"
            "  node fly.js\n"
            f"  uv run python python/train.py {args.config}",
            file=sys.stderr,
        )
        sys.exit(2)

    os.environ.setdefault("WANDB_BASE_URL", (cfg.get("wandb") or {}).get("base_url") or "http://127.0.0.1:8080")
    try:
        import wandb
        wandb.init(
            project=(cfg.get("wandb") or {}).get("project") or "fly-mc",
            name=cfg["run"]["name"],
            config=cfg,
            mode=(cfg.get("wandb") or {}).get("mode") or "online",
            tags=(cfg.get("wandb") or {}).get("tags") or [],
        )
    except Exception as exc:
        print("wandb init failed, continuing without:", exc)

    print("minecraft is up — spawn the fly with: node fly.js")
    print("PPO-in-minecraft worker is the next loop; this run verified YAML + live server.")
    if "wandb" in sys.modules:
        import wandb
        wandb.log({"minecraft_up": 1})
        wandb.finish()


if __name__ == "__main__":
    main()
