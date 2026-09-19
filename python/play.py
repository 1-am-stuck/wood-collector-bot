#!/usr/bin/env python3
"""Open-world FruitFly: load a checkpoint, walk Minecraft with no GoalSpec, livestream."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from dashboard.hub import publish
from fly_policy.graph import FlyGraph
from fly_policy.mc_env import MinecraftEnv
from fly_policy.policy import FlyPolicy, default_graph_path
from load_env import load_repo_env
from tools.build_mini_graph import build


def latest_ckpt(root: Path) -> Path:
    for name in ("fly_mc_rarest.pt", "fly_mc_ppo.pt", "fly_mc.pt"):
        path = root / "checkpoints" / name
        if path.exists():
            return path
    raise FileNotFoundError("no fly checkpoint in checkpoints/")


def minecraft_up(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_dashboard(host: str, port: int):
    import threading

    def _run():
        import uvicorn
        from dashboard.app import app
        uvicorn.run(app, host=host, port=port, log_level="warning")

    threading.Thread(target=_run, daemon=True, name="fly-dashboard").start()
    print(f"dashboard http://{host}:{port}/", flush=True)


def emit(step: int, brain: dict, pkt: dict, ckpt: Path):
    publish({
        "t": time.time(),
        "mode": "play",
        "ckpt": str(ckpt),
        "step": step,
        "action": brain.get("action"),
        "value": brain.get("value"),
        "reward": float(pkt.get("reward") or 0.0),
        "frame": pkt.get("frame") or {},
        "facts": pkt.get("facts") or {},
        "eye": pkt.get("eye"),
        "fly": pkt.get("fly"),
        "brain": brain,
    })


def main():
    load_repo_env(ROOT)
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="", help="checkpoint path; default = newest trained model")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=25565)
    p.add_argument("--username", default="FruitFlyPlay")
    p.add_argument("--dashboard-port", type=int, default=8767)
    p.add_argument("--dt-ms", type=int, default=1000)
    p.add_argument("--eye-w", type=int, default=160)
    p.add_argument("--eye-h", type=int, default=90)
    p.add_argument("--greedy", action="store_true", default=True)
    args = p.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(ROOT)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    if not minecraft_up(args.host, args.port):
        print(
            f"Minecraft is not running at {args.host}:{args.port}.\n"
            "Start Paper (./start-server.sh) then rerun play.py",
            file=sys.stderr,
        )
        sys.exit(2)

    gpath = default_graph_path(ROOT)
    if not gpath.exists():
        build()
    model = FlyPolicy.load(FlyGraph(gpath), ckpt)
    model.eval()
    print(json.dumps({
        "ckpt": str(ckpt),
        "n": model.n,
        "n_edges": model.graph.n_edges,
        "username": args.username,
        "goal": None,
    }, indent=2), flush=True)

    start_dashboard("127.0.0.1", args.dashboard_port)
    time.sleep(0.5)

    cfg = {
        "minecraft": {"host": args.host, "port": args.port, "username": args.username},
        "goal": {"catalog": {}, "explore_radius": 32},
        "view": {"eye": {"width": args.eye_w, "height": args.eye_h, "maxDist": 24}},
    }
    env = MinecraftEnv(cfg)
    hello = env.start()
    print(f"joined minecraft as {hello.get('username')} (no GoalSpec)", flush=True)
    pkt = env.play_reset()
    step = 0
    dt = max(0.05, args.dt_ms / 1000.0)
    try:
        while True:
            t0 = time.monotonic()
            obs = model.frames_to_obs(pkt.get("frame") or {})
            with torch.no_grad():
                brain = model.inspect(obs, greedy=args.greedy)
            emit(step, brain, pkt, ckpt)
            pkt = env.step(brain["action"])
            step += 1
            leftover = dt - (time.monotonic() - t0)
            if leftover > 0:
                time.sleep(leftover)
    except KeyboardInterrupt:
        print("play stopped", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
