#!/usr/bin/env python3
"""JSONL TCP server: SensoryFrame in, action out. Loads a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from load_env import load_repo_env
from fly_policy.graph import FlyGraph
from fly_policy.policy import ACTIONS, FlyPolicy, default_graph_path
from tools.build_mini_graph import build


def load_policy(ckpt: Path | None):
    gpath = default_graph_path(ROOT)
    if not gpath.exists():
        build()
    graph = FlyGraph(gpath)
    if ckpt and ckpt.exists():
        model = FlyPolicy.load(graph, ckpt)
    else:
        model = FlyPolicy(graph, n_obs=0)
        # rebuild with correct obs size from first request — placeholder
        from sense.frame import vector_size
        model = FlyPolicy(graph, vector_size(64))
    model.eval()
    return model


def handle_line(model, obj):
    typ = obj.get("type") or "act"
    if typ == "ping":
        return {"ok": True, "actions": ACTIONS}
    frame = obj.get("frame") or obj
    obs = model.frames_to_obs(frame)
    with torch.no_grad():
        action, _, value, logits = model.act(obs, greedy=bool(obj.get("greedy", True)))
    idx = int(action.item()) if action.ndim == 0 else int(action[0].item())
    return {
        "action": ACTIONS[idx],
        "index": idx,
        "value": float(value.reshape(-1)[0].item()),
        "logits": [float(x) for x in logits.reshape(-1).tolist()],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "fly_mc.pt"))
    args = p.parse_args()
    model = load_policy(Path(args.ckpt) if args.ckpt else None)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    print(f"infer_server listening on {args.host}:{args.port}", flush=True)
    while True:
        conn, _ = srv.accept()
        buf = ""
        with conn:
            conn_file = conn.makefile("rw")
            for line in conn_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    resp = handle_line(model, obj)
                except Exception as exc:
                    resp = {"error": str(exc)}
                conn_file.write(json.dumps(resp) + "\n")
                conn_file.flush()


if __name__ == "__main__":
    load_repo_env()
    main()
