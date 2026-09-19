#!/usr/bin/env python3
"""Behavioral cloning on encoder/gains/decoder. JSONL demos or synthetic expert."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from fly_policy.export import export_safetensors
from fly_policy.graph import FlyGraph
from fly_policy.policy import ACTIONS, FlyPolicy, default_graph_path
from fly_policy.synth_env import collect_expert_jsonl
from sense.frame import frame_to_vector, vector_size
from sense.goal_to_sense import load_goal
from tools.build_mini_graph import build


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rows_to_tensors(rows):
    xs, ys = [], []
    for row in rows:
        frame = row.get("frame") or row.get("sensory")
        act = row.get("action")
        if frame is None or act is None:
            continue
        idx = ACTIONS.index(act) if isinstance(act, str) else int(act)
        xs.append(frame_to_vector(frame))
        ys.append(idx)
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def train(x, y, graph, epochs=20, lr=1e-3, batch=64):
    model = FlyPolicy(graph, n_obs=x.shape[1], n_actions=len(ACTIONS))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    counts = torch.bincount(y, minlength=len(ACTIONS)).float().clamp(min=1)
    weight = (counts.sum() / (len(ACTIONS) * counts))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=min(batch, len(ds)), shuffle=True)
    last = 0.0
    for ep in range(epochs):
        total = 0.0
        n = 0
        for xb, yb in loader:
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        last = total / max(1, n)
        print(f"il epoch {ep} loss={last:.4f}")
    return model, last


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        pred = logits.argmax(-1)
        return float((pred == y).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default="")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out", default=str(ROOT / "checkpoints" / "fly_mc.pt"))
    args = p.parse_args()
    gpath = default_graph_path(ROOT)
    if not gpath.exists():
        build()
    graph = FlyGraph(gpath)
    if args.demos:
        rows = load_jsonl(Path(args.demos))
    else:
        goal = load_goal(ROOT / "configs" / "goals" / "collect_oak.json")
        rows = collect_expert_jsonl(goal, n_episodes=40, seed=1)
        demo_path = ROOT / "logs" / "expert_synth.jsonl"
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        with demo_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {demo_path} n={len(rows)}")
    x, y = rows_to_tensors(rows)
    assert x.shape[1] == vector_size(64)
    model, _ = train(x, y, graph, epochs=args.epochs)
    acc = accuracy(model, x, y)
    print(f"il train accuracy={acc:.3f}")
    out = Path(args.out)
    model.save_checkpoint(out, extra={"train_accuracy": acc})
    export_safetensors(model, out.with_suffix(".safetensors"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
