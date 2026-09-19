import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fly_policy.graph import FlyGraph
from fly_policy.policy import ACTIONS, FlyPolicy
from fly_policy.synth_env import OdorTaxisEnv, collect_expert_jsonl
from sense.frame import frame_to_vector, vector_size
from sense.goal_to_sense import load_goal
from tools.build_mini_graph import build
from train_il import accuracy, rows_to_tensors, train


def test_il_beats_random():
    goal = load_goal(ROOT / "configs/goals/collect_oak.json")
    rows = collect_expert_jsonl(goal, n_episodes=24, seed=2)
    x, y = rows_to_tensors(rows)
    g = FlyGraph(build())
    model, _ = train(x, y, g, epochs=30, lr=2e-3, batch=64)
    acc = accuracy(model, x, y)
    rand = 1.0 / len(ACTIONS)
    assert acc > rand + 0.15, f"IL acc {acc:.3f} not above random {rand:.3f}"
    mine = ACTIONS.index("mine")
    mask = y == mine
    assert mask.any()
    model.eval()
    with torch.no_grad():
        pred = model(x)[0].argmax(-1)
    mine_recall = float((pred[mask] == mine).float().mean())
    assert mine_recall > 0.3, f"mine recall {mine_recall:.3f} too low"
