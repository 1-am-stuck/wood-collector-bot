#!/usr/bin/env python3
"""Compare random / expert / trained policy on a GoalSpec."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from load_env import load_repo_env
from fly_policy.graph import FlyGraph
from fly_policy.policy import ACTIONS, FlyPolicy, default_graph_path
from fly_policy.synth_env import OdorTaxisEnv
from sense.frame import frame_to_vector
from sense.goal_to_sense import load_goal
from tools.build_mini_graph import build


def run_policy(env, choose, episodes=20):
    wins = 0
    steps_ok = []
    for i in range(episodes):
        env.rng.seed(1000 + i)
        frame, facts, _ = env.reset()
        done = False
        while not done:
            act = choose(frame, facts)
            frame, _, done, facts = env.step(act)
        if env.have:
            wins += 1
            steps_ok.append(env.steps)
    return {
        "success_rate": wins / episodes,
        "mean_steps_success": (sum(steps_ok) / len(steps_ok)) if steps_ok else None,
        "episodes": episodes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "fly_mc.pt"))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default=str(ROOT / "logs" / "eval.json"))
    args = p.parse_args()
    goal = load_goal(ROOT / "configs" / "goals" / "collect_oak.json")
    env = OdorTaxisEnv(goal, seed=0)

    def random_choose(frame, facts):
        return random.choice(ACTIONS)

    def expert_choose(frame, facts):
        return env.expert_action(facts)

    report = {
        "goal": goal["id"],
        "random": run_policy(OdorTaxisEnv(goal, seed=0), random_choose, args.episodes),
        "expert": run_policy(OdorTaxisEnv(goal, seed=0), expert_choose, args.episodes),
    }

    ckpt = Path(args.ckpt)
    if ckpt.exists():
        gpath = default_graph_path(ROOT)
        if not gpath.exists():
            build()
        model = FlyPolicy.load(FlyGraph(gpath), ckpt)
        model.eval()

        def trained_choose(frame, facts):
            o = torch.tensor(frame_to_vector(frame), dtype=torch.float32)
            with torch.no_grad():
                action, _, _, _ = model.act(o, greedy=True)
            return ACTIONS[int(action.item())]

        report["trained"] = run_policy(OdorTaxisEnv(goal, seed=0), trained_choose, args.episodes)
    else:
        report["trained"] = None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report.get("trained") and report.get("random"):
        if report["trained"]["success_rate"] < report["random"]["success_rate"]:
            print("WARN: trained success < random", file=sys.stderr)
    return report


if __name__ == "__main__":
    load_repo_env()
    main()
