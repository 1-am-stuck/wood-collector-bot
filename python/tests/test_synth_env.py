from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fly_policy.synth_env import OdorTaxisEnv
from sense.goal_to_sense import load_goal


def test_expert_solves_oak_taxis():
    goal = load_goal(ROOT / "configs/goals/collect_oak.json")
    wins = 0
    for i in range(20):
        env = OdorTaxisEnv(goal, seed=i)
        frame, facts, _ = env.reset()
        done = False
        while not done:
            act = env.expert_action(facts)
            frame, _, done, facts = env.step(act)
        wins += int(env.have)
    assert wins >= 16, f"expert should usually collect oak, got {wins}/20"
