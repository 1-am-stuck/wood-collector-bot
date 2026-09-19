import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fly_policy.graph import FlyGraph
from fly_policy.policy import FlyPolicy
from sense.frame import vector_size
from sense.goal_to_sense import load_goal
from tools.build_mini_graph import build
from train_rl import ppo


def test_ppo_smoke_finite_return():
    g = FlyGraph(build())
    model = FlyPolicy(g, vector_size(64))
    goal = load_goal(ROOT / "configs/goals/collect_oak.json")
    model, ret = ppo(model, goal, epochs=2, episodes=4, lr=3e-4)
    assert ret == ret  # not NaN
    assert isinstance(ret, float)
