import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from conftest import graph_or_skip
from fly_policy.policy import FlyPolicy
from sense.frame import vector_size
from sense.goal_to_sense import load_goal
from train_rl import ppo, ppo_epoch_update


def test_ppo_smoke_finite_return():
    g = graph_or_skip()
    model = FlyPolicy(g, vector_size(g.n_retina))
    goal = load_goal(ROOT / "configs/goals/collect_oak.json")
    model, ret = ppo(model, goal, epochs=2, episodes=4, lr=3e-4)
    assert ret == ret  # not NaN
    assert isinstance(ret, float)


def test_minibatching_covers_every_sample():
    """Every step must contribute a gradient, whatever the minibatch size.

    The live Minecraft runs have to minibatch, because backpropagating through a
    settle over 6.29M edges for a whole epoch's batch at once needs tens of
    gigabytes. A silent off-by-one in the chunking would quietly drop the tail of
    each epoch's experience.
    """
    g = graph_or_skip()
    n_obs = vector_size(g.n_retina)
    torch.manual_seed(0)
    n = 7  # deliberately not a multiple of the minibatch size
    obs = torch.rand(n, n_obs)
    act = torch.randint(0, len(FlyPolicy(g, n_obs).decoder.bias), (n,))
    adv = [1.0] * n
    ret = [0.5] * n

    seen = []
    model = FlyPolicy(g, n_obs)
    with torch.no_grad():
        logits, _ = model(obs)
    oldlp = torch.distributions.Categorical(logits=logits).log_prob(act)

    class Spy(torch.optim.Adam):
        def step(self, *a, **k):
            seen.append(int(model.decoder.weight.grad.abs().sum() > 0))
            return super().step(*a, **k)

    ppo_epoch_update(model, obs, act, oldlp, adv, ret, Spy(model.parameters(), lr=1e-4),
                     minibatch=3)
    assert len(seen) == 3, f"7 samples at minibatch 3 should be 3 updates, got {len(seen)}"
    assert all(seen), "every minibatch must produce a gradient"
