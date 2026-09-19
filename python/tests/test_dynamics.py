import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fly_policy.graph import FlyGraph
from fly_policy.policy import FlyPolicy
from sense.frame import vector_size
from tools.build_mini_graph import build


def test_forward_shapes():
    path = build()
    g = FlyGraph(path)
    m = FlyPolicy(g, vector_size(64))
    obs = torch.zeros(4, vector_size(64))
    logits, value = m(obs)
    assert logits.shape == (4, 9)
    assert value.shape == (4,)
    # topology frozen: log_gain is per-edge, signs are buffers
    assert m.log_gain.numel() == g.n_edges
    assert m.sign.numel() == g.n_edges


