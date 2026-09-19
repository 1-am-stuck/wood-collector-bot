from pathlib import Path

import torch

from fly_policy.graph import FlyGraph
from fly_policy.policy import ACTIONS, FlyPolicy
from play import latest_ckpt
from sense.frame import vector_size
from tools.build_mini_graph import build

ROOT = Path(__file__).resolve().parents[2]


def test_latest_ckpt_prefers_rarest_then_ppo(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "fly_mc.pt").write_bytes(b"a")
    (ckpt_dir / "fly_mc_ppo.pt").write_bytes(b"b")
    assert latest_ckpt(tmp_path).name == "fly_mc_ppo.pt"
    (ckpt_dir / "fly_mc_rarest.pt").write_bytes(b"c")
    assert latest_ckpt(tmp_path).name == "fly_mc_rarest.pt"


def test_inspect_returns_named_hidden_rates():
    g = FlyGraph(build())
    model = FlyPolicy(g, vector_size(64))
    obs = torch.zeros(vector_size(64))
    brain = model.inspect(obs, greedy=True)
    assert brain["action"] in ACTIONS
    assert len(brain["h"]) == g.n
    assert len(brain["names"]) == g.n
    assert "DNp09" in brain["names"]
    assert brain["n_edges"] == g.n_edges
    assert brain["n"] < 200
