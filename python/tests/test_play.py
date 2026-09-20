from pathlib import Path

import torch

from conftest import graph_or_skip
from fly_policy.policy import ACTIONS, FlyPolicy
from play import latest_ckpt
from sense.frame import vector_size

ROOT = Path(__file__).resolve().parents[2]


def test_latest_ckpt_prefers_feed_then_navigate_then_rarest(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "fly_mc.pt").write_bytes(b"a")
    (ckpt_dir / "fly_mc_ppo.pt").write_bytes(b"b")
    assert latest_ckpt(tmp_path).name == "fly_mc_ppo.pt"
    (ckpt_dir / "fly_mc_rarest.pt").write_bytes(b"c")
    assert latest_ckpt(tmp_path).name == "fly_mc_rarest.pt"
    (ckpt_dir / "fly_mc_navigate.pt").write_bytes(b"d")
    assert latest_ckpt(tmp_path).name == "fly_mc_navigate.pt"
    (ckpt_dir / "fly_mc_feed.pt").write_bytes(b"e")
    assert latest_ckpt(tmp_path).name == "fly_mc_feed.pt"


def test_feed_checkpoint_injects_the_feed_goalspec():
    from play import uses_feed_goal

    assert uses_feed_goal(Path("checkpoints/fly_mc_feed.pt"), False) is True
    assert uses_feed_goal(Path("checkpoints/fly_mc_navigate.pt"), False) is False
    assert uses_feed_goal(Path("checkpoints/fly_mc_feed.pt"), True) is False


def test_inspect_returns_named_hidden_rates():
    g = graph_or_skip()
    model = FlyPolicy(g, vector_size(g.n_retina))
    obs = torch.zeros(vector_size(g.n_retina))
    topo = model.topology()
    brain = model.inspect(obs, greedy=True)
    assert brain["action"] in ACTIONS
    # The tick is the tracked sample; names and wiring come from the topology, once.
    assert len(brain["h"]) == len(topo["tracked"])
    assert len(topo["names"]) == len(topo["tracked"])
    assert brain["n"] == g.n
    assert brain["n_edges"] == g.n_edges
    forward = topo["descending"]["forward"]
    assert any(n.startswith("DNp09") for n in forward), forward
    assert topo["descending_what"]["forward"][0] == "forward walking"
    assert "group_stats" in brain
    merged = {**topo, **brain}
    assert merged["names"][0] == topo["names"][0]
    assert len(merged["h"]) == len(merged["names"])
