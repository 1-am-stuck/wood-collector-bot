import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from conftest import graph_or_skip
from fly_policy.policy import FlyPolicy
from sense.frame import vector_size


def test_forward_shapes():
    g = graph_or_skip()
    m = FlyPolicy(g, vector_size(g.n_retina))
    obs = torch.zeros(4, vector_size(g.n_retina))
    logits, value = m(obs)
    assert logits.shape == (4, 9)
    assert value.shape == (4,)
    # topology frozen: log_gain is per-edge, signs are buffers
    assert m.log_gain.numel() == g.n_edges
    assert m.sign.numel() == g.n_edges


def test_calibrate_stays_bounded():
    """Homeostasis may not invent 10^8 rates. Five-fold boost is the ceiling."""
    g = graph_or_skip()
    m = FlyPolicy(g, vector_size(g.n_retina))
    obs = torch.full((4, vector_size(g.n_retina)), 0.55)
    report = m.calibrate(obs)
    assert report["rate_peak"] < 200, report
    assert torch.isfinite(m.forward_hidden(obs[0])).all()


def test_the_settle_is_finite_on_a_bright_and_a_dark_world():
    """Whatever the graph size, the rates must not run away."""
    g = graph_or_skip()
    m = FlyPolicy(g, vector_size(g.n_retina))
    for fill in (0.0, 0.5, 1.0):
        obs = torch.full((vector_size(g.n_retina),), fill)
        h = m.forward_hidden(obs)
        assert torch.isfinite(h).all(), f"non-finite rates at luminance {fill}"
        assert h.min() >= 0.0, "rectified rates cannot go negative"


def test_decode_is_mean_rate_so_a_large_pool_cannot_outvote_mn9():
    """Logit = mean of an action's units, not the sum.

    camera_up used to sum 16 MNnm cells. At the same firing rate that drowned
    MN9's two cells, so an untrained fly locked onto camera/jump and never ate.
    """
    from fly_policy.policy import ACTIONS

    g = graph_or_skip()
    m = FlyPolicy(g, vector_size(g.n_retina))
    dn = torch.ones(int(m.dn_idx.numel()))
    with torch.no_grad():
        logits = m.decode(dn)
        by = {a: float(logits[i]) for i, a in enumerate(ACTIONS)}
    assert abs(by["mine"] - by["camera_up"]) < 0.05, by
    assert abs(by["mine"] - by["jump"]) < 0.05, by
    assert abs(by["mine"] - 1.0) < 0.05, by
