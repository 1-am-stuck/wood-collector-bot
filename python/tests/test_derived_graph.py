"""The derived graph must stay the connectome, not drift back into a hand drawing.

Three kinds of check here.

*Measured counts.* A handful of synapse counts read off male-cns:v1.0. If a rebuild
changes them, the extractor is no longer reporting the dataset.

*Structural guarantees.* Dale's law, modality routing, and the rule that every action
reads only its own descending population. These are the properties that make the graph
trainable without becoming an MLP wearing neuron names.

*Physiology.* The untrained network should already behave like a fly on the reflexes
the wiring is strong enough to determine. A dark object on the left must excite the
left giant fibre, because LC4 -> DNp01 is strictly ipsilateral. Nothing is trained for
these; they come out of real synapse counts and real transmitters.

Tests needing the 23 MB connectome skip when it is absent, since it is not committed.
The derived NPZ is small enough to commit, so the checks that matter most always run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fly_policy.graph import FlyGraph
from fly_policy.policy import ACTIONS, FlyPolicy
from sense.frame import vector_size

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "connectome" / "malecns_path.npz"

# Both-side totals, read from male-cns:v1.0 at the >=5-synapse threshold.
KNOWN_SYNAPSES = {
    ("LC4_L", "DNp01_L"): 3782,
    ("LC4_R", "DNp01_R"): 2580,
    ("LPLC2_L", "DNp01_L"): 2636,
    ("LPLC2_R", "DNp01_R"): 2200,
    ("LC9_L", "DNp09_L"): 1098,
    ("LC9_R", "DNp09_R"): 806,
    ("HSS_L", "DNa02_L"): 47,
    ("HSS_R", "DNa02_R"): 36,
    ("DNp01_L", "TTMn_L"): 20,
    ("DNp01_R", "TTMn_R"): 70,
    # DNg60 inhibits the *opposite* side's turn neuron. There is no ipsilateral
    # DNg60 -> DNa02 connection at all, which is worth pinning down: the halt signal
    # crosses the midline.
    ("DNg60_L", "DNa02_R"): 41,
    ("DNg60_R", "DNa02_L"): 28,
}

# Edges the old hand-built graph asserted that male-cns does not contain.
INVENTED_EDGES = [
    ("DNa02_L", "DNa02_R"), ("DNa02_R", "DNa02_L"),
    ("DNg60_L", "DNa02_L"), ("DNg60_R", "DNa02_R"),
    ("DNg60_L", "DNp09_L"), ("DNp09_L", "MDN_L"), ("MDN_L", "DNp09_L"),
    ("DNp01_L", "DNg02_a_L"),
]


@pytest.fixture(scope="module")
def graph() -> FlyGraph:
    if not DERIVED.exists():
        pytest.skip(
            f"{DERIVED.name} missing; build it with "
            "`uv run python python/tools/build_graph_from_flyb.py --variant path`"
        )
    return FlyGraph(DERIVED)


@pytest.fixture(scope="module")
def policy(graph: FlyGraph) -> FlyPolicy:
    torch.manual_seed(0)
    return FlyPolicy(graph, vector_size(graph.n_retina), len(ACTIONS))


def test_the_graph_is_derived_from_male_cns(graph: FlyGraph):
    assert graph.derived
    assert graph.provenance["dataset"] == "male-cns:v1.0"
    assert graph.provenance["cells_total"] == 176_422
    assert graph.provenance["edges_total"] == 6_287_749
    assert graph.provenance["note"].startswith("edges are measured synapse counts")


@pytest.mark.parametrize("pair,count", sorted(KNOWN_SYNAPSES.items()))
def test_known_synapse_counts(graph: FlyGraph, pair, count):
    src, dst = pair
    assert graph.synapses(src, dst) == count


def test_the_halt_neuron_inhibits_turning_only_across_the_midline(graph: FlyGraph):
    """DNg60 is GABAergic and its DNa02 target is contralateral, never ipsilateral."""
    assert graph.sign[graph.index_of("DNg60_L")] == -1.0
    assert graph.sign[graph.index_of("DNg60_R")] == -1.0
    assert graph.synapses("DNg60_L", "DNa02_L") == 0
    assert graph.synapses("DNg60_R", "DNa02_R") == 0
    assert graph.synapses("DNg60_L", "DNa02_R") > 0
    assert graph.synapses("DNg60_R", "DNa02_L") > 0


@pytest.mark.parametrize("pair", INVENTED_EDGES)
def test_edges_the_hand_graph_invented_are_absent(graph: FlyGraph, pair):
    src, dst = pair
    if src not in graph.names or dst not in graph.names:
        pytest.skip(f"{src} or {dst} not in this variant")
    assert graph.synapses(src, dst) == 0


def test_dale_law_holds_by_construction(graph: FlyGraph):
    """Sign belongs to the neuron, so all of its edges must carry the same one."""
    assert graph.node_signed
    edge_sign = graph.edge_sign()
    assert edge_sign.shape[0] == graph.n_edges
    assert np.array_equal(edge_sign, graph.sign[graph.src])
    # No neuron can excite one target and inhibit another.
    order = np.argsort(graph.src)
    s = graph.src[order]
    signs = edge_sign[order]
    starts = np.searchsorted(s, np.unique(s))
    ends = np.append(starts[1:], s.size)
    for lo, hi in zip(starts, ends):
        assert len(np.unique(signs[lo:hi])) == 1


def test_photoreceptors_are_histaminergic_and_inhibitory(graph: FlyGraph):
    """Light inhibits the lamina. Every downstream visual sign depends on this."""
    for name in graph.names:
        if name.startswith("R1-R6_"):
            i = graph.index_of(name)
            assert graph.nt[i] == "histamine"
            assert graph.sign[i] == -1.0


def test_lamina_nodes_are_driven_by_darkness(graph: FlyGraph):
    """Inverting polarity, because histaminergic input inverts luminance."""
    lamina = [n for n in graph.names if n.split("_")[0] in ("L1", "L2", "L3")]
    assert lamina, "no lamina nodes in the graph"
    assert all(graph.polarity[n] == -1 for n in lamina if n in graph.routing)
    photo = [n for n in graph.names if n.startswith("R1-R6_")]
    assert all(graph.polarity[n] == 1 for n in photo if n in graph.routing)


def test_columns_with_no_measured_direction_are_left_blind(graph: FlyGraph):
    """A handful of connectome columns are absent from the micro-CT direction map.

    They get no routing at all rather than being pointed at some nearby ray, which
    would be inventing a viewing direction for a cell we have not measured.
    """
    unrouted = [n for n, role in zip(graph.names, graph.roles)
                if role == "sensory" and n not in graph.routing]
    assert len(unrouted) == graph.provenance["columns_without_measured_direction"]
    # Small enough to be a data gap rather than a bug in the join.
    assert len(unrouted) < 0.01 * len(graph.sensory_idx)


def test_each_retinotopic_node_reads_exactly_one_ommatidium(graph: FlyGraph):
    """A lamina node sees its own column, not the whole eye. This is the retinotopy."""
    rays = [n for n in graph.names
            if n.split("_")[0] in ("L1", "L2", "L3", "R1-R6")
            and n in graph.routing]
    assert len(rays) > 1000
    for name in rays:
        keys = graph.routing[name]
        assert len(keys) == 1, f"{name} reads {keys}"
        assert keys[0].startswith("retina:")
        assert keys[0] != "retina"
    # Different columns must not all collapse onto one ray, or there is no map left.
    used = {graph.routing[n][0] for n in rays}
    assert len(used) == graph.n_retina


def test_every_action_reads_only_its_own_population(graph: FlyGraph, policy: FlyPolicy):
    mask = policy.dec_mask
    assert mask.shape == (len(ACTIONS), len(graph.dn_idx))
    assert set(graph.action_units) == set(ACTIONS)
    for row, action in enumerate(ACTIONS):
        units = graph.action_units[action]
        assert units, f"{action} has no descending unit"
        assert mask[row].sum() == len(units)
    # No descending unit may drive two different actions.
    assert mask.sum(0).max() <= 1


def test_outputs_with_no_minecraft_action_are_kept_but_never_decoded(
    graph: FlyGraph, policy: FlyPolicy
):
    """Grooming, flight amplitude and HS yaw-gaze stay in the graph unread.
    DNOVS pitch gaze is decoded as camera_up, so those cells are not in unread."""
    assert graph.unread
    dn_names = [graph.names[int(i)] for i in graph.dn_idx]
    for name in graph.unread:
        assert policy.dec_mask[:, dn_names.index(name)].sum() == 0
    decoded = {u for units in graph.action_units.values() for u in units}
    assert not decoded & set(graph.unread)


def test_sensory_units_only_see_their_own_modality(graph: FlyGraph, policy: FlyPolicy):
    """An ORN must not read luminance, and a lamina node must not read odour."""
    from sense.channels import channel_map

    table = channel_map(graph.n_retina, graph.ray_sides)
    retina_cols = set(table["retina"])
    odour_cols = {i for key, v in table.items()
                  if key.startswith("glomerulus:") for i in v}
    mask = policy.enc_mask
    for row, node in enumerate(policy.sensory_idx.tolist()):
        name = graph.names[node]
        cols = set(torch.nonzero(mask[row]).flatten().tolist())
        if not cols:
            continue
        if name.startswith("ORN_"):
            assert not cols & retina_cols, f"{name} reads luminance"
        if name.split("_")[0] in ("L1", "L2", "L3"):
            assert not cols & odour_cols, f"{name} reads odour"
            assert len(cols) == 1


def test_gains_start_at_measured_relative_strengths(graph: FlyGraph):
    """Each edge starts at its share of its target's input, so proportions are real."""
    log_gain = graph.initial_log_gain()
    assert log_gain is not None
    w = np.exp(log_gain)
    total = np.zeros(graph.n)
    np.add.at(total, graph.dst, w)
    driven = total[total > 0]
    assert np.allclose(driven, 1.0, atol=1e-4)
    # The stronger of two edges onto the same target must still be the stronger one.
    a = graph.synapses("LC4_R", "DNp01_R")
    b = graph.synapses("LPLC2_R", "DNp01_R")
    dn = graph.index_of("DNp01_R")
    ga = w[(graph.src == graph.index_of("LC4_R")) & (graph.dst == dn)].sum()
    gb = w[(graph.src == graph.index_of("LPLC2_R")) & (graph.dst == dn)].sum()
    assert (a > b) == (ga > gb)


def _dn_rates(policy: FlyPolicy, graph: FlyGraph, luminance: np.ndarray):
    obs = torch.zeros(vector_size(graph.n_retina))
    obs[: graph.n_retina] = torch.tensor(luminance, dtype=torch.float32)
    with torch.no_grad():
        h = policy.forward_hidden(obs)
        dn = h.index_select(-1, policy.dn_idx).squeeze(0).numpy()
    names = [graph.names[int(i)] for i in policy.dn_idx]
    return dict(zip(names, dn))


def _dark_patch(graph: FlyGraph, centre_az: float, width: float = 45.0):
    from sense import retina_map

    rays = retina_map.load_config()["rays"]
    az = np.asarray([r["azimuthRightDeg"] for r in rays])
    el = np.asarray([r["elevationDeg"] for r in rays])
    lum = np.full(graph.n_retina, 0.9)
    offset = np.abs(((az - centre_az + 180) % 360) - 180)
    lum[(offset < width / 2) & (np.abs(el) < 40)] = 0.05
    return lum


@pytest.mark.parametrize("azimuth,nearer", [(-60.0, "L"), (60.0, "R")])
def test_an_untrained_fly_escapes_on_the_side_the_object_is_on(
    policy: FlyPolicy, graph: FlyGraph, azimuth: float, nearer: str
):
    """LC4 -> DNp01 is strictly ipsilateral, so the giant fibre nearer the object
    must be the more active one -- with no training at all."""
    rates = _dn_rates(policy, graph, _dark_patch(graph, azimuth))
    ipsi, contra = rates[f"DNp01_{nearer}"], rates[f"DNp01_{'R' if nearer == 'L' else 'L'}"]
    assert ipsi > contra, f"object at {azimuth} deg gave ipsi {ipsi} contra {contra}"


def test_darkening_the_whole_field_drives_the_escape_pathway(
    policy: FlyPolicy, graph: FlyGraph
):
    """A looming object darkens the eye, and darkness is what excites the lamina."""
    escape = [
        sum(_dn_rates(policy, graph, np.full(graph.n_retina, lum))[f"DNp01_{s}"]
            for s in "LR")
        for lum in (0.9, 0.6, 0.3, 0.1)
    ]
    assert escape == sorted(escape), f"not monotone in darkness: {escape}"
    assert escape[-1] > escape[0] * 2


def test_the_settle_stays_bounded(policy: FlyPolicy, graph: FlyGraph):
    """316k edges at measured strengths must not blow the network up."""
    for lum in (0.0, 0.5, 1.0):
        obs = torch.zeros(vector_size(graph.n_retina))
        obs[: graph.n_retina] = lum
        with torch.no_grad():
            h = policy.forward_hidden(obs)
            logits, value = policy(obs)
        assert torch.isfinite(h).all()
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()
        assert float(h.max()) < 100.0
