"""Inputs may only reach their own receptors, actions only their own motor units.

This used to test a hand-written `{neuron: observation indices}` table. That table is
gone; the routing now comes out of the connectome, so these checks read it back off the
graph. The properties being defended are the same ones, and they are the reason the
network is a fly rather than an MLP wearing neuron names: a photoreceptor cannot smell,
an ORN cannot see, contact chemoreception needs contact, and each action is driven by
the descending population that drives it in a real fly.

Anything about measured synapse counts or retinotopy lives in test_derived_graph.py.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from conftest import graph_or_skip
from fly_policy.policy import ACTIONS, FlyPolicy
from sense.channels import channel_map
from sense.frame import GLOMERULI, vector_size
from sense.routing import (
    ACTION_POPULATIONS,
    OLFACTORY_GLOMERULI,
    THERMO_HYGRO_GLOMERULI,
)


def spans(n_retina: int, ray_sides):
    """Observation indices per modality, taken from the channel map itself."""
    table = channel_map(n_retina, ray_sides)
    return {
        "retina": set(table["retina"]),
        "glomeruli": {i for g in GLOMERULI for i in table[f"glomerulus:{g}"]},
        "grns": {i for k, v in table.items() if k.startswith("grn:") for i in v},
        "mechano": {i for k, v in table.items() if k.startswith("mechano:") for i in v},
        "objects": {i for k, v in table.items() if k.startswith("object:") for i in v},
    }


# Channels the frame reports that no male-cns receptor population reads. This is a
# short, explicit list rather than an allowance for drift: abdominal touch is carried
# by cells the dataset leaves as `unknown_sensory`, so there is no typed population to
# anchor it to. Inventing an `SNta_abdomen` to make the coverage number look complete
# is exactly the kind of made-up neuron this rewrite removed.
DELIBERATELY_UNSENSED = ("mechano:touchAbdomen",)


def test_every_observation_channel_reaches_some_receptor():
    """Nothing the world tells the fly should fall on the floor unaccounted for."""
    graph = graph_or_skip()
    routed = graph.obs_routing(graph.n_retina)
    table = channel_map(graph.n_retina, graph.ray_sides)
    claimed = {i for idx in routed.values() for i in idx}
    allowed = {i for key in DELIBERATELY_UNSENSED for i in table[key]}
    missing = sorted(set(range(vector_size(graph.n_retina))) - claimed - allowed)
    assert not missing, (
        f"observation channels nothing can sense: {missing}. Either route them to a "
        "real population or add them to DELIBERATELY_UNSENSED with a reason."
    )
    # The declared gaps must really be gaps, so the list cannot rot into a blanket.
    for key in DELIBERATELY_UNSENSED:
        assert not (set(table[key]) & claimed), f"{key} is routed after all"


def test_receptors_only_read_their_own_modality():
    graph = graph_or_skip()
    s = spans(graph.n_retina, graph.ray_sides)
    routed = graph.obs_routing(graph.n_retina)

    checked = 0
    for name, idx in routed.items():
        got = set(idx)
        if name.startswith(("R1-R6", "R7", "R8", "L1", "L2", "L3")):
            assert got <= s["retina"], f"{name} reads outside the retina"
            checked += 1
        elif name.startswith("ORN_"):
            assert not got & s["retina"], f"{name} must not see light"
            glom = name[len("ORN_"):].rsplit("_", 1)[0]
            assert glom in OLFACTORY_GLOMERULI, f"{name} is not an olfactory glomerulus"
            checked += 1
        elif name.startswith(("JO-", "BM_", "SNta", "SNpp", "SApp")):
            assert not got & (s["retina"] | s["glomeruli"] | s["grns"]), name
            checked += 1
    assert checked > 50, f"only {checked} receptors checked -- routing looks empty"


def test_no_orn_exists_for_a_thermo_hygro_glomerulus():
    """VP2-VP5 carry temperature and humidity, and are innervated by TRNs / HRNs."""
    graph = graph_or_skip()
    routed = graph.obs_routing(graph.n_retina)
    table = channel_map(graph.n_retina, graph.ray_sides)
    for vp in THERMO_HYGRO_GLOMERULI:
        for side in ("L", "R"):
            assert f"ORN_{vp}_{side}" not in routed, f"{vp} carries no odour"
    thermo = {i for n, idx in routed.items()
              if n.startswith(("TRN_", "HRN_", "Thermo", "Hygro")) for i in idx}
    sensed = [vp for vp in THERMO_HYGRO_GLOMERULI
              if set(table[f"glomerulus:{vp}"]) & thermo]
    assert sensed, "no thermo/hygro receptor reads any VP glomerulus"


def test_encoder_mask_blocks_cross_modality_weights():
    graph = graph_or_skip()
    n_obs = vector_size(graph.n_retina)
    model = FlyPolicy(graph, n_obs)
    s = spans(graph.n_retina, graph.ray_sides)
    names = [graph.names[int(i)] for i in graph.sensory_idx]
    orn = [r for r, n in enumerate(names) if n.startswith("ORN_")]
    photo = [r for r, n in enumerate(names) if n.startswith(("R1-R6", "L1", "L2"))]
    assert orn and photo

    retina = sorted(s["retina"])
    glom = sorted(s["glomeruli"])
    assert model.enc_mask[orn][:, retina].sum() == 0, "an ORN could see light"
    assert model.enc_mask[photo][:, glom].sum() == 0, "a photoreceptor could smell"

    # The mask is structure, not an initial condition: it survives an update.
    model.encode_u(torch.ones(n_obs)).sum().backward()
    grad = model.encoder.weight.grad
    assert grad[orn][:, retina].abs().sum() == 0, "blocked pairs must get no gradient"


def test_each_action_reads_only_its_own_descending_population():
    graph = graph_or_skip()
    model = FlyPolicy(graph, vector_size(graph.n_retina))
    dn_names = [graph.names[int(i)] for i in graph.dn_idx]
    for row, action in enumerate(ACTIONS):
        reads = {dn_names[c] for c in range(len(dn_names)) if model.dec_mask[row, c] > 0}
        assert reads, f"{action} reads no motor unit"
        assert reads == set(graph.action_units[action]), f"{action} reads {reads}"

    # Standing still is a command in a fly, not an absence of one: `noop` is the
    # GABAergic halt neuron DNg60, and it must actually be inhibitory.
    halt = graph.action_units["noop"]
    assert halt and all(n.startswith("DNg60") for n in halt), halt
    for name in halt:
        assert graph.sign[list(graph.names).index(name)] < 0, f"{name} must be inhibitory"


def test_every_descending_population_in_the_index_exists_and_is_driven():
    """configs/sense/population_index.json is the contract for the output side.

    The contract is about *cells*, not about our grouping labels: every neuPrint type
    the index names must be in the graph as a real node, and must receive something.
    A named descending neuron that nothing synapses onto is a decoration.
    """
    index = json.loads((ROOT / "configs" / "sense" / "population_index.json").read_text())
    graph = graph_or_skip()
    names = list(graph.names)
    dn = {int(i) for i in graph.dn_idx}
    driven = {int(d) for d in graph.dst}

    for function, types in index["descending"].items():
        for type_name in types:
            # Node names are `<type>_<side>`, or `<type>_<side>_<bodyid>` in the full
            # graph, so a type is present if some node carries it as a prefix.
            cells = [i for i in dn if names[i] == type_name
                     or names[i].startswith(type_name + "_")]
            assert cells, f"{function}: {type_name} is not in the graph"
            assert any(i in driven for i in cells), \
                f"{function}: {type_name} receives no input"

    # Every action names a population that really exists on the motor side.
    groups = {graph.groups[i].split(":", 1)[1] for i in dn
              if graph.groups[i].startswith("motor:")}
    for action, population in ACTION_POPULATIONS.items():
        assert population in groups, f"{action} reads {population}, which is not wired"
        assert graph.action_units.get(action), f"{action} has no units"


def test_unread_outputs_are_kept_and_not_decoded():
    """Real fly outputs with no Minecraft action stay in the graph, undriven."""
    graph = graph_or_skip()
    model = FlyPolicy(graph, vector_size(graph.n_retina))
    dn_names = [graph.names[int(i)] for i in graph.dn_idx]
    assert graph.unread, "the connectome has outputs we cannot act on; none are declared"
    for unit in graph.unread:
        assert unit in dn_names, f"{unit} was trimmed out of the graph"
        col = dn_names.index(unit)
        assert model.dec_mask[:, col].sum() == 0, f"{unit} should drive no action"
