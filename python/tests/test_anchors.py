"""Anchors must resolve to real male-cns cells, never to names I made up.

The hand-built graph contained `BM_Notum`, `BM_Wing`, `BM_Leg`, `BM_Abd`, `JO-A`,
`JO-B`, `JO-C_L`, `JO-F`, `hair_plate`, `campaniform`, `noci`, `MN_neck_dorsal` and
`MN_neck_ventral`. Not one of those is a type in male-cns:v1.0. These tests exist so
that class of error fails immediately instead of training quietly for an hour.

Requires the connectome file, which is not committed, so they skip without it.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from connectome import anchors as anchors_mod
from connectome.anchors import Selector, motor_anchors, sensory_anchors

pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def fly():
    from connectome import flyb
    try:
        return flyb.load()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


# Names the hand-built graph used that do not exist in the dataset.
INVENTED_NAMES = [
    "BM_Notum", "BM_Wing", "BM_Leg", "BM_Abd",
    "JO-A", "JO-B", "JO-C", "JO-F", "JO-C_L", "JO-C_R",
    "hair_plate", "campaniform", "noci",
    "MN_neck_dorsal", "MN_neck_ventral", "HSE_L", "DNg02", "HS",
]


@pytest.mark.parametrize("name", INVENTED_NAMES)
def test_names_the_hand_graph_invented_are_not_real_types(fly, name):
    assert name not in fly.tables["types"]


def test_every_anchor_resolves_to_real_cells(fly):
    resolved = anchors_mod.resolve(fly, strict=False)
    assert not resolved.empty, "empty anchors:\n  " + "\n  ".join(resolved.empty)


def test_the_populations_of_the_index_have_the_expected_cell_counts(fly):
    """Counts from population_index.json, checked against the dataset."""
    expected = {
        "DNp09": 2, "DNg100": 2, "DNge053": 2, "DNa02": 2, "MDN": 4,
        "DNg60": 2, "DNp01": 2, "TTMn": 2, "MN9": 2, "DNg62": 2,
        "DNge078": 2, "DNp15": 2,
    }
    names = fly.names()
    counts = collections.Counter(names)
    for type_name, n in expected.items():
        assert counts[type_name] == n, f"{type_name}: {counts[type_name]} != {n}"


def test_the_halt_neuron_is_the_inhibitory_one(fly):
    """DNg60 is GABAergic; DNa02 is cholinergic and therefore cannot inhibit anything.

    The hand graph had DNa02 inhibiting its mirror twin, which Dale's law forbids.
    """
    names, nts = fly.names(), np.asarray(fly.tables["nts"])[fly.nt_idx]
    for cell in np.flatnonzero(names == "DNg60"):
        assert nts[cell] == "gaba" and fly.nt_sign[cell] == -1
    for cell in np.flatnonzero(names == "DNa02"):
        assert nts[cell] == "acetylcholine" and fly.nt_sign[cell] == 1


def test_there_are_no_orns_for_the_vp_glomeruli(fly):
    """VP glomeruli take thermo- and hygrosensory axons, not olfactory ones."""
    types = set(fly.tables["types"])
    for vp in ("VP1d", "VP1l", "VP1m", "VP2", "VP3a", "VP3b", "VP4", "VP5"):
        assert f"ORN_{vp}" not in types
    assert "TRN_VP2" in types and "HRN_VP4" in types


def test_abdominal_touch_is_deliberately_unrouted(fly):
    """male-cns annotates abdominal sensory cells as `unknown_sensory`.

    Their modality is not established, so no anchor claims them. If a future release
    labels them, this test is the place that should start failing.
    """
    keys = {a.key for a in sensory_anchors()}
    assert not any("abdomen" in k for k in keys)
    classes = np.asarray(fly.tables["classes"])[fly.class_idx]
    subclasses = np.asarray(fly.tables["subclasses"])[fly.subclass_idx]
    abdomen = subclasses == "abdomen"
    assert abdomen.sum() > 0
    assert (classes[abdomen] == "mechanosensory_tactile").sum() == 0


def test_leg_touch_covers_all_three_leg_nerves(fly):
    """ProLN carries only the front legs; the middle and hind pairs are MesoLN/MetaLN."""
    anchor = next(a for a in sensory_anchors() if a.key == "SNta_leg")
    assert set(anchor.selector.nerves) == {"ProLN", "MesoLN", "MetaLN"}
    assert anchor.selector.resolve(fly).size > 1500


def test_a_bad_selector_is_reported_rather_than_silently_empty(fly):
    bad = Selector(types=("NoSuchNeuron",))
    assert bad.resolve(fly).size == 0
    assert "NoSuchNeuron" in bad.describe()


def test_every_minecraft_action_has_a_motor_population():
    from fly_policy.policy import ACTIONS

    claimed = set()
    for anchor in motor_anchors():
        if anchor.action:
            claimed.add(anchor.action)
        if anchor.sides_split:
            from connectome.anchors import SIDE_ACTIONS
            claimed.update(a for (p, _), a in SIDE_ACTIONS.items()
                           if p == anchor.population)
    assert claimed == set(ACTIONS), f"missing {set(ACTIONS) - claimed}"


def test_outputs_without_an_action_are_still_declared():
    """Grooming, flight, HS yaw-gaze and the unpublished MNnm pool stay unread."""
    unread = {a.population for a in motor_anchors() if a.action is None}
    assert unread == {"yaw", "groom", "flight", "gazeYaw", "neck_nm"}


def test_optic_glomerulus_anchors_are_marked_analytic():
    """These channels are still computed in senseBridge rather than from the retina."""
    from connectome.anchors import OPTIC_GLOMERULI

    for anchor in sensory_anchors():
        if anchor.key in OPTIC_GLOMERULI:
            assert anchor.note.startswith("ANALYTIC")
