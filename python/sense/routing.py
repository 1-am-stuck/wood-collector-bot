"""What the sensory and descending populations of male-cns *are*.

This file used to also contain the wiring: an explicit `{neuron: observation
indices}` table, hand-written per population, plus a list of which unit each
Minecraft action reads. Both now come out of the connectome itself. The selectors
in `python/connectome/anchors.py` say which real cell types carry each modality,
`python/connectome/subgraph.py` records the resulting routing into the graph file,
and `python/sense/channels.py` resolves those keys against the current frame
layout. A hand-written copy could only drift from the graph it was describing --
and it had: its retina routing assumed an 8x4 grid per eye, which the measured
ommatidial directions replaced.

What is left here is reference knowledge that no connectome file carries: which
glomeruli are thermo/hygro rather than olfactory, what each gustatory neuron
tastes, and what each descending population does in a behaving fly. Provenance
for all of it is in docs/SENSE_PROVENANCE.md.
"""

from __future__ import annotations

from .frame import GLOMERULI

# VP glomeruli carry thermo- and hygrosensory input, not odour: hot cells target
# VP2, cold cells VP3, dry VP4, moist VP5. They get TRN/HRN units, not ORNs.
THERMO_HYGRO_GLOMERULI = ("VP2", "VP3a", "VP3b", "VP4", "VP5")
OLFACTORY_GLOMERULI = tuple(g for g in GLOMERULI if g not in THERMO_HYGRO_GLOMERULI)

EYES = ("L", "R")


# Gustatory receptor neuron identity: modality and molecular marker, per type.
# Source: the male-cns taste-feeding connectome (bioRxiv 2025.08.25.671814), the
# naming authority for LB / LgLG / LgAG / WG / PhG; table reproduced in
# .research/fly-brain-minecraft/docs/research/sensory-mapping.md.
#
# This cannot be inferred from the name prefix. LB1e sits in the LB1 series but is
# an Ir94e amino-acid cell, not a bitter cell, and LB3d sits in the LB3 series but
# is an Ir7c high-salt *avoidance* cell. Wiring valence off the prefix inverts
# both of them.
GRN_IDENTITY: dict[str, tuple[str, str]] = {
    "LB3b": ("sugar", "Gr64f"),
    "LB3c": ("sugar", "Gr64f"),
    "LB3a": ("water", "ppk28"),
    "LB3d": ("high salt — avoid", "Ir7c / ppk23, glutamatergic"),
    "LB1a": ("bitter", "Gr33a"),
    "LB1b": ("bitter", "Gr33a"),
    "LB1c": ("bitter", "Gr33a"),
    "LB1d": ("bitter", "Gr33a"),
    "LB1e": ("amino acid", "Ir94e"),
    "PhG1a": ("sugar", "Gr64e"),
    "PhG1b": ("sugar", "Gr64e"),
    "PhG1c": ("sugar", "Gr64e"),
    "PhG3": ("water", "ppk28"),
    "PhG4": ("water", "ppk28"),
    "PhG16": ("unassigned", "Ir60d"),
    "LgLG3": ("sugar", "Gr5a"),
    "LgLG4": ("sugar", "sugar Grs"),
    "LgAG1": ("bitter", "Gr33a, also Gr32a pheromone"),
    "WG2": ("sugar", "Gr43a / Gr64f / Gr5a"),
}

# Organ, from the name prefix. This part *is* systematic.
GRN_ORGAN = {
    "LB": "labellar bristle",
    "PhG": "pharyngeal",
    "LgLG": "tarsal, local to the VNC",
    "LgAG": "tarsal, ascending to the SEZ",
    "WG": "wing margin",
}

# Modalities that suppress feeding rather than promote it. Bitter and high salt
# are the two aversive gustatory channels; water and amino acid are appetitive.
AVERSIVE_MODALITIES = ("bitter", "high salt — avoid")
AVERSIVE_GRNS = tuple(
    g for g, (modality, _) in GRN_IDENTITY.items() if modality in AVERSIVE_MODALITIES
)


def grn_organ(name: str) -> str:
    return next((organ for prefix, organ in GRN_ORGAN.items() if name.startswith(prefix)), "")


# What each function is, and the male-cns type name where the common name differs.
# Sources: the descending-neuron behaviour survey in the reference clone, which
# traces each claim to a primary paper.
DESCENDING_WHAT: dict[str, tuple[str, str]] = {
    "forward": (
        "forward walking",
        "DNp09 is the command neuron (and biases an ipsilateral turn; at sustained "
        "high drive it switches to freezing). DNg100 = BDN2 drives forward walking "
        "even in headless flies and tracks forward velocity. DNge053 = BDN1.",
    ),
    "yaw": (
        "yaw steering",
        "DNa02, one type with one cell per side. Right-minus-left rate is near-linear "
        "in rotational velocity across the whole dynamic range, and unilateral "
        "activation turns the fly ipsiversively -- so `turn_left` reads DNa02_L and "
        "`turn_right` reads DNa02_R. It is one population, not two: the side is the "
        "signal, which is why the graph keeps left and right as separate nodes.",
    ),
    "backward": (
        "backward walking",
        "MDN = DNp50, the moonwalker. Activates backward walking and separately "
        "inhibits forward walking; required to back out of a dead end. 4 cells.",
    ),
    "halt": (
        "walk-off halt",
        "DNg60 = bluebell, GABAergic. Predominantly suppresses the *turning* "
        "component driven by DNp09, leaving the legs free to reposition.",
    ),
    "escape": (
        "escape takeoff / jump",
        "DNp01 is the giant fibre: one spike is enough. It is electrically coupled "
        "to TTMn, which depresses the middle legs and powers the jump. TTMn is "
        "silent otherwise, which makes it the cleanest jump readout in the VNC.",
    ),
    "feed": (
        "proboscis extension",
        "MN9 is necessary and sufficient for rostrum extension -- a better feeding "
        "readout than any of the feeding descending neurons. 2 cells.",
    ),
    "groom": (
        "antennal grooming",
        "aDN1 = DNg62 (bouts often stop before the stimulus does) and "
        "aDN2 = DNge078 (sustained for the whole stimulus).",
    ),
    "flight": (
        "wingbeat amplitude",
        "DNg02 is a graded population controller, not a command neuron: recruiting "
        "more of its 29 cells scales amplitude roughly linearly.",
    ),
    "gazeYaw": (
        "head yaw gaze stabilisation",
        "DNp15 = DNHS1, driven by horizontal-system yaw optic flow onto neck motor.",
    ),
    "neck_nm": (
        "neck motor pool, MNnm*",
        "18 cells moving the head on the neck. Which way a given MNnm pulls is not "
        "published per type, so `camera_up` reads this pool and `camera_down` reads "
        "the ADNM1/ADNM2/FNM2 pool, and the decoder learns the sign. The real pitch-"
        "gaze descending neurons DNp20 and DNp22 run on ocellar and vertical-system "
        "optic flow, neither of which the SensoryFrame carries.",
    ),
    "neck_adn": ("neck motor pool, ADNM / FNM", "The second neck motor group."),
}

# Mineflayer action -> the descending population whose rate drives it. The units in
# each population come from the graph (`graph.action_units`), which resolves these
# names against real male-cns cells; this table is only the action-to-function map.
# `noop` is the halt population, not an absence: stopping is a command in a fly.
ACTION_POPULATIONS: dict[str, str] = {
    "forward": "forward",
    "back": "backward",
    # Both turns read the same population; the graph splits it by side, and which side
    # is active is the turn direction.
    "turn_left": "yaw",
    "turn_right": "yaw",
    "jump": "escape",
    "mine": "feed",
    "camera_up": "neck_nm",
    "camera_down": "neck_adn",
    "noop": "halt",
}
