"""Which real male-cns cells carry each Minecraft sense, and which drive each action.

Every entry here is a *selector* resolved against the loaded connectome, not a
hand-written neuron name. That distinction matters: the previous hand-built graph
invented names like `BM_Notum`, `JO-C_L`, `hair_plate` and `MN_neck_dorsal`, none of
which exist in male-cns:v1.0. A selector either matches real cells or raises, so a
wrong guess fails loudly instead of quietly training on fiction.

What the dataset actually calls things, verified against the loaded file:

- Olfactory receptor neurons are `ORN_<glomerulus>`, pooled across both antennae;
  side is a separate field. There are no ORNs for the VP glomeruli, because those
  are thermo- and hygrosensory.
- Thermo/hygro are `TRN_VP2`, `TRN_VP3a/b`, `HRN_VP4`, `HRN_VP5`. `TRN_VP1m` and
  `HRN_VP1l` exist but the dataset class and the receptor literature disagree about
  them, so they are deliberately left out.
- Gustatory neurons are `LB*` labellar, `PhG*` pharyngeal, `LgLG*`/`LgAG*` tarsal,
  `WG*` wing margin.
- Johnston's organ has no plain `JO-A`/`JO-B`/`JO-C`/`JO-F` types. It has `JO-A1`..
  `JO-A4`, `JO-B1_a`.., `JO-CL`, `JO-CM`, `JO-EV1`.., `JO-FV`, `JO-FD1` and so on,
  and the useful grouping is the dataset's own subclass: `auditory`, `wind_gravity`,
  `grooming`.
- Head bristles are `BM_*` (`BM_InOm` 745 cells, `BM_Taste` 40). Body touch is not:
  it is `mechanosensory_tactile` `SNta*` cells, grouped by the nerve they enter on.
  Abdominal tactile cells are *not* on AbN3 in this dataset, so that region is
  selected by subclass instead.
- Proprioceptors are `SNpp45/19/52` (hair plates) and `SApp*` (campaniform sensilla).
- Photoreceptors are a single type `R1-R6` plus `R7p/y/d` and `R8p/y/d`; retinotopy
  lives in the `hex1`/`hex2` column coordinates, not in the type name.
- Neck motor neurons are real and named: `MNnm03`, `MNnm07,MNnm12`, `MNnm08`..
  `MNnm14`, plus `CvN4`..`CvN7`. Subclass `nm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flyb import Flyb


@dataclass(frozen=True)
class Selector:
    """A rule that picks real cells out of the connectome.

    `types` matches the type string exactly, `prefixes` by prefix (for families
    like `DNg02_a`..`DNg02_g` where no bare type exists). `cls`, `subclass` and
    `nerve` filter on the dataset's own annotation columns and may be combined.
    """

    types: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    cls: str | None = None
    subclass: str | None = None
    nerves: tuple[str, ...] = ()
    superclass: str | None = None

    def resolve(self, g: Flyb) -> np.ndarray:
        names = g.names()
        mask = np.ones(g.n, dtype=bool)
        if self.cls is not None:
            mask &= _column(g, "classes", g.class_idx) == self.cls
        if self.subclass is not None:
            mask &= _column(g, "subclasses", g.subclass_idx) == self.subclass
        if self.nerves:
            mask &= np.isin(_column(g, "nerves", g.nerve_idx),
                            np.asarray(self.nerves, dtype=object))
        if self.superclass is not None:
            mask &= _column(g, "superclasses", g.superclass_idx) == self.superclass
        if self.types or self.prefixes:
            named = np.zeros(g.n, dtype=bool)
            if self.types:
                named |= np.isin(names, np.asarray(self.types, dtype=object))
            for prefix in self.prefixes:
                named |= np.asarray([str(t).startswith(prefix) for t in names])
            mask &= named
        return np.flatnonzero(mask)

    def describe(self) -> str:
        bits = []
        if self.types:
            bits.append("type in {" + ", ".join(self.types) + "}")
        if self.prefixes:
            bits.append("type starts with " + "/".join(self.prefixes))
        if self.nerves:
            bits.append("nerve in {" + ", ".join(self.nerves) + "}")
        for label, value in (("class", self.cls), ("subclass", self.subclass),
                             ("superclass", self.superclass)):
            if value:
                bits.append(f"{label}={value}")
        return " & ".join(bits)


def _column(g: Flyb, table: str, idx: np.ndarray) -> np.ndarray:
    return np.asarray(g.tables[table], dtype=object)[idx]


# --- sensory side ------------------------------------------------------------
# Glomeruli this repo carries an odour row for. VP* are excluded on purpose: they
# receive thermo/hygro axons, not ORNs.
ODOUR_GLOMERULI = (
    "DM1", "DM2", "DM3", "DM4", "DM5",
    "VA2", "VA3", "VA6", "VA1v", "VA1d",
    "DL3", "DL4", "DL5",
    "DC2", "DA1", "DA2",
    "VM1", "VM2", "VM4", "VM5d",
    "VL2a", "VC1", "VC2", "VC5",
    "DP1l", "D", "V",
)

# Object channels this repo computes analytically, and the real cells each one
# stands for. There is no bare `HS` type: the horizontal system is HSE, HSN and HSS,
# two cells each.
OPTIC_GLOMERULI = {
    "LC4": ("LC4",),
    "LPLC2": ("LPLC2",),
    "LC11": ("LC11",),
    "LC18": ("LC18",),
    "LC10a": ("LC10a",),
    "LC15": ("LC15",),
    "HS": ("HSE", "HSN", "HSS"),
}

GUSTATORY_TYPES = (
    "LB3b", "LB3c", "LB3a", "LB3d",
    "LB1a", "LB1b", "LB1c", "LB1d", "LB1e",
    "PhG1a", "PhG1b", "PhG1c", "PhG3", "PhG4", "PhG16",
    "LgLG3", "LgLG4", "LgAG1", "WG2",
)


@dataclass(frozen=True)
class SenseAnchor:
    """A sensory population and the observation channels allowed to drive it."""

    key: str                  # stable name used in the graph and the routing map
    selector: Selector
    channels: tuple[str, ...]  # frame channel keys, resolved by sense.frame
    note: str = ""
    per_side: bool = True     # keep left and right as separate nodes
    per_column: bool = False  # keep medulla column (hex) identity
    inverting: bool = False   # driven by the absence of the channel, not its presence


def sensory_anchors() -> list[SenseAnchor]:
    """Every sensory population, in observation order."""
    out: list[SenseAnchor] = []

    # Vision, in two layers with opposite polarity, which male-cns itself dictates.
    #
    # Photoreceptors depolarise to light, so luminance drives R1-R6 directly. They are
    # histaminergic and carry sign -1 in this dataset, so light *inhibits* everything
    # they contact. That is the real first synapse, and the graph already has it.
    #
    # The consequence is that lamina monopolars depolarise to darkness, not to light:
    # a contrast decrement releases them from photoreceptor inhibition. So luminance is
    # injected into L1/L2/L3 inverted. Doing it this way rather than relying on the
    # R1-R6 synapse alone is a practical necessity with a stated reason: the settle is
    # rectified, so a unit cannot go below zero to represent hyperpolarisation, and
    # male-cns traces photoreceptors in only 819 columns against the lamina's 1,767, so
    # two thirds of the eye would be blind if the lamina were driven only through R1-R6.
    out.append(SenseAnchor(
        key="R1-R6",
        selector=Selector(types=("R1-R6",)),
        channels=("retina",),
        note="photoreceptors: depolarise to light, histaminergic, inhibit the lamina",
        per_column=True,
    ))
    for kind in ("L1", "L2", "L3"):
        out.append(SenseAnchor(
            key=kind,
            selector=Selector(types=(kind,)),
            channels=("retina",),
            note="lamina monopolar: depolarises to a contrast decrement, because "
                 "histaminergic photoreceptor input inverts luminance",
            per_column=True,
            inverting=True,
        ))

    # The optic glomeruli. These are real visual projection neurons and the graph
    # contains them, but the repo still computes their activation analytically in
    # `senseBridge.js` instead of deriving it from the retinotopic path. Injecting
    # the analytic value here is a stated shortcut, not a claim about the anatomy:
    # a real LC4 cell is driven by medulla columns, not by a scalar. Each of these
    # retires as soon as the lamina path drives its cells.
    for channel, types in OPTIC_GLOMERULI.items():
        # The horizontal system is the one wide-field population whose stimulus the
        # frame reports twice: `object:HS` is the analytic flow estimate, and
        # `rate:yaw` is the fly's own turn rate. HS cells encode horizontal wide-field
        # motion, which during self-motion *is* yaw flow, so both belong on it.
        extra = ("rate:yaw",) if channel == "HS" else ()
        out.append(SenseAnchor(
            key=channel,
            selector=Selector(types=types),
            channels=(f"object:{channel}",) + extra,
            note="ANALYTIC: value computed in senseBridge, not from the retina",
        ))

    # The vertical system, the horizontal system's counterpart: VS cells encode
    # wide-field vertical flow, i.e. pitch and roll. Minecraft gives the fly pitch
    # from its own camera, so that is what drives them. 18 cells in male-cns.
    out.append(SenseAnchor(
        key="VS",
        selector=Selector(types=("VS", "VST1", "VST2", "VSm")),
        channels=("rate:pitch",),
        note="lobula plate vertical system: pitch / roll wide-field flow",
    ))

    # Olfaction. One ORN population per glomerulus, bilateral; the left/right
    # comparison used for taxis happens downstream, so bearing is allowed in too.
    for glom in ODOUR_GLOMERULI:
        out.append(SenseAnchor(
            key=f"ORN_{glom}",
            selector=Selector(types=(f"ORN_{glom}",)),
            channels=(f"glomerulus:{glom}", "odor_bearing"),
        ))

    # Thermo and hygro receptors target VP glomeruli.
    for key, channel in (("TRN_VP2", "hot"), ("TRN_VP3a", "cold"),
                         ("TRN_VP3b", "cold"), ("HRN_VP4", "dry"),
                         ("HRN_VP5", "moist")):
        glom = key.split("_", 1)[1]
        out.append(SenseAnchor(
            key=key,
            selector=Selector(types=(key,)),
            channels=(f"glomerulus:{glom}", f"mechano:{channel}"),
        ))

    # Gustation: contact only, one population per receptor type.
    for grn in GUSTATORY_TYPES:
        out.append(SenseAnchor(
            key=grn,
            selector=Selector(types=(grn,)),
            channels=(f"grn:{grn}",),
        ))

    # Johnston's organ, by the dataset's own subclass rather than by letter.
    out.append(SenseAnchor(
        key="JO_auditory",
        selector=Selector(cls="mechanosensory", subclass="auditory"),
        channels=("mechano:soundHigh", "mechano:soundLow", "mechano:song"),
        note="JO-A/JO-B vibration cells; together they cover the song band",
    ))
    out.append(SenseAnchor(
        key="JO_wind_gravity",
        selector=Selector(cls="mechanosensory", subclass="wind_gravity"),
        channels=("mechano:windLeft", "mechano:windRight", "mechano:tilt"),
        note="JO-C/JO-E; maximally driven by static deflection of the receiver",
    ))
    out.append(SenseAnchor(
        key="JO_grooming",
        selector=Selector(cls="mechanosensory", subclass="grooming"),
        channels=("mechano:groomDust",),
        note="JO-F",
    ))

    # Head bristles. Damage has no dedicated nociceptor type in male-cns, so it
    # recruits every bristle and tactile field instead.
    out.append(SenseAnchor(
        key="BM_InOm",
        selector=Selector(types=("BM_InOm",)),
        channels=("mechano:touchHead", "mechano:damage"),
        note="interommatidial eye bristles, 745 cells",
    ))
    out.append(SenseAnchor(
        key="BM_Taste",
        selector=Selector(types=("BM_Taste",)),
        channels=("mechano:touchHead", "mechano:damage"),
        note="proboscis bristles; activation elicits proboscis grooming",
    ))

    # Body touch: SNta tactile cells, grouped by the nerve they arrive on. All three
    # leg nerves are taken together, because ProLN carries only the front pair while
    # MesoLN and MetaLN carry the middle and hind legs.
    #
    # There is deliberately no abdominal anchor. `touchAbdomen` exists in the frame,
    # but male-cns annotates its 1,145 abdominal sensory cells as `unknown_sensory`,
    # so no cell in this dataset is established as an abdominal touch receptor. The
    # channel is left unrouted rather than pointed at a population that might be
    # proprioceptive or chemosensory.
    for key, sel, channel in (
        ("SNta_leg", Selector(cls="mechanosensory_tactile",
                              nerves=("ProLN", "MesoLN", "MetaLN")), "touchLegs"),
        ("SNta_wing", Selector(cls="mechanosensory_tactile", nerves=("ADMN",)),
         "touchWing"),
        ("SNta_notum", Selector(cls="mechanosensory_tactile", nerves=("PDMN",)),
         "touchNotum"),
    ):
        out.append(SenseAnchor(
            key=key, selector=sel,
            channels=(f"mechano:{channel}", "mechano:damage"),
        ))

    # Proprioception: hair plates report leg load and posture, campaniform sensilla
    # report cuticular strain, i.e. wing load in flight.
    out.append(SenseAnchor(
        key="hair_plate",
        selector=Selector(cls="mechanosensory_proprioceptive", subclass="hair plate"),
        channels=("mechano:tilt", "mechano:legsOnGround", "mechano:airborne",
                  "rate:pitch"),
        note="SNpp45 / SNpp19 / SNpp52",
    ))
    out.append(SenseAnchor(
        key="campaniform",
        selector=Selector(cls="mechanosensory_proprioceptive",
                          subclass="campaniform sensilla"),
        channels=("mechano:wingbeat",),
        note="SApp09,SApp22 / SApp08 / SApp10",
    ))

    return out


# --- motor side --------------------------------------------------------------
@dataclass(frozen=True)
class MotorAnchor:
    """A descending or motor population, and the Minecraft action that reads it."""

    population: str
    selector: Selector
    action: str | None        # None = a real output with no Minecraft action
    body: str
    note: str = ""
    per_side: bool = True
    sides_split: bool = False  # left/right drive different actions


def motor_anchors() -> list[MotorAnchor]:
    """The descending populations of configs/sense/population_index.json."""
    return [
        MotorAnchor("forward", Selector(types=("DNp09", "DNg100", "DNge053")),
                    "forward", "legs",
                    "DNp09 is the forward command neuron and also biases an "
                    "ipsilateral turn; DNg100 = BDN2 works even in headless flies; "
                    "DNge053 = BDN1."),
        MotorAnchor("yaw", Selector(types=("DNa02",)), None, "legs",
                    "Right-minus-left DNa02 rate is near-linear in turn rate. "
                    "Split by side into turn_left / turn_right.",
                    sides_split=True),
        MotorAnchor("backward", Selector(types=("MDN",)), "back", "legs",
                    "MDN = DNp50, the moonwalker; 4 cells."),
        MotorAnchor("halt", Selector(types=("DNg60",)), "noop", "legs",
                    "DNg60 = bluebell, GABAergic. Suppresses the turn component "
                    "first: it synapses onto DNa02, not onto DNp09."),
        MotorAnchor("escape", Selector(types=("DNp01", "TTMn")), "jump", "legs",
                    "DNp01 is the giant fibre, electrically coupled to TTMn. TTMn "
                    "is silent otherwise, so it is the cleanest jump readout."),
        MotorAnchor("feed", Selector(types=("MN9",)), "mine", "proboscis",
                    "MN9 is necessary and sufficient for rostrum extension."),
        MotorAnchor("groom", Selector(types=("DNg62", "DNge078")), None, "antennae",
                    "aDN1 = DNg62, aDN2 = DNge078. Real output, no Minecraft action."),
        MotorAnchor("flight", Selector(prefixes=("DNg02_",)), None, "wings",
                    "29 cells across subtypes _a.._g; a graded amplitude "
                    "controller, not a command neuron. No bare DNg02 type exists."),
        MotorAnchor("gazeYaw", Selector(types=("DNp15",)), None, "head",
                    "DNHS1: horizontal-system yaw flow onto neck motor. Body yaw "
                    "is already DNa02 (turn_left/right), so this stays unread."),
        MotorAnchor("gazePitch", Selector(types=("DNp20", "DNp22")),
                    "camera_up", "head",
                    "DNOVS1/2: ocellar and vertical-system roll/pitch gaze. "
                    "Minecraft camera_up is that head elevation."),
        # Head pitch. FNM2 is a published levator; ADNM1/2 the other named neck
        # pool. MNnm exist but male-cns does not say which way each pulls, so
        # they stay unread rather than being assigned camera_up by convention.
        MotorAnchor("neck_nm", Selector(prefixes=("MNnm",), superclass="vnc_motor"),
                    None, "head",
                    "Neck muscle motor neurons MNnm03..MNnm14, 16 cells. Pull "
                    "direction is unpublished, so they are not decoded."),
        MotorAnchor("neck_adn",
                    Selector(types=("ADNM1 MN", "ADNM2 MN"),
                             superclass="vnc_motor"),
                    "camera_down", "head",
                    "ADNM1/2 depress the head."),
        MotorAnchor("neck_fnm",
                    Selector(types=("FNM2",),
                             superclass="vnc_motor"),
                    "camera_up", "head",
                    "FNM2 elevates the head (comparative DN/AN connectome)."),
    ]


# The one population read per side. Right-minus-left DNa02 rate is close to linear in
# turn rate, which is an established measurement, so the sides are genuinely two
# different commands rather than a convenience split.
SIDE_ACTIONS = {
    ("yaw", "L"): "turn_left",
    ("yaw", "R"): "turn_right",
}


@dataclass
class Resolved:
    """Anchor selectors applied to a loaded connectome."""

    sensory: dict[str, np.ndarray] = field(default_factory=dict)
    motor: dict[str, np.ndarray] = field(default_factory=dict)
    empty: list[str] = field(default_factory=list)


def resolve(g: Flyb, strict: bool = True) -> Resolved:
    """Turn every selector into real cell indices, complaining about empty ones."""
    out = Resolved()
    for anchor in sensory_anchors():
        idx = anchor.selector.resolve(g)
        if idx.size == 0:
            out.empty.append(f"sensory {anchor.key}: {anchor.selector.describe()}")
        out.sensory[anchor.key] = idx
    for anchor in motor_anchors():
        idx = anchor.selector.resolve(g)
        if idx.size == 0:
            out.empty.append(f"motor {anchor.population}: {anchor.selector.describe()}")
        out.motor[anchor.population] = idx
    if strict and out.empty:
        raise ValueError(
            "these anchors matched no cells in "
            f"{g.dataset}:\n  " + "\n  ".join(out.empty)
        )
    return out
