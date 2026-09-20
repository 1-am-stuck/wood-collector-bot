"""Derive a trainable subgraph from male-cns instead of drawing one by hand.

The hand-built graph in `python/tools/build_mini_graph.py` placed all 517 of its
edges from the literature and from memory. Checked against male-cns:v1.0, about half
of them do not exist — `DNa02_L ⊣ DNa02_R`, `DNp09 ⊣ MDN`, `DNg60 ⊣ DNp09`,
`LC11 → DNa02` are all zero-synapse — while the dominant real inputs were missing:
`DNp09`'s largest visual input is LC9 at 1,904 synapses, and nothing in the hand
graph knew LC9 existed. Worse, sign was attached to the edge, which let `DNa02`
inhibit its twin even though it is cholinergic.

So no edge in here is written by hand. Every edge is a measured synapse count between
two real cells, and every sign is the presynaptic cell's own transmitter, which makes
Dale's law hold by construction rather than by review.

## How the subgraph is chosen

Anchors come from `anchors.py`: sensory populations in, descending and motor
populations out. We want the neurons that actually carry signal between them, and we
want that choice to be a measurement too.

1. **Collapse.** Cells are grouped into nodes by (type, side), which is the unit the
   literature names and the unit a 15 Hz policy can afford. Photoreceptors and lamina
   monopolars additionally keep their medulla column, because collapsing `R1-R6`'s
   3,377 cells into one node per side would throw away the retinotopy that vision is.
   Edge weight between two nodes is the summed synapse count between their cells.
2. **Score.** Forward influence is propagated from the sensory anchors and backward
   influence from the motor anchors, each along out-degree-normalised weights for a
   few hops. A node's score is the product: high only if signal can plausibly reach
   it from a sense *and* leave it toward an action. Pure sensory side-branches and
   motor-side afferents both score low.
3. **Keep** the anchors unconditionally, plus the highest-scoring nodes up to a
   budget, then take the induced subgraph. Nothing is added to it afterwards.

That product score is what "on a short, heavy path" means operationally. It is a
heuristic about which neurons to *include*, and it is the only heuristic here; the
wiring among the included neurons is whatever male-cns says it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import anchors as anchors_mod
from .anchors import Resolved, SIDE_ACTIONS, motor_anchors, sensory_anchors
from .flyb import Flyb

# Types whose medulla column identity is kept, so vision stays retinotopic.
COLUMNAR_TYPES = ("R1-R6", "L1", "L2", "L3")


@dataclass
class Collapsed:
    """The whole connectome reduced to (type, side[, column]) nodes."""

    node_of_cell: np.ndarray   # int32[n_cells], -1 for cells with no usable node
    names: list[str]           # node label, e.g. 'DNp09_R' or 'L1_R_h12_04'
    type_name: list[str]
    side: list[str]
    hex1: np.ndarray
    hex2: np.ndarray
    sign: np.ndarray           # float32[n_nodes], from the presynaptic transmitter
    nt: list[str]
    soma: np.ndarray           # float32[n_nodes, 3], mean over member cells
    cells: np.ndarray          # int32[n_nodes], how many cells were merged
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray         # float32, summed synapse count
    dropped_cells: int
    dropped_untyped_weight: float
    dropped_within_node_weight: float

    @property
    def n(self) -> int:
        return len(self.names)


def collapse(g: Flyb, columnar: tuple[str, ...] = COLUMNAR_TYPES) -> Collapsed:
    """Group cells into (type, side) nodes, keeping columns for the retina.

    Cells with no type name are dropped. There are enough of them that merging them
    into one anonymous node per side would create a hub that carries more weight than
    any real neuron, and they cannot be cited or checked, so they are excluded and
    counted instead.
    """
    names_arr = g.names()
    types = np.asarray(g.tables["types"], dtype=object)
    columnar_type_idx = {
        i for i, t in enumerate(types) if t in set(columnar)
    }

    usable = np.asarray([bool(str(t)) for t in names_arr])
    is_columnar = np.isin(g.type_idx, list(columnar_type_idx)) if columnar_type_idx \
        else np.zeros(g.n, dtype=bool)

    # Column identity has to come from the retina table, not from the per-neuron hex
    # arrays. Every one of the 3,377 R1-R6 cells has hex1 = hex2 = -1 in the neuron
    # arrays; their column appears only in the retina table. Reading the neuron arrays
    # alone silently discards the entire photoreceptor layer, which is the one part of
    # the graph that must stay retinotopic.
    hex1 = g.hex1.astype(np.int64).copy()
    hex2 = g.hex2.astype(np.int64).copy()
    rows = g.retina
    hex1[rows["idx"]] = rows["hex1"]
    hex2[rows["idx"]] = rows["hex2"]

    # A columnar cell still without a column would merge into a single pooled node, so
    # it is excluded and counted rather than quietly averaged over the whole eye.
    usable &= ~(is_columnar & ((hex1 < 0) | (hex2 < 0)))

    key_hex1 = np.where(is_columnar, hex1, -1)
    key_hex2 = np.where(is_columnar, hex2, -1)
    # Photoreceptors have no side annotation of their own, but the retina table does
    # separate the eyes, so the side is taken from there where it is missing.
    side_idx = g.side_idx.astype(np.int64).copy()
    missing_side = side_idx[rows["idx"]] == 0
    side_idx[rows["idx"][missing_side]] = rows["side"][missing_side]

    keys = np.stack([
        g.type_idx.astype(np.int64),
        side_idx,
        key_hex1 + 1,
        key_hex2 + 1,
    ], axis=1)[usable]

    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    node_of_cell = np.full(g.n, -1, dtype=np.int64)
    node_of_cell[np.flatnonzero(usable)] = inverse
    n_nodes = uniq.shape[0]

    side_names = np.asarray(g.tables["sides"], dtype=object)
    nt_names = np.asarray(g.tables["nts"], dtype=object)

    labels: list[str] = []
    type_name: list[str] = []
    side: list[str] = []
    for t_idx, s_idx, h1, h2 in uniq:
        t = str(types[t_idx])
        s = str(side_names[s_idx]) or "M"
        label = f"{t}_{s}"
        if h1 > 0 or h2 > 0:
            label = f"{label}_h{h1 - 1:02d}_{h2 - 1:02d}"
        labels.append(label)
        type_name.append(t)
        side.append(s)

    # Sign and transmitter come from the member cells. A (type, side) group is one
    # cell type, so its transmitter is shared; where annotation disagrees we take the
    # majority, and an unsigned group stays at 0 so it cannot fake an effect.
    sign = np.zeros(n_nodes, dtype=np.float32)
    nt: list[str] = [""] * n_nodes
    soma = np.full((n_nodes, 3), np.nan, dtype=np.float32)
    cells = np.zeros(n_nodes, dtype=np.int64)
    order = np.argsort(node_of_cell[usable], kind="stable")
    members = np.flatnonzero(usable)[order]
    node_sorted = node_of_cell[members]
    bounds = np.searchsorted(node_sorted, np.arange(n_nodes + 1))
    for node in range(n_nodes):
        group = members[bounds[node]:bounds[node + 1]]
        cells[node] = group.size
        signs = g.nt_sign[group]
        pos, neg = int((signs > 0).sum()), int((signs < 0).sum())
        sign[node] = 1.0 if pos > neg else (-1.0 if neg > pos else 0.0)
        nts, counts = np.unique(g.nt_idx[group], return_counts=True)
        nt[node] = str(nt_names[nts[int(np.argmax(counts))]])
        with np.errstate(invalid="ignore"):
            pts = g.soma[group]
            if np.isfinite(pts).any():
                soma[node] = np.nanmean(pts, axis=0)

    # Sum synapse counts per node pair.
    e_src, e_dst, e_w = g.edge_arrays()
    ns, nd = node_of_cell[e_src], node_of_cell[e_dst]
    untyped = (ns < 0) | (nd < 0)
    # Cells of the same type on the same side connect to each other a great deal, and
    # collapsing them turns those synapses into self-loops. They are dropped rather
    # than kept as a self-edge: within a node the settle already has its own leak and
    # recurrence term, and a self-loop would double-count it.
    within = ~untyped & (ns == nd)
    keep_edge = ~untyped & ~within
    pair = ns[keep_edge] * n_nodes + nd[keep_edge]
    uniq_pair, inv_pair = np.unique(pair, return_inverse=True)
    summed = np.zeros(uniq_pair.size, dtype=np.float64)
    np.add.at(summed, inv_pair, e_w[keep_edge].astype(np.float64))

    return Collapsed(
        node_of_cell=node_of_cell,
        names=labels, type_name=type_name, side=side,
        hex1=(uniq[:, 2] - 1).astype(np.int64), hex2=(uniq[:, 3] - 1).astype(np.int64),
        sign=sign, nt=nt, soma=soma, cells=cells,
        src=(uniq_pair // n_nodes).astype(np.int64),
        dst=(uniq_pair % n_nodes).astype(np.int64),
        weight=summed.astype(np.float32),
        dropped_cells=int((~usable).sum()),
        dropped_untyped_weight=float(e_w[untyped].sum()),
        dropped_within_node_weight=float(e_w[within].sum()),
    )


def every_cell(g: Flyb) -> Collapsed:
    """The whole connectome, one node per neuron. No collapsing, no selection.

    176,422 neurons and 6,287,749 edges, exactly as published. Nothing is pooled, so
    each node is a single traced cell with its own body id, transmitter and soma, and
    every edge is one measured connection between two cells.

    Untyped cells are kept here, unlike in the collapsed graphs. There they would merge
    into an anonymous hub carrying more weight than any real neuron; as individuals they
    are just cells whose type nobody has named yet, and dropping 16,194 real neurons to
    tidy up the labels would be the worse distortion.
    """
    names_arr = g.names()
    nt_names = np.asarray(g.tables["nts"], dtype=object)[g.nt_idx]

    hex1 = g.hex1.astype(np.int64).copy()
    hex2 = g.hex2.astype(np.int64).copy()
    rows = g.retina
    hex1[rows["idx"]] = rows["hex1"]
    hex2[rows["idx"]] = rows["hex2"]
    side_idx = g.side_idx.astype(np.int64).copy()
    missing = side_idx[rows["idx"]] == 0
    side_idx[rows["idx"][missing]] = rows["side"][missing]
    side_table = np.asarray(g.tables["sides"], dtype=object)

    # Body id makes the name unique and traceable back to neuPrint.
    labels: list[str] = []
    type_name: list[str] = []
    side: list[str] = []
    for i in range(g.n):
        t = str(names_arr[i])
        s = str(side_table[side_idx[i]]) or "M"
        labels.append(f"{t or 'cell'}_{s}_{int(g.body_id[i])}")
        type_name.append(t)
        side.append(s)

    e_src, e_dst, e_w = g.edge_arrays()
    keep = e_src != e_dst
    return Collapsed(
        node_of_cell=np.arange(g.n, dtype=np.int64),
        names=labels, type_name=type_name, side=side,
        hex1=hex1, hex2=hex2,
        sign=g.nt_sign.astype(np.float32),
        nt=[str(x) for x in nt_names],
        soma=g.soma.astype(np.float32),
        cells=np.ones(g.n, dtype=np.int64),
        src=e_src[keep], dst=e_dst[keep], weight=e_w[keep].astype(np.float32),
        dropped_cells=0,
        dropped_untyped_weight=0.0,
        dropped_within_node_weight=float(e_w[~keep].sum()),
    )


def _normalised(src, dst, weight, n, by_source: bool):
    """Weights divided by each node's total out- (or in-) weight."""
    total = np.zeros(n, dtype=np.float64)
    np.add.at(total, src if by_source else dst, weight.astype(np.float64))
    total[total == 0] = 1.0
    return weight / total[src if by_source else dst]


def influence(src, dst, weight, n, seeds, hops, forward=True):
    """How much of each seed's signal can reach every node within `hops` steps.

    Propagated along out-degree-normalised weights so a node with many strong inputs
    is not scored simply for being big. The running maximum is kept rather than the
    sum, so a node's score is its best path, not the number of paths.
    """
    w = _normalised(src, dst, weight, n, by_source=forward)
    a, b = (src, dst) if forward else (dst, src)
    reach = np.zeros(n, dtype=np.float64)
    reach[seeds] = 1.0
    best = reach.copy()
    for _ in range(hops):
        step = np.zeros(n, dtype=np.float64)
        np.add.at(step, b, reach[a] * w)
        reach = step
        best = np.maximum(best, reach)
    return best


@dataclass
class Derived:
    """A subgraph ready to be saved in the NPZ contract FlyGraph consumes."""

    names: list[str]
    roles: list[str]
    groups: list[str]
    bodies: list[str]
    sign: np.ndarray
    nt: list[str]
    soma: np.ndarray
    cells: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    routing: dict[str, list[str]]
    polarity: dict[str, int]  # +1 driven by the channel, -1 by its absence
    action_units: dict[str, list[str]]
    unread: list[str]
    ray_sides: list[str]      # eye of each retina ray, in ray order
    provenance: dict

    @property
    def n(self) -> int:
        return len(self.names)


def derive(
    g: Flyb,
    *,
    max_central: int = 4096,
    min_synapses: int = 5,
    hops: int = 4,
    columnar: tuple[str, ...] = COLUMNAR_TYPES,
    resolved: Resolved | None = None,
    retina=None,
    whole: bool = False,
) -> Derived:
    """Extract the visuomotor subgraph between the sensory and motor anchors.

    `retina` is a `sense.retina_map.RetinaMap`, needed so each retinotopic node can be
    routed to the ommatidium that actually covers its column. Without one, a columnar
    graph would have to give every lamina node the whole retina, which is the input
    scrambling this rewrite exists to remove; so it is built by default.
    """
    res = resolved or anchors_mod.resolve(g)
    if retina is None:
        # Always built, even for the pooled variants. A pooled visual node reads its
        # whole eye rather than one ommatidium, but it still has to read *something*:
        # without a retina map there are no rays at all and the graph comes out blind.
        from sense import retina_map
        retina = retina_map.build()
    col = every_cell(g) if whole else collapse(g, columnar=columnar)

    heavy = col.weight >= float(min_synapses)
    src, dst, weight = col.src[heavy], col.dst[heavy], col.weight[heavy]

    def nodes_for(cell_idx: np.ndarray) -> np.ndarray:
        got = col.node_of_cell[cell_idx]
        return np.unique(got[got >= 0])

    sensory_nodes = {k: nodes_for(v) for k, v in res.sensory.items()}
    motor_nodes = {k: nodes_for(v) for k, v in res.motor.items()}

    seeds_in = np.unique(np.concatenate([v for v in sensory_nodes.values() if v.size]))
    seeds_out = np.unique(np.concatenate([v for v in motor_nodes.values() if v.size]))

    anchor_nodes = np.unique(np.concatenate([seeds_in, seeds_out]))
    if whole:
        # Nothing to select: the whole published connectome is the graph.
        keep = np.arange(col.n, dtype=np.int64)
    else:
        fwd = influence(src, dst, weight, col.n, seeds_in, hops, forward=True)
        bwd = influence(src, dst, weight, col.n, seeds_out, hops, forward=False)
        score = fwd * bwd
        # Anchors are kept unconditionally and the budget applies to the central
        # neurons added on top. Counting anchors against the budget would let the
        # retinotopic layers, thousands of nodes on their own, crowd out every
        # interneuron between the eye and the legs.
        candidates = np.setdiff1d(np.flatnonzero(score > 0), anchor_nodes)
        chosen = candidates[np.argsort(-score[candidates])[:max_central]]
        keep = np.unique(np.concatenate([anchor_nodes, chosen]))

    # Induced subgraph: every heavy edge whose endpoints both survived.
    remap = np.full(col.n, -1, dtype=np.int64)
    remap[keep] = np.arange(keep.size)
    on = (remap[src] >= 0) & (remap[dst] >= 0)
    sub_src, sub_dst, sub_w = remap[src[on]], remap[dst[on]], weight[on]

    names = [col.names[i] for i in keep]
    type_name = [col.type_name[i] for i in keep]
    side = [col.side[i] for i in keep]

    sensory_set: dict[int, str] = {}
    for key, nodes in sensory_nodes.items():
        for node in nodes:
            sensory_set.setdefault(int(remap[node]), key)
    motor_set: dict[int, str] = {}
    for key, nodes in motor_nodes.items():
        for node in nodes:
            motor_set.setdefault(int(remap[node]), key)

    roles: list[str] = []
    for i in range(keep.size):
        if i in sensory_set:
            roles.append("sensory")
        elif i in motor_set:
            roles.append("dn")
        else:
            roles.append("central")

    groups = [_group_of(t, sensory_set.get(i), motor_set.get(i))
              for i, t in enumerate(type_name)]
    body_of_population = {m.population: m.body for m in motor_anchors()}
    bodies = [body_of_population.get(motor_set.get(i), "") for i in range(keep.size)]

    # Routing: which observation channels may drive each sensory node. Generated from
    # the anchors so it cannot drift away from the graph it describes.
    channels_of = {a.key: a.channels for a in sensory_anchors()}
    inverting_keys = {a.key for a in sensory_anchors() if a.inverting}
    routing: dict[str, list[str]] = {}
    polarity: dict[str, int] = {}
    unmapped_columns = 0
    for i, key in sensory_set.items():
        channels: list[str] = []
        for channel in channels_of[key]:
            if channel != "retina":
                channels.append(channel)
                continue
            # A retinotopic node reads the one ommatidium covering its own column.
            # Handing it the whole retina would be exactly the scrambling this
            # rewrite exists to remove.
            node = int(keep[i])
            column = (col.side[node], int(col.hex1[node]), int(col.hex2[node]))
            ray = retina.ray_of_column.get(column) if retina else None
            if ray is not None:
                channels.append(f"retina:{ray}")
            elif col.hex1[node] >= 0:
                unmapped_columns += 1      # a column with no measured direction
            else:
                channels.append(f"retina:eye:{col.side[node]}")
        if channels:
            routing[names[i]] = channels
            polarity[names[i]] = -1 if key in inverting_keys else 1

    action_units, unread = _actions(names, side, motor_set)

    provenance = {
        "dataset": g.dataset,
        "meta": g.meta,
        "cells_total": g.n,
        "edges_total": g.n_edges,
        "collapsed_nodes": col.n,
        "collapsed_edges": int(col.weight.size),
        "cells_without_type_dropped": col.dropped_cells,
        "synapses_dropped_untyped": col.dropped_untyped_weight,
        "synapses_dropped_within_node": col.dropped_within_node_weight,
        "min_synapses": min_synapses,
        "hops": hops,
        "max_central": None if whole else max_central,
        "whole_connectome": whole,
        "columnar_types": list(columnar),
        "anchor_nodes": int(anchor_nodes.size),
        "retina": retina.summary() if retina else None,
        "columns_without_measured_direction": unmapped_columns,
        "n_retina_rays": retina.n_rays if retina else 0,
        "sensory_populations": {k: int(v.size) for k, v in sensory_nodes.items()},
        "motor_populations": {k: int(v.size) for k, v in motor_nodes.items()},
        "note": "edges are measured synapse counts; signs are presynaptic "
                "transmitters; no edge is hand-placed",
    }

    return Derived(
        names=names, roles=roles, groups=groups, bodies=bodies,
        sign=col.sign[keep].astype(np.float32),
        nt=[col.nt[i] for i in keep],
        soma=col.soma[keep],
        cells=col.cells[keep],
        src=sub_src, dst=sub_dst, weight=sub_w.astype(np.float32),
        routing=routing, polarity=polarity,
        action_units=action_units, unread=unread,
        ray_sides=[r.side for r in retina.rays] if retina else [],
        provenance=provenance,
    )


def _actions(names, side, motor_set: dict[int, str]) -> tuple[dict, list[str]]:
    """Map each Minecraft action to the descending units allowed to drive it.

    Two populations are read per side rather than as a whole: `DNa02`, because turn
    rate is carried by the right-minus-left difference in its rate. Everything else
    reads the whole population. Grooming, flight amplitude, HS yaw-gaze and the
    unpublished MNnm pool stay in the graph and are not decoded. DNOVS pitch gaze
    (DNp20/DNp22) and FNM2 drive camera_up; ADNM1/2 drive camera_down.
    """
    action_of = {m.population: m.action for m in motor_anchors()}
    split = {m.population for m in motor_anchors() if m.sides_split}

    actions: dict[str, list[str]] = {}
    unread: list[str] = []
    for i, population in motor_set.items():
        name = names[i]
        if population in split:
            action = SIDE_ACTIONS.get((population, side[i]))
            if action is None:            # midline or unknown side drives neither
                unread.append(name)
                continue
            actions.setdefault(action, []).append(name)
            continue
        action = action_of.get(population)
        if action is None:
            unread.append(name)
            continue
        actions.setdefault(action, []).append(name)
    return actions, sorted(unread)


_GROUP_PREFIX = (
    ("ORN_", "antennal_lobe"), ("TRN_", "antennal_lobe"), ("HRN_", "antennal_lobe"),
    ("R1-R6", "retina"), ("R7", "retina"), ("R8", "retina"),
    ("L1", "lamina"), ("L2", "lamina"), ("L3", "lamina"), ("L4", "lamina"),
    ("Mi", "medulla"), ("Tm", "medulla"), ("T4", "medulla"), ("T5", "medulla"),
    ("LC", "optic_glomeruli"), ("LPLC", "optic_glomeruli"), ("LPC", "optic_glomeruli"),
    ("HS", "lobula_plate"), ("VS", "lobula_plate"), ("CH", "lobula_plate"),
    ("BM_", "bristles"), ("SNta", "tactile"), ("SNpp", "proprioceptive"),
    ("SApp", "proprioceptive"), ("JO-", "johnstons_organ"),
    ("LB", "gustatory"), ("PhG", "gustatory"), ("LgLG", "gustatory"),
    ("LgAG", "gustatory"), ("WG", "gustatory"),
    ("KC", "mushroom_body"), ("MBON", "mushroom_body"), ("APL", "mushroom_body"),
    ("EPG", "central_complex"), ("PFL", "central_complex"), ("PFN", "central_complex"),
    ("PEN", "central_complex"), ("Delta7", "central_complex"), ("ER", "central_complex"),
    ("DNp", "descending"), ("DNa", "descending"), ("DNb", "descending"),
    ("DNd", "descending"), ("DNg", "descending"), ("DNge", "descending"),
    ("MDN", "descending"), ("MN", "motor"), ("TTMn", "motor"), ("CvN", "motor"),
    ("AN", "ascending"), ("IN", "interneuron"), ("PS", "interneuron"),
    ("LAL", "lateral_accessory_lobe"), ("GNG", "gnathal"),
)


def _group_of(type_name: str, sensory_key: str | None, motor_key: str | None) -> str:
    if motor_key:
        return f"motor:{motor_key}"
    for prefix, group in _GROUP_PREFIX:
        if type_name.startswith(prefix):
            return group
    return "sensory" if sensory_key else "central"


def save(derived: Derived, path):
    """Write the NPZ that FlyGraph loads, carrying provenance with the topology."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n=derived.n,
        src=derived.src,
        dst=derived.dst,
        sign=derived.sign,
        weight=derived.weight,
        names=np.array(derived.names),
        roles=np.array(derived.roles),
        groups=np.array(derived.groups),
        bodies=np.array(derived.bodies),
        nt=np.array(derived.nt),
        soma=derived.soma,
        cells=derived.cells,
        routing=json.dumps(derived.routing),
        polarity=json.dumps(derived.polarity),
        action_units=json.dumps(derived.action_units),
        unread=np.array(derived.unread),
        ray_sides=np.array(derived.ray_sides),
        n_retina=len(derived.ray_sides),
        provenance=json.dumps(derived.provenance),
    )
    return path
