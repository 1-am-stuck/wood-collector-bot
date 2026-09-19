"""Frozen connectome topology, derived from male-cns:v1.0.

A graph built by `python/tools/build_graph_from_flyb.py` carries more than topology:
the synapse count behind every edge, the presynaptic transmitter, soma coordinates,
and the routing that says which observation channels may drive each sensory node.
Carrying the routing with the graph is deliberate — the two are one artefact, and a
graph rebuilt with a different retina would otherwise be read through a stale map.

The older hand-built graph has none of those fields, so all of them are optional and
absent ones fall back to something honest: no weights, no transmitter, routing looked
up from `sense.routing` instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FlyGraph:
    def __init__(self, path: str | Path | None = None, data: dict | None = None):
        if data is None:
            if path is None:
                raise ValueError("path or data required")
            raw = np.load(path, allow_pickle=True)
            data = {k: raw[k] for k in raw.files}
        self.n = int(data["n"])
        self.src = np.asarray(data["src"], dtype=np.int64)
        self.dst = np.asarray(data["dst"], dtype=np.int64)
        self.sign = np.asarray(data["sign"], dtype=np.float32)
        self.names = [str(x) for x in np.asarray(data["names"]).tolist()]
        self.roles = [str(x) for x in np.asarray(data["roles"]).tolist()]
        # Anatomical group and acted-on body part, for the dashboard. Optional so
        # older checkpoints' graphs still load.
        self.groups = (
            [str(x) for x in np.asarray(data["groups"]).tolist()]
            if "groups" in data else list(self.roles)
        )
        self.bodies = (
            [str(x) for x in np.asarray(data["bodies"]).tolist()]
            if "bodies" in data else [""] * self.n
        )
        # Measured synapse count per edge. Present only on a derived graph; it is what
        # lets the network start at the connectome's own relative strengths instead of
        # at a uniform gain of 1.0.
        self.weight = (
            np.asarray(data["weight"], dtype=np.float32) if "weight" in data else None
        )
        self.nt = (
            [str(x) for x in np.asarray(data["nt"]).tolist()] if "nt" in data else None
        )
        self.soma = (
            np.asarray(data["soma"], dtype=np.float32) if "soma" in data else None
        )
        self.cells = (
            np.asarray(data["cells"], dtype=np.int64) if "cells" in data else None
        )
        self.routing = _json(data, "routing", {})
        # +1 where a sensory node is driven by its channel, -1 where it is driven by the
        # channel's absence. Lamina monopolars are the -1 case: histaminergic
        # photoreceptors invert luminance, so a contrast decrement is what excites them.
        self.polarity = _json(data, "polarity", {})
        self.action_units = _json(data, "action_units", {})
        self.unread = (
            [str(x) for x in np.asarray(data["unread"]).tolist()]
            if "unread" in data else []
        )
        self.ray_sides = (
            [str(x) for x in np.asarray(data["ray_sides"]).tolist()]
            if "ray_sides" in data else []
        )
        self.n_retina = int(data["n_retina"]) if "n_retina" in data else 0
        self.provenance = _json(data, "provenance", {})
        self.sensory_idx = np.array([i for i, r in enumerate(self.roles) if r == "sensory"], dtype=np.int64)
        self.dn_idx = np.array([i for i, r in enumerate(self.roles) if r == "dn"], dtype=np.int64)
        self.central_idx = np.array([i for i, r in enumerate(self.roles) if r in ("central", "dn")], dtype=np.int64)
        self.n_edges = int(self.src.shape[0])

    @property
    def derived(self) -> bool:
        """True when this graph came from the connectome rather than by hand."""
        return self.weight is not None and bool(self.routing)

    @property
    def node_signed(self) -> bool:
        """True when sign is stored per neuron, which is what Dale's law means."""
        return self.sign.shape[0] == self.n and self.n != self.n_edges

    def edge_sign(self) -> np.ndarray:
        """Sign of every edge, taken from its presynaptic neuron.

        A neuron releases one transmitter, so its sign belongs to it and every edge it
        makes carries that sign. Storing sign per neuron and expanding here makes that
        structural: there is no way to express a neuron that excites one target and
        inhibits another, which is exactly the error the hand-built graph made when it
        had cholinergic `DNa02` inhibiting its mirror twin.

        A legacy hand-built graph already stores sign per edge, and is passed through.
        """
        if not self.node_signed:
            return self.sign
        return self.sign[self.src]

    def initial_log_gain(self, fan_in: float = 1.0) -> np.ndarray | None:
        """Per-edge starting gain: each edge's share of its target's total input.

        `W = sign * exp(log_gain)`, so the gain is an absolute weight and raw synapse
        counts cannot be used directly. They span five orders of magnitude here — a
        5-synapse edge up to over 100,000 once a 162-cell population is collapsed into
        one node — and feeding that in raw makes the settle diverge immediately.

        What carries the biology is the *relative* strength of a neuron's inputs, not
        their absolute count: a cell that gets 60% of its drive from LC4 behaves like
        one that gets 60% of its drive from LC4 whether that is 600 synapses or 6,000.
        So each edge starts at its fraction of its target's total incoming synapses,
        which makes every neuron's total input weight `fan_in` and leaves the measured
        proportions intact. Absolute drive is then the network's to learn, through
        `log_gain` and the homeostatic terms.

        Returns None for a hand-built graph, which has no counts to start from.
        """
        if self.weight is None:
            return None
        total = np.zeros(self.n, dtype=np.float64)
        np.add.at(total, self.dst, self.weight.astype(np.float64))
        total[total <= 0] = 1.0
        share = self.weight.astype(np.float64) / total[self.dst]
        return np.log(share * float(fan_in)).astype(np.float32)

    def sparse_w(self, log_gain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.sign * np.exp(log_gain)
        return self.src, self.dst, w

    def obs_routing(self, n_retina: int) -> dict[str, list[int]] | None:
        """Observation indices each sensory node may read, or None if not derived."""
        if not self.routing:
            return None
        from sense.channels import resolve
        return resolve(self.routing, n_retina, self.ray_sides or None)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            n=self.n,
            src=self.src,
            dst=self.dst,
            sign=self.sign,
            names=np.array(self.names),
            roles=np.array(self.roles),
            groups=np.array(self.groups),
            bodies=np.array(self.bodies),
        )

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    def synapses(self, src_name: str, dst_name: str) -> float:
        """Measured synapse count between two nodes, so claims can be checked."""
        if self.weight is None:
            raise ValueError("this graph carries no synapse counts")
        a, b = self.index_of(src_name), self.index_of(dst_name)
        hit = (self.src == a) & (self.dst == b)
        return float(self.weight[hit].sum())


def _json(data: dict, key: str, default):
    if key not in data:
        return default
    raw = data[key]
    text = raw.item() if hasattr(raw, "item") and raw.ndim == 0 else raw
    return json.loads(str(text))
