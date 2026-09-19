"""Derive the policy graph from male-cns:v1.0. Replaces the hand-drawn mini graph.

    uv run python python/tools/build_graph_from_flyb.py --variant full

Variants differ only in how many nodes are kept; the wiring rule is identical.

    full   176,422 neurons   the published male-cns:v1.0 connectome, default
    path   ~10k nodes        visuomotor subgraph with retinotopic lamina
    type   ~1.5k nodes       fast tests, vision pooled per eye
    wide   ~12k central      larger subgraph

There is no hand-built fallback. If the FLYB file is missing this tool says so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connectome import anchors as anchors_mod  # noqa: E402
from connectome import flyb, subgraph  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# `max_central` counts only the interneurons added between the anchors; the anchors
# themselves are always kept, so the node total is larger than this number.
VARIANTS = {
    # Vision pooled per eye: no columnar retina, so the graph is small enough to
    # iterate on in seconds. Retinotopy is lost, which is why it is not the default.
    "type": dict(max_central=1024, hops=3, columnar=()),
    "path": dict(max_central=4096, hops=4, columnar=subgraph.COLUMNAR_TYPES),
    "wide": dict(max_central=12288, hops=5, columnar=subgraph.COLUMNAR_TYPES),
    # The published connectome, untouched: one node per traced neuron, one edge per
    # measured connection. No type collapsing and no subgraph selection at all.
    "full": dict(whole=True, columnar=subgraph.COLUMNAR_TYPES),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    ap.add_argument("--flyb", default=None, help="path to malecns-v1.0.flyb.gz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-synapses", type=int, default=5,
                    help="edge weight floor; the published file is already >=5")
    ap.add_argument("--max-central", type=int, default=None,
                    help="interneurons kept between the anchors; anchors are extra")
    ap.add_argument("--report", default=None, help="write the provenance JSON here")
    args = ap.parse_args()

    cfg = dict(VARIANTS[args.variant])
    if args.max_central:
        cfg["max_central"] = args.max_central
    out = Path(args.out) if args.out else \
        ROOT / "data" / "connectome" / f"malecns_{args.variant}.npz"

    print(f"reading {args.flyb or flyb.default_path()}")
    g = flyb.load(args.flyb)
    print(f"  {g.dataset}: {g.n:,} cells, {g.n_edges:,} edges")

    resolved = anchors_mod.resolve(g)
    print(f"  anchors: {len(resolved.sensory)} sensory populations "
          f"({sum(v.size for v in resolved.sensory.values()):,} cells), "
          f"{len(resolved.motor)} motor populations "
          f"({sum(v.size for v in resolved.motor.values())} cells)")

    derived = subgraph.derive(g, min_synapses=args.min_synapses,
                              resolved=resolved, **cfg)
    path = subgraph.save(derived, out)

    prov = derived.provenance
    print(f"\n{args.variant}: {derived.n:,} nodes, {derived.src.size:,} edges")
    print(f"  collapsed graph was {prov['collapsed_nodes']:,} nodes / "
          f"{prov['collapsed_edges']:,} edges")
    print(f"  synapses on kept edges: {derived.weight.sum():,.0f}")
    roles = {r: derived.roles.count(r) for r in ("sensory", "central", "dn")}
    print(f"  roles: {roles}")
    signs = np.asarray(derived.sign)
    print(f"  signs: +{int((signs > 0).sum())} excitatory, "
          f"{int((signs < 0).sum())} inhibitory, {int((signs == 0).sum())} unknown")
    print(f"  actions decoded: {len(derived.action_units)}; "
          f"unread outputs kept: {len(derived.unread)}")
    for action, units in sorted(derived.action_units.items()):
        print(f"    {action:12s} <- {', '.join(units)}")
    print(f"\nwrote {path.relative_to(ROOT)} "
          f"({path.stat().st_size / 1e6:.1f} MB)")

    if args.report:
        Path(args.report).write_text(json.dumps(prov, indent=2))
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
