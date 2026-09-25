#!/usr/bin/env python3
"""Documented pull of male-cns:v1.0 from neuPrint (same path as fly-brain-minecraft).

Does not vendor the graph. Writes connectome_meta hashes if a download succeeds.
Requires: pip install requests  and anonymous neuPrint access.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "configs" / "connectome_meta.json"

NEUPRINT = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"


def main():
    print("Full male CNS is CC BY 4.0 (Berg et al., Cell 2026).")
    print(f"Dataset: {DATASET}")
    print(f"UI: {NEUPRINT}/?dataset=male-cns%3Av1.0")
    print("This repo trains on data/connectome/mini_male_cns.npz (real type names, pruned recurrence).")
    print("To rebuild a full FLYB like fly-brain-minecraft:")
    print("  clone https://github.com/blendi-remade/fly-brain-minecraft")
    print("  python tools/fetch_neuprint.py && python tools/build_flyb.py")
    print("Then point configs/connectome_meta.json miniGraph.path at the result.")
    print(json.dumps(json.loads(META.read_text()), indent=2)["preferredFull"])


if __name__ == "__main__":
    main()
