#!/usr/bin/env python3
"""Build a frozen mini connectome that keeps real male-cns sensory / DN names.

This is a documented prune for Mineflayer training, not a substitute for the
176k-neuron graph. Sensory and descending identities stay; hidden units stand
in for central-brain recurrence. Fetch the full graph with fetch_connectome.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

SENSORY = [
    # ORNs (one unit per glomerulus used by SenseBridge tables)
    "ORN_DM1", "ORN_DM2", "ORN_VA2", "ORN_DL5", "ORN_DC2", "ORN_DA2",
    "ORN_VM1", "ORN_V", "ORN_VL2a", "ORN_DP1l", "ORN_VP2", "ORN_VP5",
    # GRNs
    "LB3b", "LB3c", "LB3a", "LB1a", "LgLG3", "LgAG1",
    # vision / object
    "R1-6", "L2", "LC4", "LPLC2", "LC11", "LC10a",
    # mechano
    "JO-C", "JO-F", "BM_InOm", "hair_plate",
    "TRN_VP2", "HRN_VP5",
]

CENTRAL = [f"CB_{i:02d}" for i in range(24)]

DN = [
    "DNp09", "DNa02_L", "DNa02_R", "MDN", "DNg60",
    "DNp01", "MN9", "aDN1", "DNg02",
]


def build():
    names = SENSORY + CENTRAL + DN
    roles = (["sensory"] * len(SENSORY) + ["central"] * len(CENTRAL) + ["dn"] * len(DN))
    n = len(names)
    src, dst, sign = [], [], []

    def edge(a, b, s=1.0):
        src.append(a)
        dst.append(b)
        sign.append(s)

    # Sensory → nearby central (deterministic, Dale-like mixed signs)
    rng = np.random.default_rng(7)
    for i, name in enumerate(SENSORY):
        for k in range(3):
            j = len(SENSORY) + int(rng.integers(0, len(CENTRAL)))
            s = 1.0 if (i + k) % 5 else -1.0
            edge(i, j, s)
        # a few skip connections onto DNs
        if name.startswith("ORN_") or name in ("LC4", "LPLC2", "LC11"):
            edge(i, n - len(DN) + int(rng.integers(0, 4)), 1.0)
        if name in ("LB3b", "LB3c", "LgLG3"):
            edge(i, names.index("MN9"), 1.0)
        if name == "LC4":
            edge(i, names.index("DNp01"), 1.0)
        if name == "JO-C":
            edge(i, names.index("DNp09"), 1.0)

    # Central recurrence
    for i, ci in enumerate(range(len(SENSORY), len(SENSORY) + len(CENTRAL))):
        for k in range(4):
            j = len(SENSORY) + (i + 1 + k * 3) % len(CENTRAL)
            edge(ci, j, 1.0 if k % 2 == 0 else -1.0)
        edge(ci, n - len(DN) + (i % len(DN)), 1.0 if i % 3 else -1.0)

    # DN lateral
    dn0 = n - len(DN)
    edge(names.index("DNa02_L"), names.index("DNa02_R"), -1.0)
    edge(names.index("DNa02_R"), names.index("DNa02_L"), -1.0)
    edge(names.index("DNp09"), names.index("MDN"), -1.0)

    data = {
        "n": n,
        "src": np.array(src, dtype=np.int64),
        "dst": np.array(dst, dtype=np.int64),
        "sign": np.array(sign, dtype=np.float32),
        "names": np.array(names),
        "roles": np.array(roles),
    }
    out = ROOT / "data" / "connectome" / "mini_male_cns.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)
    print(f"wrote {out} n={n} edges={len(src)}")
    return out


if __name__ == "__main__":
    build()
