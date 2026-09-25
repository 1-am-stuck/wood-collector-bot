"""Pack a SensoryFrame (fly-brain-minecraft fields) into a fixed vector."""

from __future__ import annotations

import json
from pathlib import Path

GLOMERULI = [
    "DM1", "DM2", "DM3", "DM4", "DM5",
    "VA2", "VA3", "VA6", "VA1v", "VA1d",
    "DL3", "DL4", "DL5",
    "DC2", "DA1", "DA2",
    "VM1", "VM2", "VM4", "VM5d",
    "VL2a", "VC1", "VC2", "VC5",
    "DP1l", "D", "V",
    "VP2", "VP3a", "VP3b", "VP4", "VP5",
]

GRNS = [
    "LB3b", "LB3c", "LB3a", "LB3d",
    "LB1a", "LB1b", "LB1c", "LB1d", "LB1e",
    "PhG1a", "PhG1b", "PhG1c", "PhG3", "PhG4", "PhG16",
    "LgLG3", "LgLG4", "LgAG1", "WG2",
]

OBJECT_CHS = ["LC4", "LPLC2", "LC11", "LC18", "LC10a", "LC15", "HS"]

MECHANO_KEYS = [
    "windLeft", "windRight", "tilt",
    "soundLow", "soundHigh", "song",
    "touchHead", "touchWing", "touchLegs", "touchNotum", "touchAbdomen",
    "groomDust", "damage", "hot", "cold", "dry", "moist",
    "airborne", "legsOnGround", "wingbeat",
]

FEATURE_GROUPS = {
    "glomeruli": GLOMERULI,
    "grns": GRNS,
    "object_channels": OBJECT_CHS,
    "mechano": MECHANO_KEYS,
}


def _retina_n(frame: dict) -> int:
    lum = frame.get("luminance") or []
    return len(lum)


def frame_to_vector(frame: dict, n_retina: int | None = None) -> list[float]:
    lum = list(frame.get("luminance") or [])
    n = n_retina or len(lum) or 64
    if len(lum) < n:
        lum = lum + [0.0] * (n - len(lum))
    vec = [0.0 if v is None or (isinstance(v, float) and v != v) else float(v) for v in lum[:n]]

    odor = frame.get("odor") or {}
    for g in GLOMERULI:
        vec.append(float(odor.get(g, 0.0)))
    bearing = frame.get("odorBearingDeg")
    if bearing is None:
        vec.extend([0.0, 0.0])
    else:
        import math
        rad = float(bearing) * math.pi / 180.0
        vec.extend([math.sin(rad), math.cos(rad)])

    taste = frame.get("taste") or {}
    for g in GRNS:
        vec.append(float(taste.get(g, 0.0)))

    ch = frame.get("objectChannels") or {}
    for k in OBJECT_CHS:
        vec.append(float(ch.get(k, 0.0)))

    for k in MECHANO_KEYS:
        v = frame.get(k, 0)
        vec.append(1.0 if v is True else 0.0 if v is False else float(v or 0))

    vec.append(float(frame.get("yawRateDegPerS") or 0.0) / 180.0)
    vec.append(float(frame.get("pitchRateDegPerS") or 0.0) / 90.0)
    return vec


def vector_size(n_retina: int = 64) -> int:
    return n_retina + len(GLOMERULI) + 2 + len(GRNS) + len(OBJECT_CHS) + len(MECHANO_KEYS) + 2


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())
