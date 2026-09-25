"""The fly's real eye: 1,772 measured ommatidial viewing directions.

The retina this repo used before was invented — 8 azimuths by 4 elevations per eye,
on a grid I chose. This replaces it with the measured optics. `column_directions.csv`
in fly-brain-minecraft gives a viewing direction for every medulla column of
male-cns: 1,682 of the 1,772 matched to the Zhao micro-CT eye map and 90 filled in by
local linear extrapolation. Each row is a unit vector in head coordinates plus the
azimuth and elevation it corresponds to.

Those columns join onto the connectome almost exactly: 1,766 of the 1,767 lamina L1
columns in male-cns appear in the file. So a lamina node's viewing direction is a
measurement, and the retinotopy is the fly's own.

## Why we do not cast 1,772 rays

One ray per column at 15 Hz is far more raycasting than the Minecraft loop can afford.
Instead a subset of **real ommatidia** is sampled and each remaining column reads its
nearest sampled neighbour — the same sharing the reference mod does. The subset is
chosen by farthest-point sampling over the measured directions, which spreads the
samples evenly over the eye's actual field of view rather than over a grid.

This matters for honesty in two ways. The sampled directions are real ommatidial
directions, not centroids of anything. And the cost of the approximation is measured,
not assumed: `angular_error_deg` reports how far each column's true direction is from
the ray it reads, so the blur introduced is a number we can quote instead of a hope.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CSV = (
    ROOT / ".research" / "fly-brain-minecraft" / "src" / "main" / "resources"
    / "assets" / "fruitfly" / "brain" / "column_directions.csv"
)
# Generated config the JS sense bridge reads, so both languages cast the same rays.
CONFIG = ROOT / "configs" / "sense" / "retina_columns.json"

# Rays per eye. 64 per eye keeps the raycast budget at twice the old 64-column total
# while covering a real 1,772-column eye; the resulting blur is reported by
# `summary()` rather than assumed to be acceptable.
RAYS_PER_EYE = 64


@dataclass
class Column:
    side: str
    hex1: int
    hex2: int
    front: float
    left: float
    up: float
    az_ipsi: float
    el: float
    source: str

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.side, self.hex1, self.hex2)

    @property
    def vec(self) -> tuple[float, float, float]:
        return (self.front, self.left, self.up)

    @property
    def azimuth_right_deg(self) -> float:
        """Signed azimuth with positive to the fly's right.

        The repo's olfactory bearing already uses that convention, so vision matches
        it. The file's own `az_deg_ipsi` is measured toward each eye's own side, which
        would put the two eyes in mirrored frames.
        """
        return math.degrees(math.atan2(-self.left, self.front))

    @property
    def elevation_deg(self) -> float:
        return math.degrees(math.asin(max(-1.0, min(1.0, self.up))))


def load_columns(path: str | Path | None = None) -> list[Column]:
    src = Path(path) if path else DEFAULT_CSV
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. It ships with fly-brain-minecraft; clone it into "
            ".research/ (gitignored). See docs/SENSE_PROVENANCE.md."
        )
    out: list[Column] = []
    with src.open() as handle:
        for row in csv.DictReader(handle):
            out.append(Column(
                side=row["side"], hex1=int(row["hex1"]), hex2=int(row["hex2"]),
                front=float(row["dir_front"]), left=float(row["dir_left"]),
                up=float(row["dir_up"]), az_ipsi=float(row["az_deg_ipsi"]),
                el=float(row["el_deg"]), source=row["source"],
            ))
    return out


def farthest_point_order(vecs: np.ndarray) -> np.ndarray:
    """Order directions so each one is as far as possible from all those before it.

    Greedy farthest-point traversal on the sphere. Taking the first k gives an even
    spread over whatever field of view the data actually covers, with no assumption
    that the eye is a rectangle.
    """
    n = vecs.shape[0]
    centre = vecs.mean(axis=0)
    centre /= np.linalg.norm(centre) or 1.0
    first = int(np.argmax(vecs @ centre))
    chosen = [first]
    # Angular distance stands in for cosine similarity; both are monotone in the angle.
    nearest = 1.0 - vecs @ vecs[first]
    for _ in range(n - 1):
        pick = int(np.argmax(nearest))
        chosen.append(pick)
        nearest = np.minimum(nearest, 1.0 - vecs @ vecs[pick])
    return np.asarray(chosen, dtype=np.int64)


@dataclass
class RetinaMap:
    """Which rays we cast, and which ray each real column reads."""

    rays: list[Column]                          # sampled ommatidia, in ray order
    ray_of_column: dict[tuple[str, int, int], int]
    error_deg: dict[tuple[str, int, int], float]
    rays_per_eye: int

    @property
    def n_rays(self) -> int:
        return len(self.rays)

    def summary(self) -> dict:
        errs = np.asarray(list(self.error_deg.values()), dtype=np.float64)
        sources = {}
        for ray in self.rays:
            sources[ray.source] = sources.get(ray.source, 0) + 1
        az = [r.azimuth_right_deg for r in self.rays]
        el = [r.elevation_deg for r in self.rays]
        return {
            "n_rays": self.n_rays,
            "rays_per_eye": self.rays_per_eye,
            "columns": len(self.ray_of_column),
            "angular_error_deg": {
                "mean": float(errs.mean()), "median": float(np.median(errs)),
                "p95": float(np.percentile(errs, 95)), "max": float(errs.max()),
            },
            "ray_sources": sources,
            "azimuth_right_deg": [min(az), max(az)],
            "elevation_deg": [min(el), max(el)],
        }


def build(columns: list[Column] | None = None,
          rays_per_eye: int = RAYS_PER_EYE) -> RetinaMap:
    """Sample `rays_per_eye` real ommatidia per eye and map every column onto them."""
    cols = columns if columns is not None else load_columns()
    rays: list[Column] = []
    ray_of: dict[tuple[str, int, int], int] = {}
    error: dict[tuple[str, int, int], float] = {}

    for side in ("L", "R"):
        eye = [c for c in cols if c.side == side]
        if not eye:
            continue
        vecs = np.asarray([c.vec for c in eye], dtype=np.float64)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        order = farthest_point_order(vecs)
        picked = order[:min(rays_per_eye, len(eye))]
        base = len(rays)
        rays.extend(eye[int(i)] for i in picked)

        # Every column, sampled or not, reads the nearest sampled direction.
        chosen_vecs = vecs[picked]
        cos = np.clip(vecs @ chosen_vecs.T, -1.0, 1.0)
        best = np.argmax(cos, axis=1)
        for local, (col, pick) in enumerate(zip(eye, best)):
            ray_of[col.key] = base + int(pick)
            error[col.key] = math.degrees(math.acos(float(cos[local, pick])))

    return RetinaMap(rays=rays, ray_of_column=ray_of, error_deg=error,
                     rays_per_eye=rays_per_eye)


def write_config(retina: RetinaMap, path: str | Path | None = None) -> Path:
    """Emit the ray directions for the JS sense bridge.

    Only the rays are written, not the 1,772-entry column map: Minecraft only needs
    to know where to look. The column map stays on the Python side, where it becomes
    the routing from each ray onto the lamina node that reads it.
    """
    out = Path(path) if path else CONFIG
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_provenance": (
            "Measured ommatidial viewing directions from male-cns / Zhao micro-CT, "
            "via fly-brain-minecraft column_directions.csv. Generated by "
            "python/sense/retina_map.py -- do not edit by hand."
        ),
        "raysPerEye": retina.rays_per_eye,
        "count": retina.n_rays,
        "summary": retina.summary(),
        "rays": [
            {
                "side": r.side, "hex1": r.hex1, "hex2": r.hex2,
                "azimuthRightDeg": round(r.azimuth_right_deg, 3),
                "elevationDeg": round(r.elevation_deg, 3),
                "dir": [round(r.front, 6), round(r.left, 6), round(r.up, 6)],
                "source": r.source,
            }
            for r in retina.rays
        ],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def load_config(path: str | Path | None = None) -> dict:
    src = Path(path) if path else CONFIG
    return json.loads(src.read_text())


if __name__ == "__main__":
    retina = build()
    written = write_config(retina)
    summary = retina.summary()
    print(json.dumps(summary, indent=2))
    print(f"wrote {written.relative_to(ROOT)}")
