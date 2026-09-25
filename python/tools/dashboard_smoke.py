#!/usr/bin/env python3
"""Drive the dashboard from a fresh policy, with no Minecraft, to check the panes.

Publishes real `FlyPolicy.inspect()` output on a synthetic-but-plausible frame so
the descending-population pane and the body drawing can be inspected without a
Paper server running. Not part of training.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from dashboard import hub  # noqa: E402
from dashboard.app import run  # noqa: E402
from fly_policy.policy import FlyPolicy, load_graph  # noqa: E402
from sense.frame import GLOMERULI, GRNS, MECHANO_KEYS, OBJECT_CHS, vector_size  # noqa: E402


def ray_azimuths(n: int) -> list[float]:
    """Where each ommatidium looks, so the fake trunk sweeps the eye geometrically.

    The rays are ordered by the farthest-point sampling that chose them, not by angle,
    so a stimulus indexed by ray number would scatter over both eyes instead of moving.
    """
    cfg = json.loads((ROOT / "configs/sense/retina_columns.json").read_text())
    return [float(r["azimuthRightDeg"]) for r in cfg["rays"][:n]]


def main(port: int = 8767) -> None:
    # The `path` variant: real male-cns cells with a retinotopic eye, small enough to
    # settle inside a 15 Hz frame budget. The full 176k graph runs at about 6 Hz, which
    # is what live training uses, but makes a jerky smoke test.
    graph = load_graph(ROOT, "path")
    n_retina = graph.n_retina
    model = FlyPolicy(graph, vector_size(n_retina))
    azimuth = ray_azimuths(n_retina)
    hub.publish_topology(model.topology())
    threading.Thread(target=run, kwargs={"port": port}, daemon=True).start()
    print(f"http://127.0.0.1:{port}/  ({graph.n:,} neurons, {graph.n_edges:,} edges)")

    step = 0
    while True:
        step += 1
        phase = step * 0.12
        obs = torch.zeros(vector_size(n_retina))
        # A dark trunk sweeping across the field of view, plus an occasional sugar
        # contact. Bright sky is 0.85; the trunk cuts luminance where it covers.
        trunk = 55.0 * math.sin(phase)
        for i, az in enumerate(azimuth):
            edge = math.exp(-((az - trunk) ** 2) / 200.0)
            obs[i] = 0.85 - 0.8 * edge
        g0 = n_retina
        obs[g0 + GLOMERULI.index("DM1")] = 0.4 + 0.3 * math.sin(phase * 0.5)
        obs[g0 + len(GLOMERULI)] = math.sin(phase)
        obs[g0 + len(GLOMERULI) + 1] = math.cos(phase)
        base = g0 + len(GLOMERULI) + 2
        if step % 17 == 0:
            obs[base + GRNS.index("LB3b")] = 1.0
        obs[base + len(GRNS) + OBJECT_CHS.index("LPLC2")] = max(0.0, math.sin(phase * 0.3))
        mech0 = base + len(GRNS) + len(OBJECT_CHS)
        obs[mech0 + MECHANO_KEYS.index("legsOnGround")] = 1.0
        obs[mech0 + MECHANO_KEYS.index("touchHead")] = 1.0 if step % 23 == 0 else 0.0

        brain = model.inspect(obs, greedy=True)
        frame = {
            "luminance": [float(x) for x in obs[:n_retina]],
            "odor": {g: float(obs[g0 + i]) for i, g in enumerate(GLOMERULI)},
            "taste": {g: float(obs[base + i]) for i, g in enumerate(GRNS)},
            "objectChannels": {
                c: float(obs[base + len(GRNS) + i]) for i, c in enumerate(OBJECT_CHS)
            },
            **{k: float(obs[mech0 + i]) for i, k in enumerate(MECHANO_KEYS)},
        }
        hub.publish({
            "t": time.time(),
            "mode": "play",
            "step": step,
            "action": brain["action"],
            "value": brain["value"],
            "reward": None,
            "frame": frame,
            "brain": brain,
        })
        time.sleep(1 / 15)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8767)
