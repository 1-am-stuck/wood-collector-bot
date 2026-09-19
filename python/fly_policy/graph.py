"""Frozen connectome topology. Mini graph uses real male-cns type names."""

from __future__ import annotations

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
        self.sensory_idx = np.array([i for i, r in enumerate(self.roles) if r == "sensory"], dtype=np.int64)
        self.dn_idx = np.array([i for i, r in enumerate(self.roles) if r == "dn"], dtype=np.int64)
        self.central_idx = np.array([i for i, r in enumerate(self.roles) if r in ("central", "dn")], dtype=np.int64)
        self.n_edges = int(self.src.shape[0])

    def sparse_w(self, log_gain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.sign * np.exp(log_gain)
        return self.src, self.dst, w

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
        )
