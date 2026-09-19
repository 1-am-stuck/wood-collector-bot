"""Read the male-cns FLYB binary: the real connectome, 176k neurons and 6.3M edges.

FLYB is the compact form built by fly-brain-minecraft's `tools/build_flyb.py` from
an anonymous neuPrint pull of **male-cns:v1.0** -- the dataset published as Berg et
al., *Sexual dimorphism in the complete Drosophila male central nervous system
connectome*, Cell 189(18) 2026, doi:10.1016/j.cell.2026.08.015. That is the
Janelia / Google Research reconstruction: 166k+ neurons and 125M synapses as
published, 176,422 neurons and 6,287,749 edges at the >=5-synapse threshold used
here. CC BY 4.0.

We read it rather than re-derive it, and we do not commit it: the file lives in the
gitignored research clone, or is rebuilt with `tools/fetch_neuprint.py`. See
`python/tools/fetch_connectome.py` for provenance and `docs/SENSE_PROVENANCE.md`.

Layout (little-endian, whole file gzipped) is specified in that builder; this module
is the mirror image of its writer. Everything is returned as numpy arrays that
share no state with the file, so the ~47 MB decompressed payload can be dropped.
"""

from __future__ import annotations

import gzip
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Where the prebuilt binary usually is. The research clone is gitignored, so this
# is a convenience, not a guarantee.
DEFAULT_PATHS = (
    ROOT / "data" / "connectome" / "malecns-v1.0.flyb.gz",
    ROOT / ".research" / "fly-brain-minecraft" / "src" / "main" / "resources"
    / "connectome" / "malecns-v1.0.flyb.gz",
)

TABLE_NAMES = (
    "types", "superclasses", "classes", "subclasses", "nts", "sides",
    "dimorphisms", "fruDsx", "neuromeres", "nerves",
)

# Retina `kind` codes, in the builder's order.
RETINA_KINDS = ("R1-R6", "R7", "R8", "L1", "L2", "L3")


@dataclass
class Flyb:
    """The whole connectome, as struct-of-arrays plus a CSR edge list."""

    dataset: str
    meta: dict
    tables: dict[str, list[str]]

    body_id: np.ndarray        # int64[n]
    type_idx: np.ndarray       # int32[n]  -> tables["types"]
    superclass_idx: np.ndarray
    class_idx: np.ndarray
    subclass_idx: np.ndarray
    nt_idx: np.ndarray
    nt_sign: np.ndarray        # int8[n]: +1 excitatory, -1 inhibitory, 0 unknown
    side_idx: np.ndarray
    hex1: np.ndarray           # int8[n]: medulla column coords, -1 when absent
    hex2: np.ndarray
    dimorphism_idx: np.ndarray
    fru_dsx_idx: np.ndarray
    neuromere_idx: np.ndarray
    nerve_idx: np.ndarray
    soma: np.ndarray           # float32[n, 3], NaN when absent
    pre_synapses: np.ndarray
    post_synapses: np.ndarray

    row_ptr: np.ndarray        # int32[n + 1], CSR over presynaptic neuron
    post_idx: np.ndarray       # int32[n_edges]
    weight: np.ndarray         # uint16[n_edges], synapse count

    retina: np.ndarray         # structured: idx, side, hex1, hex2, kind

    @property
    def n(self) -> int:
        return int(self.body_id.size)

    @property
    def n_edges(self) -> int:
        return int(self.post_idx.size)

    def names(self) -> np.ndarray:
        """Per-neuron type string, e.g. 'DNp09'. Empty when the type is unnamed."""
        return np.asarray(self.tables["types"], dtype=object)[self.type_idx]

    def sides(self) -> np.ndarray:
        return np.asarray(self.tables["sides"], dtype=object)[self.side_idx]

    def of_type(self, *type_names: str) -> np.ndarray:
        """Indices of every neuron whose type is exactly one of `type_names`."""
        wanted = {
            self.tables["types"].index(t)
            for t in type_names
            if t in self.tables["types"]
        }
        if not wanted:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(np.isin(self.type_idx, list(wanted)))

    def with_type_prefix(self, prefix: str) -> np.ndarray:
        """Indices of every neuron whose type starts with `prefix` (e.g. 'DNg02')."""
        hits = [i for i, t in enumerate(self.tables["types"]) if t.startswith(prefix)]
        if not hits:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(np.isin(self.type_idx, hits))

    def out_edges(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(postsynaptic indices, synapse counts) for one presynaptic neuron."""
        lo, hi = int(self.row_ptr[i]), int(self.row_ptr[i + 1])
        return self.post_idx[lo:hi], self.weight[lo:hi]

    def edge_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand CSR into flat (src, dst, weight), which is what torch wants."""
        counts = np.diff(self.row_ptr.astype(np.int64))
        src = np.repeat(np.arange(self.n, dtype=np.int64), counts)
        return src, self.post_idx.astype(np.int64), self.weight.astype(np.float32)


class _Cursor:
    """Sequential reader over the decompressed payload, with zero-copy arrays."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.at = 0

    def take(self, count: int) -> memoryview:
        end = self.at + count
        if end > len(self.buf):
            raise ValueError("FLYB truncated")
        chunk = memoryview(self.buf)[self.at:end]
        self.at = end
        return chunk

    def struct(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))

    def array(self, dtype: str, count: int) -> np.ndarray:
        dt = np.dtype(dtype)
        return np.frombuffer(self.take(dt.itemsize * count), dtype=dt).copy()

    def string(self, length_fmt: str) -> str:
        (length,) = self.struct(length_fmt)
        return bytes(self.take(length)).decode("utf-8")


def default_path() -> Path:
    for candidate in DEFAULT_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "malecns-v1.0.flyb.gz not found. Either clone "
        "https://github.com/blendi-remade/fly-brain-minecraft into .research/ , or "
        "rebuild it there with `python tools/fetch_neuprint.py && python "
        "tools/build_flyb.py` (anonymous neuPrint access, about two minutes). "
        "See python/tools/fetch_connectome.py."
    )


def load(path: str | Path | None = None) -> Flyb:
    """Decompress and parse the whole connectome. Takes a few seconds and ~250 MB."""
    src = Path(path) if path else default_path()
    with gzip.open(src, "rb") as handle:
        payload = handle.read()

    cur = _Cursor(payload)
    magic = bytes(cur.take(4))
    if magic != b"FLYB":
        raise ValueError(f"not a FLYB file: {magic!r}")
    version, n, n_edges, n_retina = cur.struct("<IIII")
    if version != 1:
        raise ValueError(f"unsupported FLYB version {version}")

    dataset = cur.string("<H")
    meta = json.loads(cur.string("<I"))

    tables: dict[str, list[str]] = {}
    for name in TABLE_NAMES:
        (count,) = cur.struct("<H")
        tables[name] = [cur.string("<H") for _ in range(count)]

    body_id = cur.array("<i8", n)
    type_idx = cur.array("<i4", n)
    superclass_idx = cur.array("u1", n)
    class_idx = cur.array("u1", n)
    subclass_idx = cur.array("<u2", n)
    nt_idx = cur.array("u1", n)
    nt_sign = cur.array("i1", n)
    side_idx = cur.array("u1", n)
    hex1 = cur.array("i1", n)
    hex2 = cur.array("i1", n)
    dimorphism_idx = cur.array("u1", n)
    fru_dsx_idx = cur.array("u1", n)
    neuromere_idx = cur.array("u1", n)
    nerve_idx = cur.array("u1", n)
    soma = cur.array("<f4", n * 3).reshape(n, 3)
    pre_synapses = cur.array("<i4", n)
    post_synapses = cur.array("<i4", n)

    row_ptr = cur.array("<i4", n + 1)
    post_idx = cur.array("<i4", n_edges)
    weight = cur.array("<u2", n_edges)

    retina_dt = np.dtype([("idx", "<i4"), ("side", "u1"), ("hex1", "i1"),
                          ("hex2", "i1"), ("kind", "u1")])
    retina = np.frombuffer(cur.take(retina_dt.itemsize * n_retina), dtype=retina_dt).copy()

    return Flyb(
        dataset=dataset, meta=meta, tables=tables,
        body_id=body_id, type_idx=type_idx, superclass_idx=superclass_idx,
        class_idx=class_idx, subclass_idx=subclass_idx, nt_idx=nt_idx,
        nt_sign=nt_sign, side_idx=side_idx, hex1=hex1, hex2=hex2,
        dimorphism_idx=dimorphism_idx, fru_dsx_idx=fru_dsx_idx,
        neuromere_idx=neuromere_idx, nerve_idx=nerve_idx, soma=soma,
        pre_synapses=pre_synapses, post_synapses=post_synapses,
        row_ptr=row_ptr, post_idx=post_idx, weight=weight, retina=retina,
    )
