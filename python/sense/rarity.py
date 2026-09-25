"""Explored-region wood census → rarest log type.

Rarest = fewest occurrences among log types already seen (count > 0).
Ties break alphabetically so the goal is deterministic.
"""

from __future__ import annotations

WOOD_TYPES = [
    "oak", "spruce", "birch", "jungle", "acacia",
    "dark_oak", "mangrove", "cherry", "pale_oak",
]
WOOD_LOGS = [f"{t}_log" for t in WOOD_TYPES]


def strip_ns(name: str) -> str:
    return str(name or "").replace("minecraft:", "")


def is_log(name: str) -> bool:
    return strip_ns(name) in WOOD_LOGS


def block_in_explored(block: dict, visited, radius: float) -> bool:
    if not visited:
        return False
    bx, bz = float(block["x"]), float(block["z"])
    r2 = radius * radius
    for vx, vz in visited:
        dx, dz = bx - vx, bz - vz
        if dx * dx + dz * dz <= r2:
            return True
    return False


def census_logs(blocks, visited=None, radius: float = 8.0) -> dict[str, int]:
    counts = {name: 0 for name in WOOD_LOGS}
    for b in blocks or []:
        name = strip_ns(b.get("name") or "")
        if name not in counts:
            continue
        if visited is not None and not block_in_explored(b, visited, radius):
            continue
        counts[name] += 1
    return {k: v for k, v in counts.items() if v > 0}


def rarest_log(counts: dict[str, int] | None) -> str | None:
    seen = {k: v for k, v in (counts or {}).items() if v > 0}
    if not seen:
        return None
    return min(seen.items(), key=lambda kv: (kv[1], kv[0]))[0]
