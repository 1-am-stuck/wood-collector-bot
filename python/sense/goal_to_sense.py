"""GoalSpec → additive SensoryInjection. Mirrors js/sense/goalToSense.js."""

from __future__ import annotations

import json
from pathlib import Path


def load_goal(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _clone(frame: dict) -> dict:
    out = dict(frame)
    out["luminance"] = list(frame.get("luminance") or [])
    out["objects"] = [dict(o) for o in frame.get("objects") or []]
    out["odor"] = dict(frame.get("odor") or {})
    out["taste"] = dict(frame.get("taste") or {})
    out["objectChannels"] = dict(frame.get("objectChannels") or {})
    return out


def _nearest(world: dict | None, query: dict | None):
    if not world or not query:
        return None
    name = query.get("name")
    kind = query.get("kind")
    pos = world.get("position") or {"x": 0, "y": 0, "z": 0}
    if kind == "entity":
        pool = world.get("entities") or []
    elif kind == "item":
        pool = world.get("items") or []
    else:
        pool = world.get("blocks") or []
    best = None
    best_d = 1e18
    for src in pool:
        n = str(src.get("name") or src.get("type") or "").replace("minecraft:", "")
        if n != name and not (kind == "entity" and src.get("type") == name):
            continue
        d = ((src["x"] - pos["x"]) ** 2 + (src["y"] - pos["y"]) ** 2 + (src["z"] - pos["z"]) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best = {**src, "distance": d, "name": n}
    return best


def _weight(inj: dict, world: dict | None) -> float:
    fall = inj.get("falloff") or "tonic"
    strength = float(inj.get("strength") or 0)
    if fall == "tonic":
        return strength
    hit = _nearest(world, inj.get("source_query"))
    if not hit:
        return 0.0
    if fall == "when_seen":
        return strength
    if fall == "when_near":
        return strength / (1.0 + hit["distance"])
    if fall == "distance":
        import math
        return strength * math.exp(-hit["distance"] / float(inj.get("lambda") or 6))
    return strength


def apply_goal(frame: dict, goal: dict | None, world: dict | None = None) -> dict:
    out = _clone(frame)
    if not goal:
        return out
    for inj in goal.get("injections") or []:
        w = _weight(inj, world)
        if w <= 0:
            continue
        mod = inj.get("modality")
        if mod == "olfaction" and inj.get("pattern"):
            for glom, aff in inj["pattern"].items():
                out["odor"][glom] = out["odor"].get(glom, 0.0) + float(aff) * w
        elif mod == "gustation" and inj.get("pattern"):
            for grn, aff in inj["pattern"].items():
                out["taste"][grn] = max(out["taste"].get(grn, 0.0), float(aff) * w)
        elif mod == "vision":
            pop = inj.get("population") or "LC11"
            out["objectChannels"][pop] = min(1.5, out["objectChannels"].get(pop, 0.0) + w)
        elif mod == "mechano":
            field = inj.get("field") or "groomDust"
            out[field] = min(1.0, float(out.get(field) or 0) + w)
    return out


def goal_success(goal: dict, facts: dict) -> bool:
    if not goal or not goal.get("success"):
        return False
    s = goal["success"]
    if s["type"] == "inventory_contains":
        return (facts.get("inventory") or {}).get(s["item"], 0) >= s.get("count", 1)
    if s["type"] == "taste_contact":
        return (facts.get("taste") or {}).get(s["grn"], 0) >= s.get("min", 0.5)
    if s["type"] == "distance_above":
        d = (facts.get("distances") or {}).get(s["entity"])
        return d is None or d >= s["blocks"]
    return False


def reward_for(goal: dict, prev: dict | None, facts: dict) -> float:
    r = 0.0
    if goal_success(goal, facts) and not goal_success(goal, prev or {}):
        r += 10.0
    s = (goal or {}).get("success") or {}
    if s.get("type") == "inventory_contains":
        a = (facts.get("inventory") or {}).get(s["item"], 0)
        b = ((prev or {}).get("inventory") or {}).get(s["item"], 0)
        r += (a - b) * 5.0
    if facts.get("closerToQuery") is not None:
        r += float(facts["closerToQuery"])
    return r
