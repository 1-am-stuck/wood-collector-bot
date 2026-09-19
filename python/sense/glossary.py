"""Human-readable identity for every channel the fly senses.

Two different things get attached to each channel:

- `what`: what the neuron is in *Drosophila* -- its receptor and the odour,
  taste or stimulus class it is tuned to. This is literature, not code, so
  entries we cannot state confidently are left blank rather than guessed at.
  Provenance is docs/SENSE_PROVENANCE.md.
- `keys`: what actually excites it in *this repo*, computed from the sense
  tables at call time. Nothing here is hand-written, so it cannot drift away
  from configs/sense/*.json.

The dashboard renders both next to the channel name, so "DM1 0.18" stops being
an opaque number. Nothing in this module is used for training or inference.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .frame import GLOMERULI, GRNS, MECHANO_KEYS, OBJECT_CHS
from .routing import GRN_IDENTITY, grn_organ

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sense"

# Olfactory receptor neuron classes, by the glomerulus their axons converge on:
# (tuning receptor, ligand and valence, confidence).
#
# "verified" entries trace to a primary source naming that glomerulus explicitly;
# "default" entries are standard receptor-to-odorant pairings that the reference
# survey did not re-check against DoOR, so they are plausible rather than
# established. The distinction is surfaced in the dashboard rather than hidden.
# Source table: .research/fly-brain-minecraft/docs/research/sensory-mapping.md.
GLOMERULUS_WHAT: dict[str, tuple[str, str, str]] = {
    "DM1": ("Or42b", "ethyl acetate, cider vinegar — attractive; with VA2 this is "
            "*the* vinegar attraction channel", "verified"),
    "VA2": ("Or92a", "acetoin, 2,3-butanedione — attractive, paired with DM1", "verified"),
    "DM5": ("Or85a, Or33b", "high-concentration vinegar — aversive, and strong enough to "
            "override DM1 attraction", "verified"),
    "DA2": ("Or56a, Or33a", "geosmin — strongly aversive dedicated line; suppresses "
            "chemotaxis, oviposition and feeding", "verified"),
    "V": ("Gr21a, Gr63a", "CO2 — aversive down to 0.1%", "verified"),
    "DL4": ("Or49a, Or85f", "parasitoid wasp odour (iridomyrmecin) — dedicated avoidance",
            "verified"),
    "DA1": ("Or67d", "cVA — male pheromone; the largest ORN population in the dataset",
            "verified"),
    "DL3": ("Or65a/b/c", "cVA, slower and longer-lasting — modulates aggression and courtship",
            "default"),
    "VA1v": ("Or47b", "methyl laurate — copulation-promoting", "verified"),
    "VA1d": ("Or88a", "methyl palmitate — attractive in both sexes", "verified"),
    "VL2a": ("Ir84a", "phenylacetaldehyde — a food odour that also promotes male courtship",
             "verified"),
    "VM1": ("Ir92a", "ammonia and amines — attractive", "verified"),
    "VC5": ("Ir41a", "polyamines: putrescine, cadaverine — attractive by smell, while the "
            "same compounds are aversive by taste", "verified"),
    "VM4": ("Ir76a", "amines, phenylethylamine — attractive", "verified"),
    "DP1l": ("Ir75a", "acetic and propionic acid — a fermentation marker", "verified"),
    "DM2": ("Or22a, Or22b", "ethyl hexanoate — ripe fruit", "default"),
    "DM3": ("Or47a, Or33b", "pentyl acetate — fruit ester", "default"),
    "DM4": ("Or59b", "methyl acetate — vinegar and fruit", "default"),
    "DL5": ("Or7a", "E2-hexenal — the green-leaf volatile channel", "default"),
    "VA3": ("Or67b", "plant and mushroom volatiles — vegetation", "default"),
    "VA6": ("Or82a", "geranyl acetate — floral", "default"),
    "VM2": ("Or43b", "ethyl butyrate — fruit", "default"),
    "VM5d": ("Or85b, Or98b (both uncertain in the dataset)", "2-heptanone — overripe",
             "default"),
    "VC1": ("Or33c, Or85e", "ligand not assigned", "default"),
    "VC2": ("Or71a", "4-ethylguaiacol — smoky, phenolic", "default"),
    "DC2": ("Or13a", "1-octen-3-ol — mushroom, damp wood", "default"),
    "D": ("Or69aA, Or69aB", "dual food and pheromone ligand — generic attractant", "default"),
    # The VP glomeruli are thermo/hygrosensory, not olfactory: they receive TRN and
    # HRN axons from the sacculus, which is why routing.py gives them no ORN.
    "VP2": ("TRN, hot cells", "rising temperature", "verified"),
    "VP3a": ("TRN, cold cells", "falling temperature", "verified"),
    "VP3b": ("TRN, cold cells", "falling temperature", "verified"),
    "VP4": ("HRN, dry cells", "falling humidity", "verified"),
    "VP5": ("HRN, moist cells", "rising humidity", "verified"),
}

def grn_what(name: str) -> str:
    """Modality, marker and organ, from the routing contract's GRN_IDENTITY."""
    modality, marker = GRN_IDENTITY.get(name, ("", ""))
    organ = grn_organ(name)
    parts = [p for p in (marker, modality) if p]
    return f"{' — '.join(parts)}; {organ}" if organ else " — ".join(parts)

# Visual projection neurons. The functional note lives in retina.json and is read
# from there; this adds what the cell is and what it drives.
OBJECT_WHAT = {
    "LC4": "lobula columnar; angular *speed* of a looming edge, drives escape latency",
    "LPLC2": "lobula plate/lobula columnar; radial expansion, the giant fibre's main input",
    "LC11": "lobula columnar; small discrete objects, object tracking",
    "LC18": "lobula columnar; small-object motion over a broader field than LC11",
    "LC10a": "lobula columnar; frontal fly-sized movers, courtship pursuit",
    "LC15": "lobula columnar; long bars and edges, e.g. a trunk at the horizon",
    "HS": "horizontal system tangential cell; wide-field yaw optic flow, gaze stabilisation",
}

MECHANO_WHAT = {
    "windLeft": "Johnston's organ C/E — airflow on the left antenna; anemotaxis",
    "windRight": "Johnston's organ C/E — airflow on the right antenna",
    "tilt": "antennal + leg proprioception — body tilt away from level",
    "soundLow": "Johnston's organ B — low-frequency vibration, footfalls",
    "soundHigh": "Johnston's organ A — sharp transients, startle",
    "song": "Johnston's organ A — conspecific courtship song band",
    "touchHead": "BM_InOm bristles — head contact, also triggers eye grooming",
    "touchWing": "ADMN tactile bristles — wing surface contact",
    "touchLegs": "ProLN tactile bristles — leg contact with a surface",
    "touchNotum": "PDMN tactile bristles — thorax contact from above",
    "touchAbdomen": "AbN3 tactile bristles — abdominal contact",
    "groomDust": "JO-F grooming subgroup — particulate on the antennae",
    "damage": "nociceptor — tissue damage",
    "hot": "TRN VP2 — above the comfortable range",
    "cold": "TRN VP3 — below the comfortable range",
    "dry": "HRN VP4 — low humidity",
    "moist": "HRN VP5 — high humidity",
    "airborne": "hair plates + haltere — no tarsal contact, the fly is off the ground",
    "legsOnGround": "tarsal hair plates — standing on a surface",
    "wingbeat": "campaniform sensilla — wing load during flight",
}

# What the Minecraft-side keys mean, where the frame is filled from world state
# rather than from an odour or taste table.
MECHANO_KEYS_MINECRAFT = {
    "windLeft": ["flight / movement airflow"],
    "windRight": ["flight / movement airflow"],
    "soundHigh": ["explosion", "thunder"],
    "soundLow": ["footsteps"],
    "touchHead": ["head collision"],
    "touchLegs": ["leg collision", "water"],
    "groomDust": ["rain"],
    "moist": ["rain", "water"],
    "damage": ["health loss"],
    "airborne": ["not on ground"],
    "legsOnGround": ["on ground"],
    "wingbeat": ["flying"],
}


def _load(name: str) -> dict:
    return json.loads((CONFIG / name).read_text())


def _top(pairs: list[tuple[str, float]], limit: int) -> list[str]:
    ranked = sorted(pairs, key=lambda kv: (-kv[1], kv[0]))
    seen: list[str] = []
    for key, _ in ranked:
        if key not in seen:
            seen.append(key)
        if len(seen) >= limit:
            break
    return seen


@lru_cache(maxsize=1)
def glossary(limit: int = 4) -> dict:
    """{group: {channel: {"what": str, "keys": [str]}}} for the whole frame."""
    odor_hits: dict[str, list[tuple[str, float]]] = {g: [] for g in GLOMERULI}
    for table in ("block_odor.json", "item_odor.json", "entity_odor.json"):
        for source, affinities in _load(table)["sources"].items():
            for glom, strength in affinities.items():
                odor_hits.setdefault(glom, []).append((source, float(strength)))

    # Aliased foods taste identically, so only canonical entries are reported.
    taste = _load("taste_table.json")["contact"]
    grn_hits: dict[str, list[tuple[str, float]]] = {g: [] for g in GRNS}
    for food, spec in taste.items():
        if "alias" in spec:
            continue
        for organ in ("tarsal", "labellar"):
            for grn, strength in (spec.get(organ) or {}).items():
                grn_hits.setdefault(grn, []).append((food, float(strength)))
    notes = _load("retina.json").get("objectChannels", {})

    return {
        "glomeruli": {
            g: {"what": GLOMERULUS_WHAT.get(g, ""), "keys": _top(odor_hits.get(g, []), limit)}
            for g in GLOMERULI
        },
        "grns": {
            g: {"what": grn_what(g), "keys": _top(grn_hits.get(g, []), limit)}
            for g in GRNS
        },
        "object_channels": {
            c: {
                "what": OBJECT_WHAT.get(c, ""),
                "keys": [notes.get(c, {}).get("note", "")] if notes.get(c, {}).get("note") else [],
            }
            for c in OBJECT_CHS
        },
        "mechano": {
            k: {"what": MECHANO_WHAT.get(k, ""), "keys": MECHANO_KEYS_MINECRAFT.get(k, [])}
            for k in MECHANO_KEYS
        },
    }
