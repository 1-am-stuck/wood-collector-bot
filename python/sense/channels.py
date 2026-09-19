"""Resolve the routing keys carried by a derived graph onto observation indices.

A derived graph stores routing as strings — `glomerulus:DM1`, `retina:37`,
`mechano:touchHead` — rather than as raw integers. The reason is that the graph and
the observation vector change at different times: rebuilding the connectome must not
silently shift what a lamina node is reading because the frame layout moved by two
slots. Names fail loudly; indices fail quietly.

The keys are:

    retina:<i>          one ray, i.e. one sampled ommatidium
    retina:eye:<L|R>    every ray of one eye, for graphs with no columnar retina
    retina                every ray
    glomerulus:<name>   one antennal-lobe glomerulus
    odor_bearing        the sin/cos pair of the odour bearing
    grn:<type>          one gustatory receptor type
    object:<channel>    one analytically computed optic glomerulus
    mechano:<key>       one mechano / thermo / hygro field
    rate:yaw, rate:pitch  the two self-motion channels
"""

from __future__ import annotations

from .frame import GLOMERULI, GRNS, MECHANO_KEYS, OBJECT_CHS


class UnknownChannel(KeyError):
    """A graph asked for an observation channel the frame does not have."""


def channel_map(n_retina: int, ray_sides: list[str] | None = None) -> dict[str, list[int]]:
    """Every routing key the frame can satisfy, mapped to observation indices.

    `ray_sides` is the eye each ray belongs to, in ray order, as written by
    `retina_map.write_config`. Without it the per-eye keys are unavailable, which is
    correct rather than approximated: a graph that wants one eye should not silently
    be handed both.
    """
    out: dict[str, list[int]] = {}
    retina = list(range(n_retina))
    out["retina"] = retina
    for i in retina:
        out[f"retina:{i}"] = [i]
    if ray_sides:
        for side in ("L", "R"):
            hits = [i for i, s in enumerate(ray_sides[:n_retina]) if s == side]
            if hits:
                out[f"retina:eye:{side}"] = hits

    at = n_retina
    for name in GLOMERULI:
        out[f"glomerulus:{name}"] = [at]
        at += 1
    out["odor_bearing"] = [at, at + 1]
    at += 2
    for name in GRNS:
        out[f"grn:{name}"] = [at]
        at += 1
    for name in OBJECT_CHS:
        out[f"object:{name}"] = [at]
        at += 1
    for name in MECHANO_KEYS:
        out[f"mechano:{name}"] = [at]
        at += 1
    out["rate:yaw"] = [at]
    out["rate:pitch"] = [at + 1]
    at += 2

    # The layout above has to be the layout `frame_to_vector` produces. If it drifts,
    # every routed channel is off by however much it drifted, so it is checked here
    # once rather than debugged later as a mysteriously untrainable network.
    from .frame import vector_size
    expected = vector_size(n_retina)
    if at != expected:
        raise AssertionError(
            f"channel map covers {at} slots but frame_to_vector produces {expected}; "
            "sense/channels.py and sense/frame.py have diverged"
        )
    return out


def resolve(routing: dict[str, list[str]], n_retina: int,
            ray_sides: list[str] | None = None,
            strict: bool = True) -> dict[str, list[int]]:
    """Turn {node: [channel keys]} into {node: [observation indices]}."""
    table = channel_map(n_retina, ray_sides)
    out: dict[str, list[int]] = {}
    missing: set[str] = set()
    for node, keys in routing.items():
        idx: list[int] = []
        for key in keys:
            hit = table.get(key)
            if hit is None:
                missing.add(key)
                continue
            idx.extend(hit)
        out[node] = sorted(set(idx))
    if missing and strict:
        raise UnknownChannel(
            "the graph routes channels the frame does not provide: "
            + ", ".join(sorted(missing))
        )
    return out
