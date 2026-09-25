"""The dashboard's channel glossary must cover the frame and match the tables."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from sense.frame import GLOMERULI, GRNS, MECHANO_KEYS, OBJECT_CHS
from sense.glossary import glossary
from sense.routing import AVERSIVE_GRNS, GRN_IDENTITY


def test_every_channel_has_an_entry():
    g = glossary()
    assert list(g["glomeruli"]) == GLOMERULI
    assert list(g["grns"]) == GRNS
    assert list(g["object_channels"]) == OBJECT_CHS
    assert list(g["mechano"]) == MECHANO_KEYS


def test_every_grn_and_object_channel_says_what_it_is():
    g = glossary()
    for group in ("grns", "object_channels", "mechano"):
        blank = [k for k, v in g[group].items() if not v["what"]]
        assert not blank, f"{group} channels with no description: {blank}"


def test_minecraft_keywords_come_from_the_sense_tables():
    """`keys` is derived, so a table edit must show up without touching glossary.py."""
    g = glossary()
    # `keys` is ranked by affinity and truncated, so ask for enough to see oak.
    deep = glossary(limit=60)
    assert "oak_log" in deep["glomeruli"]["DL5"]["keys"], "logs are a green-leaf DL5 odour"
    assert len(g["glomeruli"]["DL5"]["keys"]) <= 4, "the default view stays short"
    # Sugar reaches both the labellum and the tarsi.
    assert "sugar" in g["grns"]["LB3b"]["keys"]
    assert "sugar" in g["grns"]["LgLG3"]["keys"]
    # Bitter is aversive, and lands on the bitter family only.
    assert "spider_eye" in g["grns"]["LB1a"]["keys"]
    assert "spider_eye" not in g["grns"]["LB3b"]["keys"]
    # Creeper smells of geosmin in this repo, which is DA2.
    assert "creeper" in g["glomeruli"]["DA2"]["keys"]


def test_grn_valence_follows_modality_not_the_name_prefix():
    """The two types whose number series disagrees with their modality.

    Per the male-cns taste-feeding connectome: LB1e is an Ir94e amino-acid cell
    despite sitting in the bitter-numbered LB1 series, and LB3d is an Ir7c
    high-salt *avoidance* cell despite sitting in the sugar-numbered LB3 series.
    Deriving sign from the prefix inverts both.
    """
    assert GRN_IDENTITY["LB1e"][0] == "amino acid"
    assert GRN_IDENTITY["LB3d"][0] == "high salt — avoid"
    assert "LB1e" not in AVERSIVE_GRNS, "an amino-acid cell must not suppress feeding"
    assert "LB3d" in AVERSIVE_GRNS, "high salt is an avoidance channel"
    assert set(AVERSIVE_GRNS) == {"LB1a", "LB1b", "LB1c", "LB1d", "LB3d", "LgAG1"}


def test_taste_valence_is_downstream_of_the_receptor_not_in_its_sign():
    """Bitter neurons are excitatory. The avoidance happens after them.

    The old hand-built graph gave each aversive GRN an inhibitory edge onto a feeding
    node, which read plausibly and was wrong: bitter and high-salt GRNs are cholinergic,
    like the sugar ones. They excite their own second-order neurons, and suppression of
    feeding is implemented by those downstream circuits. Writing the valence into the
    receptor's transmitter would have made `AVERSIVE_GRNS` a claim about anatomy instead
    of a claim about behaviour, so this test now pins the real polarity.
    """
    from conftest import graph_or_skip

    graph = graph_or_skip()
    sign = {name: s for name, s in zip(graph.names, graph.sign)}
    nt = {name: t for name, t in zip(graph.names, graph.nt)}

    found = 0
    for grn in GRN_IDENTITY:
        for name in (n for n in graph.names if n.startswith(grn)):
            assert sign[name] > 0, (
                f"{name} is signed inhibitory; gustatory receptor neurons are "
                f"cholinergic (this one reads as {nt[name]})"
            )
            found += 1
    assert found, "no gustatory receptor neurons in the graph"
    # And the labels still say which channels are aversive, for the dashboard.
    assert "LB1a" in AVERSIVE_GRNS and "LB3b" not in AVERSIVE_GRNS
