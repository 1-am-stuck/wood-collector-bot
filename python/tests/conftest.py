"""Shared graph fixture.

Tests that just need *a* working policy use the `type` variant: it is the same
male-cns data as the default `full` graph, pooled per type with vision pooled per eye,
so it is 1,120 neurons instead of 176,422 and a PPO smoke test finishes in seconds. The
checks that care about retinotopy or about specific measured synapse counts name the
variant they need instead (see test_derived_graph.py).

There is no built-in fallback graph any more, so these skip rather than fail when the
connectome has not been built. The derived NPZs are small enough to commit, so in
practice they run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fly_policy.policy import default_graph_path, load_graph

ROOT = Path(__file__).resolve().parents[2]


def graph_or_skip(variant: str = "type"):
    path = default_graph_path(ROOT, variant)
    if not path.exists():
        pytest.skip(
            f"{path.name} not built -- "
            f"python/tools/build_graph_from_flyb.py --variant {variant}"
        )
    return load_graph(ROOT, variant)


@pytest.fixture(scope="session")
def small_graph():
    return graph_or_skip("type")
