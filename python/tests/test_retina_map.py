"""The eye must be the fly's measured optics, not a grid I chose.

The retina this repo started with was 8 azimuths by 4 elevations per eye, invented.
These tests check the replacement against things independently known about the
Drosophila eye: interommatidial angle near 4-5 degrees, each eye covering its own
hemifield from a little across the midline round to behind the head, and a narrow
frontal binocular overlap. If a future change quietly reintroduces a synthetic grid,
the interommatidial and hemifield checks are what should fail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sense import retina_map


@pytest.fixture(scope="module")
def columns():
    try:
        return retina_map.load_columns()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def built(columns):
    return retina_map.build(columns)


def test_the_map_covers_the_published_column_count(columns):
    assert len(columns) == 1772
    assert {c.side for c in columns} == {"L", "R"}
    # Most directions are matched to the micro-CT eye map rather than extrapolated.
    measured = sum(1 for c in columns if c.source == "zhao_uCT_match")
    assert measured == 1682
    assert measured / len(columns) > 0.94


def test_directions_are_unit_vectors(columns):
    for column in columns:
        assert math.isclose(np.linalg.norm(column.vec), 1.0, abs_tol=2e-3)


def test_interommatidial_angle_matches_the_real_eye(columns):
    """Neighbouring ommatidia sit about 4-5 degrees apart in Drosophila."""
    for side in ("L", "R"):
        vecs = np.asarray([c.vec for c in columns if c.side == side])
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        cos = np.clip(vecs @ vecs.T, -1.0, 1.0)
        np.fill_diagonal(cos, -1.0)
        nearest = np.degrees(np.arccos(cos.max(axis=1)))
        assert 3.0 < float(np.median(nearest)) < 6.0


def test_each_eye_looks_at_its_own_side(columns):
    """The left eye points left. Positive azimuth is the fly's right, as olfaction uses."""
    for side, sign in (("L", 1.0), ("R", -1.0)):
        eye = [c for c in columns if c.side == side]
        pointing = sum(1 for c in eye if math.copysign(1.0, c.left) == sign)
        assert pointing / len(eye) > 0.9


def test_the_hemifields_meet_at_the_midline_with_a_narrow_overlap(columns):
    """Near the horizon each eye runs from slightly across the midline to behind.

    Away from the horizon azimuth degenerates -- a column looking almost straight up
    has no meaningful azimuth -- so the check is restricted to the horizontal band.
    """
    band = [c for c in columns if abs(c.elevation_deg) < 20]
    left = [c.azimuth_right_deg for c in band if c.side == "L"]
    right = [c.azimuth_right_deg for c in band if c.side == "R"]
    assert min(left) < -140 and max(left) < 20      # leftward, crossing slightly over
    assert max(right) > 140 and min(right) > -20    # mirror image
    overlap = max(left) - min(right)
    assert 0 < overlap < 40, f"binocular overlap {overlap:.1f} deg"


def test_sampling_picks_real_ommatidia_not_invented_directions(built, columns):
    """Every ray must be one of the measured columns, so no direction is made up."""
    real = {c.key for c in columns}
    for ray in built.rays:
        assert ray.key in real


def test_every_column_reads_a_ray_and_the_blur_is_measured(built, columns):
    assert len(built.ray_of_column) == len(columns)
    assert built.n_rays == 2 * built.rays_per_eye
    summary = built.summary()
    # Sharing rays between neighbouring columns costs angular precision. The cost is
    # reported rather than assumed; it should stay within a few interommatidial angles.
    assert summary["angular_error_deg"]["mean"] < 10.0
    assert summary["angular_error_deg"]["max"] < 20.0


def test_a_column_always_reads_a_ray_from_its_own_eye(built):
    by_index = {i: r for i, r in enumerate(built.rays)}
    for (side, _, _), ray in built.ray_of_column.items():
        assert by_index[ray].side == side


def test_rays_are_split_evenly_between_the_eyes(built):
    per_side = {s: sum(1 for r in built.rays if r.side == s) for s in ("L", "R")}
    assert per_side == {"L": built.rays_per_eye, "R": built.rays_per_eye}


def test_the_generated_config_matches_the_built_map(built):
    """The JS sense bridge casts rays from this file, so it has to agree with Python."""
    config = retina_map.load_config()
    assert config["count"] == built.n_rays
    assert config["raysPerEye"] == built.rays_per_eye
    assert len(config["rays"]) == built.n_rays
    for written, ray in zip(config["rays"], built.rays):
        assert written["side"] == ray.side
        assert written["hex1"] == ray.hex1 and written["hex2"] == ray.hex2
        assert math.isclose(written["azimuthRightDeg"], ray.azimuth_right_deg,
                            abs_tol=1e-3)
        assert math.isclose(written["elevationDeg"], ray.elevation_deg, abs_tol=1e-3)


def test_farthest_point_sampling_spreads_over_the_sphere():
    """Given an even shell of directions, the first picks should be far apart."""
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(500, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    order = retina_map.farthest_point_order(vecs)
    assert len(order) == 500 and len(set(order.tolist())) == 500
    first = vecs[order[:8]]
    cos = np.clip(first @ first.T, -1.0, 1.0)
    np.fill_diagonal(cos, -1.0)
    # No two of the first eight should be within 30 degrees of each other.
    assert np.degrees(np.arccos(cos.max())) > 30.0
