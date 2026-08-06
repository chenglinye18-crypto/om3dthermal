"""Tests for the material tensor / rotation / normal-conductivity layer.

These tests target the geometric algebra of the conductivity tensor
(``K_global = R K_local R^T``) and the normal projection
``k_n = n^T K_global n`` for axis-aligned faces. They are independent
of the discretiser and the benchmark; failures point straight at the
tensor module.
"""
from __future__ import annotations

import math

import pytest

from om3dthermal.thermal import (
    canonical_rotation_key,
    global_conductivity_tensor,
    is_signed_axis_permutation,
    normal_conductivity,
    validate_rotation_matrix,
)
from om3dthermal.thermal.errors import (
    InvalidRotationMatrixError,
    UnsupportedMaterialRotationError,
)


# Common test rotations.
IDENTITY = ((1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0))


def _rot_x(angle_deg: float) -> tuple[tuple[float, float, float], ...]:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return ((1.0, 0.0, 0.0),
            (0.0,   c,  -s),
            (0.0,   s,   c))


def _rot_y(angle_deg: float) -> tuple[tuple[float, float, float], ...]:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return ((  c, 0.0,   s),
            (0.0, 1.0, 0.0),
            ( -s, 0.0,   c))


def _rot_z(angle_deg: float) -> tuple[tuple[float, float, float], ...]:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return ((  c,  -s, 0.0),
            (  s,   c, 0.0),
            (0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# A. Identity material tensor
# ---------------------------------------------------------------------------

def test_identity_rotation_returns_kx_ky_kz_for_x_y_z_axes():
    k_local = (10.0, 20.0, 30.0)
    assert normal_conductivity(k_local, IDENTITY, "x") == pytest.approx(10.0)
    assert normal_conductivity(k_local, IDENTITY, "y") == pytest.approx(20.0)
    assert normal_conductivity(k_local, IDENTITY, "z") == pytest.approx(30.0)
    assert normal_conductivity(k_local, IDENTITY, 0) == pytest.approx(10.0)
    assert normal_conductivity(k_local, IDENTITY, 1) == pytest.approx(20.0)
    assert normal_conductivity(k_local, IDENTITY, 2) == pytest.approx(30.0)


def test_global_conductivity_tensor_identity_is_diagonal():
    k_local = (10.0, 20.0, 30.0)
    K = global_conductivity_tensor(k_local, IDENTITY)
    assert K == [[10.0, 0.0, 0.0],
                 [0.0, 20.0, 0.0],
                 [0.0, 0.0, 30.0]]


# ---------------------------------------------------------------------------
# B. 90 degree rotations
# ---------------------------------------------------------------------------

def test_rot_y_90_swaps_local_x_and_local_z():
    # A 90 degree rotation about y maps local x -> global z and
    # local z -> -global x. k_n(global x) therefore reads the
    # original local z (30) and k_n(global z) reads the original
    # local x (10). y is unchanged.
    k_local = (10.0, 20.0, 30.0)
    R = _rot_y(90)
    assert is_signed_axis_permutation(R)
    assert normal_conductivity(k_local, R, "x") == pytest.approx(30.0)
    assert normal_conductivity(k_local, R, "y") == pytest.approx(20.0)
    assert normal_conductivity(k_local, R, "z") == pytest.approx(10.0)


def test_rot_x_90_swaps_local_y_and_local_z():
    k_local = (10.0, 20.0, 30.0)
    R = _rot_x(90)
    assert is_signed_axis_permutation(R)
    assert normal_conductivity(k_local, R, "x") == pytest.approx(10.0)
    assert normal_conductivity(k_local, R, "y") == pytest.approx(30.0)
    assert normal_conductivity(k_local, R, "z") == pytest.approx(20.0)


def test_rot_z_90_swaps_local_x_and_local_y():
    k_local = (10.0, 20.0, 30.0)
    R = _rot_z(90)
    assert is_signed_axis_permutation(R)
    assert normal_conductivity(k_local, R, "x") == pytest.approx(20.0)
    assert normal_conductivity(k_local, R, "y") == pytest.approx(10.0)
    assert normal_conductivity(k_local, R, "z") == pytest.approx(30.0)


def test_180_degree_rotation_returns_same_k_n_as_identity():
    # Any 180 degree axis-aligned rotation permutes the local axes
    # with possible sign flips; k_n only depends on squared entries
    # of R, so the normal conductivity is identical to identity.
    k_local = (10.0, 20.0, 30.0)
    for R in (_rot_x(180), _rot_y(180), _rot_z(180)):
        assert normal_conductivity(k_local, R, "x") == pytest.approx(10.0)
        assert normal_conductivity(k_local, R, "y") == pytest.approx(20.0)
        assert normal_conductivity(k_local, R, "z") == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# C. Invalid rotations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    None,
    ((1, 0, 0),),                              # wrong row count
    ((1, 0, 0), (0, 1, 0)),                     # missing row
    ((1, 0, 0), (0, 1, 0), (0, 0)),             # wrong column count
    ((1, 0, 0), (0, 1, 0), (0, 0, float("nan"))),
    ((1, 0, 0), (0, 1, 0), (0, 0, float("inf"))),
    ((1, 0, 0), (0, 1, 0), (0, 0, -1)),        # det = -1
    ((0, 0, 1), (0, 1, 0), (1, 0, 0)),         # det = -1 permutation
    ((0.5, 0, 0), (0, 1, 0), (0, 0, 1)),       # not orthogonal
    ((1, 0, 0), (1, 0, 0), (0, 0, 1)),         # not orthogonal
    ((1, 0, 0), (0, math.cos(math.radians(45)),
                 -math.sin(math.radians(45))),
     (0, math.sin(math.radians(45)),
      math.cos(math.radians(45)))),            # arbitrary 45 deg
    "not a matrix",
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],  # too many rows
])
def test_validate_rotation_matrix_rejects_invalid_matrices(bad):
    try:
        validate_rotation_matrix(bad)
    except (InvalidRotationMatrixError, TypeError):
        return
    raise AssertionError(
        f"validate_rotation_matrix({bad!r}) did not raise")


def test_is_signed_axis_permutation_rejects_arbitrary_angle():
    R = _rot_x(45)
    assert not is_signed_axis_permutation(R)


def test_is_signed_axis_permutation_accepts_identity_and_90_180_270():
    assert is_signed_axis_permutation(IDENTITY)
    for angle in (90, 180, 270):
        assert is_signed_axis_permutation(_rot_x(angle))
        assert is_signed_axis_permutation(_rot_y(angle))
        assert is_signed_axis_permutation(_rot_z(angle))


def test_normal_conductivity_rejects_arbitrary_rotation():
    k_local = (10.0, 20.0, 30.0)
    R = _rot_x(45)
    with pytest.raises(UnsupportedMaterialRotationError):
        normal_conductivity(k_local, R, "x")


# ---------------------------------------------------------------------------
# Canonical rotation key
# ---------------------------------------------------------------------------

def test_canonical_rotation_key_collapses_R_with_same_R_squared():
    # 90 and 270 about the same axis have the same R^2 entries
    # (the off-diagonal signs flip but the squares don't), so they
    # produce the same k_n for any diagonal K and should hash to the
    # same canonical key.
    assert canonical_rotation_key(_rot_x(90)) == canonical_rotation_key(
        _rot_x(270))
    assert canonical_rotation_key(_rot_y(90)) == canonical_rotation_key(
        _rot_y(270))
    assert canonical_rotation_key(_rot_z(90)) == canonical_rotation_key(
        _rot_z(270))


def test_canonical_rotation_key_distinct_for_distinct_permutations():
    # Four distinct k_n signatures arise from the 10 axis-aligned
    # rotations: identity / 180-about-any-axis (kx, ky, kz), x-90/270
    # (kx, kz, ky), y-90/270 (kz, ky, kx), z-90/270 (ky, kx, kz).
    rotations = (
        IDENTITY,
        _rot_x(90), _rot_x(180), _rot_x(270),
        _rot_y(90), _rot_y(180), _rot_y(270),
        _rot_z(90), _rot_z(180), _rot_z(270),
    )
    keys = {canonical_rotation_key(R) for R in rotations}
    assert len(keys) == 4
    # 180 about any axis collapses to the identity key.
    assert canonical_rotation_key(_rot_x(180)) == canonical_rotation_key(IDENTITY)
    assert canonical_rotation_key(_rot_y(180)) == canonical_rotation_key(IDENTITY)
    assert canonical_rotation_key(_rot_z(180)) == canonical_rotation_key(IDENTITY)
