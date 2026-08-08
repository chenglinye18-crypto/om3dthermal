"""Tests for the matrix-free thermal operator.

These tests build a small thermal network, hand-construct a dense
``A_ref`` and ``b_ref`` for the same network, and verify the
matrix-free operator matches the dense reference on every
vector it sees. A toy 1-D 5-cell slab is used as the canonical
example.
"""
from __future__ import annotations

import numpy as np
import pytest

from om3dthermal.thermal import build_matrix_free_operator
from om3dthermal.thermal.boundary import BoundaryLinkTable
from om3dthermal.thermal.conductance import ConductanceTable


# ---------------------------------------------------------------------------
# 1-D 5-cell slab: dense reference vs matrix-free
# ---------------------------------------------------------------------------

def _toy_1d_slab():
    """Five cells along x, each 0.2 mm wide; 4 internal edges; one
    fixed-temperature link on cell 0 (left) and one convection link
    on cell 4 (right). No power.

    The dense A and b are constructed by hand so we can compare the
    matrix-free ``apply(T)`` to ``A @ T`` and the residual to
    ``b - A @ T``.
    """
    N = 5
    cell_a = np.arange(N - 1, dtype=np.int64)
    cell_b = np.arange(1, N, dtype=np.int64)
    # Uniform k = 10 W/(m*K), face area = 1e-6 m^2, half-extent
    # = 0.1 mm = 1e-4 m. G_internal = 1e-6 / (2 * 1e-4 / 10) = 0.05.
    G_int = 1.0e-6 / (2.0 * 1.0e-4 / 10.0)
    G_internal = np.full(N - 1, G_int, dtype=np.float64)
    # Boundary on cell 0: fixed T = 350 K. With half-extent 1e-4 m,
    # k = 10, face area 1e-6, the per-cell G_fixed = 1e-6 / 1e-4 / 10
    # = 1e-3, and T_ref = 350.
    # Boundary on cell 4: convection h = 100, T_amb = 300 K.
    # G_conv = 1 / (1e-4/10 + 1/100) = 1 / 0.02 = 50, but we need
    # the per-area decomposition: 1e-6 / (1e-4/10 + 1/(100*1e-6))
    # would mix units; the boundary table stores the final G in W/K
    # so we use G_conv = 1e-6 / (1e-4/10 + 1/100) = 1e-6 / 0.02 = 5e-5.
    G_conv = 1.0e-6 / (1.0e-4 / 10.0 + 1.0 / 100.0)
    boundary_cell = np.array([0, 4], dtype=np.int64)
    boundary_G = np.array([1.0e-6 / (1.0e-4 / 10.0), G_conv],
                           dtype=np.float64)
    boundary_Tref = np.array([350.0, 300.0], dtype=np.float64)

    # Dense A: symmetric, internal + boundary links.
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N - 1):
        A[i, i] += G_int
        A[i + 1, i + 1] += G_int
        A[i, i + 1] -= G_int
        A[i + 1, i] -= G_int
    for c, G, Tref in zip(boundary_cell, boundary_G, boundary_Tref):
        # Fixed T: G contribution to A[c, c] (and Tref to b).
        A[c, c] += G
    # RHS.
    power = np.zeros(N, dtype=np.float64)
    b = power + sum(G * Tref for c, G, Tref in zip(
        boundary_cell, boundary_G, boundary_Tref))
    # The contribution to b for fixed-T is G * T_ref; for convection
    # it is also G * T_ref. Both are in the column c.
    b = np.zeros(N, dtype=np.float64)
    for c, G, Tref in zip(boundary_cell, boundary_G, boundary_Tref):
        b[c] += G * Tref
    return N, cell_a, cell_b, G_internal, boundary_cell, boundary_G, boundary_Tref, A, b


def test_apply_matches_dense_A_ref():
    (N, cell_a, cell_b, G_internal, bc, bG, bT, A, b) = _toy_1d_slab()
    conductance = ConductanceTable(
        edge_id=np.arange(N - 1, dtype=np.int64),
        cell_a=cell_a, cell_b=cell_b, axis=np.zeros(N - 1, dtype=np.int8),
        face_area_m2=np.zeros(N - 1, dtype=np.float64),
        half_distance_a_m=np.zeros(N - 1, dtype=np.float64),
        half_distance_b_m=np.zeros(N - 1, dtype=np.float64),
        k_normal_a_W_mK=np.zeros(N - 1, dtype=np.float64),
        k_normal_b_W_mK=np.zeros(N - 1, dtype=np.float64),
        interface_areal_resistance_m2K_W=np.zeros(N - 1, dtype=np.float64),
        resistance_K_W=np.zeros(N - 1, dtype=np.float64),
        conductance_W_K=G_internal,
        material_interface=np.zeros(N - 1, dtype=bool),
        interface_rule_index=np.full(N - 1, -1, dtype=np.int32),
    )
    boundary = BoundaryLinkTable(
        boundary_face_id=np.arange(2, dtype=np.int64),
        cell_id=bc, kind=np.array([2, 1], dtype=np.int8),
        axis=np.array([0, 0], dtype=np.int8),
        side=np.array([0, 1], dtype=np.int8),
        face_area_m2=np.zeros(2, dtype=np.float64),
        half_distance_m=np.zeros(2, dtype=np.float64),
        k_normal_W_mK=np.zeros(2, dtype=np.float64),
        areal_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        external_film_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        conductance_W_K=bG,
        reference_temperature_K=bT,
        rule_index=np.zeros(2, dtype=np.int32),
    )
    op = build_matrix_free_operator(
        conductance, boundary, np.zeros(N, dtype=np.float64))
    np.testing.assert_allclose(op.rhs_W, b, rtol=1e-10, atol=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(5):
        T = rng.standard_normal(N)
        out = op.apply(T)
        ref = A @ T
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)
    T = np.array([310.0, 320.0, 330.0, 320.0, 310.0])
    np.testing.assert_allclose(op.residual(T), b - A @ T,
                               rtol=1e-10, atol=1e-12)
    # diagonal is the per-row sum of incident conductances, which
    # equals A[i, i] in the dense reference.
    np.testing.assert_allclose(op.diagonal_W_K, np.diag(A),
                               rtol=1e-10, atol=1e-12)


def test_diagonal_is_per_row_sum_of_conductances():
    (N, cell_a, cell_b, G_internal, bc, bG, bT, A, _) = _toy_1d_slab()
    conductance = ConductanceTable(
        edge_id=np.arange(N - 1, dtype=np.int64),
        cell_a=cell_a, cell_b=cell_b, axis=np.zeros(N - 1, dtype=np.int8),
        face_area_m2=np.zeros(N - 1, dtype=np.float64),
        half_distance_a_m=np.zeros(N - 1, dtype=np.float64),
        half_distance_b_m=np.zeros(N - 1, dtype=np.float64),
        k_normal_a_W_mK=np.zeros(N - 1, dtype=np.float64),
        k_normal_b_W_mK=np.zeros(N - 1, dtype=np.float64),
        interface_areal_resistance_m2K_W=np.zeros(N - 1, dtype=np.float64),
        resistance_K_W=np.zeros(N - 1, dtype=np.float64),
        conductance_W_K=G_internal,
        material_interface=np.zeros(N - 1, dtype=bool),
        interface_rule_index=np.full(N - 1, -1, dtype=np.int32),
    )
    boundary = BoundaryLinkTable(
        boundary_face_id=np.arange(2, dtype=np.int64),
        cell_id=bc, kind=np.array([2, 1], dtype=np.int8),
        axis=np.array([0, 0], dtype=np.int8),
        side=np.array([0, 1], dtype=np.int8),
        face_area_m2=np.zeros(2, dtype=np.float64),
        half_distance_m=np.zeros(2, dtype=np.float64),
        k_normal_W_mK=np.zeros(2, dtype=np.float64),
        areal_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        external_film_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        conductance_W_K=bG,
        reference_temperature_K=bT,
        rule_index=np.zeros(2, dtype=np.int32),
    )
    op = build_matrix_free_operator(
        conductance, boundary, np.zeros(N, dtype=np.float64))
    # diagonal is the per-row sum of incident conductances, which
    # equals A[i, i] in the dense reference.
    np.testing.assert_allclose(op.diagonal_W_K, np.diag(A),
                               rtol=1e-12, atol=1e-15)


def test_apply_reuses_out_buffer():
    (N, cell_a, cell_b, G_internal, bc, bG, bT, A, _) = _toy_1d_slab()
    conductance = ConductanceTable(
        edge_id=np.arange(N - 1, dtype=np.int64),
        cell_a=cell_a, cell_b=cell_b, axis=np.zeros(N - 1, dtype=np.int8),
        face_area_m2=np.zeros(N - 1, dtype=np.float64),
        half_distance_a_m=np.zeros(N - 1, dtype=np.float64),
        half_distance_b_m=np.zeros(N - 1, dtype=np.float64),
        k_normal_a_W_mK=np.zeros(N - 1, dtype=np.float64),
        k_normal_b_W_mK=np.zeros(N - 1, dtype=np.float64),
        interface_areal_resistance_m2K_W=np.zeros(N - 1, dtype=np.float64),
        resistance_K_W=np.zeros(N - 1, dtype=np.float64),
        conductance_W_K=G_internal,
        material_interface=np.zeros(N - 1, dtype=bool),
        interface_rule_index=np.full(N - 1, -1, dtype=np.int32),
    )
    boundary = BoundaryLinkTable(
        boundary_face_id=np.arange(2, dtype=np.int64),
        cell_id=bc, kind=np.array([2, 1], dtype=np.int8),
        axis=np.array([0, 0], dtype=np.int8),
        side=np.array([0, 1], dtype=np.int8),
        face_area_m2=np.zeros(2, dtype=np.float64),
        half_distance_m=np.zeros(2, dtype=np.float64),
        k_normal_W_mK=np.zeros(2, dtype=np.float64),
        areal_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        external_film_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        conductance_W_K=bG,
        reference_temperature_K=bT,
        rule_index=np.zeros(2, dtype=np.int32),
    )
    op = build_matrix_free_operator(
        conductance, boundary, np.zeros(N, dtype=np.float64))
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = np.zeros(N, dtype=np.float64)
    ret = op.apply(T, out=out)
    assert ret is out
    np.testing.assert_allclose(out, A @ T, rtol=1e-12, atol=1e-15)


def test_matvec_count_increments_per_apply():
    (N, cell_a, cell_b, G_internal, bc, bG, bT, _, _) = _toy_1d_slab()
    conductance = ConductanceTable(
        edge_id=np.arange(N - 1, dtype=np.int64),
        cell_a=cell_a, cell_b=cell_b, axis=np.zeros(N - 1, dtype=np.int8),
        face_area_m2=np.zeros(N - 1, dtype=np.float64),
        half_distance_a_m=np.zeros(N - 1, dtype=np.float64),
        half_distance_b_m=np.zeros(N - 1, dtype=np.float64),
        k_normal_a_W_mK=np.zeros(N - 1, dtype=np.float64),
        k_normal_b_W_mK=np.zeros(N - 1, dtype=np.float64),
        interface_areal_resistance_m2K_W=np.zeros(N - 1, dtype=np.float64),
        resistance_K_W=np.zeros(N - 1, dtype=np.float64),
        conductance_W_K=G_internal,
        material_interface=np.zeros(N - 1, dtype=bool),
        interface_rule_index=np.full(N - 1, -1, dtype=np.int32),
    )
    boundary = BoundaryLinkTable(
        boundary_face_id=np.arange(2, dtype=np.int64),
        cell_id=bc, kind=np.array([2, 1], dtype=np.int8),
        axis=np.array([0, 0], dtype=np.int8),
        side=np.array([0, 1], dtype=np.int8),
        face_area_m2=np.zeros(2, dtype=np.float64),
        half_distance_m=np.zeros(2, dtype=np.float64),
        k_normal_W_mK=np.zeros(2, dtype=np.float64),
        areal_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        external_film_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        conductance_W_K=bG,
        reference_temperature_K=bT,
        rule_index=np.zeros(2, dtype=np.int32),
    )
    op = build_matrix_free_operator(
        conductance, boundary, np.zeros(N, dtype=np.float64))
    T = np.zeros(N, dtype=np.float64)
    before = op.matvec_count
    op.apply(T)
    op.apply(T)
    op.apply(T)
    assert op.matvec_count == before + 3


# ---------------------------------------------------------------------------
# Anchored-component check
# ---------------------------------------------------------------------------

def test_validate_anchored_components_passes_when_all_anchored():
    (N, cell_a, cell_b, G_internal, bc, bG, bT, _, _) = _toy_1d_slab()
    boundary = BoundaryLinkTable(
        boundary_face_id=np.arange(2, dtype=np.int64),
        cell_id=bc, kind=np.array([2, 1], dtype=np.int8),
        axis=np.array([0, 0], dtype=np.int8),
        side=np.array([0, 1], dtype=np.int8),
        face_area_m2=np.zeros(2, dtype=np.float64),
        half_distance_m=np.zeros(2, dtype=np.float64),
        k_normal_W_mK=np.zeros(2, dtype=np.float64),
        areal_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        external_film_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        conductance_W_K=bG,
        reference_temperature_K=bT,
        rule_index=np.zeros(2, dtype=np.int32),
    )
    # Should not raise.
    from om3dthermal.thermal import validate_anchored_components
    validate_anchored_components(N, cell_a, cell_b, boundary)


def test_validate_anchored_components_raises_when_unanchored():
    # 2-cell network, no boundary links.
    cell_a = np.array([0], dtype=np.int64)
    cell_b = np.array([1], dtype=np.int64)
    boundary = BoundaryLinkTable(
        boundary_face_id=np.empty(0, dtype=np.int64),
        cell_id=np.empty(0, dtype=np.int64),
        kind=np.empty(0, dtype=np.int8),
        axis=np.empty(0, dtype=np.int8),
        side=np.empty(0, dtype=np.int8),
        face_area_m2=np.empty(0, dtype=np.float64),
        half_distance_m=np.empty(0, dtype=np.float64),
        k_normal_W_mK=np.empty(0, dtype=np.float64),
        areal_resistance_m2K_W=np.empty(0, dtype=np.float64),
        external_film_resistance_m2K_W=np.empty(0, dtype=np.float64),
        conductance_W_K=np.empty(0, dtype=np.float64),
        reference_temperature_K=np.empty(0, dtype=np.float64),
        rule_index=np.empty(0, dtype=np.int32),
    )
    from om3dthermal.thermal import validate_anchored_components
    from om3dthermal.thermal.steady_state import UnanchoredThermalComponentError
    with pytest.raises(UnanchoredThermalComponentError):
        validate_anchored_components(2, cell_a, cell_b, boundary)


def test_validate_anchored_components_unanchored_even_with_zero_power():
    # Same as above but with a 1-cell component carrying zero power.
    cell_a = np.array([0], dtype=np.int64)
    cell_b = np.array([0], dtype=np.int64)
    boundary = BoundaryLinkTable(
        boundary_face_id=np.empty(0, dtype=np.int64),
        cell_id=np.empty(0, dtype=np.int64),
        kind=np.empty(0, dtype=np.int8),
        axis=np.empty(0, dtype=np.int8),
        side=np.empty(0, dtype=np.int8),
        face_area_m2=np.empty(0, dtype=np.float64),
        half_distance_m=np.empty(0, dtype=np.float64),
        k_normal_W_mK=np.empty(0, dtype=np.float64),
        areal_resistance_m2K_W=np.empty(0, dtype=np.float64),
        external_film_resistance_m2K_W=np.empty(0, dtype=np.float64),
        conductance_W_K=np.empty(0, dtype=np.float64),
        reference_temperature_K=np.empty(0, dtype=np.float64),
        rule_index=np.empty(0, dtype=np.int32),
    )
    from om3dthermal.thermal import validate_anchored_components
    from om3dthermal.thermal.steady_state import UnanchoredThermalComponentError
    with pytest.raises(UnanchoredThermalComponentError):
        validate_anchored_components(1, cell_a, cell_b, boundary)
