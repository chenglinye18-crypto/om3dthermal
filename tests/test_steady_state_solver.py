"""Tests for the matrix-free weighted Jacobi and PCG steady-state
solvers on small hand-built problems where the answer is known
analytically.
"""
from __future__ import annotations

import numpy as np
import pytest

from om3dthermal.thermal import (
    build_matrix_free_operator,
    solve_pcg,
    solve_weighted_jacobi,
)
from om3dthermal.thermal.boundary import BoundaryLinkTable
from om3dthermal.thermal.conductance import ConductanceTable
from om3dthermal.thermal.steady_state import validate_anchored_components


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_conductance() -> ConductanceTable:
    return ConductanceTable(
        edge_id=np.empty(0, dtype=np.int64),
        cell_a=np.empty(0, dtype=np.int64),
        cell_b=np.empty(0, dtype=np.int64),
        axis=np.empty(0, dtype=np.int8),
        face_area_m2=np.empty(0, dtype=np.float64),
        half_distance_a_m=np.empty(0, dtype=np.float64),
        half_distance_b_m=np.empty(0, dtype=np.float64),
        k_normal_a_W_mK=np.empty(0, dtype=np.float64),
        k_normal_b_W_mK=np.empty(0, dtype=np.float64),
        interface_areal_resistance_m2K_W=np.empty(0, dtype=np.float64),
        resistance_K_W=np.empty(0, dtype=np.float64),
        conductance_W_K=np.empty(0, dtype=np.float64),
        material_interface=np.empty(0, dtype=bool),
        interface_rule_index=np.empty(0, dtype=np.int32),
    )


def _boundary(cell_ids, G_arr, T_ref_arr, kinds=None) -> BoundaryLinkTable:
    n = len(cell_ids)
    if kinds is None:
        kinds = np.full(n, 1, dtype=np.int8)  # convection
    return BoundaryLinkTable(
        boundary_face_id=np.arange(n, dtype=np.int64),
        cell_id=np.asarray(cell_ids, dtype=np.int64),
        kind=kinds,
        axis=np.zeros(n, dtype=np.int8),
        side=np.zeros(n, dtype=np.int8),
        face_area_m2=np.zeros(n, dtype=np.float64),
        half_distance_m=np.zeros(n, dtype=np.float64),
        k_normal_W_mK=np.zeros(n, dtype=np.float64),
        areal_resistance_m2K_W=np.zeros(n, dtype=np.float64),
        external_film_resistance_m2K_W=np.zeros(n, dtype=np.float64),
        conductance_W_K=np.asarray(G_arr, dtype=np.float64),
        reference_temperature_K=np.asarray(T_ref_arr, dtype=np.float64),
        rule_index=np.zeros(n, dtype=np.int32),
    )


def _1d_slab(n_cells: int, *, L: float = 1e-3, A: float = 1e-6,
             k: float = 10.0, T_left: float = 350.0,
             T_right: float = 300.0, power: float = 0.0):
    """Build a 1-D slab of ``n_cells`` cells, fixed T on both ends.

    Internal edge between cell ``i`` and ``i+1`` carries the bulk
    resistance of the two half-cells sharing the interface:

        G_int = A / (d_a/k + d_b/k) = k*A/dx.

    The boundary link on cell 0 (or cell N-1) adds the bulk
    resistance from the cell centre to the fixed-temperature face:

        G_face = A / (d_face/k) = 2*k*A/dx.

    A fixed-T face is then a perfect conductor (R = 0) on the
    outside; ``G_face`` together with the per-link ref temperature
    enforces the face temperature.
    """
    dx = L / n_cells
    G_int = A * k / dx
    G_face = 2.0 * A * k / dx
    conductance = ConductanceTable(
        edge_id=np.arange(n_cells - 1, dtype=np.int64),
        cell_a=np.arange(n_cells - 1, dtype=np.int64),
        cell_b=np.arange(1, n_cells, dtype=np.int64),
        axis=np.zeros(n_cells - 1, dtype=np.int8),
        face_area_m2=np.full(n_cells - 1, A, dtype=np.float64),
        half_distance_a_m=np.full(n_cells - 1, dx / 2, dtype=np.float64),
        half_distance_b_m=np.full(n_cells - 1, dx / 2, dtype=np.float64),
        k_normal_a_W_mK=np.full(n_cells - 1, k, dtype=np.float64),
        k_normal_b_W_mK=np.full(n_cells - 1, k, dtype=np.float64),
        interface_areal_resistance_m2K_W=np.zeros(n_cells - 1, dtype=np.float64),
        resistance_K_W=np.zeros(n_cells - 1, dtype=np.float64),
        conductance_W_K=np.full(n_cells - 1, G_int, dtype=np.float64),
        material_interface=np.zeros(n_cells - 1, dtype=bool),
        interface_rule_index=np.full(n_cells - 1, -1, dtype=np.int32),
    )
    boundary = _boundary(
        cell_ids=[0, n_cells - 1],
        G_arr=[G_face, G_face],
        T_ref_arr=[T_left, T_right],
        kinds=np.array([2, 2], dtype=np.int8),
    )
    pwr = np.full(n_cells, power / n_cells, dtype=np.float64)
    op = build_matrix_free_operator(conductance, boundary, pwr)
    return op, boundary, G_int, dx, k, A


# ---------------------------------------------------------------------------
# A. Single cell, two fixed-temperature sides
# ---------------------------------------------------------------------------

def test_single_cell_two_fixed_t_sides_weighted_average():
    # 1 cell with G_left = 2 to T_left = 400 K, G_right = 3 to T_right = 300 K.
    # Analytic T = (G_left * T_left + G_right * T_right) / (G_left + G_right)
    #            = (2*400 + 3*300) / 5 = (800 + 900) / 5 = 340 K.
    conductance = _empty_conductance()
    boundary = _boundary(
        cell_ids=[0, 0], G_arr=[2.0, 3.0], T_ref_arr=[400.0, 300.0],
        kinds=np.array([2, 2], dtype=np.int8),
    )
    op = build_matrix_free_operator(
        conductance, boundary, np.zeros(1, dtype=np.float64))
    T0 = np.array([293.15])
    result = solve_pcg(op, T0, boundary, relative_residual_tolerance=1e-10,
                       max_iterations=1000)
    assert float(result.temperature_K[0]) == pytest.approx(340.0, abs=1e-6)
    assert result.converged
    # Power balance: 0 in, 0 out, 0 imbalance.
    assert result.total_input_power_W == pytest.approx(0.0)
    assert result.total_boundary_heat_out_W == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# B. 1-D slab, fixed T ends, uniform subdivision
# ---------------------------------------------------------------------------

def test_1d_uniform_slab_pcg_matches_linear_profile():
    for n_cells in (2, 5, 10):
        op, boundary, G_int, dx, k, A = _1d_slab(
            n_cells, T_left=350.0, T_right=300.0, power=0.0)
        T0 = np.full(n_cells, 293.15)
        result = solve_pcg(op, T0, boundary,
                           relative_residual_tolerance=1e-10,
                           max_iterations=2000)
        centres = np.array([(i + 0.5) * dx for i in range(n_cells)])
        expected = 350.0 + (300.0 - 350.0) * centres / (n_cells * dx)
        np.testing.assert_allclose(result.temperature_K, expected,
                                   atol=1e-6)


def test_1d_nonuniform_slab_pcg_matches_linear_profile():
    # 4 cells of widths 100, 200, 300, 400 um. Total L = 1 mm.
    widths = [100e-6, 200e-6, 300e-6, 400e-6]
    n = len(widths)
    cell_a = np.arange(n - 1, dtype=np.int64)
    cell_b = np.arange(1, n, dtype=np.int64)
    k = 10.0
    A = 1e-6
    G_int = np.array([
        A / ((widths[i] / 2) / k + (widths[i + 1] / 2) / k)
        for i in range(n - 1)
    ], dtype=np.float64)
    conductance = ConductanceTable(
        edge_id=np.arange(n - 1, dtype=np.int64),
        cell_a=cell_a, cell_b=cell_b, axis=np.zeros(n - 1, dtype=np.int8),
        face_area_m2=np.full(n - 1, A, dtype=np.float64),
        half_distance_a_m=np.array(
            [widths[i] / 2 for i in range(n - 1)], dtype=np.float64),
        half_distance_b_m=np.array(
            [widths[i + 1] / 2 for i in range(n - 1)], dtype=np.float64),
        k_normal_a_W_mK=np.full(n - 1, k, dtype=np.float64),
        k_normal_b_W_mK=np.full(n - 1, k, dtype=np.float64),
        interface_areal_resistance_m2K_W=np.zeros(n - 1, dtype=np.float64),
        resistance_K_W=np.zeros(n - 1, dtype=np.float64),
        conductance_W_K=G_int,
        material_interface=np.zeros(n - 1, dtype=bool),
        interface_rule_index=np.full(n - 1, -1, dtype=np.int32),
    )
    boundary = _boundary(
        cell_ids=[0, n - 1], G_arr=[1.0, 1.0], T_ref_arr=[350.0, 300.0],
        kinds=np.array([2, 2], dtype=np.int8),
    )
    # Boundary: G_face on cell 0 and cell N-1 are the half-cell
    # bulk to the fixed-temperature face.
    G_face_left = A / (widths[0] / (2 * k))
    G_face_right = A / (widths[-1] / (2 * k))
    boundary = _boundary(
        cell_ids=[0, n - 1], G_arr=[G_face_left, G_face_right],
        T_ref_arr=[350.0, 300.0], kinds=np.array([2, 2], dtype=np.int8),
    )
    op = build_matrix_free_operator(conductance, boundary,
                                    np.zeros(n, dtype=np.float64))
    T0 = np.full(n, 293.15)
    result = solve_pcg(op, T0, boundary, relative_residual_tolerance=1e-10,
                       max_iterations=2000)
    # Compute expected cell-centre positions and the linear profile.
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    centres = 0.5 * (edges[:-1] + edges[1:])
    expected = 350.0 + (300.0 - 350.0) * centres / edges[-1]
    np.testing.assert_allclose(result.temperature_K, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# C. 1-D slab, uniform volume generation, symmetric convection
# ---------------------------------------------------------------------------

def test_1d_slab_with_uniform_generation_pcg_matches_parabolic_profile():
    # 5 cells, length 1 mm, uniform q = 1e9 W/m^3, k = 10, A = 1e-6.
    # Symmetric convection ends: both at T_amb = 300 K, h = 100.
    # The per-cell power is q * V / 5. V = A * dx.
    n = 5
    dx = 1e-3 / n
    k = 10.0
    A = 1e-6
    q = 1e9  # W/m^3
    V = A * dx
    pwr_per_cell = q * V  # 1.999...e-4 W each
    G_int = A * k / dx
    conductance = ConductanceTable(
        edge_id=np.arange(n - 1, dtype=np.int64),
        cell_a=np.arange(n - 1, dtype=np.int64),
        cell_b=np.arange(1, n, dtype=np.int64),
        axis=np.zeros(n - 1, dtype=np.int8),
        face_area_m2=np.full(n - 1, A, dtype=np.float64),
        half_distance_a_m=np.full(n - 1, dx / 2, dtype=np.float64),
        half_distance_b_m=np.full(n - 1, dx / 2, dtype=np.float64),
        k_normal_a_W_mK=np.full(n - 1, k, dtype=np.float64),
        k_normal_b_W_mK=np.full(n - 1, k, dtype=np.float64),
        interface_areal_resistance_m2K_W=np.zeros(n - 1, dtype=np.float64),
        resistance_K_W=np.zeros(n - 1, dtype=np.float64),
        conductance_W_K=np.full(n - 1, G_int, dtype=np.float64),
        material_interface=np.zeros(n - 1, dtype=bool),
        interface_rule_index=np.full(n - 1, -1, dtype=np.int32),
    )
    # Convection ends: per-cell G = A / (dx/(2k) + 1/h).
    h = 100.0
    G_end = A / (dx / (2 * k) + 1 / h)
    boundary = _boundary(
        cell_ids=[0, n - 1], G_arr=[G_end, G_end],
        T_ref_arr=[300.0, 300.0], kinds=np.array([1, 1], dtype=np.int8),
    )
    pwr = np.full(n, pwr_per_cell, dtype=np.float64)
    op = build_matrix_free_operator(conductance, boundary, pwr)
    T0 = np.full(n, 293.15)
    result = solve_pcg(op, T0, boundary,
                       relative_residual_tolerance=1e-10, max_iterations=2000)
    # The solution must be symmetric around the centre.
    centre_T = result.temperature_K[len(result.temperature_K) // 2]
    edges_T = result.temperature_K[0]
    assert centre_T > edges_T
    np.testing.assert_allclose(result.temperature_K,
                               result.temperature_K[::-1],
                               atol=1e-4)
    # Power balance.
    assert result.relative_power_imbalance < 1e-8


# ---------------------------------------------------------------------------
# D. Convection boundary uses d/k + 1/h
# ---------------------------------------------------------------------------

def test_convection_boundary_uses_half_cell_bulk_resistance():
    # 1 cell with d = 1e-4, k = 10, h = 100, A = 1e-6.
    # G = A / (d/k + 1/h) = 1e-6 / (1e-5 + 0.01) = 1e-6 / 0.01001
    # = 9.99e-5 W/K.
    d = 1e-4
    k = 10.0
    h = 100.0
    A = 1e-6
    expected_G = A / (d / k + 1 / h)
    boundary = _boundary(
        cell_ids=[0], G_arr=[expected_G], T_ref_arr=[300.0])
    conductance = _empty_conductance()
    op = build_matrix_free_operator(conductance, boundary,
                                    np.zeros(1, dtype=np.float64))
    # The diagonal equals the G (since 1 cell, 1 link, no internal).
    assert op.diagonal_W_K[0] == pytest.approx(expected_G, rel=1e-10)


# ---------------------------------------------------------------------------
# E. Fixed temperature boundary
# ---------------------------------------------------------------------------

def test_fixed_temperature_boundary_uses_half_cell_bulk_resistance():
    d = 1e-4
    k = 10.0
    A = 1e-6
    expected_G = A / (d / k)
    boundary = _boundary(
        cell_ids=[0], G_arr=[expected_G], T_ref_arr=[350.0],
        kinds=np.array([2], dtype=np.int8))
    conductance = _empty_conductance()
    op = build_matrix_free_operator(conductance, boundary,
                                    np.zeros(1, dtype=np.float64))
    assert op.diagonal_W_K[0] == pytest.approx(expected_G, rel=1e-10)


# ---------------------------------------------------------------------------
# F. Fully adiabatic network is rejected
# ---------------------------------------------------------------------------

def test_fully_adiabatic_network_is_rejected():
    conductance = ConductanceTable(
        edge_id=np.array([0], dtype=np.int64),
        cell_a=np.array([0], dtype=np.int64),
        cell_b=np.array([1], dtype=np.int64),
        axis=np.array([0], dtype=np.int8),
        face_area_m2=np.array([1e-6], dtype=np.float64),
        half_distance_a_m=np.array([1e-4], dtype=np.float64),
        half_distance_b_m=np.array([1e-4], dtype=np.float64),
        k_normal_a_W_mK=np.array([10.0], dtype=np.float64),
        k_normal_b_W_mK=np.array([10.0], dtype=np.float64),
        interface_areal_resistance_m2K_W=np.array([0.0], dtype=np.float64),
        resistance_K_W=np.array([0.0], dtype=np.float64),
        conductance_W_K=np.array([0.05], dtype=np.float64),
        material_interface=np.array([False]),
        interface_rule_index=np.array([-1], dtype=np.int32),
    )
    empty_b = BoundaryLinkTable(
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
    with pytest.raises(Exception):
        validate_anchored_components(
            2, conductance.cell_a, conductance.cell_b, empty_b)


# ---------------------------------------------------------------------------
# I. Jacobi update formula matches textbook
# ---------------------------------------------------------------------------

def test_jacobi_update_matches_textbook_formula():
    op, boundary, _, _, _, _ = _1d_slab(
        5, T_left=350.0, T_right=300.0, power=0.0)
    T = np.full(5, 293.15)
    # Manual update.
    A_T = op.apply(T)
    residual = op.rhs_W - A_T
    delta = 0.7 * residual / op.diagonal_W_K
    T_new = T + delta
    # Apply the solver for one check.
    result = solve_weighted_jacobi(
        op, T, boundary, omega=0.7, max_iterations=1,
        check_interval=1)
    np.testing.assert_allclose(result.temperature_K, T_new, atol=1e-12)


# ---------------------------------------------------------------------------
# J. PCG and Jacobi agree on a small slab
# ---------------------------------------------------------------------------

def test_pcg_and_jacobi_agree_on_small_slab():
    op, boundary, _, _, _, _ = _1d_slab(
        5, T_left=350.0, T_right=300.0, power=0.0)
    T0 = np.full(5, 293.15)
    r_pcg = solve_pcg(op, T0, boundary,
                      relative_residual_tolerance=1e-10, max_iterations=2000)
    # Jacobi with a higher iteration budget.
    r_jac = solve_weighted_jacobi(
        op, T0, boundary, omega=0.7, max_iterations=200_000,
        relative_residual_tolerance=1e-10, max_temperature_update_tolerance=1e-10,
        check_interval=1)
    np.testing.assert_allclose(r_pcg.temperature_K, r_jac.temperature_K,
                               atol=1e-5)


# ---------------------------------------------------------------------------
# M. Operator diagnostics
# ---------------------------------------------------------------------------

def test_relative_residual_decreases_across_iterations():
    op, boundary, _, _, _, _ = _1d_slab(
        5, T_left=350.0, T_right=300.0, power=1.0)
    T0 = np.full(5, 293.15)
    r0 = op.relative_residual(T0)
    result = solve_pcg(op, T0, boundary,
                       relative_residual_tolerance=1e-12,
                       max_iterations=2000)
    assert result.final_relative_residual < r0
    assert result.converged
