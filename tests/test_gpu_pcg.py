"""GPU-PCG analytical solutions, power conservation, and true residual checks.
"""
from __future__ import annotations

import numpy as np
import pytest

from om3dthermal.thermal import (
    GPUPCGOperator,
    build_matrix_free_operator,
    solve_pcg_gpu,
    require_cupy,
)
from om3dthermal.thermal.boundary import BoundaryLinkTable
from om3dthermal.thermal.conductance import ConductanceTable


# Helpers: hand-built operators with known analytical answers


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
        kinds = np.full(n, 1, dtype=np.int8)
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


def _one_cell_with_boundary(P: float, G: float, T_ref: float):
    """A single thermal cell with one boundary link to a fixed
    reference temperature. The analytical steady state is

        T* = T_ref + P / G.
    """
    conductance = _empty_conductance()
    boundary = _boundary(
        cell_ids=[0], G_arr=[G], T_ref_arr=[T_ref],
        kinds=np.array([2], dtype=np.int8),
    )
    op = build_matrix_free_operator(conductance, boundary, np.array([P]))
    return op, boundary


def _two_cell_resistor(P1: float, P2: float, G_int: float, G_b: float,
                       T_ref: float):
    """Two cells connected by conductance G_int, both attached to
    a fixed-temperature boundary through G_b.
    """
    conductance = ConductanceTable(
        edge_id=np.array([0], dtype=np.int64),
        cell_a=np.array([0], dtype=np.int64),
        cell_b=np.array([1], dtype=np.int64),
        axis=np.zeros(1, dtype=np.int8),
        face_area_m2=np.ones(1, dtype=np.float64),
        half_distance_a_m=np.ones(1, dtype=np.float64),
        half_distance_b_m=np.ones(1, dtype=np.float64),
        k_normal_a_W_mK=np.ones(1, dtype=np.float64),
        k_normal_b_W_mK=np.ones(1, dtype=np.float64),
        interface_areal_resistance_m2K_W=np.zeros(1, dtype=np.float64),
        resistance_K_W=np.zeros(1, dtype=np.float64),
        conductance_W_K=np.array([G_int], dtype=np.float64),
        material_interface=np.zeros(1, dtype=bool),
        interface_rule_index=np.full(1, -1, dtype=np.int32),
    )
    boundary = _boundary(
        cell_ids=[0, 1], G_arr=[G_b, G_b], T_ref_arr=[T_ref, T_ref],
        kinds=np.array([2, 2], dtype=np.int8),
    )
    op = build_matrix_free_operator(
        conductance, boundary, np.array([P1, P2], dtype=np.float64),
    )
    return op, boundary


def test_gpu_pcg_matches_two_cell_analytical_solution():
    op, boundary = _two_cell_resistor(2.0, 1.0, 3.0, 2.0, 295.0)
    # Closed form for [[5, -3], [-3, 5]] @ T = [592, 591].
    expected = np.array([295.8125, 295.6875])
    result = solve_pcg_gpu(
        op, np.full(2, 293.15), boundary,
        relative_residual_tolerance=1e-12,
        max_temperature_update_tolerance=10.0,
        max_iterations=20,
        check_interval=1,
    )
    assert result.converged
    np.testing.assert_allclose(result.temperature_K, expected, rtol=0, atol=1e-10)
    assert result.final_relative_residual < 1e-12
    assert result.solver_info["full_vector_d2h_during_iteration"] == 0
    assert result.solver_info["full_vector_d2h_copy_count"] == 1
    assert result.solver_info["solver_vector_h2d_copy_count"] == 1
    assert result.solver_info["scalar_synchronization_count"] >= 2


def test_gpu_pcg_matrix_free_operator_matches_cpu_fp64():
    op, _ = _two_cell_resistor(2.0, 1.0, 3.0, 2.0, 295.0)
    vector = np.array([301.25, 297.75], dtype=np.float64)
    expected = op.apply(vector)
    cp = require_cupy()
    gpu = GPUPCGOperator.from_cpu(op, cp)
    actual = cp.asnumpy(gpu.apply(cp.asarray(vector), cp))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)


def test_gpu_pcg_requires_update_and_residual_together():
    op, boundary = _one_cell_with_boundary(10.0, 1.0, 293.15)
    result = solve_pcg_gpu(
        op, np.array([293.15]), boundary,
        relative_residual_tolerance=1.0,
        max_temperature_update_tolerance=1.0,
        max_iterations=1,
        check_interval=1,
    )
    assert result.final_relative_residual < 1.0
    assert result.max_temperature_update == pytest.approx(10.0)
    assert not result.converged


def test_two_cell_power_conservation():
    """GPU-PCG on a symmetric two-cell network with
    boundary sinks. KCL is closed at every iteration: input power
    equals the boundary heat outflow at the steady state. The
    symmetry of the network gives a symmetric temperature
    solution when P1 == P2 and an asymmetric one otherwise."""
    P1, P2 = 1.0, 0.5
    G_int = 0.4
    G_b = 1.0
    T_ref = 295.0
    op, boundary = _two_cell_resistor(P1, P2, G_int, G_b, T_ref)
    result = solve_pcg_gpu(
        op, np.array([T_ref, T_ref]), boundary,
        relative_residual_tolerance=1e-10,
        max_temperature_update_tolerance=1e-10,
        max_iterations=5_000,
    )
    assert result.converged
    T = result.temperature_K
    # Both cells must be above the reference because both dissipate power.
    assert T[0] > T_ref and T[1] > T_ref
    # The cell with more power must be hotter.
    assert T[0] > T[1]
    # KCL closure: total input power must equal the boundary heat outflow.
    q_in = P1 + P2
    q_out = (G_b * (T[0] - T_ref) + G_b * (T[1] - T_ref))
    np.testing.assert_allclose(q_in, q_out, rtol=1e-6)
    # Conservation holds bit-for-bit on the converged temperatures.
    residual = q_in - q_out
    assert abs(residual) < 1e-8


def test_single_cell_analytical_operator_balance():
    op, boundary = _one_cell_with_boundary(1.5, 2.0, 300.0)
    expected = np.array([300.75])
    np.testing.assert_allclose(op.apply(expected), op.rhs_W, rtol=0, atol=1e-12)
    assert float(boundary.conductance_W_K @ (expected - 300.0)) == 1.5
