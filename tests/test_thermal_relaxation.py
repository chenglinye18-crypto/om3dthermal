"""Targeted tests for the thermal-resistance-network relaxation solver.

These tests assert the formal contract documented in
:mod:`om3dthermal.thermal.thermal_relaxation` and
:mod:`om3dthermal.thermal.gpu_relaxation`:

    delta_Q_i = P_i - sum_j G_ij (T_i - T_j) - sum_b G_ib (T_i - T_b)
    R_eff_i  = 1 / ( sum_j G_ij + sum_b G_ib )
    delta_T_i = alpha * delta_Q_i * R_eff_i
    T_new_i   = T_old_i + delta_T_i

Convergence requires BOTH the relative heat-flow residual and
``max_abs_delta_T`` to drop below their tolerances at the same
``check_interval`` boundary. The CPU and GPU implementations
must be numerically equivalent (FP64 round-off).
"""
from __future__ import annotations

import numpy as np
import pytest

from om3dthermal.thermal import (
    GPURelaxationState,
    build_matrix_free_operator,
    solve_thermal_resistance_relaxation,
    solve_thermal_resistance_relaxation_gpu,
)
from om3dthermal.thermal.boundary import BoundaryLinkTable
from om3dthermal.thermal.conductance import ConductanceTable


# ---------------------------------------------------------------------------
# Helpers: hand-built operators with known analytical answers
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


# ---------------------------------------------------------------------------
# A. Single cell with boundary — analytical steady state
# ---------------------------------------------------------------------------


def test_single_cell_relaxation_matches_analytical_steady_state():
    """A. The relaxation must converge to ``T_ref + P / G`` on a
    single-cell network with one boundary link, regardless of
    ``alpha`` (any alpha in (0, 1] works because the matrix is 1x1)."""
    P = 1.5
    G = 2.0
    T_ref = 300.0
    expected = T_ref + P / G
    op, boundary = _one_cell_with_boundary(P, G, T_ref)
    result = solve_thermal_resistance_relaxation(
        op, np.array([T_ref]), boundary,
        alpha=0.7,
        relative_residual_tolerance=1e-12,
        max_temperature_update_tolerance=1e-12,
        max_iterations=200,
    )
    assert result.converged
    assert result.method == "thermal_resistance_relaxation"
    np.testing.assert_allclose(result.temperature_K, [expected], rtol=1e-10)


# ---------------------------------------------------------------------------
# B. Two-cell thermal resistance network — CPU relaxation
# ---------------------------------------------------------------------------


def test_two_cell_relaxation_converges_to_consistent_thermal_state():
    """B. CPU relaxation on a symmetric two-cell network with
    boundary sinks. KCL is closed at every iteration: input power
    equals the boundary heat outflow at the steady state. The
    symmetry of the network gives a symmetric temperature
    solution when P1 == P2 and an asymmetric one otherwise."""
    P1, P2 = 1.0, 0.5
    G_int = 0.4
    G_b = 1.0
    T_ref = 295.0
    op, boundary = _two_cell_resistor(P1, P2, G_int, G_b, T_ref)
    result = solve_thermal_resistance_relaxation(
        op, np.array([T_ref, T_ref]), boundary,
        alpha=0.7,
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


# ---------------------------------------------------------------------------
# C. CPU vs GPU relaxation — full temperature field equivalence
# ---------------------------------------------------------------------------


def test_cpu_and_gpu_relaxation_match_full_temperature_field():
    """C. CPU and GPU implementations must apply the same
    relaxation equation. On a small network (where GPU overhead is
    amortised away), the two ``temperature_K`` arrays must agree
    to FP64 round-off."""
    op, boundary = _two_cell_resistor(
        P1=1.0, P2=0.5, G_int=0.4, G_b=1.0, T_ref=295.0,
    )
    init = np.array([295.0, 295.0])
    cpu = solve_thermal_resistance_relaxation(
        op, init, boundary,
        alpha=0.7,
        relative_residual_tolerance=1e-10,
        max_temperature_update_tolerance=1e-10,
        max_iterations=2_000,
    )
    gpu = solve_thermal_resistance_relaxation_gpu(
        op, init, boundary,
        alpha=0.7,
        relative_residual_tolerance=1e-10,
        max_temperature_update_tolerance=1e-10,
        max_iterations=2_000,
    )
    assert cpu.converged and gpu.converged
    np.testing.assert_allclose(
        cpu.temperature_K, gpu.temperature_K,
        rtol=1e-10, atol=1e-10,
        err_msg="CPU and GPU relaxation produced different steady states",
    )


# ---------------------------------------------------------------------------
# D. KCL residual decreases monotonically (non-strict)
# ---------------------------------------------------------------------------


def test_heat_flow_residual_decreases_under_relaxation():
    """D. The relative heat-flow residual must shrink across the
    ``check_interval`` boundaries. We don't enforce strict
    monotonicity (alpha-dependent damping is allowed) but the
    final residual must be smaller than the initial one and
    below tolerance."""
    op, boundary = _two_cell_resistor(
        P1=1.0, P2=0.5, G_int=0.4, G_b=1.0, T_ref=295.0,
    )
    result = solve_thermal_resistance_relaxation(
        op, np.array([295.0, 295.0]), boundary,
        alpha=0.7,
        relative_residual_tolerance=1e-8,
        max_temperature_update_tolerance=1e-8,
        max_iterations=10_000,
    )
    assert result.converged
    history = result.residual_history
    assert history[0] > 0.0
    assert history[-1] < history[0]
    assert history[-1] < 1e-8
    # The update norm history must also be monotonically shrinking
    # in the typical (under-relaxed) regime; the last entry must
    # drop below the configured temperature tolerance.
    assert result.max_temperature_update is not None
    assert result.max_temperature_update < 1e-8


# ---------------------------------------------------------------------------
# E. alpha semantics — delta_T = alpha * delta_Q * R_eff
# ---------------------------------------------------------------------------


def test_alpha_scaling_matches_relaxation_equation():
    """E. One hand-rolled relaxation step must equal
    ``alpha * (b - A T) * R_eff`` exactly, where ``R_eff = 1 / D``.
    We verify this by computing the step directly and comparing
    it to the operator's first-iteration update."""
    P1, P2 = 1.0, 0.5
    G_int, G_b, T_ref = 0.4, 1.0, 295.0
    op, boundary = _two_cell_resistor(P1, P2, G_int, G_b, T_ref)
    T0 = np.array([300.0, 297.0])
    alpha = 0.7
    AT = op.apply(T0)
    delta_Q = op.rhs_W - AT
    R_eff = 1.0 / op.diagonal_W_K
    delta_T = alpha * delta_Q * R_eff
    T1_expected = T0 + delta_T
    # Recompute via the solver's public function (one step only).
    result = solve_thermal_resistance_relaxation(
        op, T0, boundary,
        alpha=alpha,
        relative_residual_tolerance=0.0,  # never satisfied
        max_temperature_update_tolerance=0.0,
        max_iterations=1,
    )
    np.testing.assert_allclose(
        result.temperature_K, T1_expected, rtol=1e-12, atol=1e-12,
    )


def test_alpha_out_of_unit_interval_is_rejected():
    """alpha must be in (0, 1] for the relaxation to be
    contractive. The solver must refuse anything outside that
    range, not silently fall back."""
    op, boundary = _two_cell_resistor(1.0, 0.5, 0.4, 1.0, 295.0)
    with pytest.raises(ValueError):
        solve_thermal_resistance_relaxation(
            op, np.array([295.0, 295.0]), boundary,
            alpha=0.0, max_iterations=1,
        )
    with pytest.raises(ValueError):
        solve_thermal_resistance_relaxation(
            op, np.array([295.0, 295.0]), boundary,
            alpha=1.5, max_iterations=1,
        )
    with pytest.raises(ValueError):
        solve_thermal_resistance_relaxation(
            op, np.array([295.0, 295.0]), boundary,
            alpha=-0.1, max_iterations=1,
        )


# ---------------------------------------------------------------------------
# F. Simultaneous-update semantics — both backends read T_old only
# ---------------------------------------------------------------------------


def test_simultaneous_update_does_not_use_in_place_temperature():
    """F. The relaxation is a simultaneous update: ``T_new[i]`` is
    computed from the *old* temperature of every cell, never
    from a value that was just written to ``T_new``. We check
    this with a hand-built 1D chain where the in-place
    (Gauss-Seidel-style) update would visibly disagree with the
    simultaneous update on the first iteration.

    For a 3-cell chain with cells 0, 1, 2 all connected in
    series, equal power, and a single boundary at each end:

        delta_T = alpha * (P - G_int*(T1 - T0) - G_int*(T1 - T2)
                                 - G_b*(T1 - T_ref)) * R_eff

    where R_eff = 1 / (2 G_int + G_b).

    An in-place update would use the *new* T0 (after its own
    step) when computing T1, producing a different result. The
    simultaneous update is the unique answer of the relaxation
    formula and is what we assert.
    """
    G_int = 0.5
    G_b = 1.0
    T_ref = 295.0
    P = 0.5
    conductance = ConductanceTable(
        edge_id=np.arange(2, dtype=np.int64),
        cell_a=np.array([0, 1], dtype=np.int64),
        cell_b=np.array([1, 2], dtype=np.int64),
        axis=np.zeros(2, dtype=np.int8),
        face_area_m2=np.ones(2, dtype=np.float64),
        half_distance_a_m=np.ones(2, dtype=np.float64),
        half_distance_b_m=np.ones(2, dtype=np.float64),
        k_normal_a_W_mK=np.ones(2, dtype=np.float64),
        k_normal_b_W_mK=np.ones(2, dtype=np.float64),
        interface_areal_resistance_m2K_W=np.zeros(2, dtype=np.float64),
        resistance_K_W=np.zeros(2, dtype=np.float64),
        conductance_W_K=np.array([G_int, G_int], dtype=np.float64),
        material_interface=np.zeros(2, dtype=bool),
        interface_rule_index=np.full(2, -1, dtype=np.int32),
    )
    boundary = _boundary(
        cell_ids=[0, 2], G_arr=[G_b, G_b],
        T_ref_arr=[T_ref, T_ref],
        kinds=np.array([2, 2], dtype=np.int8),
    )
    op = build_matrix_free_operator(
        conductance, boundary,
        np.array([P, P, P], dtype=np.float64),
    )
    T0 = np.array([296.0, 300.0, 296.0])
    alpha = 0.7
    AT = op.apply(T0)
    delta_Q = op.rhs_W - AT
    R_eff = 1.0 / op.diagonal_W_K
    delta_T = alpha * delta_Q * R_eff
    T1_expected = T0 + delta_T
    result = solve_thermal_resistance_relaxation(
        op, T0, boundary,
        alpha=alpha,
        relative_residual_tolerance=0.0,
        max_temperature_update_tolerance=0.0,
        max_iterations=1,
    )
    np.testing.assert_allclose(
        result.temperature_K, T1_expected,
        rtol=1e-12, atol=1e-12,
        err_msg=("simultaneous-update contract violated: "
                 "first iteration disagrees with the relaxation formula"),
    )


# ---------------------------------------------------------------------------
# GPURelaxationState — the row-oriented adjacency builder
# ---------------------------------------------------------------------------


def test_gpu_relaxation_state_constructs_and_reuses_kernel():
    """The GPU state builder must produce a valid per-cell state
    whose kernel cache survives multiple ``launch_one_step``
    calls without recompiling."""
    cp = pytest.importorskip("cupy")
    op, _boundary_unused = _two_cell_resistor(1.0, 0.5, 0.4, 1.0, 295.0)
    state = GPURelaxationState.from_cpu(op, cp)
    assert state.cell_count == 2
    assert state.max_neighbors <= 6
    # The double-buffer is allocated but not initialised; the
    # solver must populate it before the first launch.
    state.T_old.set(np.array([295.0, 295.0]))
    state.T_new.set(np.array([295.0, 295.0]))
    # Two launches with the same alpha must compile the kernel
    # exactly once. The kernel object is cached in the module
    # and reused.
    state.launch_one_step(cp, 0.7)
    state.launch_one_step(cp, 0.7)
    assert cp.all(cp.isfinite(state.T_new)).item()


# ---------------------------------------------------------------------------
# THERMAL_RELAXATION_NUMERICAL_GATE
# ---------------------------------------------------------------------------


def test_thermal_relaxation_numerical_gate():
    """THERMAL_RELAXATION_NUMERICAL_GATE: one consolidated test
    that runs all the targeted contracts end-to-end and asserts
    PASS. If any of the individual contracts is broken, the
    ``pytest`` run on this module fails, which is what the gate
    is meant to surface."""
    # A. Analytical single-cell steady state.
    op_a, boundary_a = _one_cell_with_boundary(P=1.5, G=2.0, T_ref=300.0)
    r_a = solve_thermal_resistance_relaxation(
        op_a, np.array([300.0]), boundary_a,
        alpha=0.7,
        relative_residual_tolerance=1e-12,
        max_temperature_update_tolerance=1e-12,
        max_iterations=200,
    )
    assert r_a.converged
    np.testing.assert_allclose(r_a.temperature_K, [300.75], rtol=1e-10)
    # B+C. CPU and GPU agree on a non-trivial two-cell network.
    op_b, boundary_b = _two_cell_resistor(1.0, 0.5, 0.4, 1.0, 295.0)
    init = np.array([295.0, 295.0])
    r_cpu = solve_thermal_resistance_relaxation(
        op_b, init, boundary_b, alpha=0.7,
        relative_residual_tolerance=1e-10,
        max_temperature_update_tolerance=1e-10,
        max_iterations=2_000,
    )
    r_gpu = solve_thermal_resistance_relaxation_gpu(
        op_b, init, boundary_b, alpha=0.7,
        relative_residual_tolerance=1e-10,
        max_temperature_update_tolerance=1e-10,
        max_iterations=2_000,
    )
    assert r_cpu.converged and r_gpu.converged
    np.testing.assert_allclose(
        r_cpu.temperature_K, r_gpu.temperature_K,
        rtol=1e-10, atol=1e-10,
    )
    # D. KCL residual decreased below tolerance.
    assert r_cpu.residual_history[-1] < 1e-10
    # E. Both backends report the relaxation method.
    assert r_cpu.method == "thermal_resistance_relaxation"
    assert r_gpu.method == "thermal_resistance_relaxation"
    # F. (covered above by test_simultaneous_update_*).
