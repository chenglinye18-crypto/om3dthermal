"""CPU thermal-resistance-network relaxation solver.

The solver is the standard simultaneous-update Jacobi-style
relaxation of the 3D thermal resistance network.  The per-cell
update is

    delta_Q_i = P_i - sum_j G_ij (T_i - T_j) - sum_b G_ib (T_i - T_b)
    R_eff_i  = 1 / ( sum_j G_ij + sum_b G_ib )
    delta_T_i = alpha * delta_Q_i * R_eff_i
    T_new_i   = T_old_i + delta_T_i

CPU and GPU implementations are required to be exactly the same
formula.  The CPU path below uses the existing
:class:`MatrixFreeThermalOperator` for ``A T`` and
:func:`om3dthermal.thermal.steady_state._build_thermal_resistance`
for ``R_eff``; the GPU path lives in
:mod:`om3dthermal.thermal.gpu_relaxation`.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .boundary import BoundaryLinkTable
from .operator import MatrixFreeThermalOperator
from .steady_state import (
    SteadyStateResult,
    _build_thermal_resistance,
    _global_power_balance,
)


def solve_thermal_resistance_relaxation(
    operator: MatrixFreeThermalOperator,
    initial_temperature: np.ndarray,
    boundary: BoundaryLinkTable,
    *,
    alpha: float = 0.7,
    relative_residual_tolerance: float = 1e-8,
    max_temperature_update_tolerance: float = 1e-6,
    max_iterations: int = 100_000,
    check_interval: int = 10,
) -> SteadyStateResult:
    """Solve the thermal-resistance network by simultaneous relaxation.

    Convergence requires BOTH
    ``relative_heat_flow_residual < relative_residual_tolerance`` AND
    ``max_abs_delta_T < max_temperature_update_tolerance`` to be
    satisfied at the same ``check_interval`` boundary.  NaN / inf
    and temperatures below 0 K are divergence signals that abort
    the iteration.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    T_old = np.array(initial_temperature, dtype=np.float64, copy=True)
    if T_old.shape != (operator.cell_count,):
        raise ValueError(
            f"initial_temperature has shape {T_old.shape}; expected "
            f"({operator.cell_count},)")
    if np.any(T_old < 0):
        raise ValueError(
            "initial_temperature contains values below 0 K")
    R_eff = _build_thermal_resistance(operator)
    workspace = np.empty(operator.cell_count, dtype=np.float64)
    t0 = time.perf_counter()
    initial_residual = operator.relative_residual(T_old)
    residual_history: list[float] = [float(initial_residual)]
    update_history: list[float] = []
    converged = False
    iterations = 0
    last_update = float("inf")
    last_relative = float(initial_residual)
    while iterations < max_iterations:
        # delta_Q = b - A T
        operator.apply(T_old, out=workspace)
        delta_Q = operator.rhs_W - workspace
        if not np.all(np.isfinite(delta_Q)):
            break
        delta_T = alpha * delta_Q * R_eff
        if not np.all(np.isfinite(delta_T)):
            break
        T_new = T_old + delta_T
        if not np.all(np.isfinite(T_new)) or np.any(T_new < 0):
            break
        last_update = float(np.max(np.abs(delta_T)))
        T_old = T_new
        iterations += 1
        if iterations % check_interval == 0 or iterations == max_iterations:
            operator.apply(T_old, out=workspace)
            last_relative = float(np.linalg.norm(
                operator.rhs_W - workspace)) / max(
                float(np.linalg.norm(operator.rhs_W)), 1e-30)
            residual_history.append(last_relative)
            update_history.append(last_update)
            if (last_relative < relative_residual_tolerance
                    and last_update < max_temperature_update_tolerance):
                converged = True
                break
    # Always include the final residual / update in the history so the
    # last entry is the actual converged value.
    operator.apply(T_old, out=workspace)
    last_relative = float(np.linalg.norm(
        operator.rhs_W - workspace)) / max(
        float(np.linalg.norm(operator.rhs_W)), 1e-30)
    residual_history.append(last_relative)
    update_history.append(last_update)
    elapsed = time.perf_counter() - t0
    q_input, q_out, imbalance, rel_imbalance = _global_power_balance(
        operator, boundary, T_old)
    return SteadyStateResult(
        temperature_K=T_old,
        method="thermal_resistance_relaxation",
        converged=converged,
        iterations=iterations,
        solver_info={
            "alpha": alpha,
            "check_interval": check_interval,
            "relative_residual_tolerance": relative_residual_tolerance,
            "max_temperature_update_tolerance":
                max_temperature_update_tolerance,
            "max_iterations": max_iterations,
            "matvec_count": operator.matvec_count,
        },
        initial_residual=float(initial_residual),
        final_absolute_residual=float(np.linalg.norm(
            operator.rhs_W - workspace)),
        final_relative_residual=float(last_relative),
        max_temperature_update=last_update if iterations else None,
        min_temperature_K=float(T_old.min()),
        max_temperature_K=float(T_old.max()),
        mean_temperature_K=float(T_old.mean()),
        total_input_power_W=q_input,
        total_boundary_heat_out_W=q_out,
        global_power_imbalance_W=imbalance,
        relative_power_imbalance=rel_imbalance,
        residual_history=residual_history,
        update_norm_history=update_history,
        solve_seconds=elapsed,
    )


__all__ = ["solve_thermal_resistance_relaxation"]
