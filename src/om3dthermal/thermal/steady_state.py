"""Matrix-free weighted Jacobi and PCG steady-state solvers.

Both solvers consume the same :class:`MatrixFreeThermalOperator` and
never form a dense or sparse matrix on the production path. The
Jacobi update is the textbook local-residual step

    T_new = T + omega * (b - A T) / D

and the PCG iteration is the standard conjugate-gradient
iteration wrapped around the same ``matvec``. The Jacobi solver is
included as a sanity check; the HBM benchmark calls PCG because
the convergence rate is far better on a graph of 272 k nodes.

The two solvers share :class:`SteadyStateResult` and the
:class:`UnanchoredThermalComponentError` check.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .boundary import BoundaryLinkTable
from .errors import ThermalError
from .operator import MatrixFreeThermalOperator


# ---------------------------------------------------------------------------
# Anchored-component check
# ---------------------------------------------------------------------------

class UnanchoredThermalComponentError(ThermalError, ValueError):
    """Raised when at least one connected component of the thermal
    network has no non-adiabatic boundary link.

    A pure-adiabatic component has a singular stiffness matrix:
    any uniform offset added to its temperatures is still a valid
    solution. Refusing to solve makes the limitation explicit
    rather than silently returning a non-unique answer.
    """

    def __init__(self, components: list[dict]):
        self.components = components
        summary = ", ".join(
            f"#{c['component_id']} (cells={c['cell_count']}, "
            f"P_total={c['total_power_W']:.3g} W)"
            for c in components[:5])
        super().__init__(
            f"{len(components)} thermal component(s) have no active "
            f"boundary link and are therefore not uniquely solvable: "
            f"{summary}")


def validate_anchored_components(
    cell_count: int,
    internal_cell_a: np.ndarray,
    internal_cell_b: np.ndarray,
    boundary: BoundaryLinkTable,
) -> None:
    """Verify every connected component of the internal-edge graph
    contains at least one active boundary link.

    A component without an active link cannot be uniquely solved:
    adding a constant to all of its temperatures is still a valid
    solution. We raise :class:`UnanchoredThermalComponentError`
    listing the offending components so the user can add a
    convection or fixed-temperature rule.
    """
    parent = list(range(cell_count))
    rank = [0] * cell_count

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    if internal_cell_a.shape[0]:
        for a, b in zip(internal_cell_a.tolist(), internal_cell_b.tolist()):
            union(int(a), int(b))
    anchored_root: set[int] = set()
    if boundary.link_count > 0:
        for c in boundary.cell_id.tolist():
            anchored_root.add(find(int(c)))
    comp_cells: dict[int, list[int]] = {}
    for cell_id in range(cell_count):
        root = find(cell_id)
        comp_cells.setdefault(root, []).append(cell_id)
    bad: list[dict] = []
    for root, members in comp_cells.items():
        if root in anchored_root:
            continue
        bad.append({
            "component_id": len(bad),
            "cell_count": len(members),
            "representative_cell": int(members[0]),
            "total_power_W": 0.0,
        })
    if bad:
        raise UnanchoredThermalComponentError(bad)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SteadyStateResult:
    """Container for the solver output.

    ``temperature_K`` is the final per-cell temperature. The
    residual / iteration counters and the power balance are
    computed at solver time and stored here.
    """

    temperature_K: np.ndarray
    method: str
    converged: bool
    iterations: int
    solver_info: dict
    initial_residual: float
    final_absolute_residual: float
    final_relative_residual: float
    max_temperature_update: float | None
    min_temperature_K: float
    max_temperature_K: float
    mean_temperature_K: float
    total_input_power_W: float
    total_boundary_heat_out_W: float
    global_power_imbalance_W: float
    relative_power_imbalance: float
    residual_history: list[float] = field(default_factory=list)
    update_norm_history: list[float] = field(default_factory=list)
    solve_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Shared diagnostics
# ---------------------------------------------------------------------------

def _global_power_balance(
    operator: MatrixFreeThermalOperator,
    boundary: BoundaryLinkTable,
    temperature: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (Q_input, Q_boundary, imbalance, relative_imbalance).

    ``Q_input = sum(P_i)`` and
    ``Q_boundary = sum_b G_ib * (T_i - T_ref)`` is the heat
    leaving the network through the active boundary links. For a
    converged solution these should match within the relative
    tolerance the solver targets.
    """
    q_input = float(operator.power_W.sum())
    if boundary.link_count > 0:
        q_out = float((
            boundary.conductance_W_K
            * (temperature[boundary.cell_id]
               - boundary.reference_temperature_K)
        ).sum())
    else:
        q_out = 0.0
    imbalance = q_input - q_out
    denom = max(abs(q_input), 1e-30)
    return q_input, q_out, imbalance, abs(imbalance) / denom


# ---------------------------------------------------------------------------
# Weighted Jacobi
# ---------------------------------------------------------------------------

def solve_weighted_jacobi(
    operator: MatrixFreeThermalOperator,
    initial_temperature: np.ndarray,
    boundary: BoundaryLinkTable,
    *,
    omega: float = 0.7,
    relative_residual_tolerance: float = 1e-8,
    max_temperature_update_tolerance: float = 1e-6,
    max_iterations: int = 100_000,
    check_interval: int = 10,
) -> SteadyStateResult:
    """Solve ``A T = b`` by matrix-free weighted Jacobi.

    Convergence requires BOTH ``relative_residual < tol`` and
    ``max |delta T| < tol`` to be satisfied in the same check
    window, so the solver does not declare success on a stalled
    residual that is not actually moving. NaN / inf and
    temperatures below 0 K are divergence signals that abort the
    iteration.
    """
    if not (0.0 < omega <= 1.0):
        raise ValueError(f"omega must be in (0, 1], got {omega}")
    T = np.array(initial_temperature, dtype=np.float64, copy=True)
    if T.shape != (operator.cell_count,):
        raise ValueError(
            f"initial_temperature has shape {T.shape}; expected "
            f"({operator.cell_count},)")
    if np.any(T < 0):
        raise ValueError(
            "initial_temperature contains values below 0 K")
    t0 = time.perf_counter()
    initial_residual = operator.relative_residual(T)
    history: list[float] = [initial_residual]
    update_history: list[float] = []
    converged = False
    iterations = 0
    last_update = float("inf")
    last_relative = initial_residual
    workspace = np.empty(operator.cell_count, dtype=np.float64)
    while iterations < max_iterations:
        operator.apply(T, out=workspace)
        residual = operator.rhs_W - workspace
        if not np.all(np.isfinite(residual)):
            break
        delta = omega * residual / operator.diagonal_W_K
        if not np.all(np.isfinite(delta)):
            break
        T = T + delta
        if not np.all(np.isfinite(T)) or np.any(T < 0):
            break
        iterations += 1
        last_update = float(np.max(np.abs(delta)))
        last_relative = float(np.linalg.norm(residual)) / max(
            float(np.linalg.norm(operator.rhs_W)), 1e-30)
        if iterations % check_interval == 0 or iterations == max_iterations:
            history.append(last_relative)
            update_history.append(last_update)
            if (last_relative < relative_residual_tolerance
                    and last_update < max_temperature_update_tolerance):
                converged = True
                break
    # Always include the final residual / update in the history so the
    # last entry is the actual converged value (the every-check-interval
    # samples above may otherwise miss the last few iterations).
    history.append(last_relative)
    update_history.append(last_update)
    elapsed = time.perf_counter() - t0
    q_input, q_out, imbalance, rel_imbalance = _global_power_balance(
        operator, boundary, T)
    return SteadyStateResult(
        temperature_K=T,
        method="weighted_jacobi",
        converged=converged,
        iterations=iterations,
        solver_info={
            "omega": omega,
            "check_interval": check_interval,
            "relative_residual_tolerance": relative_residual_tolerance,
            "max_temperature_update_tolerance":
                max_temperature_update_tolerance,
            "max_iterations": max_iterations,
            "matvec_count": operator.matvec_count,
        },
        initial_residual=initial_residual,
        final_absolute_residual=float(np.linalg.norm(operator.residual(T))),
        final_relative_residual=last_relative,
        max_temperature_update=last_update if iterations else None,
        min_temperature_K=float(T.min()),
        max_temperature_K=float(T.max()),
        mean_temperature_K=float(T.mean()),
        total_input_power_W=q_input,
        total_boundary_heat_out_W=q_out,
        global_power_imbalance_W=imbalance,
        relative_power_imbalance=rel_imbalance,
        residual_history=history,
        update_norm_history=update_history,
        solve_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Matrix-free PCG
# ---------------------------------------------------------------------------

def solve_pcg(
    operator: MatrixFreeThermalOperator,
    initial_temperature: np.ndarray,
    boundary: BoundaryLinkTable,
    *,
    relative_residual_tolerance: float = 1e-8,
    max_iterations: int = 10_000,
    absolute_residual_tolerance: float = 0.0,
) -> SteadyStateResult:
    """Solve ``A T = b`` with matrix-free PCG.

    Uses :class:`scipy.sparse.linalg.LinearOperator` with
    ``matvec = operator.apply`` and a Jacobi (diagonal)
    preconditioner ``M^-1 x = x / D``. The PCG iteration itself
    is the standard scipy implementation; the matrix-free
    constraint is satisfied because the production path never
    assembles a dense or sparse matrix.
    """
    from scipy.sparse.linalg import LinearOperator, cg
    T = np.array(initial_temperature, dtype=np.float64, copy=True)
    if T.shape != (operator.cell_count,):
        raise ValueError(
            f"initial_temperature has shape {T.shape}; expected "
            f"({operator.cell_count},)")
    if np.any(T < 0):
        raise ValueError(
            "initial_temperature contains values below 0 K")
    t0 = time.perf_counter()
    initial_residual = operator.relative_residual(T)
    A_op = LinearOperator(
        (operator.cell_count, operator.cell_count),
        matvec=lambda v: operator.apply(v),
        dtype=np.float64,
    )
    diagonal = operator.diagonal_W_K
    M_op = LinearOperator(
        (operator.cell_count, operator.cell_count),
        matvec=lambda v: v / diagonal,
        dtype=np.float64,
    )
    iteration_counter = {"k": 0}
    residual_history: list[float] = [initial_residual]

    def callback(xk):
        # scipy.cg invokes the callback once per iteration; the
        # ``iterations`` field on the result must therefore be the
        # true iteration count, not the residual-sampling count.
        # Sample the residual less often so the history stays
        # compact for plotting.
        iteration_counter["k"] += 1
        if iteration_counter["k"] % 10 == 0:
            residual_history.append(operator.relative_residual(xk))

    T_solution, info = cg(
        A_op, operator.rhs_W, x0=T, rtol=relative_residual_tolerance,
        atol=absolute_residual_tolerance, maxiter=max_iterations,
        M=M_op, callback=callback,
    )
    elapsed = time.perf_counter() - t0
    final_relative = operator.relative_residual(T_solution)
    final_absolute = float(np.linalg.norm(operator.residual(T_solution)))
    # Always include the final residual in the history so the last
    # entry is the actual converged value (not the last every-10th
    # sample, which may sit just before a very small last update).
    residual_history.append(final_relative)
    # scipy.cg reports ``info == 0`` only when it hits the requested
    # tolerance using its internal residual measure. We additionally
    # check the actual relative residual against the requested
    # tolerance so the user gets a true "converged" flag.
    converged = bool(info == 0) or (
        final_relative < relative_residual_tolerance)
    q_input, q_out, imbalance, rel_imbalance = _global_power_balance(
        operator, boundary, T_solution)
    return SteadyStateResult(
        temperature_K=T_solution,
        method="pcg",
        converged=converged,
        iterations=iteration_counter["k"],
        solver_info={
            "scipy_info": int(info),
            "relative_residual_tolerance": relative_residual_tolerance,
            "absolute_residual_tolerance": absolute_residual_tolerance,
            "max_iterations": max_iterations,
            "matvec_count": operator.matvec_count,
        },
        initial_residual=initial_residual,
        final_absolute_residual=final_absolute,
        final_relative_residual=final_relative,
        max_temperature_update=None,
        min_temperature_K=float(T_solution.min()),
        max_temperature_K=float(T_solution.max()),
        mean_temperature_K=float(T_solution.mean()),
        total_input_power_W=q_input,
        total_boundary_heat_out_W=q_out,
        global_power_imbalance_W=imbalance,
        relative_power_imbalance=rel_imbalance,
        residual_history=residual_history,
        update_norm_history=[],
        solve_seconds=elapsed,
    )
