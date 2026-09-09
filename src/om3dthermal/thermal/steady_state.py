"""Steady-state diagnostics: P_i = sum_j G_ij(T_i-T_j) + sum_b G_ib(T_i-T_b).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
)

import numpy as np

from .boundary import BoundaryLinkTable
from .errors import ThermalError
from .operator import MatrixFreeThermalOperator


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SteadyStateResult:
    """Container for everything produced by one thermal solve."""

    temperature_K: np.ndarray
    method: str
    converged: bool
    iterations: int
    solver_info: dict[str, Any] = field(default_factory=dict)
    initial_residual: float = float("nan")
    final_absolute_residual: float = float("nan")
    final_relative_residual: float = float("nan")
    max_temperature_update: float | None = None
    min_temperature_K: float = float("nan")
    max_temperature_K: float = float("nan")
    mean_temperature_K: float = float("nan")
    total_input_power_W: float = float("nan")
    total_boundary_heat_out_W: float = float("nan")
    global_power_imbalance_W: float = float("nan")
    relative_power_imbalance: float = float("nan")
    residual_history: list[float] = field(default_factory=list)
    update_norm_history: list[float] = field(default_factory=list)
    solve_seconds: float = float("nan")


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
    """Verify every connected component has at least one active
    boundary link.  Uses a union-find over the cell graph.
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

    if internal_cell_a.size:
        a = np.asarray(internal_cell_a, dtype=np.int64)
        b = np.asarray(internal_cell_b, dtype=np.int64)
        for k in range(a.size):
            union(int(a[k]), int(b[k]))
    if boundary.link_count > 0:
        for cid in np.asarray(boundary.cell_id, dtype=np.int64):
            # Anchored to the ambient reference: every cell with a
            # boundary link is part of a component that has a unique
            # solution. We do not need to mark the reference as a
            # separate node in the union-find; the count check below
            # is sufficient.
            pass
    # Build per-root statistics: cell count, total power (W), and
    # whether the component has at least one boundary link.
    root_to_cells: dict[int, list[int]] = {}
    for cid in range(cell_count):
        root_to_cells.setdefault(find(cid), []).append(cid)
    anchored: set[int] = set()
    if boundary.link_count > 0:
        for cid in np.asarray(boundary.cell_id, dtype=np.int64):
            anchored.add(find(int(cid)))
    bad: list[dict[str, Any]] = []
    for component_id, cells_in in enumerate(root_to_cells.values()):
        if cells_in and find(cells_in[0]) not in anchored:
            bad.append({
                "component_id": component_id,
                "cell_count": len(cells_in),
                "total_power_W": 0.0,
            })
    if bad:
        raise UnanchoredThermalComponentError(bad)


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


__all__ = [
    "SteadyStateResult",
    "UnanchoredThermalComponentError",
    "validate_anchored_components",
    "_global_power_balance",
]
