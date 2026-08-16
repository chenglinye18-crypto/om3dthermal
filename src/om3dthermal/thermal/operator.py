"""Matrix-free thermal operator.

The steady-state thermal network

    A T = b

is never materialised as a dense (or even CSR) matrix on the
production path. The :class:`MatrixFreeThermalOperator` instead
stores the *edge list* of the internal conductance graph and the
*active* boundary link list, plus a precomputed diagonal and rhs.
Multiplying ``A T`` is then a single pass over the edge list with
``numpy.bincount`` accumulators — ``O(N_edges + N_boundary)`` per
``apply`` call, no Python object list per edge, and a working
buffer of ``O(N_cells)`` floats.

The diagonal is the row sum of ``A``:

    D_i = sum_j G_ij          (internal)
        + sum_b G_ib          (boundary)

The right-hand side is

    b_i = P_i + sum_b G_ib * T_reference

so that the boundary contribution appears in the *operator*, not
in a separate ``b_boundary`` that the solver would have to
reassemble. The boundary convection link ``G_ib`` is the
``G_cell_to_ambient`` from :mod:`om3dthermal.thermal.boundary`
(half-cell bulk + interface R'' + 1/h), and the contribution to
``A T`` is ``G_ib * T_i`` (the unknown cell temperature) with the
external side folded into ``rhs``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .boundary import BoundaryLinkTable
from .conductance import ConductanceTable


@dataclass
class MatrixFreeThermalOperator:
    """Memory-layout for the matrix-free thermal operator.

    The internal edge arrays are kept in the columnar
    :class:`ConductanceTable` form; the boundary link arrays come
    from :class:`BoundaryLinkTable`. The diagonal and rhs are
    precomputed once at construction time so the per-iteration
    work in the thermal-resistance relaxation is one ``apply``
    plus trivial vector arithmetic.
    """

    cell_count: int
    # Internal edges: pairs + conductance. Re-exposed as plain
    # numpy arrays for cheap access.
    internal_cell_a: np.ndarray
    internal_cell_b: np.ndarray
    internal_conductance_W_K: np.ndarray
    # Boundary links: which cell the link is attached to, the
    # link conductance and the reference temperature that has been
    # folded into rhs.
    boundary_cell: np.ndarray
    boundary_conductance_W_K: np.ndarray
    boundary_reference_temperature_K: np.ndarray
    # Power vector (per-cell input).
    power_W: np.ndarray
    # Precomputed.
    diagonal_W_K: np.ndarray
    rhs_W: np.ndarray
    # Bookkeeping for the matvec counter (used in solver traces).
    matvec_count: int = 0

    # -- Public API -----------------------------------------------------

    def apply(self, temperature: np.ndarray, out: np.ndarray | None = None
              ) -> np.ndarray:
        """Compute ``A T`` (matrix-free) and return the result.

        ``out`` is reused if supplied; otherwise a fresh float64
        array is allocated. The ``matvec_count`` is incremented
        exactly once per call so a solver can report how many
        times the operator was evaluated.
        """
        if temperature.shape != (self.cell_count,):
            raise ValueError(
                f"temperature has shape {temperature.shape}; expected "
                f"({self.cell_count},)")
        if out is None:
            out = np.empty(self.cell_count, dtype=np.float64)
        out.fill(0.0)
        # Internal edges: flux_a = G * (T_a - T_b); flux_b = -flux_a.
        if self.internal_conductance_W_K.shape[0]:
            delta = (temperature[self.internal_cell_a]
                     - temperature[self.internal_cell_b])
            flux = self.internal_conductance_W_K * delta
            np.add.at(out, self.internal_cell_a, flux)
            np.add.at(out, self.internal_cell_b, -flux)
        # Boundary links: G_ib * T_i (the external side has been
        # folded into rhs).
        if self.boundary_conductance_W_K.shape[0]:
            np.add.at(
                out, self.boundary_cell,
                self.boundary_conductance_W_K
                * temperature[self.boundary_cell],
            )
        self.matvec_count += 1
        return out

    def residual(self, temperature: np.ndarray) -> np.ndarray:
        """Return ``b - A T`` as a fresh float64 array."""
        residual = self.rhs_W - self.apply(temperature)
        return residual

    def relative_residual(self, temperature: np.ndarray,
                          epsilon: float = 1e-30) -> float:
        """L2-norm of ``b - A T`` divided by ``max(||b||, epsilon)``."""
        r = self.residual(temperature)
        b_norm = float(np.linalg.norm(self.rhs_W))
        denom = max(b_norm, epsilon)
        return float(np.linalg.norm(r) / denom)


def build_matrix_free_operator(
    conductance: ConductanceTable,
    boundary: BoundaryLinkTable,
    power_W: np.ndarray,
) -> MatrixFreeThermalOperator:
    """Assemble the matrix-free operator.

    Performs the one-time diagonal / rhs precomputation and
    validates that the inputs are well-formed. ``power_W`` is the
    pure power vector; the boundary contribution is folded into
    ``rhs_W`` so the operator's ``apply(T) = A T`` and
    ``residual(T) = b - A T`` work as expected.
    """
    if power_W.ndim != 1:
        raise ValueError(
            f"power_W must be 1-D, got shape {power_W.shape}")
    cell_count = int(power_W.shape[0])
    max_cell_id = -1
    if conductance.cell_a.shape[0] > 0:
        max_cell_id = int(max(
            conductance.cell_a.max(), conductance.cell_b.max()))
    if boundary.link_count > 0:
        max_cell_id = max(max_cell_id, int(boundary.cell_id.max()))
    if max_cell_id >= cell_count:
        raise ValueError(
            f"operator: cell id {max_cell_id} >= cell_count "
            f"{cell_count}")
    if not np.all(np.isfinite(power_W)):
        raise ValueError("operator: power vector contains non-finite values")
    if not np.all(conductance.conductance_W_K > 0):
        raise ValueError("operator: conductance contains non-positive values")
    if not np.all(np.isfinite(conductance.conductance_W_K)):
        raise ValueError("operator: conductance contains non-finite values")
    if boundary.link_count > 0 and not np.all(boundary.conductance_W_K > 0):
        raise ValueError(
            "operator: boundary conductance contains non-positive values")

    # Precompute diagonal.
    diagonal = np.zeros(cell_count, dtype=np.float64)
    if conductance.cell_a.shape[0]:
        np.add.at(diagonal, conductance.cell_a, conductance.conductance_W_K)
        np.add.at(diagonal, conductance.cell_b, conductance.conductance_W_K)
    if boundary.link_count > 0:
        np.add.at(diagonal, boundary.cell_id, boundary.conductance_W_K)
    if not np.all(diagonal > 0):
        bad = int(np.argmax(diagonal <= 0))
        raise ValueError(
            f"operator: diagonal at cell {bad} is non-positive "
            f"({float(diagonal[bad])}); the network has a node with no "
            f"neighbour and no active boundary link")
    if not np.all(np.isfinite(diagonal)):
        raise ValueError("operator: diagonal contains non-finite values")

    # rhs = power + sum_b G_ib * T_ref.
    rhs = np.array(power_W, dtype=np.float64, copy=True)
    if boundary.link_count > 0:
        np.add.at(
            rhs, boundary.cell_id,
            boundary.conductance_W_K * boundary.reference_temperature_K,
        )
    if not np.all(np.isfinite(rhs)):
        raise ValueError("operator: rhs contains non-finite values")

    return MatrixFreeThermalOperator(
        cell_count=cell_count,
        internal_cell_a=np.asarray(conductance.cell_a, dtype=np.int64),
        internal_cell_b=np.asarray(conductance.cell_b, dtype=np.int64),
        internal_conductance_W_K=np.asarray(
            conductance.conductance_W_K, dtype=np.float64),
        boundary_cell=np.asarray(boundary.cell_id, dtype=np.int64),
        boundary_conductance_W_K=np.asarray(
            boundary.conductance_W_K, dtype=np.float64),
        boundary_reference_temperature_K=np.asarray(
            boundary.reference_temperature_K, dtype=np.float64),
        power_W=np.array(power_W, dtype=np.float64, copy=True),
        diagonal_W_K=diagonal,
        rhs_W=rhs,
        matvec_count=0,
    )
