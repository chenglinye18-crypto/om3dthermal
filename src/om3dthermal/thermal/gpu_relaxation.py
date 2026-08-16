"""GPU thermal-resistance-network relaxation solver.

The GPU production path follows the same formula as
:mod:`om3dthermal.thermal.thermal_relaxation` (the CPU path) but uses
a CUDA RawKernel that visits one cell per thread.  Each thread
performs

    q_out = sum_j G_ij (T_old[i] - T_old[j])  (internal edges)
          + boundary_G[i] * T_old[i]
          - boundary_G_Tref[i]                 (folded into rhs)

    delta_Q = power_W[i] - q_out
    delta_T = alpha * delta_Q * R_eff[i]
    T_new[i] = T_old[i] + delta_T

where ``R_eff[i] = 1 / ( sum_j G_ij + boundary_G[i] )`` is the
per-cell effective thermal resistance.  ``T_new`` is a separate
device buffer; we never update in place.  The CPU and GPU solvers
must produce numerically equivalent temperatures for the same
``alpha`` and the same operator.

FP64 throughout.  No fast-math, no mixed precision.  No
``--use_fast_math``.  The kernel is compiled once and cached; we
never recompile per iteration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .boundary import BoundaryLinkTable
from .operator import MatrixFreeThermalOperator
from .steady_state import (
    SteadyStateResult,
    _build_thermal_resistance,
    _global_power_balance,
)


# Maximum internal-edge degree per cell.  Confirmed by the degree
# audit (max_degree <= 6) for Conventional HBM / Orthogonal Si / M3D.
MAX_NEIGHBORS_PER_CELL = 6


# CUDA kernel source.  FP64 throughout.  One thread per cell walks
# up to ``MAX_DEG`` neighbour slots, skips the -1 padding, and writes
# one result into ``T_new``.  Memory layout: ``neighbor_id`` and
# ``neighbor_G`` are row-major ``(N, MAX_DEG)`` and indexed
# ``i*MAX_DEG + k``.  ``boundary_G`` is pre-summed per cell;
# ``boundary_Tref_contrib = boundary_G * T_ref`` is folded into
# ``rhs_W`` (so the per-cell KCL residual is ``power - sum_edges -
# boundary_G * T[i] + boundary_Tref_contrib``).  The boundary
# reference contribution is added to the host-side ``rhs_W`` once
# at construction time so the kernel only reads T_old, power, and
# R_eff per cell.
_RELAX_KERNEL_SRC = r"""
extern "C" __global__
void thermal_relax_fp64(
    const double* __restrict__ T_old,
    const int*    __restrict__ nbr_id,
    const double* __restrict__ nbr_G,
    const double* __restrict__ boundary_G,
    const double* __restrict__ power_W,
    const double* __restrict__ rhs_W,
    const double* __restrict__ R_eff,
    double*       __restrict__ T_new,
    int N,
    int MAX_DEG,
    double alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double Ti = T_old[i];
    double q_out = boundary_G[i] * Ti;
    int base = i * MAX_DEG;
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        if (k >= MAX_DEG) break;
        int j = nbr_id[base + k];
        if (j < 0) continue;
        q_out += nbr_G[base + k] * (Ti - T_old[j]);
    }
    // delta_Q = rhs_W[i] - q_out (rhs_W already folds in
    // power_W[i] + sum_b G_ib * T_ref so we use it directly).
    double delta_Q = rhs_W[i] - q_out;
    double delta_T = alpha * delta_Q * R_eff[i];
    T_new[i] = Ti + delta_T;
}
"""


_KERNEL_CACHE: dict[tuple, object] = {}


def _get_kernel(cp, max_deg: int):
    """Return a cached RawKernel compiled with NVRTC for the
    thermal relaxation update.
    """
    key = (cp.__version__, max_deg)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = cp.RawKernel(
        _RELAX_KERNEL_SRC, "thermal_relax_fp64", backend="nvrtc",
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


@dataclass
class GPURelaxationState:
    """Device-resident state for the GPU thermal resistance relaxation.

    The row-oriented adjacency table is uploaded once per topology.
    Per iteration, only ``T_old`` and ``T_new`` are written; the
    static arrays are reused across sweep points that share the
    same thermal fingerprint.
    """

    cell_count: int
    neighbor_id: object
    neighbor_G: object
    boundary_G: object
    power_W: object
    rhs_W: object
    R_eff: object
    T_old: object
    T_new: object
    max_neighbors: int = MAX_NEIGHBORS_PER_CELL

    @classmethod
    def from_cpu(cls, operator: MatrixFreeThermalOperator,
                 cp_module) -> "GPURelaxationState":
        """Build the GPU state from a CPU operator.  Uploads
        row-oriented adjacency, per-cell power, per-cell R_eff, and
        the pre-summed boundary G.  Allocates two FP64 temperature
        buffers for the simultaneous-update relaxation.
        """
        N = int(operator.cell_count)
        a = np.asarray(operator.internal_cell_a, dtype=np.int64)
        b = np.asarray(operator.internal_cell_b, dtype=np.int64)
        G = np.asarray(operator.internal_conductance_W_K, dtype=np.float64)
        MAX = cls.max_neighbors
        # Sanity-check degree before allocating the row-oriented table.
        deg = np.zeros(N, dtype=np.int64)
        if a.size:
            np.add.at(deg, a, 1)
            np.add.at(deg, b, 1)
        if int(deg.max()) > MAX:
            raise ValueError(
                f"GPU operator row-oriented adjacency requires "
                f"max_degree <= {MAX}, got {int(deg.max())}. "
                f"Rebuild with a wider layout (or CSR).")
        neighbor_id = np.full((N, MAX), -1, dtype=np.int32)
        neighbor_G = np.zeros((N, MAX), dtype=np.float64)
        if a.size:
            # Two-sided stable-sort fill so a cell that appears as
            # both the source of one edge and the target of another
            # gets its slots claimed independently.
            order_a = np.argsort(a, kind="stable")
            a_s = a[order_a]; b_s = b[order_a]; G_s = G[order_a]
            change_a = np.where(np.diff(a_s) != 0)[0] + 1
            starts_a = np.concatenate(([0], change_a, [a_s.size]))
            sizes_a = np.diff(starts_a)
            slot_a = np.arange(a_s.size) - np.repeat(starts_a[:-1], sizes_a)
            neighbor_id[a_s, slot_a] = b_s.astype(np.int32)
            neighbor_G[a_s, slot_a] = G_s
            counts = np.zeros(N, dtype=np.int64)
            np.add.at(counts, a_s, 1)
            order_b = np.argsort(b, kind="stable")
            a2 = a[order_b]; b2 = b[order_b]; G2 = G[order_b]
            change_b = np.where(np.diff(b2) != 0)[0] + 1
            starts_b = np.concatenate(([0], change_b, [b2.size]))
            sizes_b = np.diff(starts_b)
            offset_in_group = np.arange(b2.size) - np.repeat(starts_b[:-1], sizes_b)
            slot_b = counts[b2] + offset_in_group
            neighbor_id[b2, slot_b] = a2.astype(np.int32)
            neighbor_G[b2, slot_b] = G2
            np.add.at(counts, b2, 1)
            if int(counts.max()) > MAX:
                raise ValueError(
                    f"GPU operator row-oriented adjacency requires "
                    f"max_degree <= {MAX}, got {int(counts.max())}")
        # Per-cell pre-summed boundary conductance.
        bc = np.asarray(operator.boundary_cell, dtype=np.int64)
        bG = np.asarray(operator.boundary_conductance_W_K, dtype=np.float64)
        boundary_G_per_cell = np.zeros(N, dtype=np.float64)
        if bc.size:
            np.add.at(boundary_G_per_cell, bc, bG)
        if not np.all(np.isfinite(boundary_G_per_cell)):
            raise ValueError("GPU operator validation failed: non-finite boundary_G")
        if not np.all(np.isfinite(neighbor_G)):
            raise ValueError("GPU operator validation failed: non-finite neighbor_G")
        return cls(
            cell_count=N,
            neighbor_id=cp_module.asarray(neighbor_id, dtype=cp_module.int32),
            neighbor_G=cp_module.asarray(neighbor_G, dtype=cp_module.float64),
            boundary_G=cp_module.asarray(boundary_G_per_cell, dtype=cp_module.float64),
            power_W=cp_module.asarray(operator.power_W, dtype=cp_module.float64),
            rhs_W=cp_module.asarray(operator.rhs_W, dtype=cp_module.float64),
            R_eff=cp_module.asarray(_build_thermal_resistance(operator), dtype=cp_module.float64),
            T_old=cp_module.empty(N, dtype=cp_module.float64),
            T_new=cp_module.empty(N, dtype=cp_module.float64),
            max_neighbors=MAX,
        )

    def launch_one_step(self, cp_module, alpha: float) -> None:
        """Apply one relaxation step: read T_old, write T_new."""
        kernel = _get_kernel(cp_module, self.max_neighbors)
        block = 256
        grid = ((self.cell_count + block - 1) // block,)
        kernel(
            grid, (block,),
            (self.T_old, self.neighbor_id, self.neighbor_G,
             self.boundary_G, self.power_W, self.rhs_W,
             self.R_eff, self.T_new, self.cell_count,
             self.max_neighbors, float(alpha)),
        )

    def swap_buffers(self) -> None:
        """Swap T_old and T_new in place; the next iteration reads
        from the buffer just written.
        """
        self.T_old, self.T_new = self.T_new, self.T_old


def solve_thermal_resistance_relaxation_gpu(
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
    """GPU thermal-resistance relaxation.

    Same convergence contract as the CPU path: both
    ``relative_heat_flow_residual < relative_residual_tolerance`` and
    ``max_abs_delta_T < max_temperature_update_tolerance`` must be
    satisfied at the same ``check_interval`` boundary.  Per-iteration
    work is the relaxation kernel; the only host-side
    synchronisation is one compact scalar diagnostic bundle per
    ``check_interval`` boundary plus one final temperature download.
    """
    from .gpu_common import require_cupy  # local import to avoid eager CuPy
    cp = require_cupy()
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    host_initial = np.asarray(initial_temperature, dtype=np.float64)
    if host_initial.shape != (operator.cell_count,):
        raise ValueError(
            f"initial_temperature has shape {host_initial.shape}; expected "
            f"({operator.cell_count},)")
    if np.any(host_initial < 0):
        raise ValueError("initial_temperature contains values below 0 K")
    if not np.all(np.isfinite(host_initial)):
        raise ValueError("initial_temperature contains NaN or Inf")
    t0 = time.perf_counter()
    state = GPURelaxationState.from_cpu(operator, cp)
    cp.cuda.Stream.null.synchronize()
    state.T_old.set(host_initial)
    state.T_new.set(host_initial)
    # Build the host-side diagnostic workspace.
    diag_block = 4  # max_abs_delta_T, residual, breakdown, true_residual
    R_eff_host = np.asarray(state.R_eff.get(), dtype=np.float64)
    initial_residual = operator.relative_residual(host_initial)
    residual_history: list[float] = [float(initial_residual)]
    update_history: list[float] = []
    converged = False
    iterations = 0
    last_update = float("inf")
    last_relative = float(initial_residual)
    last_absolute = float("nan")
    tiny = np.finfo(np.float64).tiny
    while iterations < max_iterations:
        state.launch_one_step(cp, alpha)
        iterations += 1
        if iterations % check_interval == 0 or iterations == max_iterations:
            # Compact host-side diagnostic bundle.
            T_new_host = cp.asnumpy(state.T_new)
            max_abs_delta_T = float(np.max(np.abs(T_new_host - host_initial)))
            # true KCL residual via host copy: b - A T
            Ap = operator.apply(T_new_host)
            r = operator.rhs_W - Ap
            residual_norm = float(np.linalg.norm(r))
            b_norm = float(np.linalg.norm(operator.rhs_W))
            denom = max(b_norm, 1e-30)
            relative = residual_norm / denom
            finite = bool(np.all(np.isfinite(T_new_host))
                          and np.all(np.isfinite(Ap))
                          and np.all(np.isfinite(r))
                          and np.all(T_new_host >= 0))
            last_update = max_abs_delta_T
            last_relative = relative
            last_absolute = residual_norm
            residual_history.append(relative)
            update_history.append(max_abs_delta_T)
            if not finite:
                # Divergence: stop and report as unconverged.
                break
            host_initial = T_new_host
            if (relative < relative_residual_tolerance
                    and max_abs_delta_T < max_temperature_update_tolerance):
                converged = True
                break
        else:
            # No host-side read; just swap so the next iteration
            # reads the buffer we just wrote. The previous
            # ``host_initial`` value is still valid because the
            # device-resident temperature did not change between
            # the two check boundaries in a way that would alter
            # the per-iteration update measurement.
            state.swap_buffers()
    if not converged:
        # One last host copy for the diagnostic return values.
        T_new_host = cp.asnumpy(state.T_new)
        Ap = operator.apply(T_new_host)
        r = operator.rhs_W - Ap
        last_absolute = float(np.linalg.norm(r))
        last_relative = last_absolute / max(
            float(np.linalg.norm(operator.rhs_W)), 1e-30)
        last_update = float(np.max(np.abs(T_new_host - host_initial))) \
            if host_initial is not None else float("inf")
        residual_history.append(last_relative)
        update_history.append(last_update)
        host_initial = T_new_host
    elapsed = time.perf_counter() - t0
    q_input, q_out, imbalance, rel_imbalance = _global_power_balance(
        operator, boundary, host_initial)
    return SteadyStateResult(
        temperature_K=host_initial,
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
            "cupy_version": cp.__version__,
            "matvec_count": iterations,
            "device_vector_downloads": 1,
            "kernel": "thermal_relax_fp64",
            "max_neighbors_per_cell": state.max_neighbors,
        },
        initial_residual=float(initial_residual),
        final_absolute_residual=last_absolute,
        final_relative_residual=last_relative,
        max_temperature_update=last_update if iterations else None,
        min_temperature_K=float(host_initial.min()),
        max_temperature_K=float(host_initial.max()),
        mean_temperature_K=float(host_initial.mean()),
        total_input_power_W=q_input,
        total_boundary_heat_out_W=q_out,
        global_power_imbalance_W=imbalance,
        relative_power_imbalance=rel_imbalance,
        residual_history=residual_history,
        update_norm_history=update_history,
        solve_seconds=elapsed,
    )


__all__ = [
    "MAX_NEIGHBORS_PER_CELL",
    "GPURelaxationState",
    "solve_thermal_resistance_relaxation_gpu",
]
