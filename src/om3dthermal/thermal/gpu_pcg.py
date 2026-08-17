"""Device-resident FP64 matrix-free GPU PCG thermal solver.

The solver uses the same row-oriented thermal operator as the GPU
relaxation path.  Static operator arrays and all PCG vectors remain on the
device for the complete iteration.  Only compact scalar diagnostics cross
to the host at check boundaries, followed by one final temperature-field
download.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .boundary import BoundaryLinkTable
from .gpu_common import require_cupy
from .gpu_relaxation import MAX_NEIGHBORS_PER_CELL
from .operator import MatrixFreeThermalOperator
from .steady_state import SteadyStateResult, _global_power_balance


class GPUSolverBreakdownError(RuntimeError):
    """PCG encountered a non-finite value or a non-SPD denominator."""


_MATVEC_KERNEL_SRC = r"""
extern "C" __global__
void thermal_matvec_fp64(
    const double* __restrict__ x,
    const int*    __restrict__ nbr_id,
    const double* __restrict__ nbr_G,
    const double* __restrict__ boundary_G,
    double*       __restrict__ out,
    int N,
    int MAX_DEG)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double xi = x[i];
    double value = boundary_G[i] * xi;
    int base = i * MAX_DEG;
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        if (k >= MAX_DEG) break;
        int j = nbr_id[base + k];
        if (j >= 0) value += nbr_G[base + k] * (xi - x[j]);
    }
    out[i] = value;
}
"""


_KERNEL_CACHE: dict[tuple, object] = {}


def _matvec_kernel(cp, max_deg: int):
    key = (cp.__version__, max_deg)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = cp.RawKernel(
            _MATVEC_KERNEL_SRC, "thermal_matvec_fp64", backend="nvrtc")
        _KERNEL_CACHE[key] = kernel
    return kernel


@dataclass
class GPUPCGOperator:
    """Row-oriented, matrix-free operator stored entirely on the GPU."""

    cell_count: int
    neighbor_id: object
    neighbor_G: object
    boundary_G: object
    diagonal_inverse_K_W: object
    rhs_W: object
    out: object
    max_neighbors: int = MAX_NEIGHBORS_PER_CELL
    matvec_count: int = 0
    operator_h2d_copy_count: int = 5

    @classmethod
    def from_cpu(cls, operator: MatrixFreeThermalOperator, cp_module):
        n = int(operator.cell_count)
        max_deg = cls.max_neighbors
        a = np.asarray(operator.internal_cell_a, dtype=np.int64)
        b = np.asarray(operator.internal_cell_b, dtype=np.int64)
        conductance = np.asarray(
            operator.internal_conductance_W_K, dtype=np.float64)

        degree = np.zeros(n, dtype=np.int64)
        if a.size:
            np.add.at(degree, a, 1)
            np.add.at(degree, b, 1)
        observed_max = int(degree.max()) if degree.size else 0
        if observed_max > max_deg:
            raise ValueError(
                "GPU PCG row adjacency requires max_degree <= "
                f"{max_deg}, got {observed_max}")

        neighbor_id = np.full((n, max_deg), -1, dtype=np.int32)
        neighbor_G = np.zeros((n, max_deg), dtype=np.float64)
        if a.size:
            order_a = np.argsort(a, kind="stable")
            a_sorted = a[order_a]
            starts_a = np.concatenate((
                [0], np.flatnonzero(np.diff(a_sorted)) + 1, [a.size]))
            sizes_a = np.diff(starts_a)
            slots_a = np.arange(a.size) - np.repeat(starts_a[:-1], sizes_a)
            neighbor_id[a_sorted, slots_a] = b[order_a].astype(np.int32)
            neighbor_G[a_sorted, slots_a] = conductance[order_a]

            a_side_count = np.zeros(n, dtype=np.int64)
            np.add.at(a_side_count, a_sorted, 1)
            order_b = np.argsort(b, kind="stable")
            b_sorted = b[order_b]
            starts_b = np.concatenate((
                [0], np.flatnonzero(np.diff(b_sorted)) + 1, [b.size]))
            sizes_b = np.diff(starts_b)
            offsets_b = np.arange(b.size) - np.repeat(starts_b[:-1], sizes_b)
            slots_b = a_side_count[b_sorted] + offsets_b
            neighbor_id[b_sorted, slots_b] = a[order_b].astype(np.int32)
            neighbor_G[b_sorted, slots_b] = conductance[order_b]

        boundary_G = np.zeros(n, dtype=np.float64)
        boundary_cell = np.asarray(operator.boundary_cell, dtype=np.int64)
        if boundary_cell.size:
            np.add.at(
                boundary_G, boundary_cell,
                np.asarray(operator.boundary_conductance_W_K,
                           dtype=np.float64))
        diagonal = np.asarray(operator.diagonal_W_K, dtype=np.float64)
        if (not np.all(diagonal > 0.0)
                or not np.all(np.isfinite(diagonal))
                or not np.all(np.isfinite(neighbor_G))
                or not np.all(np.isfinite(boundary_G))):
            raise ValueError("GPU PCG operator contains invalid coefficients")

        return cls(
            cell_count=n,
            neighbor_id=cp_module.asarray(neighbor_id, dtype=cp_module.int32),
            neighbor_G=cp_module.asarray(neighbor_G, dtype=cp_module.float64),
            boundary_G=cp_module.asarray(boundary_G, dtype=cp_module.float64),
            diagonal_inverse_K_W=cp_module.asarray(
                1.0 / diagonal, dtype=cp_module.float64),
            rhs_W=cp_module.asarray(operator.rhs_W, dtype=cp_module.float64),
            out=cp_module.empty(n, dtype=cp_module.float64),
            max_neighbors=max_deg,
        )

    def apply(self, vector, cp_module):
        if vector.shape != (self.cell_count,):
            raise ValueError(
                f"GPU vector shape {vector.shape}; expected "
                f"({self.cell_count},)")
        if vector.dtype != cp_module.float64:
            raise TypeError("GPU PCG requires FP64 vectors")
        block = 256
        grid = ((self.cell_count + block - 1) // block,)
        _matvec_kernel(cp_module, self.max_neighbors)(
            grid, (block,),
            (vector, self.neighbor_id, self.neighbor_G, self.boundary_G,
             self.out, self.cell_count, self.max_neighbors))
        self.matvec_count += 1
        return self.out


def solve_pcg_gpu(
    operator: MatrixFreeThermalOperator,
    initial_temperature: np.ndarray,
    boundary: BoundaryLinkTable,
    *,
    relative_residual_tolerance: float = 1e-3,
    max_temperature_update_tolerance: float = 1e-2,
    max_iterations: int = 100_000,
    check_interval: int = 10,
) -> SteadyStateResult:
    """Solve ``A T = b`` with Jacobi-preconditioned GPU PCG.

    Convergence requires both the recursive relative residual and the current
    PCG update ``max(abs(pcg_alpha * p))`` to satisfy their thresholds at the
    same check boundary.  A true KCL residual is evaluated on-device before
    the final temperature download.
    """
    if relative_residual_tolerance < 0.0:
        raise ValueError("relative_residual_tolerance must be non-negative")
    if max_temperature_update_tolerance < 0.0:
        raise ValueError(
            "max_temperature_update_tolerance must be non-negative")
    if max_iterations <= 0 or check_interval <= 0:
        raise ValueError("max_iterations and check_interval must be positive")
    host_initial = np.asarray(initial_temperature, dtype=np.float64)
    if host_initial.shape != (operator.cell_count,):
        raise ValueError(
            f"initial_temperature shape {host_initial.shape}; expected "
            f"({operator.cell_count},)")
    if not np.all(np.isfinite(host_initial)) or np.any(host_initial < 0.0):
        raise ValueError("initial_temperature must be finite and non-negative")

    cp = require_cupy()
    started = time.perf_counter()
    gpu = GPUPCGOperator.from_cpu(operator, cp)
    x = cp.asarray(host_initial, dtype=cp.float64)
    r = gpu.rhs_W - gpu.apply(x, cp)
    z = gpu.diagonal_inverse_K_W * r
    p = z.copy()
    rz_old = cp.dot(r, z)
    b_norm_device = cp.linalg.norm(gpu.rhs_W)
    initial_absolute_device = cp.linalg.norm(r)
    initial_diag = cp.asnumpy(cp.stack((
        initial_absolute_device, b_norm_device, rz_old)))
    scalar_sync_count = 1
    initial_absolute = float(initial_diag[0])
    denominator = max(float(initial_diag[1]), 1e-30)
    initial_relative = initial_absolute / denominator
    if not np.all(np.isfinite(initial_diag)) or float(initial_diag[2]) < 0.0:
        raise GPUSolverBreakdownError("invalid initial GPU PCG state")

    iterations = 0
    converged = False
    last_absolute = initial_absolute
    last_relative = initial_relative
    last_update = float("inf")
    residual_history = [initial_relative]
    update_history: list[float] = []
    tiny = np.finfo(np.float64).tiny
    breakdown = False

    while iterations < max_iterations:
        Ap = gpu.apply(p, cp)
        pAp = cp.dot(p, Ap)
        step = rz_old / pAp
        update = step * p
        x += update
        r -= step * Ap
        iterations += 1

        residual_norm_device = cp.linalg.norm(r)
        max_update_device = cp.max(cp.abs(update))
        z = gpu.diagonal_inverse_K_W * r
        rz_new = cp.dot(r, z)

        should_check = (
            iterations % check_interval == 0
            or iterations == max_iterations)
        if should_check:
            diag = cp.asnumpy(cp.stack((
                residual_norm_device,
                max_update_device,
                pAp,
                rz_old,
                rz_new,
                cp.all(cp.isfinite(x)).astype(cp.float64),
                cp.all(cp.isfinite(r)).astype(cp.float64),
            )))
            scalar_sync_count += 1
            last_absolute = float(diag[0])
            last_relative = last_absolute / denominator
            last_update = float(diag[1])
            residual_history.append(last_relative)
            update_history.append(last_update)
            finite = bool(diag[5]) and bool(diag[6])
            valid_spd = (
                np.all(np.isfinite(diag[:5]))
                and float(diag[2]) > tiny
                and float(diag[3]) > tiny
                and float(diag[4]) >= 0.0)
            if not finite or not valid_spd:
                breakdown = True
                break
            if (last_relative < relative_residual_tolerance
                    and last_update < max_temperature_update_tolerance):
                converged = True
                break

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    # Validate with a true matrix-free KCL residual while x is still on GPU.
    true_residual = gpu.rhs_W - gpu.apply(x, cp)
    final_diag = cp.asnumpy(cp.stack((
        cp.linalg.norm(true_residual),
        cp.all(cp.isfinite(x)).astype(cp.float64),
        cp.all(x >= 0.0).astype(cp.float64),
    )))
    scalar_sync_count += 1
    final_absolute = float(final_diag[0])
    final_relative = final_absolute / denominator
    if not bool(final_diag[1]) or not bool(final_diag[2]):
        breakdown = True
    converged = bool(
        converged and not breakdown
        and final_relative < relative_residual_tolerance
        and last_update < max_temperature_update_tolerance)

    # The sole full-vector D2H transfer occurs after the iteration terminates.
    temperature = cp.asnumpy(x)
    elapsed = time.perf_counter() - started
    q_input, q_out, imbalance, relative_imbalance = _global_power_balance(
        operator, boundary, temperature)
    return SteadyStateResult(
        temperature_K=temperature,
        method="pcg",
        converged=converged,
        iterations=iterations,
        solver_info={
            "backend": "gpu_pcg",
            "preconditioner": "jacobi_diagonal",
            "relative_residual_tolerance": relative_residual_tolerance,
            "max_temperature_update_tolerance":
                max_temperature_update_tolerance,
            "max_iterations": max_iterations,
            "check_interval": check_interval,
            "matvec_count": gpu.matvec_count,
            "dtype": "float64",
            "operator_h2d_copy_count": gpu.operator_h2d_copy_count,
            "solver_vector_h2d_copy_count": 1,
            "full_vector_h2d_copy_count":
                gpu.operator_h2d_copy_count + 1,
            "full_vector_d2h_copy_count": 1,
            "full_vector_d2h_during_iteration": 0,
            "scalar_synchronization_count": scalar_sync_count,
            "breakdown": breakdown,
        },
        initial_residual=initial_relative,
        final_absolute_residual=final_absolute,
        final_relative_residual=final_relative,
        max_temperature_update=(last_update if iterations else None),
        min_temperature_K=float(temperature.min()),
        max_temperature_K=float(temperature.max()),
        mean_temperature_K=float(temperature.mean()),
        total_input_power_W=q_input,
        total_boundary_heat_out_W=q_out,
        global_power_imbalance_W=imbalance,
        relative_power_imbalance=relative_imbalance,
        residual_history=residual_history + [final_relative],
        update_norm_history=update_history,
        solve_seconds=elapsed,
    )


__all__ = [
    "GPUPCGOperator",
    "GPUSolverBreakdownError",
    "solve_pcg_gpu",
]
