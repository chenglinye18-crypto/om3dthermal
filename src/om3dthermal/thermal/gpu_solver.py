"""CuPy FP64 matrix-free operator and PCG backend.

CuPy is imported lazily so CPU-only installations can import and run
``om3dthermal`` without a CUDA runtime.  All graph arrays and PCG vectors stay
on the device for the solve; only compact scalar convergence diagnostics are
synchronized periodically, followed by one final temperature-vector download.

The GPU thermal matvec is row-oriented: the cell graph adjacency is uploaded
once as a fixed-width ``(N, MAX_DEG)`` pair of ``(neighbor_id, neighbor_G)``
arrays plus a pre-aggregated ``boundary_G`` vector, and a single RawKernel
writes one result per cell.  No ``bincount`` scatter, no edge-sized
``delta`` / ``flux`` temporaries, no atomic accumulations.  Boundary
contribution is folded into the per-cell ``boundary_G`` scalar so the
internal + boundary sum is produced in a single pass.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np

from .boundary import BoundaryLinkTable
from .operator import MatrixFreeThermalOperator
from .steady_state import SteadyStateResult, _global_power_balance


class GPUBackendUnavailableError(RuntimeError):
    """The GPU backend was requested without a usable CuPy/CUDA runtime."""


class GPUSolverBreakdownError(RuntimeError):
    """PCG encountered non-finite state or lost its SPD denominator."""


_CUPY_MODULE = None
_CUDA_DLL_HANDLES: list[object] = []


# Maximum internal-edge degree per cell.  The Conventional HBM / Orthogonal
# Si / M3D nominal operators all have max_degree <= 6; if a future geometry
# exceeds this, ``from_cpu`` will assert and the operator should be
# rebuilt with a wider layout (or a CSR adjacency list).
MAX_NEIGHBORS_PER_CELL = 6


# CUDA kernel source.  FP64 throughout.  One thread per cell, walks up to
# ``MAX_DEG`` neighbor slots, skips the -1 padding, and writes one result.
# Memory layout: ``neighbor_id`` and ``neighbor_G`` are row-major
# ``(N, MAX_DEG)`` and indexed ``i*MAX_DEG + k``.  ``boundary_G[i]`` is
# pre-summed over all boundary faces of cell ``i``.
_MATVEC_KERNEL_SRC = r"""
extern "C" __global__
void matvec_row_fp64(
    const double* __restrict__ T,
    const int*    __restrict__ nbr_id,
    const double* __restrict__ nbr_G,
    const double* __restrict__ boundary_G,
    double*       __restrict__ out,
    int N,
    int MAX_DEG)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double v = boundary_G[i] * T[i];
    int base = i * MAX_DEG;
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        if (k >= MAX_DEG) break;
        int j = nbr_id[base + k];
        if (j < 0) continue;
        v += nbr_G[base + k] * (T[i] - T[j]);
    }
    out[i] = v;
}
"""


def _register_pip_cuda_dll_directories() -> None:
    """Expose NVIDIA pip-wheel DLLs to CuPy on Windows, if installed."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    runtime_root = site_packages / "nvidia" / "cuda_runtime"
    candidates = [
        runtime_root / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
        site_packages / "nvidia" / "cublas" / "bin",
    ]
    dll_paths: list[str] = []
    for directory in candidates:
        if directory.is_dir():
            dll_paths.append(str(directory))
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    if dll_paths:
        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(dll_paths + [existing_path])
    if runtime_root.is_dir():
        os.environ.setdefault("CUDA_PATH", str(runtime_root))


def require_cupy():
    """Lazy-import CuPy and verify that at least one CUDA device is usable."""
    global _CUPY_MODULE
    if _CUPY_MODULE is not None:
        return _CUPY_MODULE
    _register_pip_cuda_dll_directories()
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise GPUBackendUnavailableError(
            "GPU backend requested but CuPy/CUDA is unavailable. Install the "
            "matching optional CuPy package (for example cupy-cuda12x).") from exc
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            raise RuntimeError("no CUDA device detected")
        # Force context creation now, rather than failing midway through upload.
        cp.cuda.Device().compute_capability
        probe = cp.arange(1, dtype=cp.float64)
        probe += 1.0
        cp.cuda.Stream.null.synchronize()
    except Exception as exc:
        raise GPUBackendUnavailableError(
            "GPU backend requested but CuPy/CUDA is unavailable or unusable: "
            f"{exc}") from exc
    _CUPY_MODULE = cp
    return cp


# Module-level cache for the compiled RawKernel; one kernel handles every
# pipeline because MAX_DEG is a runtime argument (the inlined inner loop
# branches on the device-side ``k >= MAX_DEG``).
_KERNEL_CACHE: dict[tuple, object] = {}


def _matvec_kernel(cp, max_deg: int):
    """Return a cached RawKernel for ``matvec_row_fp64``."""
    key = (cp.__version__, max_deg)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = cp.RawKernel(
        _MATVEC_KERNEL_SRC, "matvec_row_fp64", backend="nvrtc",
        options=("--use_fast_math",))
    _KERNEL_CACHE[key] = kernel
    return kernel


@dataclass
class CuPyMatrixFreeThermalOperator:
    """Device-resident, row-oriented representation of the thermal operator.

    Only the data the GPU actually consumes is held on the device: the
    fixed-width neighbor table, the pre-aggregated per-cell boundary
    conductance, the per-cell power, the Jacobi diagonal inverse, the
    RHS, and one reusable ``out_buffer`` that ``apply`` writes into.
    The CPU edge-list arrays are *not* re-uploaded alongside the new
    representation.
    """

    cell_count: int
    neighbor_id: object           # (N, MAX_DEG) int32, padded with -1
    neighbor_G: object            # (N, MAX_DEG) float64
    boundary_G: object            # (N,) float64, pre-summed per cell
    power_W: object               # (N,) float64
    diagonal_inverse_K_W: object  # (N,) float64
    rhs_W: object                 # (N,) float64
    out_buffer: object            # (N,) float64, reused by apply
    matvec_count: int = 0
    max_neighbors: int = MAX_NEIGHBORS_PER_CELL

    @classmethod
    def from_cpu(cls, operator: MatrixFreeThermalOperator
                 ) -> "CuPyMatrixFreeThermalOperator":
        """Build the row-oriented GPU operator from a CPU one.

        No edge-list arrays are kept on the GPU; the row-oriented
        ``(neighbor_id, neighbor_G)`` pair is the only adjacency
        representation uploaded.
        """
        cp = require_cupy()
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

        # Build a-side entries first via argsort, then b-side entries at
        # slot = post-a-side count for that cell.  Without this two-sided
        # assignment, the a- and b-side writes collide at slot 0 for cells
        # that appear on both ends of different edges.
        if a.size:
            order_a = np.argsort(a, kind="stable")
            a_s = a[order_a]
            b_s = b[order_a]
            G_s = G[order_a]
            change_a = np.where(np.diff(a_s) != 0)[0] + 1
            starts_a = np.concatenate(([0], change_a, [a_s.size]))
            sizes_a = np.diff(starts_a)
            slot_a = np.arange(a_s.size) - np.repeat(starts_a[:-1], sizes_a)
            neighbor_id[a_s, slot_a] = b_s.astype(np.int32)
            neighbor_G[a_s, slot_a] = G_s

            # count[i] = number of a-side neighbors placed for cell i
            counts = np.zeros(N, dtype=np.int64)
            np.add.at(counts, a_s, 1)

            order_b = np.argsort(b, kind="stable")
            a2 = a[order_b]
            b2 = b[order_b]
            G2 = G[order_b]
            # For each b2 entry, its slot = counts[b2] BEFORE placement
            # (use np.add to claim).  Compute slots with a per-cell offset.
            # Vectorized: for each k, slot_b[k] = counts[b2[k]] + offset_in_b_group[k]
            # offset_in_b_group is the position within the b-group (0,1,...).
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

        # Pre-aggregate boundary conductance per cell so the matvec kernel
        # sees a single scalar per cell.
        bc = np.asarray(operator.boundary_cell, dtype=np.int64)
        bG = np.asarray(operator.boundary_conductance_W_K, dtype=np.float64)
        boundary_G_per_cell = np.zeros(N, dtype=np.float64)
        if bc.size:
            np.add.at(boundary_G_per_cell, bc, bG)

        # Diagonal inverse on device.
        diag = np.asarray(operator.diagonal_W_K, dtype=np.float64)
        diag_inv = 1.0 / diag
        if not np.all(np.isfinite(diag_inv)) or not np.all(diag > 0.0):
            raise ValueError("GPU operator validation failed: non-finite/positive diagonal")
        if not np.all(np.isfinite(boundary_G_per_cell)):
            raise ValueError("GPU operator validation failed: non-finite boundary_G")
        if not np.all(np.isfinite(neighbor_G)):
            raise ValueError("GPU operator validation failed: non-finite neighbor_G")

        return cls(
            cell_count=N,
            neighbor_id=cp.asarray(neighbor_id, dtype=cp.int32),
            neighbor_G=cp.asarray(neighbor_G, dtype=cp.float64),
            boundary_G=cp.asarray(boundary_G_per_cell, dtype=cp.float64),
            power_W=cp.asarray(operator.power_W, dtype=cp.float64),
            diagonal_inverse_K_W=cp.asarray(diag_inv, dtype=cp.float64),
            rhs_W=cp.asarray(operator.rhs_W, dtype=cp.float64),
            out_buffer=cp.zeros(N, dtype=cp.float64),
            max_neighbors=MAX,
        )

    def apply(self, vector):
        """Return ``A @ vector`` using a one-thread-per-cell RawKernel.

        The result reuses the device-resident ``out_buffer``; callers
        that need a separate handle should explicitly copy.  PCG only
        uses ``Ap`` once before overwriting it on the next call, so
        reuse is safe in production.
        """
        cp = require_cupy()
        if vector.shape != (self.cell_count,):
            raise ValueError(
                f"GPU vector has shape {vector.shape}; expected "
                f"({self.cell_count},)")
        if vector.dtype != cp.float64:
            raise TypeError("GPU matrix-free operator requires float64 vectors")
        kernel = _matvec_kernel(cp, self.max_neighbors)
        block = 256
        grid = ((self.cell_count + block - 1) // block,)
        kernel(
            grid, (block,),
            (vector, self.neighbor_id, self.neighbor_G,
             self.boundary_G, self.out_buffer, self.cell_count,
             self.max_neighbors),
        )
        self.matvec_count += 1
        return self.out_buffer


def solve_pcg_gpu(
    operator: MatrixFreeThermalOperator,
    initial_temperature: np.ndarray,
    boundary: BoundaryLinkTable,
    *,
    relative_residual_tolerance: float = 1e-8,
    max_iterations: int = 10_000,
    absolute_residual_tolerance: float = 0.0,
    diagnostic_interval: int = 10,
) -> SteadyStateResult:
    """Solve ``A T = b`` with device-resident CuPy FP64 PCG.

    The Jacobi preconditioner is identical to the CPU path: ``z = r / D``.
    PCG vectors, reductions, alpha, and beta stay on device.  Every
    ``diagnostic_interval`` iterations one small scalar diagnostic bundle is
    synchronized for convergence/breakdown handling; no iteration transfers a
    cell-sized vector.  The matvec uses a row-oriented RawKernel that visits
    one cell per GPU thread.
    """
    if relative_residual_tolerance < 0 or absolute_residual_tolerance < 0:
        raise ValueError("PCG residual tolerances must be non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if diagnostic_interval <= 0:
        raise ValueError("diagnostic_interval must be positive")
    host_initial = np.asarray(initial_temperature, dtype=np.float64)
    if host_initial.shape != (operator.cell_count,):
        raise ValueError(
            f"initial_temperature has shape {host_initial.shape}; expected "
            f"({operator.cell_count},)")
    if not np.all(np.isfinite(host_initial)):
        raise ValueError("initial_temperature contains NaN or Inf")
    if np.any(host_initial < 0):
        raise ValueError("initial_temperature contains values below 0 K")

    cp = require_cupy()
    started = time.perf_counter()
    gpu = CuPyMatrixFreeThermalOperator.from_cpu(operator)
    x = cp.asarray(host_initial, dtype=cp.float64)
    r = gpu.rhs_W - gpu.apply(x)
    z = gpu.diagonal_inverse_K_W * r
    p = z.copy()
    rz_old = cp.dot(r, z)
    b_norm_device = cp.linalg.norm(gpu.rhs_W)
    initial_norm_device = cp.linalg.norm(r)
    initial_norm, b_norm, rz_initial = cp.asnumpy(cp.asarray(
        [initial_norm_device, b_norm_device, rz_old], dtype=cp.float64))
    if not np.all(np.isfinite([initial_norm, b_norm, rz_initial])):
        raise GPUSolverBreakdownError("non-finite initial GPU PCG residual")
    threshold = max(
        float(absolute_residual_tolerance),
        float(relative_residual_tolerance) * float(b_norm))
    denominator_norm = max(float(b_norm), 1e-30)
    initial_relative = float(initial_norm) / denominator_norm
    residual_history = [initial_relative]
    converged = bool(initial_norm <= threshold)
    iterations = 0
    last_absolute = float(initial_norm)
    tiny = np.finfo(np.float64).tiny
    active = cp.asarray(not converged, dtype=cp.bool_)
    breakdown = cp.asarray(False, dtype=cp.bool_)
    first_converged_iteration = cp.asarray(
        0 if converged else -1, dtype=cp.int64)

    while not converged and iterations < max_iterations:
        Ap = gpu.apply(p)
        pAp = cp.dot(p, Ap)
        valid_p_ap = cp.isfinite(pAp) & (pAp > tiny)
        breakdown |= active & ~valid_p_ap
        alpha = cp.where(
            active & valid_p_ap,
            rz_old / cp.where(valid_p_ap, pAp, 1.0),
            0.0,
        )
        x += alpha * p
        r -= alpha * Ap
        residual_norm_device = cp.linalg.norm(r)
        finite_residual_norm = cp.isfinite(residual_norm_device)
        newly_converged = (
            active & finite_residual_norm
            & (residual_norm_device <= threshold))
        first_converged_iteration = cp.where(
            (first_converged_iteration < 0) & newly_converged,
            iterations + 1,
            first_converged_iteration,
        )
        still_active = active & ~newly_converged & valid_p_ap
        z = gpu.diagonal_inverse_K_W * r
        rz_new = cp.dot(r, z)
        valid_rz = (
            cp.isfinite(rz_old) & cp.isfinite(rz_new)
            & (rz_old > tiny) & (rz_new >= 0.0))
        breakdown |= still_active & ~valid_rz
        beta = cp.where(
            still_active & valid_rz,
            rz_new / cp.where(valid_rz, rz_old, 1.0),
            0.0,
        )
        p = z + beta * p
        rz_old = rz_new
        active = still_active & valid_rz & finite_residual_norm
        iterations += 1

        should_diagnose = (
            iterations % diagnostic_interval == 0
            or iterations == max_iterations)
        if should_diagnose:
            diagnostic = cp.asnumpy(cp.asarray([
                residual_norm_device, breakdown, active,
                first_converged_iteration,
                cp.all(cp.isfinite(x)), cp.all(cp.isfinite(r)),
            ], dtype=cp.float64))
            (residual_norm, broke_down, is_active, first_converged,
             finite_x, finite_r) = diagnostic
            if not np.isfinite(residual_norm):
                raise GPUSolverBreakdownError(
                    f"non-finite GPU PCG residual at iteration {iterations}")
            if not bool(finite_x) or not bool(finite_r):
                raise GPUSolverBreakdownError(
                    f"NaN/Inf GPU PCG vector at iteration {iterations}")
            if bool(broke_down):
                raise GPUSolverBreakdownError(
                    f"GPU PCG denominator breakdown detected by iteration "
                    f"{iterations}")
            last_absolute = float(residual_norm)
            residual_history.append(last_absolute / denominator_norm)
            converged = bool(first_converged >= 0 and not is_active)
            if converged:
                iterations = int(first_converged)

    if not converged:
        raise RuntimeError(
            "GPU PCG reached maximum iterations without convergence: "
            f"iterations={iterations}, relative_residual="
            f"{last_absolute / denominator_norm:.6e}, requested="
            f"{relative_residual_tolerance:.6e}")

    # Recompute the true residual once on device before the only vector download.
    true_residual = gpu.rhs_W - gpu.apply(x)
    final_absolute_device = cp.linalg.norm(true_residual)
    final_checks = cp.asnumpy(cp.asarray([
        final_absolute_device, cp.all(cp.isfinite(x)), cp.all(x >= 0),
    ], dtype=cp.float64))
    final_absolute = float(final_checks[0])
    if not bool(final_checks[1]) or not bool(final_checks[2]):
        raise GPUSolverBreakdownError(
            "GPU PCG produced NaN/Inf or temperature below 0 K")
    cp.cuda.Stream.null.synchronize()
    temperature = cp.asnumpy(x)
    elapsed = time.perf_counter() - started
    final_relative = final_absolute / denominator_norm
    residual_history.append(final_relative)
    q_input, q_out, imbalance, rel_imbalance = _global_power_balance(
        operator, boundary, temperature)
    return SteadyStateResult(
        temperature_K=temperature,
        method="pcg",
        converged=True,
        iterations=iterations,
        solver_info={
            "backend": "gpu",
            "cupy_version": cp.__version__,
            "relative_residual_tolerance": relative_residual_tolerance,
            "absolute_residual_tolerance": absolute_residual_tolerance,
            "max_iterations": max_iterations,
            "diagnostic_interval": diagnostic_interval,
            "matvec_count": gpu.matvec_count,
            "device_vector_downloads": 1,
            "per_iteration_vector_transfers": 0,
            "dtype": "float64",
            "matvec_kernel": "row_oriented_raw_fp64",
            "max_neighbors_per_cell": gpu.max_neighbors,
        },
        initial_residual=initial_relative,
        final_absolute_residual=final_absolute,
        final_relative_residual=final_relative,
        max_temperature_update=None,
        min_temperature_K=float(temperature.min()),
        max_temperature_K=float(temperature.max()),
        mean_temperature_K=float(temperature.mean()),
        total_input_power_W=q_input,
        total_boundary_heat_out_W=q_out,
        global_power_imbalance_W=imbalance,
        relative_power_imbalance=rel_imbalance,
        residual_history=residual_history,
        update_norm_history=[],
        solve_seconds=elapsed,
    )
