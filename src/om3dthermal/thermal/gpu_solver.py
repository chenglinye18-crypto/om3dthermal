"""CuPy FP64 matrix-free operator and PCG backend.

CuPy is imported lazily so CPU-only installations can import and run
``om3dthermal`` without a CUDA runtime.  All graph arrays and PCG vectors stay
on the device for the solve; only compact scalar convergence diagnostics are
synchronized periodically, followed by one final temperature-vector download.
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


@dataclass
class CuPyMatrixFreeThermalOperator:
    """Device-resident representation of the existing thermal operator."""

    cell_count: int
    internal_cell_a: object
    internal_cell_b: object
    internal_conductance_W_K: object
    boundary_cell: object
    boundary_conductance_W_K: object
    power_W: object
    diagonal_W_K: object
    diagonal_inverse_K_W: object
    rhs_W: object
    matvec_count: int = 0

    @classmethod
    def from_cpu(cls, operator: MatrixFreeThermalOperator
                 ) -> "CuPyMatrixFreeThermalOperator":
        """Upload every solver array once, preserving int64/float64 types."""
        cp = require_cupy()
        arrays = {
            "internal_cell_a": cp.asarray(
                operator.internal_cell_a, dtype=cp.int64),
            "internal_cell_b": cp.asarray(
                operator.internal_cell_b, dtype=cp.int64),
            "internal_conductance_W_K": cp.asarray(
                operator.internal_conductance_W_K, dtype=cp.float64),
            "boundary_cell": cp.asarray(operator.boundary_cell, dtype=cp.int64),
            "boundary_conductance_W_K": cp.asarray(
                operator.boundary_conductance_W_K, dtype=cp.float64),
            "power_W": cp.asarray(operator.power_W, dtype=cp.float64),
            "diagonal_W_K": cp.asarray(
                operator.diagonal_W_K, dtype=cp.float64),
            "rhs_W": cp.asarray(operator.rhs_W, dtype=cp.float64),
        }
        diagonal = arrays["diagonal_W_K"]
        checks = cp.asnumpy(cp.asarray([
            cp.all(cp.isfinite(arrays["internal_conductance_W_K"])),
            cp.all(arrays["internal_conductance_W_K"] > 0),
            cp.all(cp.isfinite(arrays["boundary_conductance_W_K"])),
            cp.all(arrays["boundary_conductance_W_K"] > 0),
            cp.all(cp.isfinite(diagonal)),
            cp.all(diagonal > 0),
            cp.all(cp.isfinite(arrays["rhs_W"])),
        ], dtype=cp.bool_))
        labels = (
            "finite internal conductance", "positive internal conductance",
            "finite boundary conductance", "positive boundary conductance",
            "finite diagonal", "positive diagonal", "finite rhs")
        for valid, label in zip(checks.tolist(), labels):
            if not valid:
                raise ValueError(f"GPU operator validation failed: {label}")
        arrays["diagonal_inverse_K_W"] = 1.0 / diagonal
        cp.cuda.Stream.null.synchronize()
        return cls(cell_count=operator.cell_count, **arrays)

    def apply(self, vector):
        """Return ``A @ vector`` using CuPy gather/bincount scatter sums."""
        cp = require_cupy()
        if vector.shape != (self.cell_count,):
            raise ValueError(
                f"GPU vector has shape {vector.shape}; expected "
                f"({self.cell_count},)")
        if vector.dtype != cp.float64:
            raise TypeError("GPU matrix-free operator requires float64 vectors")
        if self.internal_conductance_W_K.size:
            delta = vector[self.internal_cell_a] - vector[self.internal_cell_b]
            flux = self.internal_conductance_W_K * delta
            result = cp.bincount(
                self.internal_cell_a, weights=flux, minlength=self.cell_count)
            result += cp.bincount(
                self.internal_cell_b, weights=-flux, minlength=self.cell_count)
        else:
            result = cp.zeros(self.cell_count, dtype=cp.float64)
        if self.boundary_conductance_W_K.size:
            result += cp.bincount(
                self.boundary_cell,
                weights=(self.boundary_conductance_W_K
                         * vector[self.boundary_cell]),
                minlength=self.cell_count,
            )
        self.matvec_count += 1
        return result


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
    cell-sized vector.
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
