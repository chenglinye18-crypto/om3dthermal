"""Lazy CuPy loader and native Windows Conda CUDA probe.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


class GPUBackendUnavailableError(RuntimeError):
    """The GPU backend was requested without a usable CuPy/CUDA runtime."""


_CUPY_MODULE = None
_CUDA_DLL_HANDLES: list[object] = []


def _register_conda_cuda_dll_directory() -> None:
    """Keep the Conda CUDA DLL search handle alive for NVRTC."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    candidates = [Path(sys.prefix) / "Library" / "bin"]
    dll_paths: list[str] = []
    for directory in candidates:
        if directory.is_dir():
            dll_paths.append(str(directory))
            try:
                _CUDA_DLL_HANDLES.append(
                    os.add_dll_directory(str(directory)))
            except OSError:
                pass
    if dll_paths:
        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(dll_paths + [existing_path])


def require_cupy():
    """Lazy-import CuPy and verify that at least one CUDA device is usable."""
    global _CUPY_MODULE
    if _CUPY_MODULE is not None:
        return _CUPY_MODULE
    _register_conda_cuda_dll_directory()
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise GPUBackendUnavailableError(
            "GPU backend requested but CuPy/CUDA is unavailable. Install "
            "or restore CuPy 13.x in the native Windows om3dthermal "
            "Conda environment.") from exc
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            raise RuntimeError("no CUDA device detected")
        # Force context creation now, rather than failing midway
        # through upload.
        cp.cuda.Device().compute_capability
        probe = cp.arange(1, dtype=cp.float64)
        probe += 1.0
        cp.cuda.Stream.null.synchronize()
    except Exception as exc:
        raise GPUBackendUnavailableError(
            "GPU backend requested but CuPy/CUDA is unavailable or "
            f"unusable: {exc}") from exc
    _CUPY_MODULE = cp
    return cp


__all__ = ["GPUBackendUnavailableError", "require_cupy"]

# Maximum internal face degree in the block-structured mesh.
MAX_NEIGHBORS_PER_CELL = 6
