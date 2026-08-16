"""Common helpers for GPU thermal backends.

Holds the lazy CuPy loader (``require_cupy``) and the custom
exception raised when the GPU backend is requested but the local
runtime is not usable.  All GPU-specific thermal backends
(:mod:`om3dthermal.thermal.gpu_relaxation` and any future kernels)
share this entry point so the import path and probe semantics stay
identical across backends.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


class GPUBackendUnavailableError(RuntimeError):
    """The GPU backend was requested without a usable CuPy/CUDA runtime."""


_CUPY_MODULE = None
_CUDA_DLL_HANDLES: list[object] = []


def _register_pip_cuda_dll_directories() -> None:
    """Expose NVIDIA DLLs to CuPy on Windows.

    Both the pip-style ``<env>\\Lib\\site-packages\\nvidia\\*`` tree
    and the Conda-style ``<env>\\Library\\bin`` layout are registered
    so the lazy CuPy probe works regardless of how the GPU stack
    was installed.  NVRTC's built-ins DLL in particular
    (``nvrtc-builtins64_*.dll``) lives under ``Library\\bin`` for
    Conda-installed cupy and is not in the default DLL search path.
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    runtime_root = site_packages / "nvidia" / "cuda_runtime"
    candidates = [
        runtime_root / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
        site_packages / "nvidia" / "cublas" / "bin",
        # Conda-installed CUDA: <env>/Library/bin (Windows only).
        Path(sys.prefix) / "Library" / "bin",
    ]
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
            "GPU backend requested but CuPy/CUDA is unavailable. Install "
            "the matching optional CuPy package (for example "
            "cupy-cuda12x) or activate a Conda environment with CuPy "
            "pre-installed.") from exc
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
