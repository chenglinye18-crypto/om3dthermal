"""Typed temperature observations extracted from a thermal pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermalObservables:
    memory_Tmax_degC: float
    gpu_Tmax_degC: float
    package_Tmax_degC: float


def extract_temperature_observables(pipeline) -> ThermalObservables:
    memory, gpu, package = _temperature_maxima(pipeline)
    return ThermalObservables(
        memory_Tmax_degC=memory,
        gpu_Tmax_degC=gpu,
        package_Tmax_degC=package,
    )


def _temperature_maxima(pipeline) -> tuple[float, float, float]:
    temperatures_C = pipeline.result.temperature_K - 273.15
    gpu = np.array([
        cell.component == "gpu" for cell in pipeline.cells], dtype=bool)
    memory = np.array([
        (str(cell.component).startswith("memory_column:")
         or str(cell.component).startswith("orthogonal_hbm:"))
        for cell in pipeline.cells], dtype=bool)
    if not np.any(gpu) or not np.any(memory):
        raise RuntimeError("GPU or memory thermal region is absent")
    return (float(np.max(temperatures_C[memory])),
            float(np.max(temperatures_C[gpu])),
            float(np.max(temperatures_C)))
