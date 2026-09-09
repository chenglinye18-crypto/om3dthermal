"""Typed temperature observations extracted from a thermal pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from ..architecture_comparison import _temperature_maxima


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
