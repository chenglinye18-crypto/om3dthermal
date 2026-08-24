"""Public workload-to-thermal boundary over the validated legacy compiler."""

from __future__ import annotations

from dataclasses import dataclass

from om3dthermal.architecture_comparison import (
    _temperature_maxima,
    compile_case_thermal,
)
from om3dthermal.config import SimulationConfig
from om3dthermal.power.config import CanonicalCaseConfig
from om3dthermal.power.system import ResolvedSystemPower


@dataclass(frozen=True)
class ThermalObservables:
    memory_Tmax_degC: float
    gpu_Tmax_degC: float
    package_Tmax_degC: float


def compile_canonical_thermal_case(
    case: CanonicalCaseConfig,
    system: ResolvedSystemPower,
) -> SimulationConfig:
    """Compatibility façade; does not alter geometry, power, or numerics."""
    return compile_case_thermal(case, system)


def extract_temperature_observables(pipeline) -> ThermalObservables:
    """Public typed form of the existing region-wise Tmax extraction."""
    memory, gpu, package = _temperature_maxima(pipeline)
    return ThermalObservables(
        memory_Tmax_degC=memory,
        gpu_Tmax_degC=gpu,
        package_Tmax_degC=package,
    )
