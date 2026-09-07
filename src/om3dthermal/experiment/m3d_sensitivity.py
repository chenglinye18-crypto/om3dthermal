"""Minimal M3D interface and logic/background parameter sensitivities."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

from pydantic import BaseModel

from om3dthermal.evaluation import ArchitectureCapacityFeasibility
from om3dthermal.evaluator import (
    LLMDecodePerformanceMetrics,
    LLMDecodeWorkloadThermalMetrics,
    evaluate_architecture_decode_memory_energy,
    evaluate_gpu_decode_energy,
    evaluate_llm_decode_workload_power,
    map_workload_power_to_thermal,
    run_llm_decode_workload_thermal,
)
from om3dthermal.power.config import CanonicalCaseConfig
from om3dthermal.power.system import ResolvedSystemPower
from om3dthermal.platform import (
    AffineGPUDecodePowerSpec,
    LocalMemoryGPUTransferOperatingPoint,
)
from om3dthermal.workload import LLMDecodeMetrics


SENSITIVITY_STATUS = "PARAMETRIC_SENSITIVITY"


class M3DInterfaceSensitivityRow(BaseModel):
    interface_energy_pj_per_bit: float
    read_total_energy_pj_per_bit: float
    interface_fraction_of_read_total: float
    interface_power_at_actual_bandwidth_W: float
    actual_bandwidth_bits_per_second: float
    status: str
    boundary_status: str


class M3DLogicBackgroundSensitivityRow(BaseModel):
    logic_background_power_W: float
    memory_dynamic_power_W: float
    refresh_power_W: float
    memory_background_power_W: float
    memory_total_power_W: float
    package_total_power_W: float
    memory_Tmax_degC: float
    package_Tmax_degC: float
    status: str


class M3DParameterSensitivityResult(BaseModel):
    architecture: str
    status: str
    interface_rows: tuple[M3DInterfaceSensitivityRow, ...]
    logic_background_rows: tuple[M3DLogicBackgroundSensitivityRow, ...]


def run_m3d_parameter_sensitivity(
    *,
    case: CanonicalCaseConfig,
    system: ResolvedSystemPower,
    workload: LLMDecodeMetrics,
    capacity: ArchitectureCapacityFeasibility,
    performance: LLMDecodePerformanceMetrics,
    interface_energy_values_pj_per_bit: Sequence[float],
    logic_background_values_W: Sequence[float],
    thermal_runner: Callable = run_llm_decode_workload_thermal,
    gpu_decode_power: AffineGPUDecodePowerSpec,
    transfer_operating_point: LocalMemoryGPUTransferOperatingPoint,
) -> M3DParameterSensitivityResult:
    """Run interface-only energy and logic-only power/thermal sensitivities."""
    if case.geometry.type != "orthogonal_m3d":
        raise ValueError("M3D parameter sensitivity requires orthogonal_m3d")
    if system.memory_result is None:
        raise ValueError("M3D parameter sensitivity requires resolved memory")
    bandwidth_bps = (
        transfer_operating_point.bandwidth_actual_bytes_per_s * 8.0)
    interface_rows = []
    for value in interface_energy_values_pj_per_bit:
        energy = evaluate_architecture_decode_memory_energy(
            workload, capacity, system, rho=1.0,
            interface_energy_sensitivity_pj_per_bit=value)
        assert energy.read_energy_pj_per_bit is not None
        assert energy.interface_energy_pj_per_bit is not None
        interface_rows.append(M3DInterfaceSensitivityRow(
            interface_energy_pj_per_bit=energy.interface_energy_pj_per_bit,
            read_total_energy_pj_per_bit=energy.read_energy_pj_per_bit,
            interface_fraction_of_read_total=(
                energy.interface_energy_pj_per_bit
                / energy.read_energy_pj_per_bit),
            interface_power_at_actual_bandwidth_W=(
                bandwidth_bps * energy.interface_energy_pj_per_bit * 1e-12),
            actual_bandwidth_bits_per_second=bandwidth_bps,
            status=SENSITIVITY_STATUS,
            boundary_status="CONDITIONAL_INTERFACE_ASSUMPTION_NOT_COMPLETE_PHY",
        ))

    nominal_energy = evaluate_architecture_decode_memory_energy(
        workload, capacity, system, rho=1.0)
    gpu_energy = evaluate_gpu_decode_energy(
        performance, nominal_energy, gpu_decode_power,
        transfer_operating_point=transfer_operating_point)
    operating_system = replace(
        system, gpu_power_W=gpu_energy.gpu_power_operating_point.gpu_power_W)
    logic_rows = []
    for value in logic_background_values_W:
        power = evaluate_llm_decode_workload_power(
            nominal_energy,
            performance,
            operating_system,
            unresolved_logic_background_policy=SENSITIVITY_STATUS,
            logic_background_sensitivity_W=value,
            gpu_decode_energy=gpu_energy,
        )
        mapping = map_workload_power_to_thermal(case, operating_system, power)
        thermal = thermal_runner(mapping)
        assert power.memory_dynamic_access_power_W is not None
        assert power.refresh_power_W is not None
        assert power.memory_background_power_W is not None
        assert power.memory_workload_total_W is not None
        assert power.package_workload_total_W is not None
        logic_rows.append(M3DLogicBackgroundSensitivityRow(
            logic_background_power_W=value,
            memory_dynamic_power_W=power.memory_dynamic_access_power_W,
            refresh_power_W=power.refresh_power_W,
            memory_background_power_W=power.memory_background_power_W,
            memory_total_power_W=power.memory_workload_total_W,
            package_total_power_W=power.package_workload_total_W,
            memory_Tmax_degC=thermal.memory_Tmax_degC,
            package_Tmax_degC=thermal.package_Tmax_degC,
            status=SENSITIVITY_STATUS,
        ))
    return M3DParameterSensitivityResult(
        architecture=case.name,
        status=SENSITIVITY_STATUS,
        interface_rows=tuple(interface_rows),
        logic_background_rows=tuple(logic_rows),
    )
