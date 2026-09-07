"""Case-level system-power resolution and coarse thermal source mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from om3dthermal.platform import (
    GPUBandwidthServiceOperatingPoint,
    GPUComputePowerOperatingPoint,
    GPUDecodePowerOperatingPoint,
    LocalMemoryGPUTransferOperatingPoint,
)

from .config import CanonicalCaseConfig
from .geometry import ResolvedGeometry
from .model import calculate_memory_power
from .result import MemoryPowerResult


@dataclass(frozen=True)
class ResolvedSystemPower:
    case_name: str
    architecture_type: str
    gpu_power_W: float
    memory_power_model: str
    memory_power_status: str
    read_bandwidth_gbps: float
    memory_gpu_bandwidth_demand_bytes_per_s: float | None
    memory_raw_bandwidth_capability_bytes_per_s: float | None
    gpu_peak_bandwidth_bytes_per_s: float | None
    memory_gpu_actual_bandwidth_bytes_per_s: float | None
    memory_gpu_transfer_bottleneck: str | None
    memory_dynamic_power_bandwidth_source: str
    memory_access_energy_pJ_per_bit: float | None
    memory_access_power_W: float | None
    refresh_power_W: float | None
    resolved_total_memory_power_W: float | None
    memory_result: MemoryPowerResult | None
    diagnostics: dict[str, Any]
    gpu_bandwidth_utilization: float | None = None
    gpu_sustained_bandwidth_bytes_per_s: float | None = None
    gpu_bandwidth_utilization_status: str | None = None

    def as_dict(self, *, display_na: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if display_na:
            return {key: ("N/A" if value is None else value)
                    for key, value in data.items()}
        return data


@dataclass(frozen=True)
class ThermalPowerTarget:
    name: str
    target_region: str
    power_W: float
    mapping_provenance: str


@dataclass(frozen=True)
class ResolvedThermalPowerMapping:
    case_name: str
    sources: tuple[ThermalPowerTarget, ...]
    total_mapped_power_W: float
    unresolved: bool


def resolve_system_power(
        case: CanonicalCaseConfig, *, project_root: Path,
        geometry: ResolvedGeometry,
        gpu_operating_point: (
            GPUDecodePowerOperatingPoint | GPUComputePowerOperatingPoint),
        transfer_operating_point: (
            LocalMemoryGPUTransferOperatingPoint | None),
        bandwidth_service_operating_point: (
            GPUBandwidthServiceOperatingPoint | None) = None,
) -> ResolvedSystemPower:
    """Resolve package power from explicit GPU and transfer operating points."""
    assert case.power.memory is not None
    mode = case.power.memory
    gpu_power_W = gpu_operating_point.gpu_power_W
    is_m3d = case.geometry.type == "orthogonal_m3d"
    if isinstance(gpu_operating_point, GPUDecodePowerOperatingPoint):
        if bandwidth_service_operating_point is not None and not math.isclose(
            bandwidth_service_operating_point.sustained_bandwidth_bytes_per_s,
            gpu_operating_point.bandwidth_actual_bytes_per_s,
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "GPU sustained service must equal the GPU power "
                "operating-point bandwidth")
        if is_m3d and mode.model == "analytical":
            if transfer_operating_point is None:
                raise ValueError(
                    "bandwidth-bound M3D power requires a shared "
                    "memory-to-GPU transfer operating point")
            if (bandwidth_service_operating_point is None and not math.isclose(
                gpu_operating_point.bandwidth_actual_bytes_per_s,
                transfer_operating_point.bandwidth_actual_bytes_per_s,
                rel_tol=1e-12,
                abs_tol=1e-6,
            )):
                raise ValueError(
                    "GPU and M3D power must share the same actual bandwidth")
            expected_gpu_demand = (
                min(
                    transfer_operating_point.bandwidth_demand_bytes_per_s,
                    transfer_operating_point.memory_capability_bytes_per_s,
                )
                if bandwidth_service_operating_point is None
                else bandwidth_service_operating_point
                .sustained_bandwidth_bytes_per_s
            )
            if not math.isclose(
                gpu_operating_point.bandwidth_demand_bytes_per_s,
                expected_gpu_demand,
                rel_tol=1e-12,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "GPU demand must equal memory-deliverable transfer demand")
            expected_utilization = (
                gpu_operating_point.bandwidth_actual_bytes_per_s
                / transfer_operating_point.gpu_peak_bandwidth_bytes_per_s)
            if not math.isclose(
                gpu_operating_point.bandwidth_utilization,
                expected_utilization,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "GPU utilization must use the transfer GPU peak")
            if (bandwidth_service_operating_point is not None
                    and not math.isclose(
                        bandwidth_service_operating_point
                        .transfer_ceiling_bytes_per_s,
                        transfer_operating_point.bandwidth_actual_bytes_per_s,
                        rel_tol=1e-12,
                        abs_tol=1e-6)):
                raise ValueError(
                    "sustained GPU service must consume the shared transfer "
                    "ceiling")
        elif transfer_operating_point is not None:
            raise ValueError(
                "local M3D-to-GPU transfer is invalid for non-M3D memory")

        if bandwidth_service_operating_point is not None:
            read_bandwidth_gbps = (
                bandwidth_service_operating_point
                .sustained_bandwidth_bytes_per_s * 8.0 / 1e9)
            bandwidth_source = (
                "GPU_SUSTAINED_BANDWIDTH_SERVICE_OPERATING_POINT")
        else:
            read_bandwidth_gbps = (
                transfer_operating_point.bandwidth_actual_bytes_per_s
                * 8.0 / 1e9
                if transfer_operating_point is not None
                else case.workload.read_bandwidth_gbps)
            bandwidth_source = (
                "SHARED_MEMORY_GPU_TRANSFER_OPERATING_POINT"
                if transfer_operating_point is not None
                else "SCENARIO_DEMAND_NON_M3D_UNCHANGED")
    else:
        if transfer_operating_point is not None:
            raise ValueError(
                "compute-bound GPU power does not consume a bandwidth transfer")
        if bandwidth_service_operating_point is not None:
            raise ValueError(
                "compute-bound GPU power does not consume GPU bandwidth service")
        read_bandwidth_gbps = case.workload.read_bandwidth_gbps
        bandwidth_source = "COMPUTE_REGIME_SCENARIO_DEMAND_UNCHANGED"

    transfer_fields = {
        "memory_gpu_bandwidth_demand_bytes_per_s": (
            None if transfer_operating_point is None
            else transfer_operating_point.bandwidth_demand_bytes_per_s),
        "memory_raw_bandwidth_capability_bytes_per_s": (
            None if transfer_operating_point is None
            else transfer_operating_point.memory_capability_bytes_per_s),
        "gpu_peak_bandwidth_bytes_per_s": (
            None if transfer_operating_point is None
            else transfer_operating_point.gpu_peak_bandwidth_bytes_per_s),
        "memory_gpu_actual_bandwidth_bytes_per_s": (
            None if transfer_operating_point is None
            else transfer_operating_point.bandwidth_actual_bytes_per_s),
        "memory_gpu_transfer_bottleneck": (
            None if transfer_operating_point is None
            else transfer_operating_point.bottleneck),
        "gpu_bandwidth_utilization": (
            None if bandwidth_service_operating_point is None
            else bandwidth_service_operating_point.gpu_bandwidth_utilization),
        "gpu_sustained_bandwidth_bytes_per_s": (
            None if bandwidth_service_operating_point is None
            else bandwidth_service_operating_point
            .sustained_bandwidth_bytes_per_s),
        "gpu_bandwidth_utilization_status": (
            None if bandwidth_service_operating_point is None
            else bandwidth_service_operating_point.utilization_status),
        "memory_dynamic_power_bandwidth_source": bandwidth_source,
    }
    if mode.model == "unresolved":
        return ResolvedSystemPower(
            case_name=case.name,
            architecture_type=case.geometry.type,
            gpu_power_W=gpu_power_W,
            memory_power_model=mode.model,
            memory_power_status=mode.status,
            read_bandwidth_gbps=read_bandwidth_gbps,
            **transfer_fields,
            memory_access_energy_pJ_per_bit=None,
            memory_access_power_W=None,
            refresh_power_W=None,
            resolved_total_memory_power_W=None,
            memory_result=None,
            diagnostics={"memory_power_reason": "NO_VALIDATED_M3D_SI_PRIMITIVE"},
        )
    if mode.model == "reference_fixed":
        return ResolvedSystemPower(
            case_name=case.name,
            architecture_type=case.geometry.type,
            gpu_power_W=gpu_power_W,
            memory_power_model=mode.model,
            memory_power_status=mode.status,
            read_bandwidth_gbps=read_bandwidth_gbps,
            **transfer_fields,
            memory_access_energy_pJ_per_bit=None,
            memory_access_power_W=None,
            refresh_power_W=None,
            resolved_total_memory_power_W=mode.total_power_W,
            memory_result=None,
            diagnostics={
                "reference_source": mode.source,
                "reference_provenance": mode.provenance,
                "accounting_level": mode.accounting_level,
            },
        )

    memory = calculate_memory_power(
        case, project_root=project_root, geometry=geometry,
        read_bandwidth_gbps=read_bandwidth_gbps)
    total = (
        memory.P_access_W + float(memory.P_refresh_W or 0.0)
        + float(memory.P_memory_background_W or 0.0)
        + float(memory.P_logic_background_W or 0.0))
    diagnostics = {
        "case_name": case.name,
        "architecture_type": case.geometry.type,
        "gpu_power_W": gpu_power_W,
        "memory_power_model": mode.model,
        "memory_power_status": mode.status,
        "resolved_total_memory_power_W": total,
        **transfer_fields,
        "E_memory_internal_pj_bit": memory.E_memory_internal_pj_bit,
        "E_vertical_pj_bit": memory.E_vertical_pj_bit,
        "E_base_route_pj_bit": memory.E_base_route_pj_bit,
        "E_interface_pj_bit": memory.E_interface_pj_bit,
        "E_access_total_pj_bit": memory.E_access_total_pj_bit,
        "P_refresh_W": memory.P_refresh_W,
        "P_base_FEOL_logic_W": memory.P_logic_background_W,
        **memory.diagnostics,
    }
    if case.geometry.type == "dreamram_hbm":
        bandwidth_scale = read_bandwidth_gbps * 1e-3
        diagnostics.update({
            "P_DRAM_access_W": (
                memory.E_memory_internal_pj_bit
                + memory.E_vertical_pj_bit
                + memory.E_interface_pj_bit
                + memory.E_feol_route_pj_bit) * bandwidth_scale,
            "P_base_route_W": (
                memory.E_base_route_pj_bit * bandwidth_scale),
            "P_base_FEOL_logic_status": "NOT_SEPARATELY_MODELED",
            "P_base_FEOL_logic_provenance": (
                "PLACEHOLDER_ZERO_FOR_FUTURE_LOGIC_MODEL"),
        })
    return ResolvedSystemPower(
        case_name=case.name,
        architecture_type=case.geometry.type,
        gpu_power_W=gpu_power_W,
        memory_power_model=mode.model,
        memory_power_status=mode.status,
        read_bandwidth_gbps=read_bandwidth_gbps,
        **transfer_fields,
        memory_access_energy_pJ_per_bit=memory.E_access_total_pj_bit,
        memory_access_power_W=memory.P_access_W,
        refresh_power_W=memory.P_refresh_W,
        resolved_total_memory_power_W=total,
        memory_result=memory,
        diagnostics=diagnostics,
    )


def map_system_power_to_thermal(
        case: CanonicalCaseConfig,
        system: ResolvedSystemPower) -> ResolvedThermalPowerMapping:
    """Map resolved totals to existing coarse thermal carrier regions."""
    gpu = ThermalPowerTarget(
        name="gpu", target_region="GPU_FEOL",
        power_W=system.gpu_power_W,
        mapping_provenance="EXISTING_UNIFORM_ACTIVE_REGION_MODEL")
    if system.resolved_total_memory_power_W is None:
        return ResolvedThermalPowerMapping(
            case_name=case.name, sources=(gpu,),
            total_mapped_power_W=gpu.power_W, unresolved=True)

    if case.geometry.type == "dreamram_hbm":
        if system.memory_result is None:
            raise ValueError("analytical HBM mapping requires component energy")
        group_count = int(case.geometry.layout["visible_group_count"])
        result = system.memory_result
        bandwidth = system.read_bandwidth_gbps
        base_route_power = result.E_base_route_pj_bit * bandwidth * 1e-3
        dram_power = (
            (result.E_memory_internal_pj_bit + result.E_vertical_pj_bit
             + result.E_interface_pj_bit + result.E_feol_route_pj_bit)
            * bandwidth * 1e-3
            + float(result.P_refresh_W or 0.0)
            + float(result.P_memory_background_W or 0.0))
        base_logic_power = float(result.P_logic_background_W or 0.0)
        if abs(dram_power + base_route_power + base_logic_power
               - system.resolved_total_memory_power_W) > 1e-10:
            raise RuntimeError("component-aware HBM power mapping does not close")
        memory_sources = tuple(
            source
            for index in range(group_count)
            for source in (
                ThermalPowerTarget(
                    name=f"dram_group_{index}",
                    target_region=f"DRAM_BEOL_GROUP_{index}",
                    power_W=dram_power / group_count,
                    mapping_provenance=(
                        "MODELING_CHOICE_COARSE_DRAM_TSV_DQ_REFRESH_TO_DRAM_BEOL")),
                ThermalPowerTarget(
                    name=f"base_route_group_{index}",
                    target_region=f"HBM_BASE_BEOL_GROUP_{index}",
                    power_W=base_route_power / group_count,
                    mapping_provenance=(
                        "DREAMRAM_BASE_ROUTE_TO_PHYSICAL_HBM_BASE_BEOL")),
            ))
    elif case.geometry.type == "orthogonal_si":
        memory_sources = (ThermalPowerTarget(
            name="orthogonal_si_memory", target_region="ORTHOGONAL_DRAM_BEOL",
            power_W=system.resolved_total_memory_power_W,
            mapping_provenance="MODELING_CHOICE_COARSE_ACTIVE_BEOL"),)
    else:
        assert system.memory_result is not None or (
            system.memory_power_model == "reference_fixed")
        if system.memory_result is None:
            memory_sources = (ThermalPowerTarget(
                name="m3d_reference_memory",
                target_region="M3D_BITCELL_BEOL_STACK",
                power_W=system.resolved_total_memory_power_W,
                mapping_provenance=(
                    "REFERENCE_FIXED_UNIFORM_M3D_BITCELL_BEOL_MAPPING")),)
        else:
            memory_sources = (ThermalPowerTarget(
                name="m3d_memory_bitcell_beol",
                target_region="M3D_BITCELL_BEOL_STACK",
                power_W=system.resolved_total_memory_power_W,
                mapping_provenance=(
                    "MODELING_CHOICE_UNIFORM_COMPLETE_M3D_BITCELL_BEOL")),)
    sources = (gpu, *memory_sources)
    return ResolvedThermalPowerMapping(
        case_name=case.name, sources=sources,
        total_mapped_power_W=sum(source.power_W for source in sources),
        unresolved=False)
