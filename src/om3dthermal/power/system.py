"""Case-level system-power resolution and coarse thermal source mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import CanonicalCaseConfig, find_project_root, load_case_config
from .geometry import ResolvedGeometry, resolve_case_geometry
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
    memory_access_energy_pJ_per_bit: float | None
    memory_access_power_W: float | None
    refresh_power_W: float | None
    resolved_total_memory_power_W: float | None
    memory_result: MemoryPowerResult | None
    diagnostics: dict[str, Any]

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
        geometry: ResolvedGeometry) -> ResolvedSystemPower:
    """Resolve GPU and memory power from one canonical case."""
    assert case.power.gpu is not None
    assert case.power.memory is not None
    mode = case.power.memory
    if mode.model == "unresolved":
        return ResolvedSystemPower(
            case_name=case.name,
            architecture_type=case.geometry.type,
            gpu_power_W=case.power.gpu.power_W,
            memory_power_model=mode.model,
            memory_power_status=mode.status,
            read_bandwidth_gbps=case.workload.read_bandwidth_gbps,
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
            gpu_power_W=case.power.gpu.power_W,
            memory_power_model=mode.model,
            memory_power_status=mode.status,
            read_bandwidth_gbps=case.workload.read_bandwidth_gbps,
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
        case, project_root=project_root, geometry=geometry)
    total = (
        memory.P_access_W + float(memory.P_refresh_W or 0.0)
        + float(memory.P_memory_background_W or 0.0)
        + float(memory.P_logic_background_W or 0.0))
    diagnostics = {
        "case_name": case.name,
        "architecture_type": case.geometry.type,
        "gpu_power_W": case.power.gpu.power_W,
        "memory_power_model": mode.model,
        "memory_power_status": mode.status,
        "resolved_total_memory_power_W": total,
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
        bandwidth_scale = case.workload.read_bandwidth_gbps * 1e-3
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
        gpu_power_W=case.power.gpu.power_W,
        memory_power_model=mode.model,
        memory_power_status=mode.status,
        read_bandwidth_gbps=case.workload.read_bandwidth_gbps,
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
                name="m3d_reference_memory", target_region="M3D_BITCELL_STACK",
                power_W=system.resolved_total_memory_power_W,
                mapping_provenance="REFERENCE_FIXED_COARSE_BITCELL_MAPPING"),)
        else:
            stack = case.geometry.m3d_stack
            if stack is None:
                raise ValueError("orthogonal M3D mapping requires m3d_stack")
            bitcell_um = (
                stack.bitcell_layers * stack.bitcell_layer_pitch_nm * 1e-3)
            interconnect_um = stack.beol_interconnect_um
            beol_um = bitcell_um + interconnect_um
            bitcell_power = (
                system.resolved_total_memory_power_W * bitcell_um / beol_um)
            interconnect_power = (
                system.resolved_total_memory_power_W - bitcell_power)
            memory_sources = (
                ThermalPowerTarget(
                    name="m3d_memory_beol_bitcell",
                    target_region="M3D_BITCELL_STACK",
                    power_W=bitcell_power,
                    mapping_provenance=(
                        "MODELING_CHOICE_UNIFORM_VOLUME_COMPLETE_M3D_BEOL")),
                ThermalPowerTarget(
                    name="m3d_memory_beol_interconnect",
                    target_region="M3D_BEOL_INTERCONNECT",
                    power_W=interconnect_power,
                    mapping_provenance=(
                        "MODELING_CHOICE_UNIFORM_VOLUME_COMPLETE_M3D_BEOL")),
            )
    sources = (gpu, *memory_sources)
    return ResolvedThermalPowerMapping(
        case_name=case.name, sources=sources,
        total_mapped_power_W=sum(source.power_W for source in sources),
        unresolved=False)


def run_case_system_power(path: str | Path) -> ResolvedSystemPower:
    case_path = Path(path).resolve()
    case = load_case_config(case_path)
    geometry = resolve_case_geometry(case)
    return resolve_system_power(
        case, project_root=find_project_root(case_path), geometry=geometry)
