"""E6 workload-dependent power mapping and frozen GPU-PCG integration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from pydantic import BaseModel

from om3dthermal.case_runner import run_steady_pipeline
from om3dthermal.config import (
    PowerSourceConfig,
    SimulationConfig,
    ThermalPowerSourcesConfig,
)
from om3dthermal.power.config import CanonicalCaseConfig
from om3dthermal.power.system import ResolvedSystemPower
from om3dthermal.thermal.case_adapter import (
    compile_canonical_thermal_case,
    extract_temperature_observables,
)

from .llm_decode_workload_power import LLMDecodeWorkloadPowerMetrics


# Backward-compatible public name retained for existing tests/callers while
# the implementation is now owned by the explicit thermal adapter boundary.
compile_case_thermal = compile_canonical_thermal_case


WRITE_SPATIAL_STATUS = (
    "WRITE_SPATIAL_DISTRIBUTION_READ_SHAPE_SENSITIVITY_ONLY")
M3D_LOWER_BOUND_STATUS = (
    "CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND")
M3D_PARAMETRIC_STATUS = "PARAMETRIC_SENSITIVITY"
THERMAL_SCENARIO_STATUS = "CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY"
THERMAL_BACKEND = "gpu_pcg"
POWER_CLOSURE_ABS_TOL_W = 1e-9


class WorkloadPowerBlockedError(ValueError):
    """Raised before thermal construction for a blocked E5 result."""


class WorkloadThermalSource(BaseModel):
    name: str
    power_W: float
    selector: dict[str, object]
    mapping_provenance: str


@dataclass(frozen=True)
class WorkloadThermalMapping:
    architecture: str
    rho: float
    simulation: SimulationConfig
    sources: tuple[WorkloadThermalSource, ...]
    mapped_total_power_W: float
    expected_package_total_power_W: float
    absolute_closure_error_W: float
    relative_closure_error: float
    write_spatial_distribution_status: str
    memory_total_completeness_status: str


class LLMDecodeWorkloadThermalMetrics(BaseModel):
    architecture: str
    rho: float
    mapped_package_power_W: float
    expected_package_power_W: float
    source_power_breakdown_W: dict[str, float]
    power_closure_absolute_error_W: float
    power_closure_relative_error: float

    memory_Tmax_degC: float
    gpu_Tmax_degC: float
    package_Tmax_degC: float

    converged: bool
    iterations: int
    final_relative_residual: float
    max_temperature_update_K: float
    relative_power_imbalance: float
    cell_count: int
    internal_edge_count: int
    full_vector_d2h_during_iteration: int

    thermal_backend: Literal["gpu_pcg"]
    precision_status: Literal["FP64"]
    preconditioner_status: Literal["JACOBI_DIAGONAL"]
    initial_temperature_K: Literal[293.15]
    relative_residual_tolerance: Literal[0.001]
    max_temperature_update_tolerance_K: Literal[0.01]
    max_iterations: Literal[100000]
    check_interval: Literal[10]
    warm_start_status: Literal["FRESH_SOLVE_NO_WARM_START"]
    write_spatial_distribution_status: Literal[
        "WRITE_SPATIAL_DISTRIBUTION_READ_SHAPE_SENSITIVITY_ONLY"]
    memory_total_completeness_status: str
    scenario_status: Literal[
        "CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY"]


def _finite_nonnegative(name: str, value: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _old_source(compiled: SimulationConfig, name: str) -> PowerSourceConfig:
    assert compiled.thermal_power_sources is not None
    matches = [source for source in compiled.thermal_power_sources.sources
               if source.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one existing source {name!r}")
    return matches[0]


def _replacement(
    old: PowerSourceConfig,
    power_W: float,
    provenance: str,
) -> tuple[PowerSourceConfig, WorkloadThermalSource]:
    source = PowerSourceConfig(
        name=old.name,
        total_power=power_W,
        selector=old.selector,
        distribution=old.distribution,
        metadata={
            **old.metadata,
            "workload_mapping_provenance": provenance,
            "write_spatial_distribution_status": WRITE_SPATIAL_STATUS,
        },
    )
    audit = WorkloadThermalSource(
        name=source.name,
        power_W=power_W,
        selector=source.selector.model_dump(exclude_none=True),
        mapping_provenance=provenance,
    )
    return source, audit


def map_workload_power_to_thermal(
    case: CanonicalCaseConfig,
    system: ResolvedSystemPower,
    power: LLMDecodeWorkloadPowerMetrics,
) -> WorkloadThermalMapping:
    """Replace every compiled source power with E5 workload power."""
    if power.evaluation_status != "EVALUATED_WORKLOAD_DEPENDENT_MEMORY_POWER":
        raise WorkloadPowerBlockedError(
            "blocked E5 workload power cannot enter thermal construction")
    if case.name != system.case_name or power.architecture != case.name:
        raise ValueError("case/system/workload-power architecture mismatch")
    if system.memory_result is None:
        raise ValueError("resolved memory decomposition is required")

    gpu = _finite_nonnegative("fixed_gpu_power_W", power.fixed_gpu_power_W)
    dynamic = _finite_nonnegative(
        "memory_dynamic_access_power_W", power.memory_dynamic_access_power_W)
    refresh = _finite_nonnegative("refresh_power_W", power.refresh_power_W)
    background = _finite_nonnegative(
        "memory_background_power_W", power.memory_background_power_W)
    logic = _finite_nonnegative(
        "logic_background_effective_W", power.logic_background_effective_W)
    expected = _finite_nonnegative(
        "package_workload_total_W", power.package_workload_total_W)

    compiled = compile_case_thermal(case, system)
    new_sources: list[PowerSourceConfig] = []
    audit_sources: list[WorkloadThermalSource] = []

    def add(name: str, watts: float, provenance: str) -> None:
        source, audit = _replacement(
            _old_source(compiled, name), watts, provenance)
        new_sources.append(source)
        audit_sources.append(audit)

    add("gpu", gpu, "E5_FIXED_GPU_POWER_REPLACES_EXISTING_SOURCE")

    if case.geometry.type == "dreamram_hbm":
        result = system.memory_result
        dram_like_e = (
            result.E_memory_internal_pj_bit + result.E_vertical_pj_bit
            + result.E_feol_route_pj_bit + result.E_interface_pj_bit)
        base_e = result.E_base_route_pj_bit
        total_e = dram_like_e + base_e
        if total_e <= 0.0 or not math.isfinite(total_e):
            raise ValueError("HBM read-energy decomposition total is invalid")
        dynamic_dram = dynamic * dram_like_e / total_e
        dynamic_base = dynamic * base_e / total_e
        if not math.isclose(dynamic_dram + dynamic_base, dynamic,
                            rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("HBM dynamic decomposition does not close")
        group_count = int(case.geometry.layout["visible_group_count"])
        if group_count <= 0:
            raise ValueError("HBM visible group count must be positive")
        dram_group = (dynamic_dram + refresh + background) / group_count
        base_group = (dynamic_base + logic) / group_count
        for index in range(group_count):
            add(
                f"dram_group_{index}", dram_group,
                "E5_DYNAMIC_DRAM_SHARE_PLUS_REFRESH_BACKGROUND_EQUAL_GROUP")
            add(
                f"base_route_group_{index}", base_group,
                "E5_DYNAMIC_BASE_SHARE_PLUS_LOGIC_EQUAL_GROUP")
    elif case.geometry.type == "orthogonal_si":
        memory_total = _finite_nonnegative(
            "memory_workload_total_W", power.memory_workload_total_W)
        add(
            "orthogonal_si_memory", memory_total,
            "E5_COMPLETE_MEMORY_WORKLOAD_TOTAL_TO_EXISTING_ORTHOGONAL_BEOL")
    elif case.geometry.type == "orthogonal_m3d":
        if power.memory_total_completeness_status not in {
                M3D_LOWER_BOUND_STATUS, M3D_PARAMETRIC_STATUS}:
            raise ValueError(
                "M3D logic-background boundary status was not preserved")
        memory_total = _finite_nonnegative(
            "memory_workload_total_W", power.memory_workload_total_W)
        add(
            "m3d_memory_bitcell_beol", memory_total,
            ("E5_PARAMETRIC_MEMORY_TOTAL_TO_EXISTING_MERGED_M3D_REGION"
             if power.memory_total_completeness_status == M3D_PARAMETRIC_STATUS
             else "E5_CONDITIONAL_MEMORY_TOTAL_TO_EXISTING_MERGED_M3D_REGION"))
    else:
        raise ValueError(f"unsupported architecture type {case.geometry.type!r}")

    mapped = sum(source.power_W for source in audit_sources)
    absolute_error = abs(mapped - expected)
    relative_error = absolute_error / expected if expected > 0.0 else 0.0
    if absolute_error > POWER_CLOSURE_ABS_TOL_W:
        raise RuntimeError("workload thermal source power does not close")
    simulation = compiled.model_copy(update={
        "thermal_power_sources": ThermalPowerSourcesConfig(sources=new_sources)})
    return WorkloadThermalMapping(
        architecture=case.name,
        rho=power.rho,
        simulation=simulation,
        sources=tuple(audit_sources),
        mapped_total_power_W=mapped,
        expected_package_total_power_W=expected,
        absolute_closure_error_W=absolute_error,
        relative_closure_error=relative_error,
        write_spatial_distribution_status=WRITE_SPATIAL_STATUS,
        memory_total_completeness_status=(
            power.memory_total_completeness_status),
    )


def run_llm_decode_workload_thermal(
    mapping: WorkloadThermalMapping,
) -> LLMDecodeWorkloadThermalMetrics:
    """Run one fresh frozen FP64 matrix-free Jacobi GPU-PCG solve."""
    pipeline = run_steady_pipeline(
        mapping.simulation,
        rtol=1e-3,
        max_delta_t_K=1e-2,
        max_iterations=100_000,
        check_interval=10,
        initial_temperature_K=293.15,
        backend=THERMAL_BACKEND,
    )
    result = pipeline.result
    info = result.solver_info
    if info.get("backend") != "gpu_pcg":
        raise RuntimeError("frozen E6 backend was not gpu_pcg")
    if info.get("dtype") != "float64":
        raise RuntimeError("frozen E6 solve was not FP64")
    if info.get("preconditioner") != "jacobi_diagonal":
        raise RuntimeError("frozen E6 preconditioner was not Jacobi")
    if info.get("full_vector_d2h_during_iteration") != 0:
        raise RuntimeError("full-vector D2H occurred during GPU iteration")
    mapped_actual = float(pipeline.power.total_power_W)
    actual_error = abs(
        mapped_actual - mapping.expected_package_total_power_W)
    if actual_error > POWER_CLOSURE_ABS_TOL_W:
        raise RuntimeError("discretized workload power does not close")
    observables = extract_temperature_observables(pipeline)
    max_update = result.max_temperature_update
    if max_update is None:
        raise RuntimeError("GPU-PCG result did not report temperature update")
    return LLMDecodeWorkloadThermalMetrics(
        architecture=mapping.architecture,
        rho=mapping.rho,
        mapped_package_power_W=mapped_actual,
        expected_package_power_W=mapping.expected_package_total_power_W,
        source_power_breakdown_W=dict(pipeline.power_by_source),
        power_closure_absolute_error_W=actual_error,
        power_closure_relative_error=(
            actual_error / mapping.expected_package_total_power_W
            if mapping.expected_package_total_power_W > 0.0 else 0.0),
        memory_Tmax_degC=observables.memory_Tmax_degC,
        gpu_Tmax_degC=observables.gpu_Tmax_degC,
        package_Tmax_degC=observables.package_Tmax_degC,
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_relative_residual=float(result.final_relative_residual),
        max_temperature_update_K=float(max_update),
        relative_power_imbalance=float(result.relative_power_imbalance),
        cell_count=int(pipeline.cell_count),
        internal_edge_count=int(pipeline.internal_edge_count),
        full_vector_d2h_during_iteration=int(
            info["full_vector_d2h_during_iteration"]),
        thermal_backend="gpu_pcg",
        precision_status="FP64",
        preconditioner_status="JACOBI_DIAGONAL",
        initial_temperature_K=293.15,
        relative_residual_tolerance=0.001,
        max_temperature_update_tolerance_K=0.01,
        max_iterations=100000,
        check_interval=10,
        warm_start_status="FRESH_SOLVE_NO_WARM_START",
        write_spatial_distribution_status=WRITE_SPATIAL_STATUS,
        memory_total_completeness_status=(
            mapping.memory_total_completeness_status),
        scenario_status=THERMAL_SCENARIO_STATUS,
    )
