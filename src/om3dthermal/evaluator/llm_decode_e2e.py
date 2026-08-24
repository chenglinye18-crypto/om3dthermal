"""Pure typed aggregation for the conditional LLM decode E2E table (E7)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal, Sequence

from pydantic import BaseModel

from om3dthermal.workload.architecture_capacity import (
    ArchitectureCapacityFeasibility,
)
from om3dthermal.workload.llm_decode import LLMDecodeInput, LLMDecodeMetrics

from .llm_decode_architecture_energy import (
    ArchitectureDecodeMemoryEnergyMetrics,
)
from .llm_decode_performance import LLMDecodePerformanceMetrics
from .llm_decode_workload_power import LLMDecodeWorkloadPowerMetrics
from .llm_decode_workload_thermal import LLMDecodeWorkloadThermalMetrics


ARCHITECTURE_DISPLAY_NAMES = {
    "conventional_hbm_2x1": "Conventional HBM",
    "orthogonal_si": "Orthogonal Si",
    "orthogonal_m3d_igzo": "Orthogonal M3D-IGZO",
}
ARCHITECTURE_ORDER = tuple(ARCHITECTURE_DISPLAY_NAMES)
FROZEN_RHOS = (0.0, 1.0, 100.0, 1000.0)

MATCHED_BANDWIDTH_STATUS = "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED"
MATCHED_PERFORMANCE_STATUS = "EVALUATED_MATCHED_REFERENCE_SCENARIO"
BLOCKED_PERFORMANCE_STATUS = "BLOCKED_BY_CAPACITY"
EVALUATED_ENERGY_STATUS = (
    "EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY")
EVALUATED_POWER_STATUS = "EVALUATED_WORKLOAD_DEPENDENT_MEMORY_POWER"
M3D_COMPLETENESS_STATUS = (
    "CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND")
WRITE_SPATIAL_STATUS = (
    "WRITE_SPATIAL_DISTRIBUTION_READ_SHAPE_SENSITIVITY_ONLY")
SCENARIO_STATUS = "CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY"
POWER_CLOSURE_ABS_TOL_W = 1e-9
NUMERIC_ABS_TOL = 1e-12


class ConditionalLLMDecodeE2ERow(BaseModel):
    """One architecture/rho row; no physical quantity is recomputed here."""

    architecture: str
    architecture_display_name: str
    rho: float
    workload_identifier: str
    batch_size: int
    context_length: int

    required_capacity_bytes: float
    usable_capacity_bytes: float
    capacity_margin_bytes: float
    capacity_utilization: float | None
    capacity_feasible: bool
    capacity_status: str

    aggregate_tokens_per_second: float | None
    per_sequence_tokens_per_second: float | None
    memory_time_per_token_s: float | None
    compute_time_per_token_s: float | None
    bottleneck: str
    performance_status: str
    bandwidth_status: str

    read_bytes_per_token: float
    write_bytes_per_token: float
    flops_per_token: int
    read_energy_pj_per_bit: float | None
    write_energy_pj_per_bit: float | None
    memory_dynamic_energy_j_per_token: float | None
    energy_status: str
    write_energy_status: str
    rho_zero_status: str | None

    memory_dynamic_power_W: float | None
    memory_total_power_W: float | None
    gpu_power_W: float | None
    package_power_W: float | None
    power_status: str
    memory_total_completeness_status: str

    memory_Tmax_degC: float | None
    gpu_Tmax_degC: float | None
    package_Tmax_degC: float | None
    thermal_converged: bool | None
    thermal_status: str
    power_closure_absolute_error_W: float | None
    power_closure_status: str
    write_spatial_distribution_status: str | None

    bandwidth_capability_status: Literal["NOT_VALIDATED"]
    write_energy_model_status: Literal["NOT_VALIDATED"]
    gpu_energy_model_status: Literal["NOT_AVAILABLE"]
    system_j_token_status: Literal["NOT_AVAILABLE"]
    m3d_logic_background_status: Literal[
        "NOT_APPLICABLE", "CONDITIONAL_LOWER_BOUND"]
    scenario_status: Literal["CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY"]


def _same(name: str, left: float | int, right: float | int,
          *, abs_tol: float = NUMERIC_ABS_TOL) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0,
                        abs_tol=abs_tol):
        raise ValueError(f"{name} mismatch")


def _all_none(name: str, values: Sequence[object | None]) -> None:
    if any(value is not None for value in values):
        raise ValueError(f"{name} must be absent when capacity is infeasible")


def assemble_conditional_llm_decode_e2e_row(
    workload_input: LLMDecodeInput,
    workload: LLMDecodeMetrics,
    capacity: ArchitectureCapacityFeasibility,
    performance: LLMDecodePerformanceMetrics,
    energy: ArchitectureDecodeMemoryEnergyMetrics,
    power: LLMDecodeWorkloadPowerMetrics,
    thermal: LLMDecodeWorkloadThermalMetrics | None,
    *,
    workload_identifier: str,
    architecture_display_name: str | None = None,
) -> ConditionalLLMDecodeE2ERow:
    """Validate cross-stage semantics and forward one conditional E2E row."""
    architecture = capacity.architecture
    display_name = architecture_display_name or ARCHITECTURE_DISPLAY_NAMES.get(
        architecture)
    if not display_name:
        raise ValueError("unsupported E7 architecture")
    if not workload_identifier or not workload_identifier.strip():
        raise ValueError("workload_identifier must be non-empty")
    identities = (performance.architecture, energy.architecture,
                  power.architecture)
    if any(value != architecture for value in identities):
        raise ValueError("architecture identity mismatch")
    if thermal is not None and thermal.architecture != architecture:
        raise ValueError("thermal architecture identity mismatch")

    _same("required capacity", workload.required_capacity_bytes,
          capacity.required_capacity_bytes)
    _same("performance read traffic", workload.read_bytes_per_token,
          performance.read_bytes_per_token)
    _same("performance write traffic", workload.write_bytes_per_token,
          performance.write_bytes_per_token)
    if workload.flops_per_token != performance.flops_per_token:
        raise ValueError("workload/performance FLOPs mismatch")
    if workload_input.batch_size != performance.batch_size:
        raise ValueError("workload/performance batch-size mismatch")
    if workload_input.weight_activity_model != workload.weight_activity_model or (
            workload_input.weight_reuse_model != workload.weight_reuse_model) or (
            workload_input.kv_read_model != workload.kv_read_model):
        raise ValueError("workload modeling identity mismatch")

    for label, observed, expected in (
        ("energy read traffic", energy.read_bytes_per_token,
         workload.read_bytes_per_token),
        ("energy write traffic", energy.write_bytes_per_token,
         workload.write_bytes_per_token),
    ):
        _same(label, observed, expected)
    feasibility_values = (performance.capacity_feasible,
                          energy.capacity_feasible, power.capacity_feasible)
    if any(value != capacity.capacity_feasible for value in feasibility_values):
        raise ValueError("cross-stage capacity feasibility mismatch")
    if energy.rho != power.rho:
        raise ValueError("energy/power rho mismatch")
    if thermal is not None and thermal.rho != energy.rho:
        raise ValueError("power/thermal rho mismatch")
    if performance.bandwidth_status != MATCHED_BANDWIDTH_STATUS:
        raise ValueError("bandwidth provenance is not matched-reference")

    if not capacity.capacity_feasible:
        if performance.performance_status != BLOCKED_PERFORMANCE_STATUS:
            raise ValueError("infeasible capacity did not block performance")
        if thermal is not None:
            raise ValueError("capacity-infeasible row cannot carry thermal result")
        _all_none("infeasible performance", (
            performance.aggregate_tokens_per_second,
            performance.memory_time_per_token_equivalent_s,
            performance.compute_time_per_token_equivalent_s))
        _all_none("infeasible energy", (
            energy.read_dynamic_energy_j_per_token,
            energy.write_dynamic_energy_j_per_token,
            energy.memory_dynamic_energy_j_per_token))
        _all_none("infeasible power", (
            power.memory_dynamic_access_power_W,
            power.memory_workload_total_W,
            power.fixed_gpu_power_W,
            power.package_workload_total_W))
        thermal_status = "BLOCKED_BY_CAPACITY"
        closure_status = "NOT_EVALUATED_CAPACITY_INFEASIBLE"
    else:
        if performance.performance_status != MATCHED_PERFORMANCE_STATUS:
            raise ValueError("feasible performance result is not evaluated")
        if energy.evaluation_status != EVALUATED_ENERGY_STATUS:
            raise ValueError("feasible energy result is not evaluated")
        if power.evaluation_status != EVALUATED_POWER_STATUS:
            raise ValueError("feasible power result is not evaluated")
        if thermal is None:
            raise ValueError("feasible E7 row requires committed E6 thermal result")
        if energy.memory_dynamic_energy_j_per_token is None:
            raise ValueError("evaluated energy result is missing dynamic energy")
        _same("energy/power dynamic energy",
              energy.memory_dynamic_energy_j_per_token,
              power.memory_dynamic_energy_j_per_token)
        if performance.aggregate_tokens_per_second is None:
            raise ValueError("evaluated performance is missing throughput")
        _same("performance/power aggregate throughput",
              performance.aggregate_tokens_per_second,
              power.aggregate_tokens_per_second)
        if power.package_workload_total_W is None:
            raise ValueError("evaluated power is missing package total")
        _same("power/thermal expected package power",
              power.package_workload_total_W,
              thermal.expected_package_power_W,
              abs_tol=POWER_CLOSURE_ABS_TOL_W)
        _same("power/thermal mapped package power",
              power.package_workload_total_W,
              thermal.mapped_package_power_W,
              abs_tol=POWER_CLOSURE_ABS_TOL_W)
        if thermal.power_closure_absolute_error_W > POWER_CLOSURE_ABS_TOL_W:
            raise ValueError("thermal source power closure failed")
        if not thermal.converged:
            raise ValueError("thermal result is not converged")
        if thermal.scenario_status != SCENARIO_STATUS or (
                power.scenario_status != SCENARIO_STATUS) or (
                energy.scenario_status != SCENARIO_STATUS):
            raise ValueError("conditional scenario status mismatch")
        if thermal.memory_total_completeness_status != (
                power.memory_total_completeness_status):
            raise ValueError("power/thermal completeness status mismatch")
        if thermal.write_spatial_distribution_status != WRITE_SPATIAL_STATUS:
            raise ValueError("write spatial provenance mismatch")
        thermal_status = "CONVERGED_FROZEN_GPU_PCG"
        closure_status = "PASS_ABSOLUTE_ERROR_LE_1E-9_W"

    if architecture == "orthogonal_m3d_igzo":
        if capacity.capacity_feasible and (
                power.memory_total_completeness_status
                != M3D_COMPLETENESS_STATUS):
            raise ValueError("M3D conditional lower-bound status was lost")
        m3d_status = "CONDITIONAL_LOWER_BOUND"
    else:
        m3d_status = "NOT_APPLICABLE"

    return ConditionalLLMDecodeE2ERow(
        architecture=architecture,
        architecture_display_name=display_name,
        rho=energy.rho,
        workload_identifier=workload_identifier,
        batch_size=workload_input.batch_size,
        context_length=workload_input.context_length,
        required_capacity_bytes=capacity.required_capacity_bytes,
        usable_capacity_bytes=float(capacity.usable_capacity_bytes),
        capacity_margin_bytes=capacity.capacity_margin_bytes,
        capacity_utilization=capacity.capacity_utilization,
        capacity_feasible=capacity.capacity_feasible,
        capacity_status=capacity.capacity_scope_status,
        aggregate_tokens_per_second=performance.aggregate_tokens_per_second,
        per_sequence_tokens_per_second=performance.per_sequence_tokens_per_second,
        memory_time_per_token_s=performance.memory_time_per_token_equivalent_s,
        compute_time_per_token_s=performance.compute_time_per_token_equivalent_s,
        bottleneck=performance.bottleneck,
        performance_status=performance.performance_status,
        bandwidth_status=performance.bandwidth_status,
        read_bytes_per_token=workload.read_bytes_per_token,
        write_bytes_per_token=workload.write_bytes_per_token,
        flops_per_token=workload.flops_per_token,
        read_energy_pj_per_bit=energy.read_energy_pj_per_bit,
        write_energy_pj_per_bit=energy.write_energy_pj_per_bit,
        memory_dynamic_energy_j_per_token=(
            energy.memory_dynamic_energy_j_per_token),
        energy_status=energy.evaluation_status,
        write_energy_status=energy.write_energy_status,
        rho_zero_status=("MATHEMATICAL_WRITE_ENERGY_LOWER_BOUND"
                         if energy.rho == 0.0 else None),
        memory_dynamic_power_W=power.memory_dynamic_access_power_W,
        memory_total_power_W=power.memory_workload_total_W,
        gpu_power_W=power.fixed_gpu_power_W,
        package_power_W=power.package_workload_total_W,
        power_status=power.evaluation_status,
        memory_total_completeness_status=(
            power.memory_total_completeness_status),
        memory_Tmax_degC=(thermal.memory_Tmax_degC if thermal else None),
        gpu_Tmax_degC=(thermal.gpu_Tmax_degC if thermal else None),
        package_Tmax_degC=(thermal.package_Tmax_degC if thermal else None),
        thermal_converged=(thermal.converged if thermal else None),
        thermal_status=thermal_status,
        power_closure_absolute_error_W=(
            thermal.power_closure_absolute_error_W if thermal else None),
        power_closure_status=closure_status,
        write_spatial_distribution_status=(
            thermal.write_spatial_distribution_status if thermal else None),
        bandwidth_capability_status="NOT_VALIDATED",
        write_energy_model_status="NOT_VALIDATED",
        gpu_energy_model_status="NOT_AVAILABLE",
        system_j_token_status="NOT_AVAILABLE",
        m3d_logic_background_status=m3d_status,
        scenario_status=SCENARIO_STATUS,
    )


def validate_conditional_llm_decode_e2e_table(
    rows: Sequence[ConditionalLLMDecodeE2ERow],
) -> tuple[ConditionalLLMDecodeE2ERow, ...]:
    """Require exactly the frozen 3x4 table and its semantic invariants."""
    frozen = tuple(rows)
    expected_keys = [(architecture, rho) for architecture in ARCHITECTURE_ORDER
                     for rho in FROZEN_RHOS]
    if [(row.architecture, row.rho) for row in frozen] != expected_keys:
        raise ValueError(
            "E7 rows must be the ordered frozen 3x4 architecture/rho set")
    return validate_conditional_llm_decode_e2e_rows(
        frozen,
        expected_architecture_ids=ARCHITECTURE_ORDER,
        expected_rhos=FROZEN_RHOS,
    )


def validate_conditional_llm_decode_e2e_rows(
    rows: Sequence[ConditionalLLMDecodeE2ERow],
    *,
    expected_architecture_ids: Sequence[str],
    expected_rhos: Sequence[float],
) -> tuple[ConditionalLLMDecodeE2ERow, ...]:
    """Validate a configured comparison without hard-coding its identities."""
    result = tuple(rows)
    architectures = tuple(expected_architecture_ids)
    rhos = tuple(float(value) for value in expected_rhos)
    if not architectures or not rhos:
        raise ValueError("expected architectures and rho values must be non-empty")
    if len(set(architectures)) != len(architectures):
        raise ValueError("expected architecture identities must be unique")
    if len(set(rhos)) != len(rhos):
        raise ValueError("expected rho values must be unique")
    expected_keys = [(architecture, rho) for architecture in architectures
                     for rho in rhos]
    keys = [(row.architecture, row.rho) for row in result]
    if keys != expected_keys:
        raise ValueError("E2E rows must match the ordered configured architecture/rho set")
    if Counter(row.architecture for row in result) != Counter({
            architecture: len(rhos) for architecture in architectures}):
        raise ValueError("E2E row count per architecture is incorrect")

    shared_fields = (
        "workload_identifier", "batch_size", "context_length",
        "required_capacity_bytes", "read_bytes_per_token",
        "write_bytes_per_token", "flops_per_token",
        "aggregate_tokens_per_second",
    )
    for field in shared_fields:
        values = [getattr(row, field) for row in result]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"{field} changed across architecture/rho comparison")

    for architecture in architectures:
        group = [row for row in result if row.architecture == architecture]
        for field in ("write_energy_pj_per_bit",
                      "memory_dynamic_energy_j_per_token",
                      "memory_dynamic_power_W", "memory_total_power_W",
                      "package_power_W", "package_Tmax_degC"):
            values = [getattr(row, field) for row in group]
            if any(value is None for value in values):
                raise ValueError(f"{field} is unavailable in frozen feasible table")
            if any(float(values[index]) > float(values[index + 1])
                   for index in range(len(values) - 1)):
                raise ValueError(f"{field} is not monotonic for {architecture}")
        for field in ("workload_identifier", "batch_size", "context_length",
                      "required_capacity_bytes", "capacity_feasible",
                      "read_bytes_per_token", "write_bytes_per_token",
                      "flops_per_token", "aggregate_tokens_per_second"):
            values = [getattr(row, field) for row in group]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"{field} changed across rho for {architecture}")
    return result
