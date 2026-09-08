"""Capacity-constrained Conventional HBM resident-wave sensitivity.

This scheduler-only model serializes fully HBM-resident Decode waves.  It
deliberately excludes one-time KV admission/swap and scheduler overhead, so
its throughput is an optimistic Conventional-HBM upper bound.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.platform import load_platform_spec_file
from om3dthermal.workload import (
    DenseLLMModelSpec,
    evaluate_llm_decode,
)
from om3dthermal.workload.dense_decode_ledger import (
    build_dense_decode_placement_units,
)

from .gpu import AnalyticalRooflineGPUModel
from .mixed_phase_e2e import (
    MixedPhaseServingCase,
    resolve_conventional_hbm_backend,
)
from .residency import ServingCapacitySource, evaluate_capacity_residency


RESIDENT_WAVE_SCHEDULER_POLICY = (
    "CAPACITY_CONSTRAINED_RUN_TO_COMPLETION_RESIDENT_WAVES")
RESIDENT_WAVE_MODEL = (
    "CAPACITY_CONSTRAINED_RUN_TO_COMPLETION_OPTIMISTIC_BOUND")
RESIDENT_WAVE_MODEL_STATUS = (
    "RESIDENT_WAVE_COMPUTE_ONLY_OPTIMISTIC_UPPER_BOUND")
OUTPUT_LENGTH_STATUS = (
    "EQUAL_OUTPUT_LENGTH__CANCELS_FROM_AGGREGATE_THROUGHPUT")
SWAP_ADMISSION_STATUS = "NOT_INCLUDED_OPTIMISTIC_UPPER_BOUND"
ADMISSION_POLICY = "SERIAL_HOST_TO_HBM_ADMISSION_BETWEEN_WAVES"
OUTPUT_LENGTH_MODEL_STATUS = (
    "EQUAL_GENERATED_LENGTH_PER_REQUEST_SENSITIVITY")
ADMISSION_AWARE_MODEL_STATUS = (
    "ADMISSION_AWARE_RESIDENT_WAVE_SENSITIVITY")


class ResidentWaveDecodeResult(BaseModel):
    """Normalized equal-output-length result; not an ordinary TPOT."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    context_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    prefill_requests: int = Field(gt=0)
    requested_decode_requests: int = Field(gt=0)
    resident_batch_limit: int = Field(gt=0)
    wave_count: int = Field(gt=0)
    wave_sizes: tuple[int, ...]
    wave_step_times_ms: tuple[float, ...]
    wave_matrix_weight_read_bytes_per_step: tuple[float, ...]
    normalized_wave_decode_makespan_ms_per_output_token_depth: float = Field(gt=0.0)
    aggregate_decode_tokens_per_s: float = Field(gt=0.0)
    host_read_bytes_per_decode_step: Literal[0] = 0
    host_write_bytes_per_decode_step: Literal[0] = 0
    total_nonresident_KV_GB: float = Field(ge=0.0)
    scheduler_policy: Literal[
        "CAPACITY_CONSTRAINED_RUN_TO_COMPLETION_RESIDENT_WAVES"]
    resident_wave_model: Literal[
        "CAPACITY_CONSTRAINED_RUN_TO_COMPLETION_OPTIMISTIC_BOUND"]
    evaluation_status: Literal[
        "RESIDENT_WAVE_COMPUTE_ONLY_OPTIMISTIC_UPPER_BOUND"]
    output_length_assumption_status: Literal[
        "EQUAL_OUTPUT_LENGTH__CANCELS_FROM_AGGREGATE_THROUGHPUT"]
    output_length_cancels_in_equal_length_wave_throughput: Literal["YES"] = "YES"
    swap_admission_overhead_status: Literal[
        "NOT_INCLUDED_OPTIMISTIC_UPPER_BOUND"]
    admission_time_status: Literal[
        "ADMISSION_TIME_NOT_INCLUDED_IN_WAVE_THROUGHPUT"] = (
            "ADMISSION_TIME_NOT_INCLUDED_IN_WAVE_THROUGHPUT")
    capacity_status: str
    weight_activity_model: Literal["dimension_derived_active_operators"]
    weight_reuse_model: Literal["tile_reuse"]
    kv_read_model: Literal["full_reread"]
    thermal: None = None
    evaluation_scope: Literal["DECODE_SERVICE_ONLY"] = "DECODE_SERVICE_ONLY"
    context_evolution_status: Literal["FIXED_CONTEXT_SNAPSHOT"] = (
        "FIXED_CONTEXT_SNAPSHOT")
    initial_state_requirement: Literal[
        "FIRST_WAVE_KV_ALREADY_LOCAL__LATER_WAVES_HOST_VALID"] = (
            "FIRST_WAVE_KV_ALREADY_LOCAL__LATER_WAVES_HOST_VALID")
    final_state_requirement: Literal[
        "NOT_MODELED_NORMALIZED_OUTPUT_TOKEN_DEPTH"] = (
            "NOT_MODELED_NORMALIZED_OUTPUT_TOKEN_DEPTH")

    @model_validator(mode="after")
    def _closures(self) -> "ResidentWaveDecodeResult":
        if len(self.wave_sizes) != self.wave_count:
            raise ValueError("wave count does not close")
        if len(self.wave_step_times_ms) != self.wave_count:
            raise ValueError("wave step-time count does not close")
        if len(self.wave_matrix_weight_read_bytes_per_step) != self.wave_count:
            raise ValueError("wave weight-read count does not close")
        if sum(self.wave_sizes) != self.requested_decode_requests:
            raise ValueError("wave sizes do not close to requested Decode requests")
        if any(size <= 0 or size > self.resident_batch_limit
               for size in self.wave_sizes):
            raise ValueError("wave size exceeds resident batch limit")
        makespan = sum(self.wave_step_times_ms)
        if not math.isclose(
            makespan,
            self.normalized_wave_decode_makespan_ms_per_output_token_depth,
            rel_tol=1e-12,
        ):
            raise ValueError("normalized wave makespan does not close")
        throughput = self.requested_decode_requests / (makespan * 1e-3)
        if not math.isclose(
            throughput, self.aggregate_decode_tokens_per_s, rel_tol=1e-12
        ):
            raise ValueError("resident-wave throughput does not close")
        return self


class ResidentWaveAdmissionAwareResult(BaseModel):
    """Resident waves with serialized host-to-HBM historical-KV admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    context_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    prefill_requests: int = Field(gt=0)
    requested_decode_requests: int = Field(gt=0)
    generated_output_tokens_per_request: int = Field(gt=0)
    resident_batch_limit: int = Field(gt=0)
    wave_count: int = Field(gt=0)
    wave_sizes: tuple[int, ...]
    wave_step_times_ms: tuple[float, ...]
    admission_bytes_per_wave: tuple[float, ...]
    admission_time_ms_per_wave: tuple[float, ...]
    total_admission_GB: float = Field(ge=0.0)
    total_admission_time_ms: float = Field(ge=0.0)
    compute_time_ms: float = Field(gt=0.0)
    total_completion_time_ms: float = Field(gt=0.0)
    total_generated_tokens: int = Field(gt=0)
    optimistic_resident_wave_tokens_per_s: float = Field(gt=0.0)
    admission_aware_tokens_per_s: float = Field(gt=0.0)
    throughput_retention_vs_optimistic: float = Field(gt=0.0, le=1.0)
    throughput_loss_vs_optimistic: float = Field(ge=0.0, lt=1.0)
    admission_impact_classification: Literal[
        "LOW_IMPACT", "MODERATE_IMPACT", "HIGH_IMPACT"]
    host_effective_bandwidth_bytes_per_s: float = Field(gt=0.0)
    scheduler_policy: Literal[
        "CAPACITY_CONSTRAINED_RUN_TO_COMPLETION_RESIDENT_WAVES"]
    admission_policy: Literal[
        "SERIAL_HOST_TO_HBM_ADMISSION_BETWEEN_WAVES"]
    output_length_model_status: Literal[
        "EQUAL_GENERATED_LENGTH_PER_REQUEST_SENSITIVITY"]
    resident_wave_model_status: Literal[
        "ADMISSION_AWARE_RESIDENT_WAVE_SENSITIVITY"]
    evaluation_scope: Literal["DECODE_SERVICE_ONLY"] = "DECODE_SERVICE_ONLY"
    context_evolution_status: Literal["FIXED_CONTEXT_SNAPSHOT"] = (
        "FIXED_CONTEXT_SNAPSHOT")
    first_wave_admission_status: Literal[
        "ZERO_ALREADY_RESIDENT"] = "ZERO_ALREADY_RESIDENT"
    scheduler_software_overhead_included: Literal["NO"] = "NO"
    thermal: None = None

    @model_validator(mode="after")
    def _closures(self) -> "ResidentWaveAdmissionAwareResult":
        if not (len(self.wave_sizes) == len(self.wave_step_times_ms)
                == len(self.admission_bytes_per_wave)
                == len(self.admission_time_ms_per_wave) == self.wave_count):
            raise ValueError("admission-aware wave vectors do not close")
        if self.admission_bytes_per_wave[0] != 0.0:
            raise ValueError("first wave admission must be zero")
        if self.admission_time_ms_per_wave[0] != 0.0:
            raise ValueError("first wave admission time must be zero")
        admission_bytes = sum(self.admission_bytes_per_wave)
        admission_ms = sum(self.admission_time_ms_per_wave)
        if not math.isclose(self.total_admission_GB, admission_bytes / 1e9,
                            rel_tol=1e-12):
            raise ValueError("total admission bytes do not close")
        if not math.isclose(self.total_admission_time_ms, admission_ms,
                            rel_tol=1e-12):
            raise ValueError("total admission time does not close")
        expected_compute = (self.generated_output_tokens_per_request
                            * sum(self.wave_step_times_ms))
        if not math.isclose(self.compute_time_ms, expected_compute,
                            rel_tol=1e-12):
            raise ValueError("generated-length compute time does not close")
        if not math.isclose(
            self.total_completion_time_ms,
            self.compute_time_ms + self.total_admission_time_ms,
            rel_tol=1e-12,
        ):
            raise ValueError("admission-aware completion time does not close")
        expected_tokens = (self.requested_decode_requests
                           * self.generated_output_tokens_per_request)
        if self.total_generated_tokens != expected_tokens:
            raise ValueError("total generated tokens do not close")
        expected_rate = expected_tokens / (self.total_completion_time_ms * 1e-3)
        if not math.isclose(self.admission_aware_tokens_per_s, expected_rate,
                            rel_tol=1e-12):
            raise ValueError("admission-aware throughput does not close")
        if self.admission_aware_tokens_per_s > (
                self.optimistic_resident_wave_tokens_per_s * (1.0 + 1e-12)):
            raise ValueError("admission-aware throughput exceeds optimistic bound")
        retention = (self.admission_aware_tokens_per_s
                     / self.optimistic_resident_wave_tokens_per_s)
        if not math.isclose(self.throughput_retention_vs_optimistic,
                            retention, rel_tol=1e-12):
            raise ValueError("throughput retention does not close")
        if not math.isclose(self.throughput_loss_vs_optimistic,
                            1.0 - retention, rel_tol=1e-12):
            raise ValueError("throughput loss does not close")
        return self


class ResidentWaveGrowingKVResult(BaseModel):
    """True G-step Decode horizon with S+j traffic and S+G reservation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    initial_context_length: int
    generated_decode_steps: int
    high_water_context_length: int
    requested_decode_requests: int
    resident_batch_limit_at_high_water: int
    wave_sizes: tuple[int, ...]
    step_context_lengths: tuple[int, ...]
    step_times_ms_by_wave: tuple[tuple[float, ...], ...]
    wave_compute_times_ms: tuple[float, ...]
    admission_bytes_per_wave: tuple[float, ...]
    total_admission_GB: float
    total_admission_time_ms: float
    total_compute_time_ms: float
    total_completion_time_ms: float
    total_generated_tokens: int
    aggregate_decode_tokens_per_s: float
    evaluation_scope: Literal["DECODE_SERVICE_ONLY"] = "DECODE_SERVICE_ONLY"
    context_evolution_status: Literal["GROWING_KV__STEP_CONTEXT_S_PLUS_J"] = (
        "GROWING_KV__STEP_CONTEXT_S_PLUS_J")
    capacity_reservation_status: Literal[
        "HIGH_WATER_S_PLUS_G_MODEL_DECLARED_FOOTPRINT"] = (
            "HIGH_WATER_S_PLUS_G_MODEL_DECLARED_FOOTPRINT")
    first_generated_token_status: Literal[
        "STEP_J0_READS_S_AND_APPENDS_TOKEN_S"] = (
            "STEP_J0_READS_S_AND_APPENDS_TOKEN_S")
    scheduler_policy: Literal[
        "CAPACITY_CONSTRAINED_RUN_TO_COMPLETION_RESIDENT_WAVES"]
    admission_policy: Literal[
        "SERIAL_HOST_TO_HBM_ADMISSION_BETWEEN_WAVES"]
    workspace_capacity_status: Literal[
        "WORKSPACE_UNRESOLVED__NOT_ASSUMED_ZERO"] = (
            "WORKSPACE_UNRESOLVED__NOT_ASSUMED_ZERO")
    thermal: None = None

    @model_validator(mode="after")
    def _growing_closures(self) -> "ResidentWaveGrowingKVResult":
        if self.high_water_context_length != (
                self.initial_context_length + self.generated_decode_steps):
            raise ValueError("S+G high-water context does not close")
        if self.step_context_lengths != tuple(
                self.initial_context_length+j
                for j in range(self.generated_decode_steps)):
            raise ValueError("step context sequence does not close")
        if sum(self.wave_sizes) != self.requested_decode_requests:
            raise ValueError("growing-KV wave sizes do not close")
        if any(size > self.resident_batch_limit_at_high_water
               for size in self.wave_sizes):
            raise ValueError("wave exceeds high-water resident limit")
        if len(self.step_times_ms_by_wave) != len(self.wave_sizes):
            raise ValueError("wave timing vector count does not close")
        if any(len(row) != self.generated_decode_steps
               for row in self.step_times_ms_by_wave):
            raise ValueError("each wave must execute exactly G Decode steps")
        expected_wave = tuple(sum(row) for row in self.step_times_ms_by_wave)
        if any(not math.isclose(a,b,rel_tol=1e-12)
               for a,b in zip(self.wave_compute_times_ms,expected_wave)):
            raise ValueError("per-wave growing compute time does not close")
        if not math.isclose(self.total_compute_time_ms,sum(expected_wave),rel_tol=1e-12):
            raise ValueError("total growing compute time does not close")
        if not math.isclose(self.total_completion_time_ms,
                            self.total_compute_time_ms+self.total_admission_time_ms,
                            rel_tol=1e-12):
            raise ValueError("growing horizon completion does not close")
        tokens=self.requested_decode_requests*self.generated_decode_steps
        if self.total_generated_tokens != tokens:
            raise ValueError("growing horizon token count does not close")
        if not math.isclose(self.aggregate_decode_tokens_per_s,
                            tokens/(self.total_completion_time_ms*1e-3),rel_tol=1e-12):
            raise ValueError("growing horizon throughput does not close")
        return self


def _matrix_weight_read_bytes_per_step(model_input) -> float:
    """Read the existing dense operator ledger without recreating formulas."""
    return sum(
        unit.active_weight_read_bytes
        for unit in build_dense_decode_placement_units(model_input)
        if unit.operator_type != "TOKEN_EMBED_LOOKUP"
    )


def evaluate_conventional_hbm_resident_wave_decode(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    case: MixedPhaseServingCase,
) -> ResidentWaveDecodeResult:
    """Evaluate serial, fully resident Decode waves with no host-step traffic."""
    if model.model_id != case.model_id:
        raise ValueError("model and mixed case model_id must match")
    root = Path(project_root).resolve()
    backend = resolve_conventional_hbm_backend(root)
    platform = load_platform_spec_file(
        root / "configs/platform/gpu_package_h200_reference.yaml")
    gpu_decode = platform.gpu_decode_power
    gpu_compute = platform.gpu_compute_power
    if gpu_decode is None or gpu_compute is None:
        raise ValueError("canonical H200 GPU Decode data is incomplete")

    requested_input = model.decode_input(
        batch_size=case.decode_requests, context_length=case.context_length)
    requested_metrics = evaluate_llm_decode(requested_input)
    capacity_source = ServingCapacitySource(
        architecture=backend.architecture,
        usable_capacity_bytes=backend.capacity_bytes,
        capacity_source_status=backend.capacity_source_status,
        provenance_status="CANONICAL_CONVENTIONAL_HBM_CASE",
    )
    residency = evaluate_capacity_residency(
        requested_metrics, capacity_source,
        requested_requests=case.decode_requests)
    resident_limit = residency.max_resident_requests
    if resident_limit is None:
        resident_limit = case.decode_requests
    if resident_limit <= 0:
        raise ValueError("Conventional HBM cannot retain model fixed footprint")

    remaining = case.decode_requests
    wave_sizes: list[int] = []
    while remaining:
        wave = min(resident_limit, remaining)
        wave_sizes.append(wave)
        remaining -= wave

    gpu_model = AnalyticalRooflineGPUModel(
        matched_payload_bandwidth_bits_per_second=(
            8.0 * backend.sustained_bandwidth_bytes_per_s),
        effective_compute_flops_per_second=(
            gpu_compute.peak_compute_BF16_dense_flops_per_s),
    )
    step_times: list[float] = []
    matrix_reads: list[float] = []
    for wave_size in wave_sizes:
        wave_input = model.decode_input(
            batch_size=wave_size, context_length=case.context_length)
        step = gpu_model.evaluate(wave_input, batch_size=wave_size)
        step_times.append(step.decode_step_time_ms)
        matrix_reads.append(_matrix_weight_read_bytes_per_step(wave_input))

    makespan_ms = sum(step_times)
    initially_nonresident = max(0, case.decode_requests - resident_limit)
    return ResidentWaveDecodeResult(
        model_id=model.model_id,
        context_length=case.context_length,
        batch_size=case.batch_size,
        prefill_requests=case.prefill_requests,
        requested_decode_requests=case.decode_requests,
        resident_batch_limit=resident_limit,
        wave_count=len(wave_sizes),
        wave_sizes=tuple(wave_sizes),
        wave_step_times_ms=tuple(step_times),
        wave_matrix_weight_read_bytes_per_step=tuple(matrix_reads),
        normalized_wave_decode_makespan_ms_per_output_token_depth=makespan_ms,
        aggregate_decode_tokens_per_s=(
            case.decode_requests / (makespan_ms * 1e-3)),
        total_nonresident_KV_GB=(
            initially_nonresident * requested_metrics.kv_bytes_per_request / 1e9),
        scheduler_policy=RESIDENT_WAVE_SCHEDULER_POLICY,
        resident_wave_model=RESIDENT_WAVE_MODEL,
        evaluation_status=RESIDENT_WAVE_MODEL_STATUS,
        output_length_assumption_status=OUTPUT_LENGTH_STATUS,
        swap_admission_overhead_status=SWAP_ADMISSION_STATUS,
        capacity_status=residency.capacity_status,
        weight_activity_model=requested_metrics.weight_activity_model,
        weight_reuse_model=requested_metrics.weight_reuse_model,
        kv_read_model=requested_metrics.kv_read_model,
    )


def evaluate_conventional_hbm_resident_wave_admission(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    case: MixedPhaseServingCase, generated_output_tokens_per_request: int,
) -> ResidentWaveAdmissionAwareResult:
    """Add serialized historical-KV admission to the existing wave model."""
    if isinstance(generated_output_tokens_per_request, bool) or not isinstance(
            generated_output_tokens_per_request, int):
        raise TypeError("generated_output_tokens_per_request must be an int")
    if generated_output_tokens_per_request <= 0:
        raise ValueError("generated_output_tokens_per_request must be positive")
    root = Path(project_root).resolve()
    optimistic = evaluate_conventional_hbm_resident_wave_decode(
        project_root=root, model=model, case=case)
    backend = resolve_conventional_hbm_backend(root)
    host_bandwidth = backend.host_offload.effective_bandwidth_bytes_per_second
    if host_bandwidth is None:
        raise ValueError("canonical host effective bandwidth is unresolved")
    metrics = evaluate_llm_decode(model.decode_input(
        batch_size=case.decode_requests, context_length=case.context_length))
    admission_bytes = tuple(
        0.0 if index == 0 else wave_size * metrics.kv_bytes_per_request
        for index, wave_size in enumerate(optimistic.wave_sizes)
    )
    admission_times_ms = tuple(
        value / host_bandwidth * 1e3 for value in admission_bytes)
    total_admission_bytes = sum(admission_bytes)
    total_admission_ms = sum(admission_times_ms)
    compute_ms = (generated_output_tokens_per_request
                  * sum(optimistic.wave_step_times_ms))
    total_ms = compute_ms + total_admission_ms
    total_tokens = (case.decode_requests
                    * generated_output_tokens_per_request)
    realistic_rate = total_tokens / (total_ms * 1e-3)
    retention = realistic_rate / optimistic.aggregate_decode_tokens_per_s
    loss = 1.0 - retention
    impact = (
        "LOW_IMPACT" if loss < 0.10
        else "MODERATE_IMPACT" if loss < 0.30
        else "HIGH_IMPACT"
    )
    return ResidentWaveAdmissionAwareResult(
        model_id=model.model_id,
        context_length=case.context_length,
        batch_size=case.batch_size,
        prefill_requests=case.prefill_requests,
        requested_decode_requests=case.decode_requests,
        generated_output_tokens_per_request=(
            generated_output_tokens_per_request),
        resident_batch_limit=optimistic.resident_batch_limit,
        wave_count=optimistic.wave_count,
        wave_sizes=optimistic.wave_sizes,
        wave_step_times_ms=optimistic.wave_step_times_ms,
        admission_bytes_per_wave=admission_bytes,
        admission_time_ms_per_wave=admission_times_ms,
        total_admission_GB=total_admission_bytes / 1e9,
        total_admission_time_ms=total_admission_ms,
        compute_time_ms=compute_ms,
        total_completion_time_ms=total_ms,
        total_generated_tokens=total_tokens,
        optimistic_resident_wave_tokens_per_s=(
            optimistic.aggregate_decode_tokens_per_s),
        admission_aware_tokens_per_s=realistic_rate,
        throughput_retention_vs_optimistic=retention,
        throughput_loss_vs_optimistic=loss,
        admission_impact_classification=impact,
        host_effective_bandwidth_bytes_per_s=host_bandwidth,
        scheduler_policy=RESIDENT_WAVE_SCHEDULER_POLICY,
        admission_policy=ADMISSION_POLICY,
        output_length_model_status=OUTPUT_LENGTH_MODEL_STATUS,
        resident_wave_model_status=ADMISSION_AWARE_MODEL_STATUS,
    )


def evaluate_conventional_hbm_resident_wave_growing_kv(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    case: MixedPhaseServingCase, generated_decode_steps: int,
) -> ResidentWaveGrowingKVResult:
    """Evaluate real S..S+G-1 Decode steps with high-water capacity gating."""
    if isinstance(generated_decode_steps,bool) or not isinstance(generated_decode_steps,int):
        raise TypeError("generated_decode_steps must be an int")
    if generated_decode_steps<=0:
        raise ValueError("generated_decode_steps must be positive")
    root=Path(project_root).resolve(); backend=resolve_conventional_hbm_backend(root)
    platform=load_platform_spec_file(
        root/"configs/platform/gpu_package_h200_reference.yaml")
    gpu_compute=platform.gpu_compute_power
    if gpu_compute is None:
        raise ValueError("canonical H200 GPU compute is unresolved")
    high_water=case.context_length+generated_decode_steps
    high_metrics=evaluate_llm_decode(model.decode_input(
        batch_size=case.decode_requests,context_length=high_water))
    capacity=ServingCapacitySource(
        architecture=backend.architecture,usable_capacity_bytes=backend.capacity_bytes,
        capacity_source_status=backend.capacity_source_status,
        provenance_status="CANONICAL_CONVENTIONAL_HBM_CASE")
    residency=evaluate_capacity_residency(
        high_metrics,capacity,requested_requests=case.decode_requests)
    limit=residency.max_resident_requests
    if limit is None: limit=case.decode_requests
    if limit<=0: raise ValueError("high-water model fixed footprint does not fit HBM")
    remaining=case.decode_requests; waves=[]
    while remaining:
        wave=min(limit,remaining); waves.append(wave); remaining-=wave
    gpu_model=AnalyticalRooflineGPUModel(
        matched_payload_bandwidth_bits_per_second=8*backend.sustained_bandwidth_bytes_per_s,
        effective_compute_flops_per_second=gpu_compute.peak_compute_BF16_dense_flops_per_s)
    contexts=tuple(case.context_length+j for j in range(generated_decode_steps))
    timings=[]
    for wave in waves:
        timings.append(tuple(gpu_model.evaluate(
            model.decode_input(batch_size=wave,context_length=context),batch_size=wave
        ).decode_step_time_ms for context in contexts))
    start_metrics=evaluate_llm_decode(model.decode_input(
        batch_size=case.decode_requests,context_length=case.context_length))
    admission=tuple(0.0 if index==0 else wave*start_metrics.kv_bytes_per_request
                    for index,wave in enumerate(waves))
    host_bw=backend.host_offload.effective_bandwidth_bytes_per_second
    if host_bw is None: raise ValueError("canonical host bandwidth is unresolved")
    admission_ms=sum(admission)/host_bw*1e3
    compute_ms=sum(sum(row) for row in timings); total_ms=compute_ms+admission_ms
    tokens=case.decode_requests*generated_decode_steps
    return ResidentWaveGrowingKVResult(
        model_id=model.model_id,initial_context_length=case.context_length,
        generated_decode_steps=generated_decode_steps,
        high_water_context_length=high_water,
        requested_decode_requests=case.decode_requests,
        resident_batch_limit_at_high_water=limit,wave_sizes=tuple(waves),
        step_context_lengths=contexts,step_times_ms_by_wave=tuple(timings),
        wave_compute_times_ms=tuple(sum(row) for row in timings),
        admission_bytes_per_wave=admission,total_admission_GB=sum(admission)/1e9,
        total_admission_time_ms=admission_ms,total_compute_time_ms=compute_ms,
        total_completion_time_ms=total_ms,total_generated_tokens=tokens,
        aggregate_decode_tokens_per_s=tokens/(total_ms*1e-3),
        scheduler_policy=RESIDENT_WAVE_SCHEDULER_POLICY,
        admission_policy=ADMISSION_POLICY)
