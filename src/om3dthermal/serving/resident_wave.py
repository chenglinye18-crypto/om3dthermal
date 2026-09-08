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
