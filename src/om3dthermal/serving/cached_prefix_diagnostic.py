"""Independent cached-prefix resident-wave workload diagnostic."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.platform import (
    load_platform_spec_file, resolve_gpu_prefill_compute_energy_calibration,
)
from om3dthermal.workload import (
    CachedPrefixIncrementalPrefillMetrics, DenseLLMModelSpec,
    evaluate_cached_prefix_incremental_prefill, evaluate_gpu_prefill_roofline,
    evaluate_llm_decode,
)

from .mixed_phase_e2e import resolve_conventional_hbm_backend
from .nmp_decode import resolve_m3d_architecture_backend
from .persistent_horizon import _gpu_decode_sum, _m3d_gpu_bandwidth
from .workspace import (
    WorkspaceExecutionConfig, evaluate_decode_workspace,
    evaluate_prefill_workspace,
)


DiagnosticSystem = Literal["CONVENTIONAL_HBM_GPU", "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY"]


class CachedPrefixDiagnosticWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: Literal["llama31_8b"] = "llama31_8b"
    cached_history_tokens: int = Field(default=128000, gt=0)
    new_prompt_tokens: int = Field(default=1024, gt=0)
    generation_tokens: int = Field(default=2048, gt=0)
    final_context_tokens: Literal[131072] = 131072
    total_requests: Literal[128] = 128
    arrival_policy: Literal["ALL_REQUESTS_AT_T0"] = "ALL_REQUESTS_AT_T0"
    cached_kv_backing_store: Literal["HOST"] = "HOST"

    @model_validator(mode="after")
    def _context_closure(self) -> "CachedPrefixDiagnosticWorkload":
        if (self.cached_history_tokens+self.new_prompt_tokens
                + self.generation_tokens != self.final_context_tokens):
            raise ValueError("cached history + new prompt + Decode must equal 128K")
        return self

    @property
    def decode_start_context_tokens(self) -> int:
        return self.cached_history_tokens+self.new_prompt_tokens


class CapacityPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    system: DiagnosticSystem
    safe_active_batch: int
    capacity_bytes: int
    peak_runtime_bytes: int
    capacity_margin_bytes: int
    limiting_phase: Literal["INCREMENTAL_PREFILL", "DECODE_FINAL"]


class PhasePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    system: DiagnosticSystem
    active_batch: int
    cached_kv_admission_bytes: int
    cached_kv_admission_time_s: float
    incremental_prefill_time_s: float
    first_decode_step_time_s: float
    decode_time_s: float
    execution_time_s: float
    serving_wave_time_s: float
    output_tokens_per_wave: int
    execution_wave_throughput_tokens_per_s: float
    prefill_fraction: float
    decode_fraction: float


class WavePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    system: DiagnosticSystem
    wave_index: int
    request_start: int
    request_end: int
    batch_size: int
    wave_start_s: float
    admission_start_s: float
    admission_end_s: float
    incremental_prefill_start_s: float
    incremental_prefill_end_s: float
    decode_start_s: float
    decode_end_s: float
    wave_end_s: float


class RequestLatencyPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    system: DiagnosticSystem
    request_id: int
    wave_index: int
    queue_delay_s: float
    cached_kv_admission_time_s: float
    ttft_s: float
    completion_latency_s: float


class SystemServingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    system: DiagnosticSystem
    safe_active_batch: int
    num_waves: int
    wave_sizes: tuple[int, ...]
    representative_execution_wave_time_s: float
    representative_serving_wave_time_s: float
    batch_makespan_s: float
    aggregate_output_tokens_per_s: float
    mean_queue_delay_s: float
    p50_queue_delay_s: float
    p95_queue_delay_s: float
    max_queue_delay_s: float
    mean_ttft_s: float
    p50_ttft_s: float
    p95_ttft_s: float
    max_ttft_s: float
    mean_completion_latency_s: float
    p50_completion_latency_s: float
    p95_completion_latency_s: float
    max_completion_latency_s: float


class CachedPrefixDiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    workload: CachedPrefixDiagnosticWorkload
    incremental_prefill_metrics_b1: CachedPrefixIncrementalPrefillMetrics
    capacities: tuple[CapacityPoint, ...]
    phases: tuple[PhasePoint, ...]
    waves: tuple[WavePoint, ...]
    requests: tuple[RequestLatencyPoint, ...]
    summaries: tuple[SystemServingSummary, ...]
    workload_phase_gate: Literal["PASS", "FAIL"]
    resident_concurrency_gain: float
    p95_ttft_gain: float
    p95_completion_gain: float
    wave_count_reduction: float
    admission_semantics: str
    nmp_enabled: Literal[False] = False
    recurring_host_offload_enabled: Literal[False] = False
    thermal: None = None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction*len(ordered))-1)]


def _prefill_time(
    root: Path, model: DenseLLMModelSpec,
    metrics: CachedPrefixIncrementalPrefillMetrics, bandwidth: float,
) -> float:
    platform = load_platform_spec_file(
        root/"configs/platform/gpu_package_h200_reference.yaml")
    compute = platform.gpu_compute_power
    prefill = platform.gpu_prefill_compute
    if compute is None or prefill is None:
        raise ValueError("canonical GPU Prefill platform data is incomplete")
    calibration = resolve_gpu_prefill_compute_energy_calibration(compute, prefill)
    roofline = evaluate_gpu_prefill_roofline(
        metrics,
        peak_compute_flops_per_s=compute.peak_compute_BF16_dense_flops_per_s,
        large_gemm_effective_flops_per_s=prefill.large_gemm_effective_tflops*1e12,
        causal_attention_effective_flops_per_s=(
            prefill.causal_attention_effective_tflops*1e12),
        sustained_memory_bandwidth_bytes_per_s=bandwidth,
        static_power_W=compute.static_power_W,
        peak_reference_dynamic_J_per_FLOP_min=(
            calibration.peak_reference_dynamic_J_per_FLOP_min),
        peak_reference_dynamic_J_per_FLOP_max=(
            calibration.peak_reference_dynamic_J_per_FLOP_max),
        nominal_gemm_dynamic_J_per_FLOP_min=(
            calibration.nominal_gemm_dynamic_J_per_FLOP_min),
        nominal_gemm_dynamic_J_per_FLOP_max=(
            calibration.nominal_gemm_dynamic_J_per_FLOP_max),
        nominal_attention_dynamic_J_per_FLOP_min=(
            calibration.nominal_attention_dynamic_J_per_FLOP_min),
        nominal_attention_dynamic_J_per_FLOP_max=(
            calibration.nominal_attention_dynamic_J_per_FLOP_max),
        compute_bound_total_power_W_min=calibration.compute_bound_total_power_W_min,
        compute_bound_total_power_W_max=calibration.compute_bound_total_power_W_max,
    )
    return roofline.nominal_prefill_latency_s


def evaluate_cached_prefix_diagnostic(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    workload: CachedPrefixDiagnosticWorkload,
    workspace_config: WorkspaceExecutionConfig,
) -> CachedPrefixDiagnosticResult:
    """Evaluate the two capacity-only resident-wave systems."""
    if model.model_id != workload.model_id:
        raise ValueError("model and workload mismatch")
    root = Path(project_root).resolve()
    hbm = resolve_conventional_hbm_backend(root)
    m3d = resolve_m3d_architecture_backend(root)
    platform = load_platform_spec_file(
        root/"configs/platform/gpu_package_h200_reference.yaml")
    compute = platform.gpu_compute_power
    host = hbm.host_offload
    if compute is None or host.effective_bandwidth_bytes_per_second is None:
        raise ValueError("canonical platform data is incomplete")
    one = evaluate_llm_decode(model.decode_input(
        batch_size=1, context_length=workload.cached_history_tokens))
    weight = int(one.weight_footprint_bytes+one.runtime_fixed_bytes)
    kv_cache = int(one.kv_bytes_per_request)
    kv_token = int(one.kv_write_bytes_per_token)
    kv_decode_start = kv_cache+workload.new_prompt_tokens*kv_token
    kv_final = kv_decode_start+workload.generation_tokens*kv_token
    if kv_final != kv_token*workload.final_context_tokens:
        raise ValueError("final KV bytes do not close at exactly 128K")

    systems: tuple[DiagnosticSystem, ...] = (
        "CONVENTIONAL_HBM_GPU", "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY")
    capacity_points: list[CapacityPoint] = []
    safe_by_system: dict[DiagnosticSystem, int] = {}
    bandwidth_by_system = {
        "CONVENTIONAL_HBM_GPU": hbm.sustained_bandwidth_bytes_per_s,
        "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY": _m3d_gpu_bandwidth(root),
    }
    capacity_by_system = {
        "CONVENTIONAL_HBM_GPU": int(hbm.capacity_bytes),
        "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY": int(m3d.layout.total_capacity_bytes),
    }

    def runtime(system: DiagnosticSystem, batch: int) -> tuple[int, int, int]:
        page = (None if system == "CONVENTIONAL_HBM_GPU"
                else m3d.layout.slot_capacity_bytes)
        physical = (lambda value: value) if page is None else (
            lambda value: math.ceil(value/page)*page)
        prefill_ws = evaluate_prefill_workspace(
            model, batch_size=batch,
            context_length=workload.new_prompt_tokens,
            config=workspace_config).peak_bytes
        decode_ws = evaluate_decode_workspace(
            model, batch_size=batch,
            context_length=workload.final_context_tokens-1,
            config=workspace_config).peak_bytes
        prefill_peak = (
            physical(weight)+batch*physical(kv_decode_start)+physical(prefill_ws))
        decode_peak = (
            physical(weight)+batch*physical(kv_final)+physical(decode_ws))
        return max(prefill_peak, decode_peak), prefill_peak, decode_peak

    for system in systems:
        capacity = capacity_by_system[system]
        safe = 0
        safe_runtime = safe_prefill = safe_decode = 0
        for batch in range(1, workload.total_requests+1):
            peak, prefill_peak, decode_peak = runtime(system, batch)
            if peak > capacity:
                break
            safe = batch
            safe_runtime, safe_prefill, safe_decode = peak, prefill_peak, decode_peak
        if safe < 1:
            raise RuntimeError(f"{system} cannot admit one request")
        safe_by_system[system] = safe
        capacity_points.append(CapacityPoint(
            system=system, safe_active_batch=safe, capacity_bytes=capacity,
            peak_runtime_bytes=safe_runtime,
            capacity_margin_bytes=capacity-safe_runtime,
            limiting_phase=("DECODE_FINAL" if safe_decode >= safe_prefill
                            else "INCREMENTAL_PREFILL"),
        ))

    phase_cache: dict[tuple[DiagnosticSystem, int], PhasePoint] = {}

    def phase(system: DiagnosticSystem, batch: int) -> PhasePoint:
        key = (system, batch)
        if key in phase_cache:
            return phase_cache[key]
        metrics = evaluate_cached_prefix_incremental_prefill(
            model.prefill_input(
                batch_size=batch, prompt_length=workload.new_prompt_tokens),
            cached_history_tokens=workload.cached_history_tokens)
        prefill_s = _prefill_time(
            root, model, metrics, bandwidth_by_system[system])
        decode_s, _ = _gpu_decode_sum(
            model, batch=batch, S=workload.decode_start_context_tokens,
            G=workload.generation_tokens,
            bandwidth=bandwidth_by_system[system],
            compute=compute.peak_compute_BF16_dense_flops_per_s)
        first_s, _ = _gpu_decode_sum(
            model, batch=batch, S=workload.decode_start_context_tokens,
            G=1, bandwidth=bandwidth_by_system[system],
            compute=compute.peak_compute_BF16_dense_flops_per_s)
        admission_bytes = int(metrics.cached_kv_bytes_before)
        admission_s = admission_bytes/host.effective_bandwidth_bytes_per_second
        execution = prefill_s+decode_s
        item = PhasePoint(
            system=system, active_batch=batch,
            cached_kv_admission_bytes=admission_bytes,
            cached_kv_admission_time_s=admission_s,
            incremental_prefill_time_s=prefill_s,
            first_decode_step_time_s=first_s,
            decode_time_s=decode_s,
            execution_time_s=execution,
            serving_wave_time_s=admission_s+execution,
            output_tokens_per_wave=batch*workload.generation_tokens,
            execution_wave_throughput_tokens_per_s=(
                batch*workload.generation_tokens/execution),
            prefill_fraction=prefill_s/execution,
            decode_fraction=decode_s/execution,
        )
        phase_cache[key] = item
        return item

    diagnostic_phases: list[PhasePoint] = []
    for system in systems:
        for batch in dict.fromkeys((1, safe_by_system[system])):
            diagnostic_phases.append(phase(system, batch))
    phase_gate = "PASS" if all(
        item.decode_fraction >= .80 for item in diagnostic_phases) else "FAIL"

    waves: list[WavePoint] = []
    requests: list[RequestLatencyPoint] = []
    summaries: list[SystemServingSummary] = []
    if phase_gate == "PASS":
        for system in systems:
            safe = safe_by_system[system]
            cursor = 0.0
            request_id = 1
            wave_index = 1
            wave_sizes: list[int] = []
            system_requests: list[RequestLatencyPoint] = []
            while request_id <= workload.total_requests:
                batch = min(safe, workload.total_requests-request_id+1)
                item = phase(system, batch)
                wave_sizes.append(batch)
                admission_end = cursor+item.cached_kv_admission_time_s
                prefill_end = admission_end+item.incremental_prefill_time_s
                decode_end = prefill_end+item.decode_time_s
                wave = WavePoint(
                    system=system, wave_index=wave_index,
                    request_start=request_id, request_end=request_id+batch-1,
                    batch_size=batch, wave_start_s=cursor,
                    admission_start_s=cursor, admission_end_s=admission_end,
                    incremental_prefill_start_s=admission_end,
                    incremental_prefill_end_s=prefill_end,
                    decode_start_s=prefill_end, decode_end_s=decode_end,
                    wave_end_s=decode_end)
                waves.append(wave)
                ttft = prefill_end+item.first_decode_step_time_s
                for current in range(request_id, request_id+batch):
                    point = RequestLatencyPoint(
                        system=system, request_id=current,
                        wave_index=wave_index, queue_delay_s=cursor,
                        cached_kv_admission_time_s=item.cached_kv_admission_time_s,
                        ttft_s=ttft, completion_latency_s=decode_end)
                    requests.append(point)
                    system_requests.append(point)
                cursor = decode_end
                request_id += batch
                wave_index += 1
            queue = [item.queue_delay_s for item in system_requests]
            ttft = [item.ttft_s for item in system_requests]
            completion = [item.completion_latency_s for item in system_requests]
            representative = phase(system, safe)
            summaries.append(SystemServingSummary(
                system=system, safe_active_batch=safe,
                num_waves=len(wave_sizes), wave_sizes=tuple(wave_sizes),
                representative_execution_wave_time_s=representative.execution_time_s,
                representative_serving_wave_time_s=representative.serving_wave_time_s,
                batch_makespan_s=cursor,
                aggregate_output_tokens_per_s=(
                    workload.total_requests*workload.generation_tokens/cursor),
                mean_queue_delay_s=sum(queue)/len(queue),
                p50_queue_delay_s=_percentile(queue, .50),
                p95_queue_delay_s=_percentile(queue, .95),
                max_queue_delay_s=max(queue),
                mean_ttft_s=sum(ttft)/len(ttft),
                p50_ttft_s=_percentile(ttft, .50),
                p95_ttft_s=_percentile(ttft, .95), max_ttft_s=max(ttft),
                mean_completion_latency_s=sum(completion)/len(completion),
                p50_completion_latency_s=_percentile(completion, .50),
                p95_completion_latency_s=_percentile(completion, .95),
                max_completion_latency_s=max(completion),
            ))

    hbm_summary = next((x for x in summaries if x.system == systems[0]), None)
    m3d_summary = next((x for x in summaries if x.system == systems[1]), None)
    return CachedPrefixDiagnosticResult(
        workload=workload,
        incremental_prefill_metrics_b1=evaluate_cached_prefix_incremental_prefill(
            model.prefill_input(
                batch_size=1, prompt_length=workload.new_prompt_tokens),
            cached_history_tokens=workload.cached_history_tokens),
        capacities=tuple(capacity_points), phases=tuple(diagnostic_phases),
        waves=tuple(waves), requests=tuple(requests), summaries=tuple(summaries),
        workload_phase_gate=phase_gate,
        resident_concurrency_gain=(
            safe_by_system[systems[1]]/safe_by_system[systems[0]]),
        p95_ttft_gain=(1.0 if hbm_summary is None else
                       hbm_summary.p95_ttft_s/m3d_summary.p95_ttft_s),
        p95_completion_gain=(1.0 if hbm_summary is None else
                             hbm_summary.p95_completion_latency_s
                             / m3d_summary.p95_completion_latency_s),
        wave_count_reduction=(1.0 if hbm_summary is None else
                              hbm_summary.num_waves/m3d_summary.num_waves),
        admission_semantics=(
            "MODELING_CHOICE_HOST_BACKING_STORE_TO_LOCAL_ONCE_PER_WAVE__"
            "CANONICAL_56_2_GBPS_LINK__NO_RECURRING_OFFLOAD"),
    )
