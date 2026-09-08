"""Auditable Conventional-HBM overflow boundary policies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.platform import load_platform_spec_file
from om3dthermal.workload import DenseLLMModelSpec, evaluate_llm_decode

from .gpu import AnalyticalRooflineGPUModel
from .mixed_phase_e2e import resolve_conventional_hbm_backend
from .persistent_horizon import PersistentMixedServiceCase, _prefill_roofline
from .workspace import WorkspaceExecutionConfig, evaluate_decode_workspace, evaluate_prefill_workspace


ConventionalOverflowPolicy = Literal[
    "RESIDENT_ONLY_QUEUE_TO_FIT", "HOST_KV_OFFLOAD"]


class HostKVDecodeStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(ge=0)
    context_length: int = Field(gt=0)
    required_historical_KV_bytes: int
    local_historical_KV_bytes: int
    host_historical_KV_bytes: int
    host_append_write_bytes: int
    host_prefill_eviction_write_bytes: int
    local_compute_memory_time_s: float
    host_transfer_time_s: float
    stage_elapsed_time_s: float

    @model_validator(mode="after")
    def _closure(self) -> "HostKVDecodeStep":
        if self.required_historical_KV_bytes != (
                self.local_historical_KV_bytes
                + self.host_historical_KV_bytes):
            raise ValueError("historical KV byte conservation failed")
        expected = max(self.local_compute_memory_time_s,
                       self.host_transfer_time_s)
        if not math.isclose(self.stage_elapsed_time_s, expected, rel_tol=1e-12):
            raise ValueError("optimistic overlap stage timing failed")
        return self


class ConventionalOverflowResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    policy: ConventionalOverflowPolicy
    policy_role: Literal["OPTIMISTIC_LOCAL_RESIDENCY_BASELINE",
                         "OPTIMISTIC_RECURRING_HOST_ACCESS_BASELINE"]
    S: int
    P: int
    D: int
    BS: int
    G: int
    status: Literal["EVALUATED", "CAPACITY_INFEASIBLE"]
    resident_limit_initial: int
    resident_limit_horizon_safe: int
    resident_wave_size: int
    num_waves: int
    queue_delay_per_wave_s: tuple[float, ...]
    per_request_completion_latency_s: tuple[float, ...]
    prefill_time_s: float | None
    decode_compute_time_s: float | None
    host_transfer_time_s: float | None
    host_critical_path_time_s: float | None
    admission_time_s: float | None
    eviction_time_s: float | None
    total_completion_time_s: float | None
    generated_tokens: int
    aggregate_tokens_per_s: float | None
    total_host_KV_read_bytes: int
    total_host_KV_write_bytes: int
    historical_host_read_bytes: int
    append_host_write_bytes: int
    admission_host_read_bytes: int
    migration_host_bytes: int
    average_host_read_GB_per_decode_step: float
    peak_host_read_GB_per_decode_step: float
    effective_host_bw_GBps: float
    host_transfer_time_fraction: float | None
    host_traffic_bytes_per_token: float
    peak_workspace_bytes: int
    peak_local_runtime_bytes: int
    capacity_bytes: int
    capacity_margin_bytes: int
    known_energy_J: float
    known_DDR_PCIe_energy_J: float
    energy_status: Literal["INCOMPLETE_CONVENTIONAL_HBM_WRITE_ENERGY_UNRESOLVED"]
    unresolved_HBM_write_bytes: int
    steps: tuple[HostKVDecodeStep, ...] = Field(exclude=True, repr=False)
    thermal: None = None


def _resident_limit(model: DenseLLMModelSpec, *, S: int, G: int,
                    capacity: int, weight: int,
                    config: WorkspaceExecutionConfig) -> int:
    kv = int(evaluate_llm_decode(model.decode_input(
        batch_size=1, context_length=S)).kv_bytes_per_request)
    kv_token = int(evaluate_llm_decode(model.decode_input(
        batch_size=1, context_length=S)).kv_write_bytes_per_token)
    context = S + G
    limit = 0
    # The relevant limits are small (tens of requests); this exact scan also
    # keeps the batch-dependent workspace visible and auditable.
    for batch in range(1, 4097):
        workspace = evaluate_decode_workspace(
            model, batch_size=batch, context_length=max(S, context-1),
            config=config).peak_bytes
        required = weight + batch*(kv+G*kv_token) + workspace
        if required > capacity:
            break
        limit = batch
    return limit


def _gpu_step(model: DenseLLMModelSpec, *, batch: int, context: int,
              bandwidth: float, compute: float):
    inp = model.decode_input(batch_size=batch, context_length=context)
    metrics = evaluate_llm_decode(inp)
    result = AnalyticalRooflineGPUModel(
        matched_payload_bandwidth_bits_per_second=8.0*bandwidth,
        effective_compute_flops_per_second=compute).evaluate(inp, batch_size=batch)
    return metrics, result.decode_step_time_ms*1e-3


def evaluate_conventional_overflow_policy(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    case: PersistentMixedServiceCase, policy: ConventionalOverflowPolicy,
    workspace_config: WorkspaceExecutionConfig,
) -> ConventionalOverflowResult:
    """Evaluate one of two explicit, non-production Conventional baselines."""
    root = Path(project_root).resolve()
    backend = resolve_conventional_hbm_backend(root)
    platform = load_platform_spec_file(
        root/"configs/platform/gpu_package_h200_reference.yaml")
    gpu = platform.gpu_decode_power
    compute = platform.gpu_compute_power
    host = backend.host_offload
    if gpu is None or compute is None or host.effective_bandwidth_bytes_per_second is None:
        raise ValueError("canonical GPU/host platform data is incomplete")
    metrics = evaluate_llm_decode(model.decode_input(
        batch_size=case.batch_size, context_length=case.context_length))
    weight = int(metrics.weight_footprint_bytes + metrics.runtime_fixed_bytes)
    kv_s = int(metrics.kv_bytes_per_request)
    kv_token = int(metrics.kv_write_bytes_per_token)
    capacity = int(backend.capacity_bytes)
    prefill_ws = evaluate_prefill_workspace(
        model, batch_size=case.prefill_requests,
        context_length=case.context_length, config=workspace_config)
    initial_limit = _resident_limit(
        model, S=case.context_length, G=0, capacity=capacity,
        weight=weight, config=workspace_config)
    safe_limit = _resident_limit(
        model, S=case.context_length, G=case.generated_decode_steps,
        capacity=capacity, weight=weight, config=workspace_config)
    prefill_metrics, roofline, _ = _prefill_roofline(
        root, model, case, backend.sustained_bandwidth_bytes_per_s)
    prefill_compute_s = roofline.nominal_prefill_latency_s
    host_bw = host.effective_bandwidth_bytes_per_second
    generated = case.D*case.G
    step_rows: list[HostKVDecodeStep] = []
    queue: list[float] = []
    completion: list[float] = []
    historical_read = append_write = migration = 0
    admission_bytes = 0
    decode_compute_s = raw_host_s = host_critical_s = 0.0
    admission_s = eviction_s = 0.0
    prefill_host_s = 0.0

    if safe_limit <= 0:
        status: Literal["EVALUATED", "CAPACITY_INFEASIBLE"] = "CAPACITY_INFEASIBLE"
        wave_size = waves = 0
        prefill_time = total = None
        peak_workspace = evaluate_decode_workspace(
            model, batch_size=1,
            context_length=case.context_length+case.generated_decode_steps-1,
            config=workspace_config).peak_bytes
        peak_runtime = weight + kv_s + case.G*kv_token + peak_workspace
    elif policy == "RESIDENT_ONLY_QUEUE_TO_FIT":
        status = "EVALUATED"
        wave_size = min(case.D, safe_limit)
        waves = math.ceil(case.D/wave_size)
        decode_ws = evaluate_decode_workspace(
            model, batch_size=wave_size,
            context_length=case.context_length+case.G-1,
            config=workspace_config)
        peak_workspace = max(prefill_ws.peak_bytes, decode_ws.peak_bytes)
        local_d_peak = wave_size*(kv_s+case.G*kv_token)
        local_p = min(case.P*kv_s,
                      max(0, capacity-weight-decode_ws.peak_bytes-local_d_peak))
        migration = case.P*kv_s-local_p
        waiting_d = case.D-wave_size
        admission_bytes = waiting_d*kv_s
        admission_s = admission_bytes/host_bw
        eviction_s = migration/host_bw
        raw_host_s = admission_s+eviction_s
        cursor = prefill_compute_s+eviction_s
        remaining = case.D
        for wave_index in range(waves):
            batch = min(wave_size, remaining)
            if wave_index:
                wave_admission = batch*kv_s/host_bw
                cursor += wave_admission
            queue.append(cursor)
            wave_compute = 0.0
            for j in range(case.G):
                _, step_s = _gpu_step(
                    model, batch=batch, context=case.S+j,
                    bandwidth=backend.sustained_bandwidth_bytes_per_s,
                    compute=compute.peak_compute_BF16_dense_flops_per_s)
                wave_compute += step_s
            decode_compute_s += wave_compute
            cursor += wave_compute
            completion.extend([cursor]*batch)
            remaining -= batch
        host_critical_s = raw_host_s
        prefill_time = prefill_compute_s
        total = prefill_compute_s+decode_compute_s+host_critical_s
        peak_runtime = weight+local_d_peak+local_p+decode_ws.peak_bytes
    else:
        status = "EVALUATED"
        wave_size = case.D
        waves = 1
        decode_ws = evaluate_decode_workspace(
            model, batch_size=case.D,
            context_length=case.S+case.G-1, config=workspace_config)
        peak_workspace = max(prefill_ws.peak_bytes, decode_ws.peak_bytes)
        local_budget = max(0, capacity-weight-decode_ws.peak_bytes)
        initial_required_d = case.D*kv_s
        local_d = min(initial_required_d, local_budget)
        initial_local_p = min(case.P*kv_s, max(0, local_budget-local_d))
        migration = case.P*kv_s-initial_local_p
        prefill_host_s = migration/host_bw
        elapsed_decode = 0.0
        for j in range(case.G):
            required = case.D*(kv_s+j*kv_token)
            next_required = required+case.D*kv_token
            local_d_now = min(required, local_budget)
            local_d_next = min(next_required, local_budget)
            local_p_now = min(initial_local_p, max(0, local_budget-local_d_now))
            local_p_next = min(initial_local_p, max(0, local_budget-local_d_next))
            host_required = required-local_d_now
            host_append = max(0, case.D*kv_token-(local_d_next-local_d_now))
            prefill_evict = max(0, local_p_now-local_p_next)
            _, compute_s = _gpu_step(
                model, batch=case.D, context=case.S+j,
                bandwidth=backend.sustained_bandwidth_bytes_per_s,
                compute=compute.peak_compute_BF16_dense_flops_per_s)
            transfer_s = (host_required+host_append+prefill_evict)/host_bw
            elapsed = max(compute_s, transfer_s)
            step_rows.append(HostKVDecodeStep(
                step_index=j, context_length=case.S+j,
                required_historical_KV_bytes=required,
                local_historical_KV_bytes=local_d_now,
                host_historical_KV_bytes=host_required,
                host_append_write_bytes=host_append,
                host_prefill_eviction_write_bytes=prefill_evict,
                local_compute_memory_time_s=compute_s,
                host_transfer_time_s=transfer_s,
                stage_elapsed_time_s=elapsed))
            historical_read += host_required
            append_write += host_append
            migration += prefill_evict
            decode_compute_s += compute_s
            raw_host_s += transfer_s
            elapsed_decode += elapsed
        host_critical_s = prefill_host_s+elapsed_decode-decode_compute_s
        queue = [prefill_compute_s+prefill_host_s]
        total = prefill_compute_s+prefill_host_s+elapsed_decode
        completion = [total]*case.D
        prefill_time = prefill_compute_s
        peak_runtime = weight+local_budget+decode_ws.peak_bytes

    total_host_read = historical_read+admission_bytes
    total_host_write = append_write+migration
    total_host_bytes = total_host_read+total_host_write
    host_bits = 8.0*total_host_bytes
    ddr_pcie_J = host_bits*(
        float(host.e_pcie_dynamic_J_per_bit or 0.0)
        + float(host.e_ddr_dynamic_J_per_bit or 0.0))
    known = ddr_pcie_J
    if total is not None:
        known += gpu.static_power_W*total
    unresolved_write = int(prefill_metrics.prefill_write_bytes+total_host_read)
    return ConventionalOverflowResult(
        model=model.model_id, policy=policy,
        policy_role=("OPTIMISTIC_LOCAL_RESIDENCY_BASELINE"
                     if policy == "RESIDENT_ONLY_QUEUE_TO_FIT" else
                     "OPTIMISTIC_RECURRING_HOST_ACCESS_BASELINE"),
        S=case.S, P=case.P, D=case.D, BS=case.BS, G=case.G,
        status=status, resident_limit_initial=initial_limit,
        resident_limit_horizon_safe=safe_limit,
        resident_wave_size=wave_size, num_waves=waves,
        queue_delay_per_wave_s=tuple(queue),
        per_request_completion_latency_s=tuple(completion),
        prefill_time_s=prefill_time,
        decode_compute_time_s=(None if total is None else decode_compute_s),
        host_transfer_time_s=(None if total is None else raw_host_s+prefill_host_s),
        host_critical_path_time_s=(None if total is None else host_critical_s),
        admission_time_s=(None if total is None else admission_s),
        eviction_time_s=(None if total is None else eviction_s),
        total_completion_time_s=total, generated_tokens=generated,
        aggregate_tokens_per_s=(None if total is None else generated/total),
        total_host_KV_read_bytes=total_host_read,
        total_host_KV_write_bytes=total_host_write,
        historical_host_read_bytes=historical_read,
        append_host_write_bytes=append_write,
        admission_host_read_bytes=admission_bytes,
        migration_host_bytes=migration,
        average_host_read_GB_per_decode_step=(historical_read/case.G/1e9),
        peak_host_read_GB_per_decode_step=(
            max((row.host_historical_KV_bytes for row in step_rows), default=0)/1e9),
        effective_host_bw_GBps=host_bw/1e9,
        host_transfer_time_fraction=(None if total is None else host_critical_s/total),
        host_traffic_bytes_per_token=total_host_bytes/generated,
        peak_workspace_bytes=peak_workspace,
        peak_local_runtime_bytes=peak_runtime,
        capacity_bytes=capacity, capacity_margin_bytes=capacity-peak_runtime,
        known_energy_J=known, known_DDR_PCIe_energy_J=ddr_pcie_J,
        energy_status="INCOMPLETE_CONVENTIONAL_HBM_WRITE_ENERGY_UNRESOLVED",
        unresolved_HBM_write_bytes=unresolved_write,
        steps=tuple(step_rows))
