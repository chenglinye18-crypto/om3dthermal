"""Frozen B x S x G long-context inference benchmark semantics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.platform import load_platform_spec_file
from om3dthermal.power.nmp_die_power import resolve_orthogonal_m3d_write_energy_pj_per_bit
from om3dthermal.workload import DenseLLMModelSpec, evaluate_llm_decode

from .gpu import AnalyticalRooflineGPUModel
from .mixed_phase_e2e import SystemId, resolve_conventional_hbm_backend
from .nmp_decode import evaluate_nmp_decode_batch, resolve_m3d_architecture_backend
from .persistent_horizon import PersistentMixedServiceCase, _m3d_gpu_bandwidth, _nmp_decode_sum, _prefill_roofline
from .workspace import WorkspaceExecutionConfig, evaluate_decode_workspace, evaluate_prefill_workspace


FormalPolicy = Literal[
    "RESIDENT_ONLY_QUEUE_TO_FIT", "HOST_KV_OFFLOAD", "FULLY_LOCAL"]


class FormalInferenceWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_id: str
    batch_size: int = Field(gt=0)
    prompt_tokens: int = Field(gt=0)
    generation_tokens: int = Field(gt=0)
    arrival_policy: Literal["ALL_REQUESTS_AT_T0"] = "ALL_REQUESTS_AT_T0"
    prefill_count_per_request: Literal[1] = 1


class FormalE2EResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str
    system: SystemId
    policy: FormalPolicy
    batch_size: int
    prompt_tokens: int
    generation_tokens: int
    status: Literal["EVALUATED", "CAPACITY_INFEASIBLE"]
    safe_resident_batch: int
    num_waves: int
    prefill_executions: int
    decode_steps_per_request: int
    initial_runtime_bytes: int
    peak_runtime_bytes: int
    capacity_bytes: int
    capacity_margin_bytes: int
    resident_fraction: float
    peak_host_resident_bytes: int
    first_capacity_violation_step: int | None
    prefill_time_s: float | None
    decode_active_time_s: float | None
    queue_time_s: float | None
    host_transfer_time_s: float | None
    host_critical_path_time_s: float | None
    batch_makespan_s: float | None
    e2e_output_tokens_per_s: float | None
    decode_tokens_per_s: float | None
    mean_ttft_s: float | None
    p95_ttft_s: float | None
    max_ttft_s: float | None
    mean_completion_latency_s: float | None
    p95_completion_latency_s: float | None
    max_completion_latency_s: float | None
    mean_tpot_s: float | None
    p95_tpot_s: float | None
    historical_host_read_bytes: int
    host_append_write_bytes: int
    migration_bytes: int
    total_host_transfer_bytes: int
    host_bytes_per_generated_token: float
    host_transfer_time_fraction: float | None
    known_energy_J: float
    known_energy_components: dict[str, float]
    energy_status: str
    unresolved_energy_terms: dict[str, int]
    absolute_J_per_token_status: Literal[
        "UNAVAILABLE_DUE_TO_UNRESOLVED_HBM_WRITE_ENERGY",
        "UNAVAILABLE_DUE_TO_UNRESOLVED_PREFILL_GPU_DYNAMIC_ENERGY",
        "AVAILABLE"]
    numerical_status: str
    thermal: None = None

    @model_validator(mode="after")
    def _closure(self) -> "FormalE2EResult":
        if self.prefill_executions != self.batch_size:
            raise ValueError("every request must execute Prefill exactly once")
        if self.decode_steps_per_request != self.generation_tokens:
            raise ValueError("every request must execute exactly G Decode steps")
        if self.batch_makespan_s is not None:
            expected = self.batch_size*self.generation_tokens/self.batch_makespan_s
            if not math.isclose(float(self.e2e_output_tokens_per_s), expected,
                                rel_tol=1e-12):
                raise ValueError("E2E output throughput does not close")
            if self.max_completion_latency_s != self.batch_makespan_s:
                raise ValueError("batch makespan must be the last completion")
        return self


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = max(0, math.ceil(fraction*len(ordered))-1)
    return ordered[rank]


def _safe_batch(model: DenseLLMModelSpec, *, S: int, G: int,
                capacity: int, config: WorkspaceExecutionConfig,
                page_size: int | None = None, proposed: bool = False,
                die_count: int = 0) -> int:
    one = evaluate_llm_decode(model.decode_input(batch_size=1, context_length=S))
    weight = int(one.weight_footprint_bytes+one.runtime_fixed_bytes)
    kv_end = int(one.kv_bytes_per_request+G*one.kv_write_bytes_per_token)
    rounded = lambda value: value if page_size is None else math.ceil(value/page_size)*page_size
    safe = 0
    for batch in range(1, 4097):
        prefill = evaluate_prefill_workspace(
            model, batch_size=batch, context_length=S, config=config).peak_bytes
        decode = evaluate_decode_workspace(
            model, batch_size=batch, context_length=S+G-1, config=config,
            proposed_nmp=proposed, nmp_die_count=die_count).peak_bytes
        peak = max(
            rounded(weight)+batch*rounded(int(one.kv_bytes_per_request))+rounded(prefill),
            rounded(weight)+batch*rounded(kv_end)+rounded(decode))
        if peak > capacity:
            break
        safe = batch
    return safe


def formal_capacity_limits(
    *, project_root: str | Path, model: DenseLLMModelSpec, S: int, G: int,
    workspace_config: WorkspaceExecutionConfig,
) -> dict[str, int]:
    root = Path(project_root).resolve()
    hbm = resolve_conventional_hbm_backend(root)
    m3d = resolve_m3d_architecture_backend(root)
    common = dict(model=model, S=S, G=G, config=workspace_config)
    return {
        "B_max_HBM_safe": _safe_batch(
            **common, capacity=int(hbm.capacity_bytes)),
        "B_max_M3D_safe": _safe_batch(
            **common, capacity=m3d.layout.total_capacity_bytes,
            page_size=m3d.layout.slot_capacity_bytes),
        "B_max_NMP_safe": _safe_batch(
            **common, capacity=m3d.layout.total_capacity_bytes,
            page_size=m3d.layout.slot_capacity_bytes, proposed=True,
            die_count=m3d.layout.slab_count),
    }


def _gpu_profile(model, *, batch: int, S: int, G: int,
                 bandwidth: float, compute: float):
    evaluator = AnalyticalRooflineGPUModel(
        matched_payload_bandwidth_bits_per_second=8.0*bandwidth,
        effective_compute_flops_per_second=compute)
    times: list[float] = []
    read = write = total = 0.0
    for j in range(G):
        inp = model.decode_input(batch_size=batch, context_length=S+j)
        metrics = evaluate_llm_decode(inp)
        times.append(evaluator.evaluate(inp, batch_size=batch).decode_step_time_ms*1e-3)
        read += batch*metrics.read_bytes_per_token
        write += batch*metrics.write_bytes_per_token
        total += batch*(metrics.read_bytes_per_token+metrics.write_bytes_per_token)
    return times, (read, write, total)


def evaluate_formal_inference_workload(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    workload: FormalInferenceWorkload, system: SystemId, policy: FormalPolicy,
    workspace_config: WorkspaceExecutionConfig,
) -> FormalE2EResult:
    if model.model_id != workload.model_id:
        raise ValueError("model and formal workload mismatch")
    if system == "CONVENTIONAL_HBM_GPU" and policy == "FULLY_LOCAL":
        raise ValueError("Conventional benchmark requires an explicit overflow policy")
    if system != "CONVENTIONAL_HBM_GPU" and policy != "FULLY_LOCAL":
        raise ValueError("M3D systems use the frozen fully-local policy")
    root = Path(project_root).resolve()
    hbm = resolve_conventional_hbm_backend(root)
    m3d = resolve_m3d_architecture_backend(root)
    platform = load_platform_spec_file(root/"configs/platform/gpu_package_h200_reference.yaml")
    gpu = platform.gpu_decode_power
    compute = platform.gpu_compute_power
    host = hbm.host_offload
    if gpu is None or compute is None or host.effective_bandwidth_bytes_per_second is None:
        raise ValueError("canonical platform data is incomplete")
    B, S, G = workload.batch_size, workload.prompt_tokens, workload.generation_tokens
    one = evaluate_llm_decode(model.decode_input(batch_size=1, context_length=S))
    weight = int(one.weight_footprint_bytes+one.runtime_fixed_bytes)
    kv_s = int(one.kv_bytes_per_request)
    kv_token = int(one.kv_write_bytes_per_token)
    proposed = system == "IOM3D_FEOL_NMP"
    capacity = int(hbm.capacity_bytes if system == "CONVENTIONAL_HBM_GPU"
                   else m3d.layout.total_capacity_bytes)
    page = None if system == "CONVENTIONAL_HBM_GPU" else m3d.layout.slot_capacity_bytes
    safe = _safe_batch(
        model, S=S, G=G, capacity=capacity, config=workspace_config,
        page_size=page, proposed=proposed, die_count=m3d.layout.slab_count)
    prefill_ws = evaluate_prefill_workspace(
        model, batch_size=B, context_length=S, config=workspace_config)
    decode_ws = evaluate_decode_workspace(
        model, batch_size=B, context_length=S+G-1, config=workspace_config,
        proposed_nmp=proposed, nmp_die_count=m3d.layout.slab_count)
    rounded = lambda value: value if page is None else math.ceil(value/page)*page
    initial_runtime = rounded(weight)+B*rounded(kv_s)+rounded(prefill_ws.peak_bytes)
    peak_runtime = rounded(weight)+B*rounded(kv_s+G*kv_token)+rounded(decode_ws.peak_bytes)
    peak_runtime = max(initial_runtime, peak_runtime)
    ttfts: list[float] = []
    completions: list[float] = []
    tpots: list[float] = []
    prefill_total = decode_compute = queue_total = 0.0
    raw_host = critical_host = 0.0
    historical = append_host = migration = 0
    peak_host_resident = 0
    known: dict[str, float] = {}
    prefill_read_bytes = prefill_write_bytes = 0.0
    unresolved: dict[str, int] = {"PREFILL_GPU_DYNAMIC_RANGE_NO_SINGLE_NOMINAL": 0}
    status: Literal["EVALUATED", "CAPACITY_INFEASIBLE"] = "EVALUATED"
    waves = 1
    first_capacity_violation_step = None

    def prefill(batch: int) -> tuple[float, object]:
        case = PersistentMixedServiceCase(
            model_id=model.model_id, context_length=S,
            prefill_requests=batch, decode_requests=1,
            generated_decode_steps=G)
        metrics, roofline, _ = _prefill_roofline(
            root, model, case, hbm.sustained_bandwidth_bytes_per_s)
        return roofline.nominal_prefill_latency_s, metrics

    prefill_metrics_all = None
    if policy == "RESIDENT_ONLY_QUEUE_TO_FIT":
        if safe <= 0:
            status = "CAPACITY_INFEASIBLE"
        else:
            wave_size = min(B, safe)
            waves = math.ceil(B/wave_size)
            cursor = 0.0
            remaining = B
            read = write = total_traffic = 0.0
            while remaining:
                batch = min(wave_size, remaining)
                wave_start = cursor
                prefill_s, prefill_metrics = prefill(batch)
                prefill_metrics_all = prefill_metrics
                prefill_read_bytes += prefill_metrics.prefill_read_bytes
                prefill_write_bytes += prefill_metrics.prefill_write_bytes
                times, traffic = _gpu_profile(
                    model, batch=batch, S=S, G=G,
                    bandwidth=hbm.sustained_bandwidth_bytes_per_s,
                    compute=compute.peak_compute_BF16_dense_flops_per_s)
                decode_s = sum(times)
                first = wave_start+prefill_s+times[0]
                done = wave_start+prefill_s+decode_s
                ttfts.extend([first]*batch)
                completions.extend([done]*batch)
                tpots.extend([decode_s/G]*batch)
                queue_total += wave_start*batch
                prefill_total += prefill_s
                decode_compute += decode_s
                read += traffic[0]; write += traffic[1]; total_traffic += traffic[2]
                cursor = done
                remaining -= batch
            known["gpu_decode_dynamic_J"] = 8.0*total_traffic*gpu.e_decode_J_per_bit
            known["hbm_read_dynamic_J"] = 8.0*read*hbm.read_energy_pJ_per_bit*1e-12
            unresolved["CONVENTIONAL_HBM_WRITE"] = int(write)
            actual_prefill_ws = evaluate_prefill_workspace(
                model, batch_size=wave_size, context_length=S,
                config=workspace_config).peak_bytes
            actual_decode_ws = evaluate_decode_workspace(
                model, batch_size=wave_size, context_length=S+G-1,
                config=workspace_config).peak_bytes
            initial_runtime = weight+wave_size*kv_s+actual_prefill_ws
            peak_runtime = max(
                initial_runtime,
                weight+wave_size*(kv_s+G*kv_token)+actual_decode_ws)
    elif policy == "HOST_KV_OFFLOAD":
        prefill_s, prefill_metrics_all = prefill(B)
        prefill_read_bytes += prefill_metrics_all.prefill_read_bytes
        prefill_write_bytes += prefill_metrics_all.prefill_write_bytes
        prefill_total = prefill_s
        times, traffic = _gpu_profile(
            model, batch=B, S=S, G=G,
            bandwidth=hbm.sustained_bandwidth_bytes_per_s,
            compute=compute.peak_compute_BF16_dense_flops_per_s)
        decode_compute = sum(times)
        local_budget = max(0, capacity-weight-decode_ws.peak_bytes)
        local_initial = min(B*kv_s, local_budget)
        migration = B*kv_s-local_initial
        prefill_host = migration/host.effective_bandwidth_bytes_per_second
        raw_host += prefill_host
        critical_host += prefill_host
        elapsed_decode = 0.0
        for j, compute_s in enumerate(times):
            required = B*(kv_s+j*kv_token)
            next_required = required+B*kv_token
            local_now = min(required, local_budget)
            local_next = min(next_required, local_budget)
            host_required = required-local_now
            host_append = max(0, B*kv_token-(local_next-local_now))
            transfer = (host_required+host_append)/host.effective_bandwidth_bytes_per_second
            elapsed = max(compute_s, transfer)
            historical += host_required
            append_host += host_append
            raw_host += transfer
            critical_host += elapsed-compute_s
            elapsed_decode += elapsed
            if j == 0:
                first_elapsed = elapsed
        makespan = prefill_s+prefill_host+elapsed_decode
        ttfts = [prefill_s+prefill_host+first_elapsed]*B
        completions = [makespan]*B
        tpots = [elapsed_decode/G]*B
        known["gpu_decode_dynamic_J"] = 8.0*traffic[2]*gpu.e_decode_J_per_bit
        known["hbm_read_dynamic_J"] = 8.0*traffic[0]*hbm.read_energy_pJ_per_bit*1e-12
        host_bits = 8.0*(historical+append_host+migration)
        known["host_DDR_PCIe_dynamic_J"] = host_bits*(
            float(host.e_ddr_dynamic_J_per_bit or 0.0)
            + float(host.e_pcie_dynamic_J_per_bit or 0.0))
        unresolved["CONVENTIONAL_HBM_WRITE"] = int(
            traffic[1]+historical)
        local_end = min(B*(kv_s+G*kv_token), local_budget)
        peak_host_resident = max(
            0, B*(kv_s+G*kv_token)-local_budget)
        peak_runtime = max(
            weight+local_initial+prefill_ws.peak_bytes,
            weight+local_end+decode_ws.peak_bytes)
        initial_runtime = weight+local_initial+prefill_ws.peak_bytes
    else:
        if B > safe:
            status = "CAPACITY_INFEASIBLE"
            if initial_runtime > capacity:
                first_capacity_violation_step = 0
            else:
                low, high = 1, G
                while low < high:
                    step = (low+high)//2
                    workspace = evaluate_decode_workspace(
                        model, batch_size=B, context_length=S+step-1,
                        config=workspace_config, proposed_nmp=proposed,
                        nmp_die_count=m3d.layout.slab_count).peak_bytes
                    runtime = (rounded(weight)
                               + B*rounded(kv_s+step*kv_token)
                               + rounded(workspace))
                    if runtime > capacity:
                        high = step
                    else:
                        low = step+1
                first_capacity_violation_step = low
        else:
            prefill_s, prefill_metrics_all = prefill(B)
            prefill_read_bytes += prefill_metrics_all.prefill_read_bytes
            prefill_write_bytes += prefill_metrics_all.prefill_write_bytes
            prefill_total = prefill_s
            if system == "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY":
                times, traffic = _gpu_profile(
                    model, batch=B, S=S, G=G,
                    bandwidth=_m3d_gpu_bandwidth(root),
                    compute=compute.peak_compute_BF16_dense_flops_per_s)
                decode_compute = sum(times)
                first_step = times[0]
                write_pj = resolve_orthogonal_m3d_write_energy_pj_per_bit(
                    m3d.case, m3d.memory)
                known["gpu_decode_dynamic_J"] = 8.0*traffic[2]*gpu.e_decode_J_per_bit
                known["m3d_decode_read_J"] = 8.0*traffic[0]*m3d.memory.E_access_total_pj_bit*1e-12
                known["m3d_decode_write_J"] = 8.0*traffic[1]*write_pj*1e-12
                numerical = "EXACT_PER_TOKEN_GROWING_CONTEXT_SUM"
            else:
                decode_compute, nmp_J = _nmp_decode_sum(
                    root, model, batch=B, resident_batch=B, S=S, G=G)
                first_result = evaluate_nmp_decode_batch(
                    model.decode_input(batch_size=B, context_length=S),
                    project_root=root)
                first_step = float(first_result.decode_step_time_ms)*1e-3
                known["nmp_decode_total_J"] = nmp_J
                numerical = "THREE_POINT_QUADRATIC_GROWING_CONTEXT_SUM__SHORT_G_VALIDATED"
            makespan = prefill_s+decode_compute
            ttfts = [prefill_s+first_step]*B
            completions = [makespan]*B
            tpots = [decode_compute/G]*B

    if status == "CAPACITY_INFEASIBLE":
        makespan = None
        numerical = "NOT_EVALUATED_CAPACITY_INFEASIBLE"
    elif policy in {"RESIDENT_ONLY_QUEUE_TO_FIT", "HOST_KV_OFFLOAD"}:
        makespan = max(completions)
        numerical = "EXACT_PER_TOKEN_GROWING_CONTEXT_SUM"
    if makespan is not None:
        known["gpu_static_J"] = gpu.static_power_W*makespan
    if prefill_metrics_all is not None:
        if system == "CONVENTIONAL_HBM_GPU":
            known["hbm_prefill_read_J"] = 8.0*prefill_read_bytes*hbm.read_energy_pJ_per_bit*1e-12
            unresolved["CONVENTIONAL_HBM_WRITE"] = unresolved.get(
                "CONVENTIONAL_HBM_WRITE", 0)+int(prefill_write_bytes)
        else:
            write_pj = resolve_orthogonal_m3d_write_energy_pj_per_bit(m3d.case, m3d.memory)
            known["m3d_prefill_read_J"] = 8.0*prefill_read_bytes*m3d.memory.E_access_total_pj_bit*1e-12
            known["m3d_prefill_write_J"] = 8.0*prefill_write_bytes*write_pj*1e-12
    host_total = historical+append_host+migration
    return FormalE2EResult(
        model=model.model_id, system=system, policy=policy, batch_size=B,
        prompt_tokens=S, generation_tokens=G, status=status,
        safe_resident_batch=safe, num_waves=(waves if status == "EVALUATED" else 0),
        prefill_executions=B, decode_steps_per_request=G,
        initial_runtime_bytes=initial_runtime, peak_runtime_bytes=peak_runtime,
        capacity_bytes=capacity, capacity_margin_bytes=capacity-peak_runtime,
        resident_fraction=min(1.0, safe/B),
        peak_host_resident_bytes=peak_host_resident,
        first_capacity_violation_step=first_capacity_violation_step,
        prefill_time_s=(prefill_total if status == "EVALUATED" else None),
        decode_active_time_s=(decode_compute if status == "EVALUATED" else None),
        queue_time_s=(queue_total/B if status == "EVALUATED" else None),
        host_transfer_time_s=(raw_host if status == "EVALUATED" else None),
        host_critical_path_time_s=(critical_host if status == "EVALUATED" else None),
        batch_makespan_s=makespan,
        e2e_output_tokens_per_s=(None if makespan is None else B*G/makespan),
        decode_tokens_per_s=(None if decode_compute == 0 else B*G/decode_compute),
        mean_ttft_s=(None if not ttfts else sum(ttfts)/B),
        p95_ttft_s=(None if not ttfts else _percentile(ttfts, .95)),
        max_ttft_s=(None if not ttfts else max(ttfts)),
        mean_completion_latency_s=(None if not completions else sum(completions)/B),
        p95_completion_latency_s=(None if not completions else _percentile(completions, .95)),
        max_completion_latency_s=(None if not completions else max(completions)),
        mean_tpot_s=(None if not tpots else sum(tpots)/B),
        p95_tpot_s=(None if not tpots else _percentile(tpots, .95)),
        historical_host_read_bytes=historical,
        host_append_write_bytes=append_host, migration_bytes=migration,
        total_host_transfer_bytes=host_total,
        host_bytes_per_generated_token=host_total/(B*G),
        host_transfer_time_fraction=(None if makespan is None else critical_host/makespan),
        known_energy_J=sum(known.values()), known_energy_components=known,
        energy_status=("INCOMPLETE_CONVENTIONAL_HBM_WRITE_ENERGY_UNRESOLVED"
                       if system == "CONVENTIONAL_HBM_GPU" else
                       "INCOMPLETE_PREFILL_GPU_DYNAMIC_RANGE_NO_SINGLE_NOMINAL"),
        unresolved_energy_terms=unresolved,
        absolute_J_per_token_status=(
            "UNAVAILABLE_DUE_TO_UNRESOLVED_HBM_WRITE_ENERGY"
            if system == "CONVENTIONAL_HBM_GPU" else
            "UNAVAILABLE_DUE_TO_UNRESOLVED_PREFILL_GPU_DYNAMIC_ENERGY"),
        numerical_status=numerical)
