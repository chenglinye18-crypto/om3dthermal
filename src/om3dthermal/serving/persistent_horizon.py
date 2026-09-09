"""Unified non-thermal persistent mixed-service horizon across three systems."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.platform import (
    load_platform_spec_file, resolve_gpu_bandwidth_service,
    resolve_gpu_prefill_compute_energy_calibration,
    resolve_local_memory_gpu_transfer,
)
from om3dthermal.power.memory_bandwidth import resolve_internal_service_bandwidth
from om3dthermal.power.nmp_die_activity import NMP_BANK_TO_LOCAL_ROUTE_DELAY_NS
from om3dthermal.power.nmp_die_power import resolve_orthogonal_m3d_write_energy_pj_per_bit
from om3dthermal.workload import (
    DenseLLMModelSpec, evaluate_gpu_prefill_roofline, evaluate_llm_decode,
    evaluate_llm_prefill,
)

from .gpu import AnalyticalRooflineGPUModel
from .mixed_phase_e2e import (
    SYSTEM_CONFIGURATIONS, SystemId, resolve_conventional_hbm_backend,
)
from .nmp_decode import evaluate_nmp_decode_batch, resolve_m3d_architecture_backend
from .state_ledger import EvaluationSemantics, require_comparable_semantics
from .workspace import (
    WorkspaceExecutionConfig, evaluate_decode_workspace,
    evaluate_prefill_workspace,
)


_NMP_STREAM_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


class PersistentMixedServiceCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    context_length: int = Field(gt=0)
    prefill_requests: int = Field(gt=0)
    decode_requests: int = Field(gt=0)
    generated_decode_steps: int = Field(gt=0)

    @property
    def batch_size(self) -> int:
        return self.prefill_requests + self.decode_requests

    @property
    def S(self) -> int:
        return self.context_length

    @property
    def P(self) -> int:
        return self.prefill_requests

    @property
    def D(self) -> int:
        return self.decode_requests

    @property
    def G(self) -> int:
        return self.generated_decode_steps

    @property
    def BS(self) -> int:
        return self.batch_size


class ServingE2EClosureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str
    model_registry_dir: str
    models: tuple[str, ...]
    context_lengths: tuple[int, ...]
    mixed_points: tuple[dict[str, int], ...]
    generation_horizons: tuple[int, ...]
    systems: tuple[SystemId, ...]
    schedule_policy: Literal["NO_OVERLAP__PREFILL_FIRST"]
    workspace: WorkspaceExecutionConfig

    @model_validator(mode="after")
    def _closure(self) -> "ServingE2EClosureSpec":
        if set(self.systems) != set(SYSTEM_CONFIGURATIONS):
            raise ValueError("closure matrix requires exactly the three systems")
        for point in self.mixed_points:
            if point["prefill_requests"] + point["decode_requests"] != point["batch_size"]:
                raise ValueError("mixed point batch does not close")
        return self


def load_serving_e2e_closure_spec(path: str | Path) -> ServingE2EClosureSpec:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return ServingE2EClosureSpec.model_validate(raw)


class PersistentMixedServiceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    system: SystemId
    scope: Literal["MIXED_SERVICE_WINDOW"] = "MIXED_SERVICE_WINDOW"
    schedule_policy: Literal["NO_OVERLAP__PREFILL_FIRST"] = (
        "NO_OVERLAP__PREFILL_FIRST")
    context_evolution: Literal["GROWING_KV"] = "GROWING_KV"
    S: int
    P: int
    D: int
    BS: int
    G: int
    status: str

    initial_weight_bytes: int
    initial_KV_bytes: int
    peak_KV_bytes: int
    peak_prefill_workspace_bytes: int
    peak_decode_workspace_bytes: int
    peak_workspace_bytes: int
    peak_runtime_bytes: int
    capacity_bytes: int
    capacity_margin_bytes: int
    first_infeasible_step: int | None
    max_feasible_G: int | None

    prefill_time_s: float | None
    decode_compute_time_s: float | None
    host_transfer_time_s: float | None
    eviction_time_s: float | None
    admission_time_s: float | None
    total_service_time_s: float | None
    generated_tokens: int
    tokens_per_s: float | None

    host_to_local_bytes: int
    local_to_host_bytes: int
    host_resident_peak_bytes: int
    resident_fraction: float
    resident_prefill_requests: int
    resident_decode_requests: int

    known_energy_J: float
    known_energy_components: dict[str, float]
    unresolved_energy_terms: dict[str, int]
    energy_status: Literal["COMPLETE", "INCOMPLETE"]
    J_per_token: float | None
    tokens_per_J: float | None
    workspace_peak_stage: str
    workspace_provenance_status: str
    allocation_status: str
    horizon_sum_status: str
    thermal: None = None

    @model_validator(mode="after")
    def _result_closure(self) -> "PersistentMixedServiceResult":
        if self.BS != self.P + self.D:
            raise ValueError("result batch does not close")
        if self.total_service_time_s is not None:
            terms = (
                self.prefill_time_s, self.decode_compute_time_s,
                self.host_transfer_time_s, self.eviction_time_s)
            if any(value is None for value in terms):
                raise ValueError("service time requires all NO_OVERLAP terms")
            if not math.isclose(self.total_service_time_s,
                                sum(float(x) for x in terms), rel_tol=1e-12):
                raise ValueError("NO_OVERLAP service time does not close")
        if self.energy_status == "INCOMPLETE" and self.J_per_token is not None:
            raise ValueError("incomplete energy cannot report absolute J/token")
        return self


def _prefill_roofline(root: Path, model: DenseLLMModelSpec,
                      case: PersistentMixedServiceCase, bandwidth: float):
    platform = load_platform_spec_file(
        root / "configs/platform/gpu_package_h200_reference.yaml")
    compute = platform.gpu_compute_power
    prefill = platform.gpu_prefill_compute
    if compute is None or prefill is None:
        raise ValueError("GPU Prefill platform data is incomplete")
    metrics = evaluate_llm_prefill(model.prefill_input(
        batch_size=case.prefill_requests, prompt_length=case.context_length))
    calibration = resolve_gpu_prefill_compute_energy_calibration(compute, prefill)
    roofline = evaluate_gpu_prefill_roofline(
        metrics,
        peak_compute_flops_per_s=compute.peak_compute_BF16_dense_flops_per_s,
        large_gemm_effective_flops_per_s=prefill.large_gemm_effective_tflops*1e12,
        causal_attention_effective_flops_per_s=prefill.causal_attention_effective_tflops*1e12,
        sustained_memory_bandwidth_bytes_per_s=bandwidth,
        static_power_W=compute.static_power_W,
        peak_reference_dynamic_J_per_FLOP_min=calibration.peak_reference_dynamic_J_per_FLOP_min,
        peak_reference_dynamic_J_per_FLOP_max=calibration.peak_reference_dynamic_J_per_FLOP_max,
        nominal_gemm_dynamic_J_per_FLOP_min=calibration.nominal_gemm_dynamic_J_per_FLOP_min,
        nominal_gemm_dynamic_J_per_FLOP_max=calibration.nominal_gemm_dynamic_J_per_FLOP_max,
        nominal_attention_dynamic_J_per_FLOP_min=calibration.nominal_attention_dynamic_J_per_FLOP_min,
        nominal_attention_dynamic_J_per_FLOP_max=calibration.nominal_attention_dynamic_J_per_FLOP_max,
        compute_bound_total_power_W_min=calibration.compute_bound_total_power_W_min,
        compute_bound_total_power_W_max=calibration.compute_bound_total_power_W_max)
    return metrics, roofline, platform


def _m3d_gpu_bandwidth(root: Path) -> float:
    architecture = resolve_m3d_architecture_backend(root)
    platform = load_platform_spec_file(
        root / "configs/platform/gpu_package_h200_reference.yaml")
    gpu = platform.gpu_decode_power
    if gpu is None:
        raise ValueError("GPU Decode platform data is incomplete")
    latency = statistics.fmean(
        item.mat_latency_ns + item.miv_latency_ns + NMP_BANK_TO_LOCAL_ROUTE_DELAY_NS
        for item in architecture.physical_latency.locations)
    internal = resolve_internal_service_bandwidth(architecture.bandwidth, latency)
    boundary = resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=internal,
        memory_capability_bytes_per_s=architecture.bandwidth.coil_bandwidth_bytes_per_s,
        gpu_peak_bandwidth_bytes_per_s=gpu.peak_memory_bandwidth_bytes_per_s)
    service = resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=boundary.bandwidth_actual_bytes_per_s,
        service_status=platform.gpu_bandwidth_service.service_status,
        provenance=platform.gpu_bandwidth_service.provenance)
    return min(internal, service.sustained_bandwidth_bytes_per_s)


def _first_infeasible_step(
    *, capacity: int, weight: int, p: int, d: int, kv_at_s: int,
    kv_per_token: int, prefill_workspace: int, model: DenseLLMModelSpec,
    S: int, config: WorkspaceExecutionConfig, proposed: bool,
    nmp_die_count: int,
    page_size: int | None,
) -> tuple[int | None, int]:
    def physical(value: int) -> int:
        return value if page_size is None else (
            (value+page_size-1)//page_size)*page_size

    prefill_peak = physical(weight) + (p+d)*physical(kv_at_s) + physical(prefill_workspace)
    if prefill_peak > capacity:
        return 0, 0

    def fits(generated: int) -> bool:
        context = S + max(0, generated-1)
        workspace = evaluate_decode_workspace(
            model, batch_size=d, context_length=context, config=config,
            proposed_nmp=proposed, nmp_die_count=nmp_die_count).peak_bytes
        return (physical(weight) + p*physical(kv_at_s)
                + d*physical(kv_at_s+generated*kv_per_token)
                + physical(workspace) <= capacity)

    if not fits(1):
        return 0, 0
    low, high = 1, 1
    while fits(high):
        low, high = high, high*2
        if high >= 1 << 30:
            return None, low
    while low + 1 < high:
        middle = (low+high)//2
        if fits(middle):
            low = middle
        else:
            high = middle
    return high-1, low


def _gpu_decode_sum(
    model: DenseLLMModelSpec, *, batch: int, S: int, G: int,
    bandwidth: float, compute: float,
) -> tuple[float, tuple[float, float, float]]:
    evaluator = AnalyticalRooflineGPUModel(
        matched_payload_bandwidth_bits_per_second=8.0*bandwidth,
        effective_compute_flops_per_second=compute)
    time_s = read_bytes = write_bytes = total_bytes = 0.0
    for j in range(G):
        inp = model.decode_input(batch_size=batch, context_length=S+j)
        metrics = evaluate_llm_decode(inp)
        step = evaluator.evaluate(inp, batch_size=batch)
        time_s += step.decode_step_time_ms*1e-3
        read_bytes += batch*metrics.read_bytes_per_token
        write_bytes += batch*metrics.write_bytes_per_token
        total_bytes += batch*(metrics.read_bytes_per_token+metrics.write_bytes_per_token)
    return time_s, (read_bytes, write_bytes, total_bytes)


def _nmp_decode_sum(
    root: Path, model: DenseLLMModelSpec, *, batch: int, resident_batch: int,
    S: int, G: int,
) -> tuple[float, float]:
    stream_key = (str(root), model.model_id, batch, resident_batch, S)
    stream = _NMP_STREAM_CACHE.get(stream_key)
    if stream is None:
        initial = evaluate_nmp_decode_batch(
            model.decode_input(batch_size=batch, context_length=S),
            project_root=root, active_capacity_requests=resident_batch,
            resident_context_length=S)
        if initial.evaluation_status != "EVALUATED" or initial.execution_trace is None:
            raise RuntimeError("feasible horizon lacks initial NMP execution placement")
        stream = {
            "placement": initial.execution_trace.resident_placement,
            "points": {0: initial},
        }
        _NMP_STREAM_CACHE[stream_key] = stream
    placement = stream["placement"]
    cache = stream["points"]

    def point(j: int):
        if j not in cache:
            cache[j] = evaluate_nmp_decode_batch(
                model.decode_input(batch_size=batch, context_length=S+j),
                project_root=root, active_capacity_requests=resident_batch,
                resident_context_length=S,
                persistent_resident_placement=placement)
        result = cache[j]
        if result.evaluation_status != "EVALUATED":
            raise RuntimeError("NMP persistent execution unexpectedly became infeasible")
        return result

    def sum_field(field: str, lo: int, hi: int) -> float:
        if lo == hi:
            return float(getattr(point(lo), field))
        if hi-lo <= 2:
            return sum(float(getattr(point(index), field))
                       for index in range(lo, hi+1))
        left = float(getattr(point(lo), field))
        right = float(getattr(point(hi), field))
        mid = (lo+hi)//2
        middle = float(getattr(point(mid), field))
        n = hi-lo
        m = mid-lo
        # Fixed-placement integer sharding adds a small periodic ripple to the
        # smooth context curve.  A three-point quadratic integrates that curve
        # without rebuilding the full placement/power stack for every token.
        # Short-horizon regression compares this numerical path to the exact
        # token loop; this is a NUMERICAL_CHOICE, not a physical coefficient.
        curvature = ((right-left)/n-(middle-left)/m)/(n-m)
        slope = (middle-left)/m-curvature*m
        count = n+1
        sum_x = n*(n+1)/2.0
        sum_x2 = n*(n+1)*(2*n+1)/6.0
        return count*left+slope*sum_x+curvature*sum_x2

    return (
        sum_field("decode_step_time_ms", 0, G-1)*1e-3,
        sum_field("total_J_per_step", 0, G-1))


def evaluate_persistent_mixed_service_horizon(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    case: PersistentMixedServiceCase, system: SystemId,
    workspace_config: WorkspaceExecutionConfig,
) -> PersistentMixedServiceResult:
    """Evaluate one shared request specification under the frozen schedule."""
    if system == "CONVENTIONAL_HBM_GPU":
        raise ValueError(
            "Conventional overflow requires an explicit "
            "RESIDENT_ONLY_QUEUE_TO_FIT or HOST_KV_OFFLOAD policy")
    if model.model_id != case.model_id:
        raise ValueError("model and case model_id must match")
    root = Path(project_root).resolve()
    backend = resolve_conventional_hbm_backend(root)
    architecture = resolve_m3d_architecture_backend(root)
    platform = load_platform_spec_file(
        root / "configs/platform/gpu_package_h200_reference.yaml")
    gpu = platform.gpu_decode_power
    compute = platform.gpu_compute_power
    host = backend.host_offload
    if gpu is None or compute is None or host.effective_bandwidth_bytes_per_second is None:
        raise ValueError("canonical platform data is incomplete")
    proposed = system == "IOM3D_FEOL_NMP"
    capacity = int(architecture.layout.total_capacity_bytes)
    metrics = evaluate_llm_decode(model.decode_input(
        batch_size=case.batch_size, context_length=case.context_length))
    weight = int(metrics.weight_footprint_bytes + metrics.runtime_fixed_bytes)
    kv_s = int(metrics.kv_bytes_per_request)
    kv_token = int(metrics.kv_write_bytes_per_token)
    initial_kv = case.batch_size*kv_s
    peak_kv = case.prefill_requests*kv_s + case.decode_requests*(
        kv_s + case.generated_decode_steps*kv_token)
    prefill_ws = evaluate_prefill_workspace(
        model, batch_size=case.prefill_requests,
        context_length=case.context_length, config=workspace_config)
    decode_ws = evaluate_decode_workspace(
        model, batch_size=case.decode_requests,
        context_length=case.context_length+case.generated_decode_steps-1,
        config=workspace_config, proposed_nmp=proposed,
        nmp_die_count=architecture.layout.slab_count)
    peak_workspace = max(prefill_ws.peak_bytes, decode_ws.peak_bytes)
    page_size = (None if system == "CONVENTIONAL_HBM_GPU"
                 else architecture.layout.slot_capacity_bytes)
    def physical(value: int) -> int:
        return value if page_size is None else (
            (value+page_size-1)//page_size)*page_size
    prefill_runtime = (physical(weight) + case.batch_size*physical(kv_s)
                       + physical(prefill_ws.peak_bytes))
    decode_runtime = (
        physical(weight) + case.prefill_requests*physical(kv_s)
        + case.decode_requests*physical(
            kv_s+case.generated_decode_steps*kv_token)
        + physical(decode_ws.peak_bytes))
    peak_runtime = max(prefill_runtime, decode_runtime)
    first_bad, max_g = _first_infeasible_step(
        capacity=capacity, weight=weight, p=case.prefill_requests,
        d=case.decode_requests, kv_at_s=kv_s, kv_per_token=kv_token,
        prefill_workspace=prefill_ws.peak_bytes, model=model, S=case.context_length,
        config=workspace_config, proposed=proposed,
        nmp_die_count=architecture.layout.slab_count, page_size=page_size)
    all_local_feasible = peak_runtime <= capacity

    # Decode-first residency uses high-water bytes and reserves workspace.
    available = max(0, capacity-weight-peak_workspace)
    resident_d = min(case.decode_requests, available//(kv_s+case.generated_decode_steps*kv_token))
    available -= resident_d*(kv_s+case.generated_decode_steps*kv_token)
    resident_p = min(case.prefill_requests, available//kv_s)
    if system != "CONVENTIONAL_HBM_GPU":
        resident_d = case.decode_requests if all_local_feasible else 0
        resident_p = case.prefill_requests if all_local_feasible else 0
    spilled_d = case.decode_requests-resident_d
    spilled_p = case.prefill_requests-resident_p
    host_to_local = spilled_d*kv_s if system == "CONVENTIONAL_HBM_GPU" else 0
    local_to_host = spilled_p*kv_s if system == "CONVENTIONAL_HBM_GPU" else 0
    host_peak = (spilled_d+spilled_p)*kv_s
    host_bw = host.effective_bandwidth_bytes_per_second
    admission_s = host_to_local/host_bw
    prefill_host_s = local_to_host/host_bw

    prefill_metrics, roofline, _ = _prefill_roofline(
        root, model, case, backend.sustained_bandwidth_bytes_per_s)
    prefill_compute_s = roofline.nominal_prefill_latency_s
    prefill_s = prefill_compute_s
    status = "EVALUATED"
    decode_s: float | None = None
    nmp_energy = 0.0
    gpu_traffic = (0.0, 0.0, 0.0)
    if system != "CONVENTIONAL_HBM_GPU" and not all_local_feasible:
        status = "CAPACITY_INFEASIBLE"
        prefill_s = decode_s = None
    elif system == "CONVENTIONAL_HBM_GPU":
        if resident_d <= 0:
            status = "CAPACITY_INFEASIBLE"
            prefill_s = None
        else:
            decode_s = 0.0
            traffic = [0.0, 0.0, 0.0]
            remaining = case.decode_requests
            while remaining:
                wave = min(resident_d, remaining)
                wave_s, wave_traffic = _gpu_decode_sum(
                    model, batch=wave, S=case.context_length,
                    G=case.generated_decode_steps,
                    bandwidth=backend.sustained_bandwidth_bytes_per_s,
                    compute=compute.peak_compute_BF16_dense_flops_per_s)
                decode_s += wave_s
                traffic = [left+right for left, right in zip(
                    traffic, wave_traffic, strict=True)]
                remaining -= wave
            gpu_traffic = tuple(traffic)
    elif system == "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY":
        decode_s, gpu_traffic = _gpu_decode_sum(
            model, batch=case.decode_requests, S=case.context_length,
            G=case.generated_decode_steps, bandwidth=_m3d_gpu_bandwidth(root),
            compute=compute.peak_compute_BF16_dense_flops_per_s)
    else:
        decode_s, nmp_energy = _nmp_decode_sum(
            root, model, batch=case.decode_requests,
            resident_batch=case.batch_size, S=case.context_length,
            G=case.generated_decode_steps)

    generated = case.decode_requests*case.generated_decode_steps
    total_s = (None if decode_s is None or prefill_s is None else
               prefill_s+decode_s+prefill_host_s+admission_s)
    known: dict[str, float] = {}
    unresolved: dict[str, int] = {
        "PREFILL_GPU_DYNAMIC_RANGE_NO_SINGLE_NOMINAL": 0}
    if total_s is not None:
        known["gpu_static_J"] = compute.static_power_W*total_s
        host_bits = 8.0*(host_to_local+local_to_host)
        known["host_link_dynamic_J"] = host_bits*float(
            host.host_link_dynamic_J_per_bit or 0.0)
        known["host_memory_dynamic_J"] = host_bits*float(
            host.host_memory_dynamic_J_per_bit or 0.0)
        if system == "CONVENTIONAL_HBM_GPU":
            read, write, total = gpu_traffic
            known["decode_gpu_dynamic_J"] = 8.0*total*gpu.e_decode_J_per_bit
            known["hbm_read_dynamic_J"] = 8.0*(
                prefill_metrics.prefill_read_bytes+read)*backend.read_energy_pJ_per_bit*1e-12
            known["hbm_refresh_J"] = backend.refresh_power_W*total_s
            unresolved["CONVENTIONAL_HBM_WRITE"] = int(
                prefill_metrics.prefill_write_bytes+write)
        elif system == "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY":
            read, write, total = gpu_traffic
            write_pj = resolve_orthogonal_m3d_write_energy_pj_per_bit(
                architecture.case, architecture.memory)
            known["decode_gpu_dynamic_J"] = 8.0*total*gpu.e_decode_J_per_bit
            known["m3d_read_dynamic_J"] = 8.0*(
                prefill_metrics.prefill_read_bytes+read)*architecture.memory.E_access_total_pj_bit*1e-12
            known["m3d_write_dynamic_J"] = 8.0*(
                prefill_metrics.prefill_write_bytes+write)*write_pj*1e-12
            known["m3d_refresh_J"] = float(architecture.memory.P_refresh_W or 0.0)*total_s
        else:
            known["nmp_decode_total_J"] = nmp_energy
            write_pj = resolve_orthogonal_m3d_write_energy_pj_per_bit(
                architecture.case, architecture.memory)
            known["prefill_m3d_read_J"] = 8.0*prefill_metrics.prefill_read_bytes*architecture.memory.E_access_total_pj_bit*1e-12
            known["prefill_m3d_write_J"] = 8.0*prefill_metrics.prefill_write_bytes*write_pj*1e-12
    known_total = sum(known.values())
    energy_status: Literal["COMPLETE", "INCOMPLETE"] = "INCOMPLETE"
    capacity_margin = capacity-peak_runtime
    return PersistentMixedServiceResult(
        model=model.model_id, system=system, S=case.context_length,
        P=case.prefill_requests, D=case.decode_requests, BS=case.batch_size,
        G=case.generated_decode_steps, status=status,
        initial_weight_bytes=weight, initial_KV_bytes=initial_kv,
        peak_KV_bytes=peak_kv,
        peak_prefill_workspace_bytes=prefill_ws.peak_bytes,
        peak_decode_workspace_bytes=decode_ws.peak_bytes,
        peak_workspace_bytes=peak_workspace, peak_runtime_bytes=peak_runtime,
        capacity_bytes=capacity, capacity_margin_bytes=capacity_margin,
        first_infeasible_step=(first_bad if first_bad is not None and
                               first_bad < case.generated_decode_steps else None),
        max_feasible_G=max_g,
        prefill_time_s=prefill_s, decode_compute_time_s=decode_s,
        host_transfer_time_s=(None if total_s is None else prefill_host_s+admission_s),
        eviction_time_s=(None if total_s is None else 0.0),
        admission_time_s=(None if total_s is None else admission_s),
        total_service_time_s=total_s, generated_tokens=generated,
        tokens_per_s=(None if total_s is None else generated/total_s),
        host_to_local_bytes=host_to_local, local_to_host_bytes=local_to_host,
        host_resident_peak_bytes=host_peak,
        resident_fraction=(resident_d+resident_p)/case.batch_size,
        resident_prefill_requests=int(resident_p),
        resident_decode_requests=int(resident_d), known_energy_J=known_total,
        known_energy_components=known, unresolved_energy_terms=unresolved,
        energy_status=energy_status, J_per_token=None, tokens_per_J=None,
        workspace_peak_stage=(prefill_ws.peak_stage if prefill_ws.peak_bytes >= decode_ws.peak_bytes
                              else decode_ws.peak_stage),
        workspace_provenance_status=workspace_config.provenance_status,
        allocation_status="DETERMINISTIC_PAGE_SLOT_AND_WHOLE_VECTOR_OWNER_CLOSED",
        horizon_sum_status=(
            "THREE_POINT_QUADRATIC_NUMERICAL_SUM__SHORT_G_EXACT_REGRESSION"
            if proposed else "EXACT_PER_TOKEN_SUM"))


class PersistentHorizonComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    baseline_system: SystemId
    system: SystemId
    status: str
    speedup: float | None
    feasibility_advantage: bool


def compare_persistent_horizons(
    baseline: PersistentMixedServiceResult,
    candidate: PersistentMixedServiceResult,
) -> PersistentHorizonComparison:
    left = EvaluationSemantics(
        evaluation_scope=baseline.scope, context_evolution=baseline.context_evolution,
        initial_state_requirement=f"{baseline.model}:{baseline.S}:{baseline.P}:{baseline.D}",
        final_state_requirement="D_REQUESTS_GENERATE_G_THEN_FREE__PREFILL_KV_RETAINED_THROUGH_HORIZON",
        generated_decode_steps=baseline.G, schedule_policy=baseline.schedule_policy)
    right = EvaluationSemantics(
        evaluation_scope=candidate.scope, context_evolution=candidate.context_evolution,
        initial_state_requirement=f"{candidate.model}:{candidate.S}:{candidate.P}:{candidate.D}",
        final_state_requirement="D_REQUESTS_GENERATE_G_THEN_FREE__PREFILL_KV_RETAINED_THROUGH_HORIZON",
        generated_decode_steps=candidate.G, schedule_policy=candidate.schedule_policy)
    require_comparable_semantics(left, right)
    if baseline.total_service_time_s is None and candidate.total_service_time_s is None:
        status = "BOTH_INFEASIBLE"
    elif baseline.total_service_time_s is None:
        status = "BASELINE_CAPACITY_INFEASIBLE"
    elif candidate.total_service_time_s is None:
        status = "PROPOSED_CAPACITY_INFEASIBLE"
    else:
        status = "COMPARABLE"
    speedup = (None if status != "COMPARABLE" else
               baseline.total_service_time_s/candidate.total_service_time_s)
    return PersistentHorizonComparison(
        baseline_system=baseline.system, system=candidate.system,
        status=status, speedup=speedup,
        feasibility_advantage=(baseline.total_service_time_s is None
                               and candidate.total_service_time_s is not None))
