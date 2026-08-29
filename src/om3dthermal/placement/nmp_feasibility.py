"""Minimal analytical NMP feasibility model for resident MoE experts.

Only the expert MLP executes on the M3D side.  Existing Mixtral analytical
traffic/FLOPs, P0/P1/P2 placement bandwidths, coil bandwidth, and GPU-only
timings are inputs; no NMP microarchitecture is implied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

from om3dthermal.workload.moe_decode import MoEDecodeInput, evaluate_moe_decode
from om3dthermal.workload.moe_published_page_demand import (
    MoEPublishedPageDemand,
)

from .moe_published_e2e import MoEHierarchicalPlacementE2ECase
from .serving_e2e import HierarchicalPlacementServingTimingResult


NMPBottleneck = Literal["MEMORY", "COMPUTE", "BALANCED"]


@dataclass(frozen=True)
class NMPWorkloadClosure:
    model_id: str
    batch_size: int
    num_layers: int
    hidden_size: int
    top_k: int
    dtype: str
    dtype_bytes: int
    expert_weight_bytes_per_decode_step: float
    expert_weight_bytes_per_token: float
    expert_flops_per_token: int
    expert_flops_per_decode_step: int
    total_flops_per_token: int
    gpu_nonexpert_flops_per_token: int
    gpu_nonexpert_flops_per_decode_step: int
    activation_bytes_per_token: int
    activation_bytes_per_decode_step: int
    shared_weight_bytes_per_decode_step: float
    kv_read_bytes_per_decode_step: float
    kv_write_bytes_per_decode_step: float
    gpu_remaining_memory_bytes_per_decode_step: float
    expert_arithmetic_intensity_flop_per_byte: float
    expert_weight_reuse_semantics: str
    activation_traffic_semantics: str
    workload_source: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GPUOnlyPlacementBaseline:
    strategy: str
    expert_weight_bytes_crossing_coil_per_decode_step: float
    activation_bytes_crossing_coil_per_decode_step: float
    remaining_gpu_memory_bytes_crossing_coil_per_decode_step: float
    total_bytes_crossing_coil_per_decode_step: float
    expert_weight_coil_transfer_time_ms: float
    gpu_expert_compute_time_ms: float
    current_total_step_time_ms: float
    current_tokens_per_s: float
    baseline_source: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NMPPlacementPoint:
    strategy: str
    internal_bandwidth_bytes_per_s: float
    coil_bandwidth_bytes_per_s: float
    remaining_gpu_memory_effective_bandwidth_bytes_per_s: float
    expert_balance_tflops: float
    expert_memory_time_ms: float
    expert_compute_time_ms: float
    expert_nmp_time_ms: float
    nmp_expert_bottleneck: NMPBottleneck
    gpu_nonexpert_compute_time_ms: float
    gpu_remaining_memory_time_ms: float
    activation_coil_time_ms: float
    serial_step_time_ms: float
    ideal_overlap_upper_bound_step_time_ms: float
    tokens_per_s: float
    expert_weight_bytes_crossing_coil_per_decode_step: float
    activation_bytes_crossing_coil_per_decode_step: float
    remaining_gpu_memory_bytes_crossing_coil_per_decode_step: float
    total_bytes_crossing_coil_per_decode_step: float
    expert_weights_residency: str
    system_model: str
    result_status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NMPFeasibilityPoint:
    effective_nmp_tflops: float
    effective_nmp_flops_per_second: float
    nmp_parameter_classification: str
    nmp_parameter_status: str
    workload: NMPWorkloadClosure
    gpu_only_p0: GPUOnlyPlacementBaseline
    gpu_only_p1: GPUOnlyPlacementBaseline
    gpu_only_p2: GPUOnlyPlacementBaseline
    p0: NMPPlacementPoint
    p1: NMPPlacementPoint
    p2: NMPPlacementPoint
    p1_over_p0_throughput_gain: float
    p2_over_p1_throughput_gain: float
    p2_over_p0_throughput_gain: float
    p0_speedup_over_gpu_only: float
    p1_speedup_over_gpu_only: float
    p2_speedup_over_gpu_only: float
    expert_to_activation_traffic_reduction_ratio: float
    p0_total_coil_traffic_reduction_ratio: float
    internal_bandwidth_scale: float

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["workload"] = self.workload.as_dict()
        for field in ("gpu_only_p0", "gpu_only_p1", "gpu_only_p2"):
            value[field] = getattr(self, field).as_dict()
        for field in ("p0", "p1", "p2"):
            value[field] = getattr(self, field).as_dict()
        return value


@dataclass(frozen=True)
class NMPSweepSummary:
    batch_size: int
    points: tuple[NMPFeasibilityPoint, ...]
    minimum_useful_nmp_tflops: float | None
    memory_saturating_nmp_tflops: float | None
    recommendation: str
    minimum_useful_rule: str
    memory_saturating_rule: str

    def as_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "points": [point.as_dict() for point in self.points],
            "minimum_useful_nmp_tflops": self.minimum_useful_nmp_tflops,
            "memory_saturating_nmp_tflops": self.memory_saturating_nmp_tflops,
            "recommendation": self.recommendation,
            "minimum_useful_rule": self.minimum_useful_rule,
            "memory_saturating_rule": self.memory_saturating_rule,
        }


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _dtype_bytes(workload: MoEDecodeInput) -> int:
    if workload.dtype != "BF16" or workload.weight_bits != 16:
        raise ValueError("NMP activation model currently requires canonical BF16")
    return workload.weight_bits // 8


def _build_workload_closure(
        workload: MoEDecodeInput, demand: MoEPublishedPageDemand,
        ) -> NMPWorkloadClosure:
    metrics = evaluate_moe_decode(workload)
    if demand.requested_requests != workload.batch_size:
        raise ValueError("NMP workload and demand batch sizes differ")
    if not math.isclose(
            demand.total_expert_read_bytes_per_decode_step,
            metrics.active_expert_weight_bytes_per_decode_step,
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("NMP expert traffic must reuse existing MoE demand")
    if not math.isclose(
            demand.total_shared_weight_read_bytes_per_decode_step,
            metrics.active_nonexpert_weight_bytes_per_decode_step,
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("NMP shared traffic must reuse existing MoE demand")
    dtype_bytes = _dtype_bytes(workload)
    hidden_bytes = workload.hidden_size * dtype_bytes
    activation_per_token = (
        2 * workload.num_experts_per_tok * hidden_bytes
        * workload.num_hidden_layers)
    expert_flops_step = (
        workload.batch_size * metrics.active_expert_flops_per_token)
    nonexpert_flops_token = (
        metrics.flops_per_token - metrics.active_expert_flops_per_token)
    if nonexpert_flops_token <= 0:
        raise ValueError("GPU non-expert FLOPs must remain positive")
    expert_bytes_step = demand.total_expert_read_bytes_per_decode_step
    remaining = (
        demand.total_shared_weight_read_bytes_per_decode_step
        + demand.total_kv_read_bytes_per_decode_step
        + demand.kv_write_bytes_per_decode_step)
    return NMPWorkloadClosure(
        model_id=workload.model_id,
        batch_size=workload.batch_size,
        num_layers=workload.num_hidden_layers,
        hidden_size=workload.hidden_size,
        top_k=workload.num_experts_per_tok,
        dtype=workload.dtype,
        dtype_bytes=dtype_bytes,
        expert_weight_bytes_per_decode_step=expert_bytes_step,
        expert_weight_bytes_per_token=(
            expert_bytes_step / workload.batch_size),
        expert_flops_per_token=metrics.active_expert_flops_per_token,
        expert_flops_per_decode_step=expert_flops_step,
        total_flops_per_token=metrics.flops_per_token,
        gpu_nonexpert_flops_per_token=nonexpert_flops_token,
        gpu_nonexpert_flops_per_decode_step=(
            workload.batch_size * nonexpert_flops_token),
        activation_bytes_per_token=activation_per_token,
        activation_bytes_per_decode_step=(
            workload.batch_size * activation_per_token),
        shared_weight_bytes_per_decode_step=(
            demand.total_shared_weight_read_bytes_per_decode_step),
        kv_read_bytes_per_decode_step=(
            demand.total_kv_read_bytes_per_decode_step),
        kv_write_bytes_per_decode_step=demand.kv_write_bytes_per_decode_step,
        gpu_remaining_memory_bytes_per_decode_step=remaining,
        expert_arithmetic_intensity_flop_per_byte=(
            expert_flops_step / expert_bytes_step),
        expert_weight_reuse_semantics=metrics.weight_reuse_status,
        activation_traffic_semantics=(
            "CONSERVATIVE_NO_ACTIVATION_AGGREGATION_MODEL__"
            "TWO_TOP_K_INPUT_OUTPUT_HIDDEN_VECTORS_PER_LAYER"),
        workload_source="EXISTING_MIXTRAL_ANALYTICAL_MODEL_AND_PAGE_DEMAND",
    )


def _strategy_timings(
        gpu_only: MoEHierarchicalPlacementE2ECase,
        ) -> tuple[tuple[str, HierarchicalPlacementServingTimingResult], ...]:
    return (
        ("P0_LATENCY_OBLIVIOUS_RANDOM_MEAN", gpu_only.random_timing),
        ("P1_FAST_REGION_ONLY", gpu_only.fast_region_timing),
        ("P2_POPULARITY_AWARE_FAST_REGION",
         gpu_only.popularity_aware_timing),
    )


def _gpu_baseline(
        *, strategy: str, timing: HierarchicalPlacementServingTimingResult,
        workload: NMPWorkloadClosure, gpu_compute_flops_per_second: float,
        ) -> GPUOnlyPlacementBaseline:
    expert_bytes = workload.expert_weight_bytes_per_decode_step
    remaining = workload.gpu_remaining_memory_bytes_per_decode_step
    coil = timing.bandwidth.coil_bandwidth_bytes_per_s
    return GPUOnlyPlacementBaseline(
        strategy=strategy,
        expert_weight_bytes_crossing_coil_per_decode_step=expert_bytes,
        activation_bytes_crossing_coil_per_decode_step=0.0,
        remaining_gpu_memory_bytes_crossing_coil_per_decode_step=remaining,
        total_bytes_crossing_coil_per_decode_step=expert_bytes + remaining,
        expert_weight_coil_transfer_time_ms=expert_bytes / coil * 1e3,
        gpu_expert_compute_time_ms=(
            workload.expert_flops_per_decode_step
            / gpu_compute_flops_per_second * 1e3),
        current_total_step_time_ms=timing.total_step_time_ms,
        current_tokens_per_s=timing.aggregate_tokens_per_s,
        baseline_source="EXISTING_HIERARCHICAL_MIXTRAL_GPU_ONLY_PATHWAY",
    )


def _classify(memory_s: float, compute_s: float) -> NMPBottleneck:
    if math.isclose(memory_s, compute_s, rel_tol=1e-12, abs_tol=0.0):
        return "BALANCED"
    return "MEMORY" if memory_s > compute_s else "COMPUTE"


def _nmp_point(
        *, strategy: str, timing: HierarchicalPlacementServingTimingResult,
        workload: NMPWorkloadClosure, nmp_flops_per_second: float,
        gpu_compute_flops_per_second: float, internal_bandwidth_scale: float,
        ) -> NMPPlacementPoint:
    internal = (
        timing.bandwidth.internal_bandwidth_bytes_per_s
        * internal_bandwidth_scale)
    coil = timing.bandwidth.coil_bandwidth_bytes_per_s
    remaining_effective = min(internal, coil)
    expert_memory_s = (
        workload.expert_weight_bytes_per_decode_step / internal)
    expert_compute_s = (
        workload.expert_flops_per_decode_step / nmp_flops_per_second)
    expert_s = max(expert_memory_s, expert_compute_s)
    gpu_nonexpert_s = (
        workload.gpu_nonexpert_flops_per_decode_step
        / gpu_compute_flops_per_second)
    activation_s = workload.activation_bytes_per_decode_step / coil
    remaining_memory_s = (
        workload.gpu_remaining_memory_bytes_per_decode_step
        / remaining_effective
        + timing.startup_step_time_ms * 1e-3)
    serial_s = (
        gpu_nonexpert_s + remaining_memory_s + activation_s + expert_s)
    ideal_s = max(
        gpu_nonexpert_s + remaining_memory_s,
        activation_s + expert_s,
    )
    return NMPPlacementPoint(
        strategy=strategy,
        internal_bandwidth_bytes_per_s=internal,
        coil_bandwidth_bytes_per_s=coil,
        remaining_gpu_memory_effective_bandwidth_bytes_per_s=(
            remaining_effective),
        expert_balance_tflops=(
            internal
            * workload.expert_arithmetic_intensity_flop_per_byte / 1e12),
        expert_memory_time_ms=expert_memory_s * 1e3,
        expert_compute_time_ms=expert_compute_s * 1e3,
        expert_nmp_time_ms=expert_s * 1e3,
        nmp_expert_bottleneck=_classify(expert_memory_s, expert_compute_s),
        gpu_nonexpert_compute_time_ms=gpu_nonexpert_s * 1e3,
        gpu_remaining_memory_time_ms=remaining_memory_s * 1e3,
        activation_coil_time_ms=activation_s * 1e3,
        serial_step_time_ms=serial_s * 1e3,
        ideal_overlap_upper_bound_step_time_ms=ideal_s * 1e3,
        tokens_per_s=workload.batch_size / serial_s,
        expert_weight_bytes_crossing_coil_per_decode_step=0.0,
        activation_bytes_crossing_coil_per_decode_step=(
            workload.activation_bytes_per_decode_step),
        remaining_gpu_memory_bytes_crossing_coil_per_decode_step=(
            workload.gpu_remaining_memory_bytes_per_decode_step),
        total_bytes_crossing_coil_per_decode_step=(
            workload.activation_bytes_per_decode_step
            + workload.gpu_remaining_memory_bytes_per_decode_step),
        expert_weights_residency="RESIDENT_IN_M3D_DO_NOT_CROSS_COIL",
        system_model=(
            "SERIAL_GPU_NMP_FIRST_ORDER_MODEL_WITH_REMAINING_GPU_MEMORY_"
            "TRAFFIC_PRESERVED"),
        result_status="NMP_RESULTS_CONDITIONAL_ON_CURRENT_INTERNAL_BW_MODEL",
    )


def evaluate_nmp_feasibility(
        workload: MoEDecodeInput, demand: MoEPublishedPageDemand,
        gpu_only: MoEHierarchicalPlacementE2ECase, *,
        effective_gpu_compute_flops_per_second: float,
        effective_nmp_tflops: float,
        internal_bandwidth_scale: float = 1.0,
        ) -> NMPFeasibilityPoint:
    """Evaluate one sustained-NMP-throughput point for P0/P1/P2."""
    gpu_rate = _positive(
        effective_gpu_compute_flops_per_second,
        "effective_gpu_compute_flops_per_second")
    nmp_tflops = _positive(effective_nmp_tflops, "effective_nmp_tflops")
    bandwidth_scale = _positive(
        internal_bandwidth_scale, "internal_bandwidth_scale")
    nmp_rate = nmp_tflops * 1e12
    closure = _build_workload_closure(workload, demand)
    timings = _strategy_timings(gpu_only)
    baselines = tuple(
        _gpu_baseline(
            strategy=strategy, timing=timing, workload=closure,
            gpu_compute_flops_per_second=gpu_rate)
        for strategy, timing in timings)
    points = tuple(
        _nmp_point(
            strategy=strategy, timing=timing, workload=closure,
            nmp_flops_per_second=nmp_rate,
            gpu_compute_flops_per_second=gpu_rate,
            internal_bandwidth_scale=bandwidth_scale)
        for strategy, timing in timings)
    p0, p1, p2 = points
    b0, b1, b2 = baselines
    return NMPFeasibilityPoint(
        effective_nmp_tflops=nmp_tflops,
        effective_nmp_flops_per_second=nmp_rate,
        nmp_parameter_classification="MODELING_CHOICE",
        nmp_parameter_status="NOT_HARDWARE_VALIDATED",
        workload=closure,
        gpu_only_p0=b0, gpu_only_p1=b1, gpu_only_p2=b2,
        p0=p0, p1=p1, p2=p2,
        p1_over_p0_throughput_gain=p1.tokens_per_s / p0.tokens_per_s - 1.0,
        p2_over_p1_throughput_gain=p2.tokens_per_s / p1.tokens_per_s - 1.0,
        p2_over_p0_throughput_gain=p2.tokens_per_s / p0.tokens_per_s - 1.0,
        p0_speedup_over_gpu_only=p0.tokens_per_s / b0.current_tokens_per_s,
        p1_speedup_over_gpu_only=p1.tokens_per_s / b1.current_tokens_per_s,
        p2_speedup_over_gpu_only=p2.tokens_per_s / b2.current_tokens_per_s,
        expert_to_activation_traffic_reduction_ratio=(
            closure.expert_weight_bytes_per_decode_step
            / closure.activation_bytes_per_decode_step),
        p0_total_coil_traffic_reduction_ratio=(
            b0.total_bytes_crossing_coil_per_decode_step
            / p0.total_bytes_crossing_coil_per_decode_step),
        internal_bandwidth_scale=bandwidth_scale,
    )


def sweep_nmp_feasibility(
        workload: MoEDecodeInput, demand: MoEPublishedPageDemand,
        gpu_only: MoEHierarchicalPlacementE2ECase, *,
        effective_gpu_compute_flops_per_second: float,
        effective_nmp_tflops_values: tuple[float, ...] = (
            8.0, 16.0, 32.0, 64.0, 128.0),
        internal_bandwidth_scale: float = 1.0,
        ) -> NMPSweepSummary:
    """Run a deterministic sustained-throughput sweep and identify regimes."""
    if (not effective_nmp_tflops_values
            or tuple(sorted(set(effective_nmp_tflops_values)))
            != effective_nmp_tflops_values):
        raise ValueError("NMP TFLOPS sweep values must be unique and increasing")
    points = tuple(
        evaluate_nmp_feasibility(
            workload, demand, gpu_only,
            effective_gpu_compute_flops_per_second=(
                effective_gpu_compute_flops_per_second),
            effective_nmp_tflops=value,
            internal_bandwidth_scale=internal_bandwidth_scale,
        )
        for value in effective_nmp_tflops_values)
    minimum_useful = next((
        point.effective_nmp_tflops for point in points
        if any(item.nmp_expert_bottleneck != "COMPUTE"
               for item in (point.p0, point.p1, point.p2))), None)
    saturating = next((
        point.effective_nmp_tflops for point in points
        if all(item.nmp_expert_bottleneck == "MEMORY"
               for item in (point.p0, point.p1, point.p2))), None)
    upper = points[-1].effective_nmp_tflops
    recommendation = (
        f"{saturating:g} TFLOP/s" if saturating is not None
        else f">{upper:g} TFLOP/s")
    return NMPSweepSummary(
        batch_size=workload.batch_size,
        points=points,
        minimum_useful_nmp_tflops=minimum_useful,
        memory_saturating_nmp_tflops=saturating,
        recommendation=recommendation,
        minimum_useful_rule=(
            "FIRST_SWEEP_POINT_WHERE_ANY_PLACEMENT_EXPERT_STAGE_IS_NOT_"
            "COMPUTE_BOUND"),
        memory_saturating_rule=(
            "FIRST_SWEEP_POINT_WHERE_ALL_P0_P1_P2_EXPERT_STAGES_ARE_"
            "MEMORY_BOUND"),
    )
