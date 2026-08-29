"""Propagate M3D placement latency through the existing decode roofline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from om3dthermal.evaluation import ArchitectureCapacityFeasibility
from om3dthermal.evaluator import evaluate_llm_decode_performance
from om3dthermal.power.physical_capacity import PhysicalCapacityLayout
from om3dthermal.workload import LLMDecodeInput, evaluate_llm_decode
from om3dthermal.workload.m3d_page_demand import M3DWorkloadPageDemand

from .fast_region import (
    FastRegionOccupancyPoint,
    FastRegionWorkloadComparison,
)


@dataclass(frozen=True)
class PlacementServingTimingResult:
    strategy: str
    requested_requests: int
    physical_access_latency_avg_ns: float
    physical_access_latency_max_ns: float
    read_page_equivalents_per_decode_step: float
    memory_access_latency_step_time_ms: float
    memory_bandwidth_step_time_ms: float
    memory_stage_step_time_ms: float
    compute_step_time_ms: float
    gpu_resource_step_time_ms: float
    host_penalty_step_time_ms: float
    other_step_time_ms: float
    total_step_time_ms: float
    inter_token_latency_ms: float
    aggregate_tokens_per_s: float
    physical_latency_fraction_of_total: float
    memory_stage_fraction_of_total: float
    bottleneck: str
    access_count_semantics: str
    overlap_semantics: str
    double_counting_status: str


@dataclass(frozen=True)
class PlacementServingComparison:
    fast_pack: PlacementServingTimingResult
    conventional: PlacementServingTimingResult
    random_mean: PlacementServingTimingResult
    physical_latency_gain_vs_conventional: float
    memory_stage_gain_vs_conventional: float
    end_to_end_latency_gain_vs_conventional: float
    tokens_per_s_gain_vs_conventional: float
    physical_latency_gain_vs_random: float
    end_to_end_latency_gain_vs_random: float
    tokens_per_s_gain_vs_random: float


@dataclass(frozen=True)
class OccupancyServingClosurePoint:
    occupancy_fraction: float
    fast_physical_latency_ns: float
    random_physical_latency_ns: float
    physical_latency_gain: float
    fast_total_step_time_ms: float
    random_total_step_time_ms: float
    end_to_end_latency_gain: float
    fast_tokens_per_s: float
    random_tokens_per_s: float
    tokens_per_s_gain: float


def evaluate_placement_serving_timing(
    workload: LLMDecodeInput,
    demand: M3DWorkloadPageDemand,
    physical_layout: PhysicalCapacityLayout,
    *,
    strategy: str,
    physical_access_latency_avg_ns: float,
    physical_access_latency_max_ns: float,
    matched_payload_bandwidth_bits_per_second: float,
    effective_compute_flops_per_second: float,
    host_penalty_step_time_ms: float = 0.0,
) -> PlacementServingTimingResult:
    """Add page-scan startup latency inside the existing memory boundary."""
    if workload.batch_size != demand.requested_requests:
        raise ValueError("workload batch and page demand request count differ")
    average_latency = _finite_nonnegative(
        physical_access_latency_avg_ns, "physical_access_latency_avg_ns")
    maximum_latency = _finite_nonnegative(
        physical_access_latency_max_ns, "physical_access_latency_max_ns")
    if maximum_latency < average_latency:
        raise ValueError("maximum physical latency cannot be below average")
    host_penalty_ms = _finite_nonnegative(
        host_penalty_step_time_ms, "host_penalty_step_time_ms")
    metrics = evaluate_llm_decode(workload)
    page_size = demand.page_layout.page_size_bytes
    if page_size <= 0:
        raise ValueError("page size must be positive")
    page_equivalents = (
        demand.total_read_bytes_per_decode_step / page_size)
    access_latency_step_s = page_equivalents * average_latency * 1e-9
    access_latency_per_token_equivalent_s = (
        access_latency_step_s / workload.batch_size)
    required = metrics.required_capacity_bytes
    capacity = ArchitectureCapacityFeasibility(
        architecture="orthogonal_m3d_igzo_placement_timing",
        physical_capacity_bytes=physical_layout.total_capacity_bytes,
        physical_capacity_GiB=physical_layout.total_capacity_gib,
        reserved_capacity_bytes=0,
        usable_capacity_bytes=physical_layout.total_capacity_bytes,
        required_capacity_bytes=required,
        capacity_margin_bytes=physical_layout.total_capacity_bytes - required,
        capacity_utilization=required / physical_layout.total_capacity_bytes,
        capacity_feasible=True,
        capacity_scope_status="M3D_ONLY_PAGE_CAPACITY_ALREADY_VALIDATED",
        capacity_source_status=physical_layout.capacity_source_status,
    )
    performance = evaluate_llm_decode_performance(
        metrics,
        capacity,
        batch_size=workload.batch_size,
        matched_payload_bandwidth_bits_per_second=(
            matched_payload_bandwidth_bits_per_second),
        effective_compute_flops_per_second=(
            effective_compute_flops_per_second),
        physical_access_latency_time_per_token_equivalent_s=(
            access_latency_per_token_equivalent_s),
    )
    assert performance.memory_bandwidth_time_per_token_equivalent_s is not None
    assert performance.physical_access_latency_time_per_token_equivalent_s is not None
    assert performance.memory_time_per_token_equivalent_s is not None
    assert performance.compute_time_per_token_equivalent_s is not None
    assert performance.aggregate_step_time_s is not None
    batch = workload.batch_size
    bandwidth_step_ms = (
        batch
        * performance.memory_bandwidth_time_per_token_equivalent_s
        * 1e3)
    physical_step_ms = (
        batch
        * performance.physical_access_latency_time_per_token_equivalent_s
        * 1e3)
    memory_stage_ms = (
        batch * performance.memory_time_per_token_equivalent_s * 1e3)
    compute_step_ms = batch * performance.compute_time_per_token_equivalent_s * 1e3
    gpu_resource_ms = performance.aggregate_step_time_s * 1e3
    total_ms = gpu_resource_ms + host_penalty_ms
    tokens_per_s = batch / (total_ms * 1e-3)
    return PlacementServingTimingResult(
        strategy=strategy,
        requested_requests=batch,
        physical_access_latency_avg_ns=average_latency,
        physical_access_latency_max_ns=maximum_latency,
        read_page_equivalents_per_decode_step=page_equivalents,
        memory_access_latency_step_time_ms=physical_step_ms,
        memory_bandwidth_step_time_ms=bandwidth_step_ms,
        memory_stage_step_time_ms=memory_stage_ms,
        compute_step_time_ms=compute_step_ms,
        gpu_resource_step_time_ms=gpu_resource_ms,
        host_penalty_step_time_ms=host_penalty_ms,
        other_step_time_ms=0.0,
        total_step_time_ms=total_ms,
        inter_token_latency_ms=total_ms,
        aggregate_tokens_per_s=tokens_per_s,
        physical_latency_fraction_of_total=physical_step_ms / total_ms,
        memory_stage_fraction_of_total=memory_stage_ms / total_ms,
        bottleneck=performance.bottleneck,
        access_count_semantics=(
            "SERIAL_ONE_LATENCY_EXPOSURE_PER_2MIB_READ_PAGE_EQUIVALENT"),
        overlap_semantics=(
            "MEMORY_STAGE_EQUALS_BANDWIDTH_PLUS_ACCESS_LATENCY_THEN_"
            "EXISTING_ROOFLINE_MAX_WITH_COMPUTE"),
        double_counting_status=(
            "PASS_BULK_BYTES_OVER_BANDWIDTH_SEPARATE_FROM_PAGE_SCAN_"
            "STARTUP_LATENCY"),
    )


def compare_placement_serving_performance(
    workload: LLMDecodeInput,
    demand: M3DWorkloadPageDemand,
    placement: FastRegionWorkloadComparison,
    physical_layout: PhysicalCapacityLayout,
    *,
    matched_payload_bandwidth_bits_per_second: float,
    effective_compute_flops_per_second: float,
) -> PlacementServingComparison:
    """Propagate fast, balanced conventional, and random-mean latencies."""
    common = {
        "workload": workload,
        "demand": demand,
        "physical_layout": physical_layout,
        "matched_payload_bandwidth_bits_per_second": (
            matched_payload_bandwidth_bits_per_second),
        "effective_compute_flops_per_second": (
            effective_compute_flops_per_second),
    }
    fast = evaluate_placement_serving_timing(
        **common,
        strategy="FAST_PACK",
        physical_access_latency_avg_ns=(
            placement.fast_pack.weighted_average_access_latency_ns),
        physical_access_latency_max_ns=(
            placement.fast_pack.max_occupied_slot_latency_ns),
    )
    conventional = evaluate_placement_serving_timing(
        **common,
        strategy="CONVENTIONAL_LATENCY_OBLIVIOUS",
        physical_access_latency_avg_ns=(
            placement.conventional.weighted_average_access_latency_ns),
        physical_access_latency_max_ns=(
            placement.conventional.max_occupied_slot_latency_ns),
    )
    random_mean = evaluate_placement_serving_timing(
        **common,
        strategy="RANDOM_MEAN",
        physical_access_latency_avg_ns=(
            placement.random.mean_average_access_latency_ns),
        physical_access_latency_max_ns=(
            placement.random.mean_max_occupied_latency_ns),
    )
    return PlacementServingComparison(
        fast_pack=fast,
        conventional=conventional,
        random_mean=random_mean,
        physical_latency_gain_vs_conventional=(
            1.0
            - fast.physical_access_latency_avg_ns
            / conventional.physical_access_latency_avg_ns),
        memory_stage_gain_vs_conventional=(
            1.0
            - fast.memory_stage_step_time_ms
            / conventional.memory_stage_step_time_ms),
        end_to_end_latency_gain_vs_conventional=(
            1.0 - fast.total_step_time_ms / conventional.total_step_time_ms),
        tokens_per_s_gain_vs_conventional=(
            fast.aggregate_tokens_per_s / conventional.aggregate_tokens_per_s
            - 1.0),
        physical_latency_gain_vs_random=(
            1.0
            - fast.physical_access_latency_avg_ns
            / random_mean.physical_access_latency_avg_ns),
        end_to_end_latency_gain_vs_random=(
            1.0 - fast.total_step_time_ms / random_mean.total_step_time_ms),
        tokens_per_s_gain_vs_random=(
            fast.aggregate_tokens_per_s / random_mean.aggregate_tokens_per_s
            - 1.0),
    )


def propagate_occupancy_sweep_to_serving(
    workload: LLMDecodeInput,
    demand: M3DWorkloadPageDemand,
    physical_layout: PhysicalCapacityLayout,
    occupancy_points: Sequence[FastRegionOccupancyPoint],
    *,
    matched_payload_bandwidth_bits_per_second: float,
    effective_compute_flops_per_second: float,
) -> tuple[OccupancyServingClosurePoint, ...]:
    """Apply existing occupancy latency summaries to one fixed workload."""
    rows: list[OccupancyServingClosurePoint] = []
    for point in occupancy_points:
        common = {
            "workload": workload,
            "demand": demand,
            "physical_layout": physical_layout,
            "matched_payload_bandwidth_bits_per_second": (
                matched_payload_bandwidth_bits_per_second),
            "effective_compute_flops_per_second": (
                effective_compute_flops_per_second),
        }
        fast = evaluate_placement_serving_timing(
            **common,
            strategy="FAST_PACK_OCCUPANCY_SENSITIVITY",
            physical_access_latency_avg_ns=(
                point.fast_pack_average_slot_latency_ns),
            physical_access_latency_max_ns=(
                point.fast_pack_max_occupied_latency_ns),
        )
        random_result = evaluate_placement_serving_timing(
            **common,
            strategy="RANDOM_MEAN_OCCUPANCY_SENSITIVITY",
            physical_access_latency_avg_ns=(
                point.random_mean_average_slot_latency_ns),
            physical_access_latency_max_ns=max(
                slot.physical_access_latency_ns
                for slot in physical_layout.slot_classes),
        )
        rows.append(OccupancyServingClosurePoint(
            occupancy_fraction=point.realized_occupancy_fraction,
            fast_physical_latency_ns=fast.physical_access_latency_avg_ns,
            random_physical_latency_ns=(
                random_result.physical_access_latency_avg_ns),
            physical_latency_gain=(
                1.0
                - fast.physical_access_latency_avg_ns
                / random_result.physical_access_latency_avg_ns),
            fast_total_step_time_ms=fast.total_step_time_ms,
            random_total_step_time_ms=random_result.total_step_time_ms,
            end_to_end_latency_gain=(
                1.0 - fast.total_step_time_ms / random_result.total_step_time_ms),
            fast_tokens_per_s=fast.aggregate_tokens_per_s,
            random_tokens_per_s=random_result.aggregate_tokens_per_s,
            tokens_per_s_gain=(
                fast.aggregate_tokens_per_s
                / random_result.aggregate_tokens_per_s
                - 1.0),
        ))
    return tuple(rows)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric
