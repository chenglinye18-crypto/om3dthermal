"""Published Mixtral routing demand through placement and frozen E2E timing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Sequence

from om3dthermal.power.physical_capacity import PhysicalCapacityLayout
from om3dthermal.workload.moe_decode import MoEDecodeInput, evaluate_moe_decode
from om3dthermal.workload.moe_published_page_demand import (
    MoEPublishedPageDemand,
    PageDemandView,
    build_published_moe_page_demand,
    expert_only_page_demand_view,
)
from om3dthermal.workload.moe_published_profile import FiddlerPublishedProfile

from .fast_region import PagePlacementResult, place_pages_on_slots
from .serving_e2e import (
    PlacementServingTimingResult,
    evaluate_metrics_placement_serving_timing,
)


@dataclass(frozen=True)
class TrafficClassLatency:
    expert_ns: float | None
    shared_weight_ns: float | None
    kv_ns: float | None
    all_read_ns: float


@dataclass(frozen=True)
class PhysicalPlacementPoint:
    strategy: str
    weighted_average_latency_ns: float
    random_std_latency_ns: float
    min_occupied_slot_latency_ns: float
    max_occupied_slot_latency_ns: float
    traffic_class_latency: TrafficClassLatency


@dataclass(frozen=True)
class PhysicalPlacementComparison:
    random: PhysicalPlacementPoint
    fast_region_only: PhysicalPlacementPoint
    popularity_aware_fast_region: PhysicalPlacementPoint
    fast_region_gain: float
    popularity_ordering_gain: float
    total_placement_gain: float
    occupied_slot_set_closure: str


@dataclass(frozen=True)
class MoEPlacementPerformanceInput:
    required_capacity_bytes: float
    read_bytes_per_token: float
    write_bytes_per_token: float
    flops_per_token: int


@dataclass(frozen=True)
class MoEPublishedPlacementE2ECase:
    requested_requests: int
    logical_working_set_bytes: int
    allocated_page_bytes: int
    occupancy_fraction: float
    page_count: int
    expert_page_count: int
    shared_weight_page_count: int
    kv_page_count: int
    all_read_physical: PhysicalPlacementComparison
    expert_only_physical: PhysicalPlacementComparison
    uniform_expert_only_physical: PhysicalPlacementComparison
    uniform_all_read_physical: PhysicalPlacementComparison
    uniform_all_read_popularity_ordering_gain: float
    random_timing: PlacementServingTimingResult
    fast_region_timing: PlacementServingTimingResult
    popularity_aware_timing: PlacementServingTimingResult
    fast_region_e2e_latency_gain: float
    popularity_e2e_latency_gain: float
    total_e2e_latency_gain: float
    fast_region_throughput_gain: float
    popularity_throughput_gain: float
    total_throughput_gain: float
    physical_latency_exposure_model: str
    verdict_scope: str


def evaluate_published_moe_placement_e2e(
    profile: FiddlerPublishedProfile,
    workload: MoEDecodeInput,
    physical_layout: PhysicalCapacityLayout,
    *,
    matched_payload_bandwidth_bits_per_second: float,
    effective_compute_flops_per_second: float,
    random_seeds: Sequence[int] = tuple(range(20)),
) -> MoEPublishedPlacementE2ECase:
    """Evaluate P0/P1/P2 without changing routing, physics, or roofline."""
    published = build_published_moe_page_demand(
        profile, workload, physical_layout,
        routing_source="FIDDLER_PUBLISHED_ROUTING_PROFILE")
    uniform = build_published_moe_page_demand(
        profile, workload, physical_layout,
        routing_source="UNIFORM_MOE_ROUTING_CONTROL")
    all_read, all_placements = _compare_physical(
        published, physical_layout, random_seeds)
    expert_only, _ = _compare_physical(
        expert_only_page_demand_view(published),
        physical_layout,
        random_seeds,
    )
    uniform_expert_only, _ = _compare_physical(
        expert_only_page_demand_view(uniform),
        physical_layout,
        random_seeds,
    )
    uniform_all_read, _ = _compare_physical(
        uniform, physical_layout, random_seeds)

    metrics = evaluate_moe_decode(workload)
    expected_read_step = (
        metrics.active_weight_bytes_per_decode_step
        + workload.batch_size * metrics.kv_read_bytes_per_token_per_request)
    if not math.isclose(
            published.total_read_bytes_per_decode_step, expected_read_step,
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("published pages changed existing MoE read accounting")
    performance_input = MoEPlacementPerformanceInput(
        required_capacity_bytes=metrics.required_capacity_bytes,
        read_bytes_per_token=(
            published.total_read_bytes_per_decode_step / workload.batch_size),
        write_bytes_per_token=(
            published.kv_write_bytes_per_decode_step / workload.batch_size),
        flops_per_token=metrics.flops_per_token,
    )
    common = {
        "metrics": performance_input,
        "demand": published,
        "physical_layout": physical_layout,
        "requested_requests": workload.batch_size,
        "matched_payload_bandwidth_bits_per_second": (
            matched_payload_bandwidth_bits_per_second),
        "effective_compute_flops_per_second": (
            effective_compute_flops_per_second),
    }
    random_timing = evaluate_metrics_placement_serving_timing(
        **common,
        strategy="P0_LATENCY_OBLIVIOUS_RANDOM_MEAN",
        physical_access_latency_avg_ns=(
            all_read.random.weighted_average_latency_ns),
        physical_access_latency_max_ns=(
            all_read.random.max_occupied_slot_latency_ns),
    )
    fast_timing = evaluate_metrics_placement_serving_timing(
        **common,
        strategy="P1_FAST_REGION_ONLY",
        physical_access_latency_avg_ns=(
            all_read.fast_region_only.weighted_average_latency_ns),
        physical_access_latency_max_ns=(
            all_read.fast_region_only.max_occupied_slot_latency_ns),
    )
    popularity_timing = evaluate_metrics_placement_serving_timing(
        **common,
        strategy="P2_POPULARITY_AWARE_FAST_REGION",
        physical_access_latency_avg_ns=(
            all_read.popularity_aware_fast_region.weighted_average_latency_ns),
        physical_access_latency_max_ns=(
            all_read.popularity_aware_fast_region.max_occupied_slot_latency_ns),
    )
    exposure_models = {
        random_timing.access_count_semantics,
        fast_timing.access_count_semantics,
        popularity_timing.access_count_semantics,
    }
    if len(exposure_models) != 1:
        raise RuntimeError("placement strategies changed service semantics")
    return MoEPublishedPlacementE2ECase(
        requested_requests=workload.batch_size,
        logical_working_set_bytes=published.logical_working_set_bytes,
        allocated_page_bytes=published.allocated_page_bytes,
        occupancy_fraction=(
            published.page_count / physical_layout.physical_slot_count),
        page_count=published.page_count,
        expert_page_count=published.expert_page_count,
        shared_weight_page_count=published.shared_weight_page_count,
        kv_page_count=published.kv_page_count,
        all_read_physical=all_read,
        expert_only_physical=expert_only,
        uniform_expert_only_physical=uniform_expert_only,
        uniform_all_read_physical=uniform_all_read,
        uniform_all_read_popularity_ordering_gain=(
            uniform_all_read.popularity_ordering_gain),
        random_timing=random_timing,
        fast_region_timing=fast_timing,
        popularity_aware_timing=popularity_timing,
        fast_region_e2e_latency_gain=(
            1.0 - fast_timing.total_step_time_ms
            / random_timing.total_step_time_ms),
        popularity_e2e_latency_gain=(
            1.0 - popularity_timing.total_step_time_ms
            / fast_timing.total_step_time_ms),
        total_e2e_latency_gain=(
            1.0 - popularity_timing.total_step_time_ms
            / random_timing.total_step_time_ms),
        fast_region_throughput_gain=(
            fast_timing.aggregate_tokens_per_s
            / random_timing.aggregate_tokens_per_s - 1.0),
        popularity_throughput_gain=(
            popularity_timing.aggregate_tokens_per_s
            / fast_timing.aggregate_tokens_per_s - 1.0),
        total_throughput_gain=(
            popularity_timing.aggregate_tokens_per_s
            / random_timing.aggregate_tokens_per_s - 1.0),
        physical_latency_exposure_model=next(iter(exposure_models)),
        verdict_scope="CONDITIONAL_ON_CURRENT_MEMORY_SERVICE_MODEL",
    )


def _compare_physical(
    demand: MoEPublishedPageDemand | PageDemandView,
    physical_layout: PhysicalCapacityLayout,
    random_seeds: Sequence[int],
) -> tuple[
    PhysicalPlacementComparison,
    tuple[PagePlacementResult, PagePlacementResult],
]:
    seeds = tuple(random_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("random seeds must be non-empty and unique")
    random_runs = tuple(
        place_pages_on_slots(
            demand, physical_layout,
            slot_policy="RANDOM",
            page_ordering="CANONICAL",
            random_seed=seed,
        )
        for seed in seeds
    )
    fast = place_pages_on_slots(
        demand, physical_layout,
        slot_policy="FASTEST", page_ordering="CANONICAL")
    aware = place_pages_on_slots(
        demand, physical_layout,
        slot_policy="FASTEST", page_ordering="DEMAND_DESCENDING")
    fast_slots = {
        (item.slab_id, item.cluster_id, item.layer_id)
        for item in fast.assignments
    }
    aware_slots = {
        (item.slab_id, item.cluster_id, item.layer_id)
        for item in aware.assignments
    }
    if fast_slots != aware_slots:
        raise RuntimeError("P1 and P2 do not use the same occupied slot set")
    random_averages = tuple(
        item.weighted_average_access_latency_ns for item in random_runs)
    random_categories = tuple(_traffic_class_latency(item) for item in random_runs)
    random_point = PhysicalPlacementPoint(
        strategy="P0_LATENCY_OBLIVIOUS_RANDOM_MEAN",
        weighted_average_latency_ns=statistics.fmean(random_averages),
        random_std_latency_ns=statistics.pstdev(random_averages),
        min_occupied_slot_latency_ns=statistics.fmean(
            item.min_occupied_slot_latency_ns for item in random_runs),
        max_occupied_slot_latency_ns=statistics.fmean(
            item.max_occupied_slot_latency_ns for item in random_runs),
        traffic_class_latency=TrafficClassLatency(
            expert_ns=_optional_mean(
                item.expert_ns for item in random_categories),
            shared_weight_ns=_optional_mean(
                item.shared_weight_ns for item in random_categories),
            kv_ns=_optional_mean(item.kv_ns for item in random_categories),
            all_read_ns=statistics.fmean(
                item.all_read_ns for item in random_categories),
        ),
    )
    fast_point = _deterministic_point("P1_FAST_REGION_ONLY", fast)
    aware_point = _deterministic_point(
        "P2_POPULARITY_AWARE_FAST_REGION", aware)
    if not (
        aware_point.weighted_average_latency_ns
        <= fast_point.weighted_average_latency_ns + 1e-12
        <= random_point.weighted_average_latency_ns + 1e-12
    ):
        raise ValueError("placement latency ordering P2 <= P1 <= P0 failed")
    return PhysicalPlacementComparison(
        random=random_point,
        fast_region_only=fast_point,
        popularity_aware_fast_region=aware_point,
        fast_region_gain=(
            1.0 - fast_point.weighted_average_latency_ns
            / random_point.weighted_average_latency_ns),
        popularity_ordering_gain=(
            1.0 - aware_point.weighted_average_latency_ns
            / fast_point.weighted_average_latency_ns),
        total_placement_gain=(
            1.0 - aware_point.weighted_average_latency_ns
            / random_point.weighted_average_latency_ns),
        occupied_slot_set_closure="P1_P2_IDENTICAL_FASTEST_CAPACITY_PREFIX",
    ), (fast, aware)


def _deterministic_point(
    strategy: str,
    placement: PagePlacementResult,
) -> PhysicalPlacementPoint:
    return PhysicalPlacementPoint(
        strategy=strategy,
        weighted_average_latency_ns=(
            placement.weighted_average_access_latency_ns),
        random_std_latency_ns=0.0,
        min_occupied_slot_latency_ns=placement.min_occupied_slot_latency_ns,
        max_occupied_slot_latency_ns=placement.max_occupied_slot_latency_ns,
        traffic_class_latency=_traffic_class_latency(placement),
    )


def _traffic_class_latency(
    placement: PagePlacementResult,
) -> TrafficClassLatency:
    expert = tuple(
        item for item in placement.assignments
        if item.parent_object_id.startswith("expert."))
    shared = tuple(
        item for item in placement.assignments
        if item.parent_object_id == "weights.shared_nonexpert")
    kv = tuple(
        item for item in placement.assignments if item.object_type == "KV")
    return TrafficClassLatency(
        expert_ns=_weighted_latency(expert),
        shared_weight_ns=_weighted_latency(shared),
        kv_ns=_weighted_latency(kv),
        all_read_ns=placement.weighted_average_access_latency_ns,
    )


def _weighted_latency(assignments: tuple) -> float | None:
    total = math.fsum(
        item.read_demand_bytes_per_decode_step for item in assignments)
    if total == 0.0:
        return None
    return math.fsum(
        item.read_demand_bytes_per_decode_step
        * item.physical_access_latency_ns
        for item in assignments
    ) / total


def _optional_mean(values) -> float | None:
    present = tuple(value for value in values if value is not None)
    return None if not present else statistics.fmean(present)
