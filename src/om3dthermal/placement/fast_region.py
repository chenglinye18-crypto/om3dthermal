"""M3D-only fast-region slot selection and placement baselines."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import random
import statistics
from typing import Literal, Sequence

from om3dthermal.power.physical_capacity import (
    PhysicalCapacityLayout,
    PhysicalSlot,
    iter_physical_slots,
)
from om3dthermal.workload.m3d_page_demand import M3DWorkloadPageDemand


SlotSelectionPolicy = Literal[
    "FASTEST",
    "CONVENTIONAL_LATENCY_OBLIVIOUS",
    "RANDOM",
    "SEQUENTIAL",
]
PageOrderingPolicy = Literal["CANONICAL", "DEMAND_DESCENDING"]


class FastRegionCapacityError(ValueError):
    """The requested all-local page count exceeds physical M3D slots."""


@dataclass(frozen=True)
class PhysicalSlotSelection:
    policy: SlotSelectionPolicy
    selected_slots: tuple[PhysicalSlot, ...]
    selected_slot_count: int
    total_physical_slot_count: int
    occupancy_fraction: float
    mean_slot_latency_ns: float
    min_slot_latency_ns: float
    max_slot_latency_ns: float
    random_seed: int | None
    selection_semantics: str


@dataclass(frozen=True)
class PageSlotAssignment:
    page_id: str
    parent_object_id: str
    object_type: str
    read_demand_bytes_per_decode_step: float
    slab_id: int
    cluster_id: int
    layer_id: int
    physical_access_latency_ns: float


@dataclass(frozen=True)
class PagePlacementResult:
    slot_selection_policy: SlotSelectionPolicy
    page_ordering_policy: PageOrderingPolicy
    random_seed: int | None
    page_count: int
    occupancy_fraction: float
    assignments: tuple[PageSlotAssignment, ...]
    total_read_demand_bytes_per_decode_step: float
    weighted_average_access_latency_ns: float
    min_occupied_slot_latency_ns: float
    max_occupied_slot_latency_ns: float
    placement_status: str


@dataclass(frozen=True)
class RandomPlacementSummary:
    seeds: tuple[int, ...]
    average_access_latency_per_seed_ns: tuple[float, ...]
    mean_average_access_latency_ns: float
    std_average_access_latency_ns: float
    min_average_access_latency_ns: float
    max_average_access_latency_ns: float
    max_occupied_latency_per_seed_ns: tuple[float, ...]
    mean_max_occupied_latency_ns: float
    baseline_semantics: str


@dataclass(frozen=True)
class FastRegionWorkloadComparison:
    page_count: int
    occupancy_fraction: float
    allocated_working_set_bytes: int
    fast_pack: PagePlacementResult
    conventional: PagePlacementResult
    sequential: PagePlacementResult
    random: RandomPlacementSummary
    fast_pack_canonical_page_order_latency_ns: float
    page_ordering_gain: float
    slot_selection_gain_vs_random: float
    slot_selection_gain_vs_conventional: float
    slot_selection_gain_vs_sequential: float
    gain_decomposition_status: str


@dataclass(frozen=True)
class FastRegionOccupancyPoint:
    requested_occupancy_fraction: float
    selected_slot_count: int
    realized_occupancy_fraction: float
    fast_pack_average_slot_latency_ns: float
    fast_pack_min_occupied_latency_ns: float
    fast_pack_max_occupied_latency_ns: float
    random_mean_average_slot_latency_ns: float
    random_std_average_slot_latency_ns: float
    slot_selection_gain_vs_random: float


def select_physical_slots(
    layout: PhysicalCapacityLayout,
    selected_slot_count: int,
    *,
    policy: SlotSelectionPolicy,
    random_seed: int | None = None,
) -> PhysicalSlotSelection:
    """Select physical capacity without changing latency or multiplicity."""
    if isinstance(selected_slot_count, bool) or not isinstance(
            selected_slot_count, int):
        raise TypeError("selected_slot_count must be an int")
    if selected_slot_count <= 0:
        raise ValueError("selected_slot_count must be positive")
    if selected_slot_count > layout.physical_slot_count:
        raise FastRegionCapacityError(
            "M3D_FAST_REGION_CAPACITY_FAIL: requested pages exceed physical slots")
    expanded = _expanded_physical_slots(layout)
    if len(expanded) != layout.physical_slot_count:
        raise ValueError("compact slot multiplicity does not close when expanded")
    if policy == "FASTEST":
        selected = _latency_ordered_physical_slots(layout)[:selected_slot_count]
        seed = None
        semantics = "GLOBAL_LOWEST_LATENCY_CAPACITY_PREFIX"
    elif policy == "CONVENTIONAL_LATENCY_OBLIVIOUS":
        selected = _balanced_interleaved_slots(layout)[:selected_slot_count]
        seed = None
        semantics = (
            "DETERMINISTIC_CLASS_AND_SLAB_BALANCED_INTERLEAVING_WITHOUT_"
            "LATENCY_RANKING")
    elif policy == "SEQUENTIAL":
        selected = expanded[:selected_slot_count]
        seed = None
        semantics = "CANONICAL_PHYSICAL_SLOT_ENUMERATION_PREFIX_REFERENCE"
    elif policy == "RANDOM":
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("RANDOM selection requires an integer random_seed")
        selected = tuple(
            random.Random(random_seed).sample(expanded, selected_slot_count))
        seed = random_seed
        semantics = "UNIFORM_SAMPLE_WITHOUT_REPLACEMENT_LATENCY_OBLIVIOUS"
    else:
        raise ValueError(f"unsupported slot selection policy: {policy}")
    identities = tuple(
        (slot.slab_id, slot.cluster_id, slot.layer_id) for slot in selected)
    if len(set(identities)) != selected_slot_count:
        raise RuntimeError("physical slot selection contains duplicates")
    latencies = tuple(slot.physical_access_latency_ns for slot in selected)
    return PhysicalSlotSelection(
        policy=policy,
        selected_slots=selected,
        selected_slot_count=selected_slot_count,
        total_physical_slot_count=layout.physical_slot_count,
        occupancy_fraction=selected_slot_count / layout.physical_slot_count,
        mean_slot_latency_ns=statistics.fmean(latencies),
        min_slot_latency_ns=min(latencies),
        max_slot_latency_ns=max(latencies),
        random_seed=seed,
        selection_semantics=semantics,
    )


def place_pages_on_slots(
    demand: M3DWorkloadPageDemand,
    layout: PhysicalCapacityLayout,
    *,
    slot_policy: SlotSelectionPolicy,
    page_ordering: PageOrderingPolicy = "CANONICAL",
    random_seed: int | None = None,
) -> PagePlacementResult:
    """Map already-resident pages one-to-one onto one selected slot set."""
    if demand.page_count != len(demand.page_demands):
        raise ValueError("page demand count does not close")
    selection = select_physical_slots(
        layout,
        demand.page_count,
        policy=slot_policy,
        random_seed=random_seed,
    )
    if page_ordering == "CANONICAL":
        pages = demand.page_demands
    elif page_ordering == "DEMAND_DESCENDING":
        pages = tuple(sorted(
            demand.page_demands,
            key=lambda page: (-page.read_demand_bytes_per_decode_step,
                              page.page_id),
        ))
    else:
        raise ValueError(f"unsupported page ordering policy: {page_ordering}")
    assignments = tuple(
        PageSlotAssignment(
            page_id=page.page_id,
            parent_object_id=page.parent_object_id,
            object_type=page.object_type,
            read_demand_bytes_per_decode_step=(
                page.read_demand_bytes_per_decode_step),
            slab_id=slot.slab_id,
            cluster_id=slot.cluster_id,
            layer_id=slot.layer_id,
            physical_access_latency_ns=slot.physical_access_latency_ns,
        )
        for page, slot in zip(pages, selection.selected_slots, strict=True)
    )
    total_demand = sum(
        assignment.read_demand_bytes_per_decode_step
        for assignment in assignments)
    if not math.isclose(
            total_demand,
            demand.total_read_bytes_per_decode_step,
            rel_tol=1e-12,
            abs_tol=1e-9):
        raise ValueError("placement changed page-demand traffic closure")
    if total_demand <= 0.0:
        raise ValueError("placement requires positive total read demand")
    weighted_latency = sum(
        assignment.read_demand_bytes_per_decode_step
        * assignment.physical_access_latency_ns
        for assignment in assignments
    ) / total_demand
    return PagePlacementResult(
        slot_selection_policy=slot_policy,
        page_ordering_policy=page_ordering,
        random_seed=selection.random_seed,
        page_count=demand.page_count,
        occupancy_fraction=selection.occupancy_fraction,
        assignments=assignments,
        total_read_demand_bytes_per_decode_step=total_demand,
        weighted_average_access_latency_ns=weighted_latency,
        min_occupied_slot_latency_ns=selection.min_slot_latency_ns,
        max_occupied_slot_latency_ns=selection.max_slot_latency_ns,
        placement_status="ONE_RESIDENT_PAGE_TO_ONE_UNIQUE_PHYSICAL_SLOT",
    )


def compare_fast_region_placements(
    demand: M3DWorkloadPageDemand,
    layout: PhysicalCapacityLayout,
    *,
    random_seeds: Sequence[int] = tuple(range(20)),
) -> FastRegionWorkloadComparison:
    """Decompose fast-region slot selection from within-set page ordering."""
    seeds = _validate_seeds(random_seeds)
    fast = place_pages_on_slots(
        demand,
        layout,
        slot_policy="FASTEST",
        page_ordering="DEMAND_DESCENDING",
    )
    fast_canonical = place_pages_on_slots(
        demand,
        layout,
        slot_policy="FASTEST",
        page_ordering="CANONICAL",
    )
    sequential = place_pages_on_slots(
        demand,
        layout,
        slot_policy="SEQUENTIAL",
        page_ordering="CANONICAL",
    )
    conventional = place_pages_on_slots(
        demand,
        layout,
        slot_policy="CONVENTIONAL_LATENCY_OBLIVIOUS",
        page_ordering="CANONICAL",
    )
    random_runs = tuple(
        place_pages_on_slots(
            demand,
            layout,
            slot_policy="RANDOM",
            page_ordering="CANONICAL",
            random_seed=seed,
        )
        for seed in seeds
    )
    random_latencies = tuple(
        run.weighted_average_access_latency_ns for run in random_runs)
    random_maxima = tuple(
        run.max_occupied_slot_latency_ns for run in random_runs)
    random_mean = statistics.fmean(random_latencies)
    random_summary = RandomPlacementSummary(
        seeds=seeds,
        average_access_latency_per_seed_ns=random_latencies,
        mean_average_access_latency_ns=random_mean,
        std_average_access_latency_ns=statistics.pstdev(random_latencies),
        min_average_access_latency_ns=min(random_latencies),
        max_average_access_latency_ns=max(random_latencies),
        max_occupied_latency_per_seed_ns=random_maxima,
        mean_max_occupied_latency_ns=statistics.fmean(random_maxima),
        baseline_semantics=(
            "FIXED_SEED_UNIFORM_SLOT_SAMPLE_WITHOUT_REPLACEMENT_"
            "LATENCY_OBLIVIOUS_NOT_AN_INDUSTRY_ALLOCATOR"),
    )
    return FastRegionWorkloadComparison(
        page_count=demand.page_count,
        occupancy_fraction=demand.page_count / layout.physical_slot_count,
        allocated_working_set_bytes=demand.allocated_page_bytes,
        fast_pack=fast,
        conventional=conventional,
        sequential=sequential,
        random=random_summary,
        fast_pack_canonical_page_order_latency_ns=(
            fast_canonical.weighted_average_access_latency_ns),
        page_ordering_gain=(
            1.0
            - fast.weighted_average_access_latency_ns
            / fast_canonical.weighted_average_access_latency_ns),
        slot_selection_gain_vs_random=(
            1.0 - fast.weighted_average_access_latency_ns / random_mean),
        slot_selection_gain_vs_conventional=(
            1.0
            - fast.weighted_average_access_latency_ns
            / conventional.weighted_average_access_latency_ns),
        slot_selection_gain_vs_sequential=(
            1.0
            - fast.weighted_average_access_latency_ns
            / sequential.weighted_average_access_latency_ns),
        gain_decomposition_status=(
            "PAGE_ORDERING_AND_SLOT_SELECTION_REPORTED_SEPARATELY"),
    )


def evaluate_fast_region_occupancy_sweep(
    layout: PhysicalCapacityLayout,
    occupancy_fractions: Sequence[float] = (
        0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00),
    *,
    random_seeds: Sequence[int] = tuple(range(20)),
) -> tuple[FastRegionOccupancyPoint, ...]:
    """Compare capacity-prefix selection against random uniform occupancy."""
    seeds = _validate_seeds(random_seeds)
    points: list[FastRegionOccupancyPoint] = []
    for fraction in occupancy_fractions:
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("occupancy fractions must be finite in (0, 1]")
        count = math.ceil(fraction * layout.physical_slot_count)
        fast = select_physical_slots(layout, count, policy="FASTEST")
        random_means = tuple(
            select_physical_slots(
                layout, count, policy="RANDOM", random_seed=seed
            ).mean_slot_latency_ns
            for seed in seeds
        )
        random_mean = statistics.fmean(random_means)
        points.append(FastRegionOccupancyPoint(
            requested_occupancy_fraction=fraction,
            selected_slot_count=count,
            realized_occupancy_fraction=count / layout.physical_slot_count,
            fast_pack_average_slot_latency_ns=fast.mean_slot_latency_ns,
            fast_pack_min_occupied_latency_ns=fast.min_slot_latency_ns,
            fast_pack_max_occupied_latency_ns=fast.max_slot_latency_ns,
            random_mean_average_slot_latency_ns=random_mean,
            random_std_average_slot_latency_ns=statistics.pstdev(random_means),
            slot_selection_gain_vs_random=(
                1.0 - fast.mean_slot_latency_ns / random_mean),
        ))
    return tuple(points)


def _validate_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds or len(set(seeds)) != len(seeds) or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in seeds):
        raise ValueError("random seeds must be unique integers")
    return seeds


@lru_cache(maxsize=4)
def _expanded_physical_slots(
    layout: PhysicalCapacityLayout,
) -> tuple[PhysicalSlot, ...]:
    expanded = tuple(iter_physical_slots(layout))
    if len(expanded) != layout.physical_slot_count:
        raise ValueError("compact slot multiplicity does not close when expanded")
    return expanded


@lru_cache(maxsize=4)
def _latency_ordered_physical_slots(
    layout: PhysicalCapacityLayout,
) -> tuple[PhysicalSlot, ...]:
    return tuple(sorted(
        _expanded_physical_slots(layout),
        key=lambda slot: (
            slot.physical_access_latency_ns,
            slot.cluster_id,
            slot.layer_id,
            slot.slab_id,
        ),
    ))


@lru_cache(maxsize=4)
def _balanced_interleaved_slots(
    layout: PhysicalCapacityLayout,
) -> tuple[PhysicalSlot, ...]:
    """Balance every prefix across classes and slabs without latency input."""
    classes = layout.slot_classes
    slots: list[PhysicalSlot] = []
    for replica_round in range(layout.slab_count):
        for class_index, slot_class in enumerate(classes):
            slab_id = (class_index + replica_round) % layout.slab_count
            slots.append(PhysicalSlot(
                slab_id=slab_id,
                cluster_id=slot_class.cluster_id,
                layer_id=slot_class.layer_id,
                capacity_bytes=slot_class.capacity_bytes,
                physical_access_latency_ns=(
                    slot_class.physical_access_latency_ns),
            ))
    if len(slots) != layout.physical_slot_count:
        raise ValueError("balanced interleaving does not close to slot count")
    return tuple(slots)
