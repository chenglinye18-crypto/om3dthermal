"""Physical placement analyses over existing M3D slots."""

from .fast_region import (
    FastRegionCapacityError,
    FastRegionOccupancyPoint,
    FastRegionWorkloadComparison,
    PagePlacementResult,
    PageSlotAssignment,
    PhysicalSlotSelection,
    RandomPlacementSummary,
    compare_fast_region_placements,
    evaluate_fast_region_occupancy_sweep,
    place_pages_on_slots,
    select_physical_slots,
)
from .serving_e2e import (
    OccupancyServingClosurePoint,
    PlacementServingComparison,
    PlacementServingTimingResult,
    compare_placement_serving_performance,
    evaluate_placement_serving_timing,
    evaluate_metrics_placement_serving_timing,
    propagate_occupancy_sweep_to_serving,
)
from .moe_published_e2e import (
    MoEPublishedPlacementE2ECase,
    PhysicalPlacementComparison,
    PhysicalPlacementPoint,
    TrafficClassLatency,
    evaluate_published_moe_placement_e2e,
)

__all__ = [
    "FastRegionCapacityError",
    "FastRegionOccupancyPoint",
    "FastRegionWorkloadComparison",
    "PagePlacementResult",
    "PageSlotAssignment",
    "PhysicalSlotSelection",
    "RandomPlacementSummary",
    "OccupancyServingClosurePoint",
    "PlacementServingComparison",
    "PlacementServingTimingResult",
    "MoEPublishedPlacementE2ECase",
    "PhysicalPlacementComparison",
    "PhysicalPlacementPoint",
    "TrafficClassLatency",
    "compare_fast_region_placements",
    "evaluate_fast_region_occupancy_sweep",
    "place_pages_on_slots",
    "select_physical_slots",
    "compare_placement_serving_performance",
    "evaluate_placement_serving_timing",
    "evaluate_metrics_placement_serving_timing",
    "evaluate_published_moe_placement_e2e",
    "propagate_occupancy_sweep_to_serving",
]
