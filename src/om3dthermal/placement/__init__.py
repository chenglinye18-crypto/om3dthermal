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

__all__ = [
    "FastRegionCapacityError",
    "FastRegionOccupancyPoint",
    "FastRegionWorkloadComparison",
    "PagePlacementResult",
    "PageSlotAssignment",
    "PhysicalSlotSelection",
    "RandomPlacementSummary",
    "compare_fast_region_placements",
    "evaluate_fast_region_occupancy_sweep",
    "place_pages_on_slots",
    "select_physical_slots",
]
