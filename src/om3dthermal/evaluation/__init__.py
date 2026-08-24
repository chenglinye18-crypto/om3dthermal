"""Cross-domain workload-to-hardware evaluation boundaries."""

from .architecture_capacity import (
    ArchitectureCapacityFeasibility,
    evaluate_architecture_capacity_feasibility,
)
from .capacity import (
    CapacityDemand,
    CapacityFeasibilityMetrics,
    evaluate_capacity_feasibility,
)

__all__ = [
    "ArchitectureCapacityFeasibility",
    "CapacityDemand",
    "CapacityFeasibilityMetrics",
    "evaluate_architecture_capacity_feasibility",
    "evaluate_capacity_feasibility",
]
