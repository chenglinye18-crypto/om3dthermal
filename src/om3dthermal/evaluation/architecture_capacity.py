"""Adapter joining resolved architecture capacity to workload demand."""

from __future__ import annotations

from pydantic import BaseModel

from om3dthermal.architecture import ResolvedPacking
from om3dthermal.architecture_capacity import ResolvedArchitectureCapacity

from .capacity import CapacityDemand, evaluate_capacity_feasibility


class ArchitectureCapacityFeasibility(BaseModel):
    """Aggregate-only capacity feasibility for one architecture."""

    architecture: str
    physical_capacity_bytes: int | float
    physical_capacity_GiB: float
    reserved_capacity_bytes: int | float
    usable_capacity_bytes: int | float
    required_capacity_bytes: float
    capacity_margin_bytes: float
    capacity_utilization: float | None
    capacity_feasible: bool
    capacity_scope_status: str
    capacity_source_status: str


def evaluate_architecture_capacity_feasibility(
    demand: CapacityDemand,
    capacity: ResolvedArchitectureCapacity | ResolvedPacking,
    *,
    reserved_capacity_bytes: int | float,
) -> ArchitectureCapacityFeasibility:
    """Apply the generic aggregate capacity gate to one architecture."""

    result = evaluate_capacity_feasibility(
        demand,
        physical_capacity_bytes=capacity.system_capacity_bytes,
        reserved_capacity_bytes=reserved_capacity_bytes,
    )
    return ArchitectureCapacityFeasibility(
        architecture=capacity.architecture,
        physical_capacity_bytes=result.physical_capacity_bytes,
        physical_capacity_GiB=capacity.system_capacity_GiB,
        reserved_capacity_bytes=result.reserved_capacity_bytes,
        usable_capacity_bytes=result.usable_capacity_bytes,
        required_capacity_bytes=result.required_capacity_bytes,
        capacity_margin_bytes=result.capacity_margin_bytes,
        capacity_utilization=result.capacity_utilization,
        capacity_feasible=result.capacity_feasible,
        capacity_scope_status=result.scope_status,
        capacity_source_status=capacity.source_status,
    )
