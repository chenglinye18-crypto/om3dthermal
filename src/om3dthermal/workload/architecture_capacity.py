"""Thin adapter between canonical hardware capacity and an LLM workload."""

from __future__ import annotations

from pydantic import BaseModel

from om3dthermal.architecture_capacity import ResolvedArchitectureCapacity
from om3dthermal.workload.capacity import evaluate_capacity_feasibility
from om3dthermal.workload.llm_decode import LLMDecodeMetrics


class ArchitectureCapacityFeasibility(BaseModel):
    """Aggregate-only capacity feasibility for one canonical architecture."""

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
    workload: LLMDecodeMetrics,
    capacity: ResolvedArchitectureCapacity,
    *,
    reserved_capacity_bytes: int | float,
) -> ArchitectureCapacityFeasibility:
    """Apply the shared aggregate feasibility primitive to one architecture.

    ``reserved_capacity_bytes`` has no default: callers must make the hardware
    reserve assumption explicit, including zero for an ideal raw-capacity
    scenario.  No capacity or feasibility equation is reimplemented here.
    """
    result = evaluate_capacity_feasibility(
        workload,
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
