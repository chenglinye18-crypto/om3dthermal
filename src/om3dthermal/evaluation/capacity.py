"""Generic aggregate hardware-capacity feasibility evaluation."""

from __future__ import annotations

import math
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel


CapacityScalar: TypeAlias = int | float
SCOPE_STATUS = "AGGREGATE_CAPACITY_FEASIBILITY_ONLY"


class CapacityDemand(Protocol):
    """Minimal workload boundary required by aggregate capacity evaluation."""

    required_capacity_bytes: float


class CapacityFeasibilityMetrics(BaseModel):
    """Aggregate capacity result, with all dimensional fields in bytes."""

    physical_capacity_bytes: CapacityScalar
    reserved_capacity_bytes: CapacityScalar
    usable_capacity_bytes: CapacityScalar
    required_capacity_bytes: float
    capacity_margin_bytes: float
    capacity_utilization: float | None
    utilization_status: Literal[
        "DEFINED",
        "DEFINED_ZERO_REQUIRED_ZERO_USABLE",
        "UNDEFINED_ZERO_USABLE_CAPACITY",
    ]
    capacity_feasible: bool
    scope_status: Literal["AGGREGATE_CAPACITY_FEASIBILITY_ONLY"]


def _validate_capacity_scalar(
    name: str, value: CapacityScalar,
) -> CapacityScalar:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float expressed in bytes")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def evaluate_capacity_feasibility(
    demand: CapacityDemand,
    *,
    physical_capacity_bytes: CapacityScalar,
    reserved_capacity_bytes: CapacityScalar,
) -> CapacityFeasibilityMetrics:
    """Evaluate aggregate workload fit without application-specific semantics."""

    physical = _validate_capacity_scalar(
        "physical_capacity_bytes", physical_capacity_bytes)
    reserved = _validate_capacity_scalar(
        "reserved_capacity_bytes", reserved_capacity_bytes)
    if reserved > physical:
        raise ValueError(
            "reserved_capacity_bytes must not exceed physical_capacity_bytes")

    required = demand.required_capacity_bytes
    if isinstance(required, bool) or not isinstance(required, (int, float)):
        raise TypeError("demand.required_capacity_bytes must be numeric")
    if not math.isfinite(required):
        raise ValueError("demand.required_capacity_bytes must be finite")
    if required < 0:
        raise ValueError("demand.required_capacity_bytes must be non-negative")

    usable = physical - reserved
    margin = usable - required
    feasible = required <= usable
    if usable > 0:
        utilization = required / usable
        utilization_status = "DEFINED"
    elif required == 0:
        utilization = 0.0
        utilization_status = "DEFINED_ZERO_REQUIRED_ZERO_USABLE"
    else:
        utilization = None
        utilization_status = "UNDEFINED_ZERO_USABLE_CAPACITY"

    return CapacityFeasibilityMetrics(
        physical_capacity_bytes=physical,
        reserved_capacity_bytes=reserved,
        usable_capacity_bytes=usable,
        required_capacity_bytes=required,
        capacity_margin_bytes=margin,
        capacity_utilization=utilization,
        utilization_status=utilization_status,
        capacity_feasible=feasible,
        scope_status=SCOPE_STATUS,
    )
