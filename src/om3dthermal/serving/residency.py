"""Capacity-only request residency for analytical LLM serving."""

from __future__ import annotations

import math
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from om3dthermal.workload import LLMDecodeMetrics


class UsableCapacity(Protocol):
    architecture: str
    usable_capacity_bytes: int | float
    capacity_source_status: str


class ServingCapacitySource(BaseModel):
    """Capacity-only reference; it is not a physical architecture model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: str = Field(min_length=1)
    usable_capacity_bytes: int | float = Field(ge=0)
    capacity_source_status: str = Field(min_length=1)
    provenance_status: str = Field(min_length=1)


class CapacityResidencyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: str
    usable_capacity_bytes: float
    weight_bytes: float
    runtime_fixed_bytes: float
    runtime_per_request_bytes: float
    kv_bytes_per_request: float
    resident_bytes_per_request: float
    available_for_requests_bytes: float
    max_resident_requests: int | None
    requested_requests: int
    local_resident_requests: int
    spilled_requests: int
    local_capacity_utilization: float | None
    capacity_status: Literal[
        "FULLY_LOCAL",
        "CAPACITY_PRESSURED",
        "WEIGHTS_NOT_RESIDENT",
        "UNBOUNDED_PER_REQUEST_FOOTPRINT",
    ]
    capacity_source_status: str
    runtime_capacity_semantics_status: str
    residency_model_status: Literal["ANALYTICAL_CAPACITY_RESIDENCY_V0"]


def evaluate_capacity_residency(
    workload: LLMDecodeMetrics,
    capacity: UsableCapacity,
    *,
    requested_requests: int,
) -> CapacityResidencyResult:
    """Resolve local KV residency without weight streaming or scheduling."""
    if isinstance(requested_requests, bool) or not isinstance(
        requested_requests, int
    ):
        raise TypeError("requested_requests must be an int")
    if requested_requests <= 0:
        raise ValueError("requested_requests must be positive")

    usable = float(capacity.usable_capacity_bytes)
    if not math.isfinite(usable) or usable < 0.0:
        raise ValueError("usable capacity must be finite and non-negative")
    weight = float(workload.weight_footprint_bytes)
    runtime_fixed = float(workload.runtime_fixed_bytes)
    runtime_per_request = float(workload.runtime_per_request_bytes)
    kv_per_request = float(workload.kv_bytes_per_request)
    values = (weight, runtime_fixed, runtime_per_request, kv_per_request)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("workload residency inputs must be finite and non-negative")

    available = usable - weight - runtime_fixed
    resident_per_request = kv_per_request + runtime_per_request
    fixed_fits = available >= 0.0
    if not fixed_fits:
        maximum = 0
        local = 0
        spilled = requested_requests
        status = "WEIGHTS_NOT_RESIDENT"
    elif resident_per_request == 0.0:
        maximum = None
        local = requested_requests
        spilled = 0
        status = "UNBOUNDED_PER_REQUEST_FOOTPRINT"
    else:
        # Python integers avoid fixed-width overflow after the floor operation.
        maximum = math.floor(available / resident_per_request)
        local = min(requested_requests, maximum)
        spilled = max(0, requested_requests - maximum)
        status = "FULLY_LOCAL" if spilled == 0 else "CAPACITY_PRESSURED"

    used = weight + runtime_fixed + local * resident_per_request
    utilization = used / usable if usable > 0.0 else None
    return CapacityResidencyResult(
        architecture=capacity.architecture,
        usable_capacity_bytes=usable,
        weight_bytes=weight,
        runtime_fixed_bytes=runtime_fixed,
        runtime_per_request_bytes=runtime_per_request,
        kv_bytes_per_request=kv_per_request,
        resident_bytes_per_request=resident_per_request,
        available_for_requests_bytes=available,
        max_resident_requests=maximum,
        requested_requests=requested_requests,
        local_resident_requests=local,
        spilled_requests=spilled,
        local_capacity_utilization=utilization,
        capacity_status=status,
        capacity_source_status=capacity.capacity_source_status,
        runtime_capacity_semantics_status=(
            workload.runtime_capacity_semantics_status),
        residency_model_status="ANALYTICAL_CAPACITY_RESIDENCY_V0",
    )
