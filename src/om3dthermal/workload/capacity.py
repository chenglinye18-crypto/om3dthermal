"""Aggregate-only hardware capacity feasibility for an LLM workload.

This primitive connects an architecture-independent :class:`LLMDecodeMetrics`
result to an explicitly supplied hardware capacity.  Every capacity value at
this boundary is expressed in bytes; callers must perform any GB (``* 10**9``)
or GiB (``* 2**30``) conversion before calling this module.

Accounting boundary:

* ``LLMDecodeMetrics.required_capacity_bytes`` already includes weights, KV
  cache, and the workload-side ``runtime_bytes``.
* ``reserved_capacity_bytes`` is the part of physical hardware capacity that
  cannot be allocated to this workload.

The same scheduler, runtime, or metadata allocation must not appear in both
values.  This evaluation checks aggregate capacity only; it does not validate
bank/die/slab placement, fragmentation, allocation granularity, bandwidth,
latency, power, or thermal placement.
"""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

from pydantic import BaseModel

from om3dthermal.workload.llm_decode import LLMDecodeMetrics


CapacityScalar: TypeAlias = int | float

SCOPE_STATUS = "AGGREGATE_CAPACITY_FEASIBILITY_ONLY"


class CapacityFeasibilityMetrics(BaseModel):
    """Aggregate capacity result, with all dimensional fields in bytes.

    ``capacity_utilization`` is ``None`` only when usable capacity is zero and
    required capacity is positive.  For the empty exact-fit case where both
    required and usable capacity are zero, utilization is defined as ``0.0``.
    """

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


def _validate_capacity_scalar(name: str, value: CapacityScalar) -> CapacityScalar:
    """Return a finite, non-negative byte value without rounding it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float expressed in bytes")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def evaluate_capacity_feasibility(
    workload: LLMDecodeMetrics,
    *,
    physical_capacity_bytes: CapacityScalar,
    reserved_capacity_bytes: CapacityScalar,
) -> CapacityFeasibilityMetrics:
    """Evaluate aggregate workload-fit feasibility using explicit byte inputs.

    The workload requirement is consumed exactly as produced by
    ``evaluate_llm_decode``.  In particular, this function does not add
    ``runtime_bytes`` again.  Hardware reserve must be supplied explicitly,
    including an explicit zero when no reserve is modeled.
    """
    physical = _validate_capacity_scalar(
        "physical_capacity_bytes", physical_capacity_bytes)
    reserved = _validate_capacity_scalar(
        "reserved_capacity_bytes", reserved_capacity_bytes)
    if reserved > physical:
        raise ValueError(
            "reserved_capacity_bytes must not exceed physical_capacity_bytes")

    required = workload.required_capacity_bytes
    if not math.isfinite(required):
        raise ValueError("workload.required_capacity_bytes must be finite")
    if required < 0:
        raise ValueError(
            "workload.required_capacity_bytes must be non-negative")

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
