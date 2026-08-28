"""Translate serving-C residency accounting into already-local A objects."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from om3dthermal.power.physical_capacity import PhysicalCapacityLayout
from om3dthermal.resident_pages import (
    ResidentCapacityExceededError,
    ResidentDataObject,
    ResidentPageLayout,
    build_resident_page_layout,
)


REQUEST_ID_SEMANTICS = "DETERMINISTIC_ACCOUNTING_ID"
REQUEST_ORDERING_SEMANTICS = (
    "RESIDENT_REQUESTS_INDEXED_0_TO_N_MINUS_1_NO_PRIORITY_HOTNESS_OR_"
    "SCHEDULING_MEANING"
)


class ServingResidencySource(Protocol):
    """Fields shared by the existing C residency and serving results."""

    architecture: str
    weight_bytes: float
    runtime_fixed_bytes: float
    runtime_per_request_bytes: float
    kv_bytes_per_request: float
    requested_requests: int
    local_resident_requests: int
    spilled_requests: int
    capacity_status: str


class CAResidentAccountingError(ValueError):
    """C aggregate residency fields cannot close to explicit A objects."""


class CACapacitySemanticsMismatchError(ValueError):
    """C logical residency and A page-allocated capacity disagree."""


@dataclass(frozen=True)
class ResidentSetAdapterResult:
    source_serving_case: str
    resident_objects: tuple[ResidentDataObject, ...]
    resident_object_count: int
    resident_weight_object_count: int
    resident_kv_object_count: int
    resident_runtime_object_count: int
    requested_request_count: int
    resident_request_count: int
    spilled_request_count: int
    resident_weight_bytes: int
    resident_kv_bytes: int
    resident_runtime_bytes: int
    total_resident_logical_bytes: int
    c_reported_local_resident_bytes: int
    byte_closure_error: int
    request_id_semantics: str
    request_ordering_semantics: str
    adapter_status: str
    provenance: str


@dataclass(frozen=True)
class ResidentSetPageIntegrationResult:
    resident_set: ResidentSetAdapterResult
    page_layout: ResidentPageLayout
    logical_capacity_status: str
    page_allocated_capacity_status: str
    integration_status: str


def build_resident_objects_from_serving_residency(
    residency: ServingResidencySource,
) -> ResidentSetAdapterResult:
    """Translate C's decision without changing who is resident or spilled."""
    requested = _nonnegative_int(
        residency.requested_requests, "requested_requests")
    local = _nonnegative_int(
        residency.local_resident_requests, "local_resident_requests")
    spilled = _nonnegative_int(residency.spilled_requests, "spilled_requests")
    if local + spilled != requested:
        raise CAResidentAccountingError(
            "C resident and spilled request counts do not close")

    weight = _whole_bytes(residency.weight_bytes, "weight_bytes")
    runtime_fixed = _whole_bytes(
        residency.runtime_fixed_bytes, "runtime_fixed_bytes")
    runtime_per_request = _whole_bytes(
        residency.runtime_per_request_bytes, "runtime_per_request_bytes")
    kv_per_request = _whole_bytes(
        residency.kv_bytes_per_request, "kv_bytes_per_request")

    # C V0 has all-or-nothing weight residency.  Its WEIGHTS_NOT_RESIDENT
    # status establishes no executable local resident set.
    weights_local = residency.capacity_status != "WEIGHTS_NOT_RESIDENT"
    if not weights_local and local != 0:
        raise CAResidentAccountingError(
            "WEIGHTS_NOT_RESIDENT cannot contain local resident requests")

    objects: list[ResidentDataObject] = []
    resident_weight = weight if weights_local else 0
    resident_runtime_fixed = runtime_fixed if weights_local else 0
    resident_runtime_per_request = local * runtime_per_request
    resident_kv = local * kv_per_request
    if resident_weight:
        objects.append(ResidentDataObject("weights", "WEIGHT", resident_weight))
    if resident_runtime_fixed:
        objects.append(ResidentDataObject(
            "runtime.fixed", "OTHER", resident_runtime_fixed))
    for request_index in range(local):
        if kv_per_request:
            objects.append(ResidentDataObject(
                f"kv.request.{request_index}", "KV", kv_per_request))
        if runtime_per_request:
            objects.append(ResidentDataObject(
                f"runtime.request.{request_index}",
                "OTHER",
                runtime_per_request,
            ))

    object_tuple = tuple(objects)
    adapter_bytes = sum(obj.size_bytes for obj in object_tuple)
    c_bytes = (
        resident_weight
        + resident_runtime_fixed
        + local * (kv_per_request + runtime_per_request)
    )
    closure_error = adapter_bytes - c_bytes
    if closure_error != 0:
        raise CAResidentAccountingError(
            f"C-to-A resident byte closure failed by {closure_error} bytes")
    source_workload = getattr(residency, "workload", None)
    source_case = residency.architecture
    if source_workload:
        source_case = f"{source_case}:{source_workload}"
    return ResidentSetAdapterResult(
        source_serving_case=source_case,
        resident_objects=object_tuple,
        resident_object_count=len(object_tuple),
        resident_weight_object_count=sum(
            obj.object_type == "WEIGHT" for obj in object_tuple),
        resident_kv_object_count=sum(
            obj.object_type == "KV" for obj in object_tuple),
        resident_runtime_object_count=sum(
            obj.object_type == "OTHER" for obj in object_tuple),
        requested_request_count=requested,
        resident_request_count=local,
        spilled_request_count=spilled,
        resident_weight_bytes=resident_weight,
        resident_kv_bytes=resident_kv,
        resident_runtime_bytes=(
            resident_runtime_fixed + resident_runtime_per_request),
        total_resident_logical_bytes=adapter_bytes,
        c_reported_local_resident_bytes=c_bytes,
        byte_closure_error=closure_error,
        request_id_semantics=REQUEST_ID_SEMANTICS,
        request_ordering_semantics=REQUEST_ORDERING_SEMANTICS,
        adapter_status="EXACT_C_RESIDENCY_TRANSLATION",
        provenance="EXISTING_TYPED_SERVING_C_RESIDENCY_FIELDS",
    )


def build_resident_pages_from_serving_residency(
    residency: ServingResidencySource,
    physical_layout: PhysicalCapacityLayout,
) -> ResidentSetPageIntegrationResult:
    """Run the existing A pager and expose logical/page capacity mismatch."""
    resident_set = build_resident_objects_from_serving_residency(residency)
    if resident_set.total_resident_logical_bytes > (
            physical_layout.total_capacity_bytes):
        raise CACapacitySemanticsMismatchError(
            "C_A_CAPACITY_SEMANTICS_MISMATCH: LOGICAL_CAPACITY_FAIL")
    try:
        page_layout = build_resident_page_layout(
            resident_set.resident_objects, physical_layout)
    except ResidentCapacityExceededError as error:
        raise CACapacitySemanticsMismatchError(
            "C_A_CAPACITY_SEMANTICS_MISMATCH: LOGICAL_CAPACITY_PASS; "
            "PAGE_ALLOCATED_CAPACITY_FAIL"
        ) from error
    if page_layout.logical_resident_bytes != (
            resident_set.total_resident_logical_bytes):
        raise CAResidentAccountingError(
            "adapter object bytes and A page logical bytes do not close")
    return ResidentSetPageIntegrationResult(
        resident_set=resident_set,
        page_layout=page_layout,
        logical_capacity_status="LOGICAL_CAPACITY_PASS",
        page_allocated_capacity_status="PAGE_ALLOCATED_CAPACITY_PASS",
        integration_status="C_TO_A_PAGE_INTEGRATION_PASS",
    )


def _whole_bytes(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CAResidentAccountingError(f"{field} must be numeric whole bytes")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise CAResidentAccountingError(
            f"{field} must be finite non-negative whole bytes")
    return int(numeric)


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CAResidentAccountingError(
            f"{field} must be a non-negative int")
    return value
