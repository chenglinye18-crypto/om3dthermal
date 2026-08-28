"""M3D-only dense-workload objects, pages, and read-demand accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math

from om3dthermal.power.physical_capacity import PhysicalCapacityLayout
from om3dthermal.resident_pages import (
    ResidentCapacityExceededError,
    ResidentDataObject,
    ResidentPageLayout,
    build_resident_page_layout,
)

from .llm_decode import LLMDecodeInput, LLMDecodeMetrics, evaluate_llm_decode


class M3DOnlyCapacityError(ValueError):
    """The all-local, page-rounded workload does not fit physical M3D."""


class PageAccessDemandTrafficMismatchError(ValueError):
    """Page-level traffic does not close to existing workload traffic."""


class M3DWorkloadPhysicalPackingError(ValueError):
    """Analytical byte-equivalents cannot form whole-byte local objects."""


@dataclass(frozen=True)
class ResidentPageAccessDemand:
    page_id: str
    parent_object_id: str
    object_type: str
    logical_size_bytes: int
    read_demand_bytes_per_decode_step: float


@dataclass(frozen=True)
class M3DWorkloadPageDemand:
    requested_requests: int
    resident_objects: tuple[ResidentDataObject, ...]
    page_layout: ResidentPageLayout
    page_count: int
    weight_page_count: int
    kv_page_count: int
    other_page_count: int
    weight_footprint_bytes: int
    kv_footprint_bytes: int
    runtime_footprint_bytes: int
    logical_working_set_bytes: int
    allocated_page_bytes: int
    total_weight_read_bytes_per_decode_step: float
    total_kv_read_bytes_per_decode_step: float
    kv_write_bytes_per_decode_step: float
    total_read_bytes_per_decode_step: float
    page_demands: tuple[ResidentPageAccessDemand, ...]
    demand_min: float
    demand_median: float
    demand_p90: float
    demand_max: float
    top_10_percent_page_traffic_share: float
    top_25_percent_page_traffic_share: float
    demand_max_min_ratio: float | None
    weight_traffic_closure_error_bytes: float
    kv_traffic_closure_error_bytes: float
    total_traffic_closure_error_bytes: float
    weight_traffic_semantics: str
    kv_traffic_semantics: str
    runtime_read_demand_semantics: str
    capacity_status: str
    demand_status: str


def build_m3d_only_workload_objects(
    workload: LLMDecodeInput,
) -> tuple[ResidentDataObject, ...]:
    """Build the workload-declared all-local set without consulting C."""
    metrics = evaluate_llm_decode(workload)
    return _build_objects(workload, metrics)


def build_m3d_workload_page_demand(
    workload: LLMDecodeInput,
    physical_layout: PhysicalCapacityLayout,
) -> M3DWorkloadPageDemand:
    """Distribute existing dense decode-step reads over existing pages."""
    metrics = evaluate_llm_decode(workload)
    objects = _build_objects(workload, metrics)
    try:
        page_layout = build_resident_page_layout(objects, physical_layout)
    except ResidentCapacityExceededError as error:
        raise M3DOnlyCapacityError(
            "M3D_ONLY_CAPACITY_FAIL: page-rounded all-local workload exceeds "
            "physical M3D capacity"
        ) from error
    if page_layout.allocated_page_bytes > physical_layout.total_capacity_bytes:
        raise M3DOnlyCapacityError(
            "M3D_ONLY_CAPACITY_FAIL: allocated pages exceed physical capacity")

    weight_read = _finite_nonnegative(
        metrics.weight_active_per_step_bytes,
        "weight_active_per_step_bytes",
    )
    kv_read_per_request = _finite_nonnegative(
        metrics.kv_read_bytes_per_token,
        "kv_read_bytes_per_token",
    )
    kv_read = workload.batch_size * kv_read_per_request
    kv_write = workload.batch_size * _finite_nonnegative(
        metrics.kv_write_bytes_per_token,
        "kv_write_bytes_per_token",
    )
    workload_aggregate_read = workload.batch_size * _finite_nonnegative(
        metrics.read_bytes_per_token,
        "read_bytes_per_token",
    )
    _require_close(
        weight_read + kv_read,
        workload_aggregate_read,
        "existing aggregate decode-step read traffic",
    )

    object_by_id = {obj.object_id: obj for obj in objects}
    demands: list[ResidentPageAccessDemand] = []
    for page in page_layout.pages:
        parent = object_by_id[page.parent_object_id]
        if parent.object_type == "WEIGHT":
            object_read = weight_read
        elif parent.object_type == "KV":
            object_read = kv_read_per_request
        else:
            object_read = 0.0
        demand = object_read * page.size_bytes / parent.size_bytes
        demands.append(ResidentPageAccessDemand(
            page_id=page.page_id,
            parent_object_id=page.parent_object_id,
            object_type=page.object_type,
            logical_size_bytes=page.size_bytes,
            read_demand_bytes_per_decode_step=demand,
        ))
    demand_tuple = tuple(demands)
    for parent_id, parent in object_by_id.items():
        object_page_read = sum(
            item.read_demand_bytes_per_decode_step for item in demand_tuple
            if item.parent_object_id == parent_id)
        if parent.object_type == "WEIGHT":
            expected_object_read = weight_read
        elif parent.object_type == "KV":
            expected_object_read = kv_read_per_request
        else:
            expected_object_read = 0.0
        _require_close(
            object_page_read,
            expected_object_read,
            f"object {parent_id} page traffic",
        )
    weight_page_read = sum(
        item.read_demand_bytes_per_decode_step for item in demand_tuple
        if item.object_type == "WEIGHT")
    kv_page_read = sum(
        item.read_demand_bytes_per_decode_step for item in demand_tuple
        if item.object_type == "KV")
    total_page_read = sum(
        item.read_demand_bytes_per_decode_step for item in demand_tuple)
    _require_close(weight_page_read, weight_read, "weight page traffic")
    _require_close(kv_page_read, kv_read, "KV page traffic")
    _require_close(total_page_read, workload_aggregate_read, "total page traffic")

    values = tuple(
        item.read_demand_bytes_per_decode_step for item in demand_tuple)
    ordered = tuple(sorted(values))
    minimum = ordered[0]
    maximum = ordered[-1]
    total = sum(values)
    return M3DWorkloadPageDemand(
        requested_requests=workload.batch_size,
        resident_objects=objects,
        page_layout=page_layout,
        page_count=len(demand_tuple),
        weight_page_count=page_layout.pages_by_object_type["WEIGHT"],
        kv_page_count=page_layout.pages_by_object_type["KV"],
        other_page_count=page_layout.pages_by_object_type["OTHER"],
        weight_footprint_bytes=_whole_bytes(
            metrics.weight_footprint_bytes, "weight_footprint_bytes"),
        kv_footprint_bytes=_whole_bytes(
            metrics.kv_footprint_bytes, "kv_footprint_bytes"),
        runtime_footprint_bytes=_whole_bytes(
            metrics.runtime_bytes, "runtime_bytes"),
        logical_working_set_bytes=page_layout.logical_resident_bytes,
        allocated_page_bytes=page_layout.allocated_page_bytes,
        total_weight_read_bytes_per_decode_step=weight_read,
        total_kv_read_bytes_per_decode_step=kv_read,
        kv_write_bytes_per_decode_step=kv_write,
        total_read_bytes_per_decode_step=workload_aggregate_read,
        page_demands=demand_tuple,
        demand_min=minimum,
        demand_median=_percentile(ordered, 0.50),
        demand_p90=_percentile(ordered, 0.90),
        demand_max=maximum,
        top_10_percent_page_traffic_share=_top_fraction_share(values, total, 0.10),
        top_25_percent_page_traffic_share=_top_fraction_share(values, total, 0.25),
        demand_max_min_ratio=(None if minimum == 0.0 else maximum / minimum),
        weight_traffic_closure_error_bytes=weight_page_read - weight_read,
        kv_traffic_closure_error_bytes=kv_page_read - kv_read,
        total_traffic_closure_error_bytes=total_page_read - workload_aggregate_read,
        weight_traffic_semantics=(
            "EXISTING_AGGREGATE_DECODE_STEP_ACTIVE_WEIGHT_TRAFFIC_WITH_"
            "BATCH_TILE_REUSE_ALREADY_INCLUDED"),
        kv_traffic_semantics=(
            "EXISTING_EQUAL_CONTEXT_PER_REQUEST_FULL_REREAD_TRAFFIC"),
        runtime_read_demand_semantics=(
            "ZERO_UNTIL_WORKLOAD_MODEL_EXPOSES_RUNTIME_READ_TRAFFIC"),
        capacity_status="M3D_ONLY_PAGE_ALLOCATED_CAPACITY_PASS",
        demand_status="EXACT_EXISTING_WORKLOAD_READ_TRAFFIC_DISTRIBUTION",
    )


def _build_objects(
    workload: LLMDecodeInput,
    metrics: LLMDecodeMetrics,
) -> tuple[ResidentDataObject, ...]:
    weight = _whole_bytes(metrics.weight_footprint_bytes, "weight_footprint_bytes")
    kv_per_request = _whole_bytes(
        metrics.kv_bytes_per_request, "kv_bytes_per_request")
    runtime = _whole_bytes(metrics.runtime_bytes, "runtime_bytes")
    objects = [ResidentDataObject("weights", "WEIGHT", weight)]
    if kv_per_request:
        objects.extend(
            ResidentDataObject(f"kv.request.{index}", "KV", kv_per_request)
            for index in range(workload.batch_size)
        )
    if runtime:
        objects.append(ResidentDataObject("runtime", "OTHER", runtime))
    expected = _whole_bytes(metrics.required_capacity_bytes, "required_capacity_bytes")
    actual = sum(obj.size_bytes for obj in objects)
    if actual != expected:
        raise M3DWorkloadPhysicalPackingError(
            "M3D workload objects do not close to required capacity")
    return tuple(objects)


def _whole_bytes(value: object, field: str) -> int:
    numeric = _finite_nonnegative(value, field)
    if not numeric.is_integer():
        raise M3DWorkloadPhysicalPackingError(
            f"{field} is an analytical fractional-byte value and cannot be "
            "silently converted to a physical resident object")
    return int(numeric)


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M3DWorkloadPhysicalPackingError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise M3DWorkloadPhysicalPackingError(
            f"{field} must be finite and non-negative")
    return numeric


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise PageAccessDemandTrafficMismatchError(
            "PAGE_ACCESS_DEMAND_TRAFFIC_MISMATCH: "
            f"{label}: actual={actual}, expected={expected}")


def _percentile(ordered: tuple[float, ...], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _top_fraction_share(
    values: tuple[float, ...],
    total: float,
    fraction: float,
) -> float:
    count = math.ceil(len(values) * fraction)
    if total == 0.0:
        return 0.0
    return sum(sorted(values, reverse=True)[:count]) / total
