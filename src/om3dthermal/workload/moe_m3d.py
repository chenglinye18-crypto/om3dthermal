"""M3D-only resident objects and page-rounded capacity for structural MoE."""

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

from .moe_decode import MoEDecodeInput, MoEDecodeMetrics, evaluate_moe_decode


class M3DMoECapacityError(ValueError):
    """The page-rounded, all-local MoE resident set does not fit M3D."""


class M3DMoEPhysicalPackingError(ValueError):
    """Analytical MoE byte-equivalents cannot form physical objects."""


@dataclass(frozen=True)
class M3DMoECapacityResult:
    requested_requests: int
    resident_objects: tuple[ResidentDataObject, ...]
    page_layout: ResidentPageLayout
    expert_object_count: int
    weight_logical_bytes: int
    kv_logical_bytes: int
    runtime_logical_bytes: int
    total_logical_bytes: int
    page_rounded_allocated_bytes: int
    occupancy_fraction: float
    capacity_status: str
    residency_semantics: str


def build_moe_resident_objects(
    workload: MoEDecodeInput,
    metrics: MoEDecodeMetrics | None = None,
) -> tuple[ResidentDataObject, ...]:
    """Represent every expert separately while keeping shared weights compact."""
    resolved = metrics if metrics is not None else evaluate_moe_decode(workload)
    expert_bytes = _whole_bytes(
        resolved.expert_footprint_bytes, "expert_footprint_bytes")
    objects = [ResidentDataObject(
        "weights.shared_nonexpert",
        "WEIGHT",
        _whole_bytes(
            resolved.nonexpert_footprint_bytes,
            "nonexpert_footprint_bytes",
        ),
    )]
    objects.extend(
        ResidentDataObject(
            f"expert.layer.{layer:02d}.expert.{expert:02d}",
            "WEIGHT",
            expert_bytes,
        )
        for layer in range(workload.num_hidden_layers)
        for expert in range(workload.num_local_experts)
    )
    kv_per_request = _whole_bytes(
        resolved.kv_bytes_per_request, "kv_bytes_per_request")
    if kv_per_request:
        objects.extend(
            ResidentDataObject(f"kv.request.{index}", "KV", kv_per_request)
            for index in range(workload.batch_size)
        )
    runtime = _whole_bytes(
        resolved.runtime_footprint_bytes, "runtime_footprint_bytes")
    if runtime:
        objects.append(ResidentDataObject("runtime", "OTHER", runtime))

    expected = _whole_bytes(
        resolved.required_capacity_bytes, "required_capacity_bytes")
    if sum(obj.size_bytes for obj in objects) != expected:
        raise M3DMoEPhysicalPackingError(
            "MoE resident objects do not close to required capacity")
    return tuple(objects)


def build_m3d_moe_capacity_layout(
    workload: MoEDecodeInput,
    physical_layout: PhysicalCapacityLayout,
) -> M3DMoECapacityResult:
    """Page-pack the explicitly all-local MoE set; never spill or offload."""
    metrics = evaluate_moe_decode(workload)
    objects = build_moe_resident_objects(workload, metrics)
    try:
        page_layout = build_resident_page_layout(objects, physical_layout)
    except ResidentCapacityExceededError as error:
        raise M3DMoECapacityError(
            "M3D_ONLY_CAPACITY_FAIL: page-rounded MoE resident set exceeds "
            "physical M3D capacity"
        ) from error
    return M3DMoECapacityResult(
        requested_requests=workload.batch_size,
        resident_objects=objects,
        page_layout=page_layout,
        expert_object_count=metrics.expert_count,
        weight_logical_bytes=_whole_bytes(
            metrics.total_weight_footprint_bytes,
            "total_weight_footprint_bytes",
        ),
        kv_logical_bytes=_whole_bytes(
            metrics.kv_footprint_bytes, "kv_footprint_bytes"),
        runtime_logical_bytes=_whole_bytes(
            metrics.runtime_footprint_bytes, "runtime_footprint_bytes"),
        total_logical_bytes=page_layout.logical_resident_bytes,
        page_rounded_allocated_bytes=page_layout.allocated_page_bytes,
        occupancy_fraction=(
            page_layout.allocated_page_bytes
            / physical_layout.total_capacity_bytes
        ),
        capacity_status="M3D_ONLY_PAGE_ALLOCATED_CAPACITY_PASS",
        residency_semantics="ALL_EXPERTS_STORED_TOP_K_EXPERTS_ACCESSED",
    )


def _whole_bytes(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M3DMoEPhysicalPackingError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise M3DMoEPhysicalPackingError(
            f"{field} must be finite, non-negative whole bytes")
    return int(numeric)
