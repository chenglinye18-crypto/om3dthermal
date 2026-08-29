"""Page-level Mixtral read demand from a published expert profile."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from om3dthermal.power.physical_capacity import PhysicalCapacityLayout
from om3dthermal.resident_pages import (
    ResidentCapacityExceededError,
    ResidentDataObject,
    ResidentPageLayout,
    build_resident_page_layout,
)

from .m3d_page_demand import ResidentPageAccessDemand
from .moe_decode import MoEDecodeInput, evaluate_moe_decode
from .moe_m3d import build_moe_resident_objects
from .moe_published_profile import FiddlerPublishedProfile


RoutingDemandSource = Literal[
    "FIDDLER_PUBLISHED_ROUTING_PROFILE",
    "UNIFORM_MOE_ROUTING_CONTROL",
]


@dataclass(frozen=True)
class MoEPublishedPageDemand:
    requested_requests: int
    resident_objects: tuple[ResidentDataObject, ...]
    page_layout: ResidentPageLayout
    page_count: int
    expert_page_count: int
    shared_weight_page_count: int
    kv_page_count: int
    other_page_count: int
    logical_working_set_bytes: int
    allocated_page_bytes: int
    total_expert_read_bytes_per_decode_step: float
    total_shared_weight_read_bytes_per_decode_step: float
    total_kv_read_bytes_per_decode_step: float
    kv_write_bytes_per_decode_step: float
    total_read_bytes_per_decode_step: float
    page_demands: tuple[ResidentPageAccessDemand, ...]
    expert_traffic_closure_error_bytes: float
    shared_weight_traffic_closure_error_bytes: float
    kv_traffic_closure_error_bytes: float
    total_traffic_closure_error_bytes: float
    routing_demand_source: RoutingDemandSource
    expert_internal_demand_semantics: str
    shared_weight_demand_semantics: str
    kv_demand_semantics: str
    capacity_status: str


@dataclass(frozen=True)
class PageDemandView:
    """Placement-compatible immutable subset of an existing page demand."""

    requested_requests: int
    page_layout: ResidentPageLayout
    page_count: int
    allocated_page_bytes: int
    total_read_bytes_per_decode_step: float
    page_demands: tuple[ResidentPageAccessDemand, ...]
    view_semantics: str


def build_published_moe_page_demand(
    profile: FiddlerPublishedProfile,
    workload: MoEDecodeInput,
    physical_layout: PhysicalCapacityLayout,
    *,
    routing_source: RoutingDemandSource = "FIDDLER_PUBLISHED_ROUTING_PROFILE",
) -> MoEPublishedPageDemand:
    """Map existing MoE objects to 2 MiB pages without synthetic hotness."""
    metrics = evaluate_moe_decode(workload)
    if profile.model_id != workload.model_id:
        raise ValueError("published profile model does not match workload")
    if (profile.num_layers, profile.num_experts, profile.top_k) != (
        workload.num_hidden_layers,
        workload.num_local_experts,
        workload.num_experts_per_tok,
    ):
        raise ValueError("published profile dimensions do not match workload")
    if routing_source == "FIDDLER_PUBLISHED_ROUTING_PROFILE":
        probabilities = profile.selection_probability
    elif routing_source == "UNIFORM_MOE_ROUTING_CONTROL":
        probability = (
            workload.num_experts_per_tok / workload.num_local_experts)
        probabilities = tuple(
            tuple(probability for _ in range(workload.num_local_experts))
            for _ in range(workload.num_hidden_layers)
        )
    else:
        raise ValueError(f"unsupported routing demand source: {routing_source}")
    for layer, row in enumerate(probabilities):
        if len(row) != workload.num_local_experts:
            raise ValueError(f"routing row {layer} has the wrong expert count")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
               for value in row):
            raise ValueError(f"routing row {layer} has invalid probabilities")
        if not math.isclose(
                math.fsum(row), workload.num_experts_per_tok,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"routing row {layer} does not close to top-k")

    objects = build_moe_resident_objects(workload, metrics)
    try:
        page_layout = build_resident_page_layout(objects, physical_layout)
    except ResidentCapacityExceededError as error:
        raise ValueError(
            "M3D_ONLY_CAPACITY_FAIL: published MoE pages exceed physical M3D"
        ) from error

    expert_reads = {
        f"expert.layer.{layer:02d}.expert.{expert:02d}": (
            probabilities[layer][expert] * metrics.expert_footprint_bytes)
        for layer in range(workload.num_hidden_layers)
        for expert in range(workload.num_local_experts)
    }
    shared_read = metrics.active_nonexpert_weight_bytes_per_decode_step
    kv_read_per_request = metrics.kv_read_bytes_per_token_per_request
    kv_write = (
        workload.batch_size * metrics.kv_write_bytes_per_token_per_request)
    object_by_id = {item.object_id: item for item in objects}
    object_reads: dict[str, float] = {}
    for item in objects:
        if item.object_id == "weights.shared_nonexpert":
            object_reads[item.object_id] = shared_read
        elif item.object_id in expert_reads:
            object_reads[item.object_id] = expert_reads[item.object_id]
        elif item.object_type == "KV":
            object_reads[item.object_id] = kv_read_per_request
        else:
            object_reads[item.object_id] = 0.0

    page_demands = tuple(
        ResidentPageAccessDemand(
            page_id=page.page_id,
            parent_object_id=page.parent_object_id,
            object_type=page.object_type,
            logical_size_bytes=page.size_bytes,
            read_demand_bytes_per_decode_step=(
                object_reads[page.parent_object_id]
                * page.size_bytes
                / object_by_id[page.parent_object_id].size_bytes
            ),
        )
        for page in page_layout.pages
    )
    for object_id, expected in object_reads.items():
        actual = math.fsum(
            page.read_demand_bytes_per_decode_step for page in page_demands
            if page.parent_object_id == object_id)
        _require_close(actual, expected, f"object {object_id}")

    expert_total = math.fsum(expert_reads.values())
    kv_total = workload.batch_size * kv_read_per_request
    expected_total = expert_total + shared_read + kv_total
    page_expert_total = math.fsum(
        page.read_demand_bytes_per_decode_step for page in page_demands
        if page.parent_object_id.startswith("expert."))
    page_shared_total = math.fsum(
        page.read_demand_bytes_per_decode_step for page in page_demands
        if page.parent_object_id == "weights.shared_nonexpert")
    page_kv_total = math.fsum(
        page.read_demand_bytes_per_decode_step for page in page_demands
        if page.object_type == "KV")
    page_total = math.fsum(
        page.read_demand_bytes_per_decode_step for page in page_demands)
    _require_close(
        expert_total, metrics.active_expert_weight_bytes_per_decode_step,
        "active expert traffic")
    _require_close(page_expert_total, expert_total, "expert page traffic")
    _require_close(page_shared_total, shared_read, "shared page traffic")
    _require_close(page_kv_total, kv_total, "KV page traffic")
    _require_close(page_total, expected_total, "all-read page traffic")

    expert_page_count = sum(
        page.parent_object_id.startswith("expert.") for page in page_demands)
    shared_page_count = sum(
        page.parent_object_id == "weights.shared_nonexpert"
        for page in page_demands)
    kv_page_count = sum(page.object_type == "KV" for page in page_demands)
    return MoEPublishedPageDemand(
        requested_requests=workload.batch_size,
        resident_objects=objects,
        page_layout=page_layout,
        page_count=len(page_demands),
        expert_page_count=expert_page_count,
        shared_weight_page_count=shared_page_count,
        kv_page_count=kv_page_count,
        other_page_count=(
            len(page_demands) - expert_page_count
            - shared_page_count - kv_page_count),
        logical_working_set_bytes=page_layout.logical_resident_bytes,
        allocated_page_bytes=page_layout.allocated_page_bytes,
        total_expert_read_bytes_per_decode_step=expert_total,
        total_shared_weight_read_bytes_per_decode_step=shared_read,
        total_kv_read_bytes_per_decode_step=kv_total,
        kv_write_bytes_per_decode_step=kv_write,
        total_read_bytes_per_decode_step=expected_total,
        page_demands=page_demands,
        expert_traffic_closure_error_bytes=page_expert_total - expert_total,
        shared_weight_traffic_closure_error_bytes=(
            page_shared_total - shared_read),
        kv_traffic_closure_error_bytes=page_kv_total - kv_total,
        total_traffic_closure_error_bytes=page_total - expected_total,
        routing_demand_source=routing_source,
        expert_internal_demand_semantics=(
            "FULL_EXPERT_READ_DISTRIBUTED_PROPORTIONAL_TO_LOGICAL_PAGE_BYTES"),
        shared_weight_demand_semantics=(
            "EXISTING_ACTIVE_NONEXPERT_FULL_READ_DISTRIBUTED_BY_LOGICAL_BYTES"),
        kv_demand_semantics=(
            "EXISTING_EQUAL_CONTEXT_PER_REQUEST_FULL_REREAD"),
        capacity_status="M3D_ONLY_PAGE_ALLOCATED_CAPACITY_PASS",
    )


def expert_only_page_demand_view(
    demand: MoEPublishedPageDemand,
) -> PageDemandView:
    pages = tuple(
        page for page in demand.page_demands
        if page.parent_object_id.startswith("expert."))
    total = math.fsum(
        page.read_demand_bytes_per_decode_step for page in pages)
    _require_close(
        total, demand.total_expert_read_bytes_per_decode_step,
        "expert-only demand view")
    return PageDemandView(
        requested_requests=demand.requested_requests,
        page_layout=demand.page_layout,
        page_count=len(pages),
        allocated_page_bytes=len(pages) * demand.page_layout.page_size_bytes,
        total_read_bytes_per_decode_step=total,
        page_demands=pages,
        view_semantics="EXPERT_PAGES_ONLY_NO_SHARED_WEIGHT_OR_KV",
    )


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            f"PUBLISHED_MOE_PAGE_TRAFFIC_MISMATCH: {label}: "
            f"actual={actual}, expected={expected}")
