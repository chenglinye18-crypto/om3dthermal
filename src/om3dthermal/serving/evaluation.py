"""Compact two-tier capacity-aware serving evaluation."""

from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.platform import HostOffloadSpec, resolve_host_offload_power
from om3dthermal.workload import LLMDecodeInput, evaluate_llm_decode

from .gpu import GPUDecodePerformanceModel
from .residency import CapacityResidencyResult, UsableCapacity, evaluate_capacity_residency


class HostOverlapSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Literal["NO_OVERLAP", "FULL_OVERLAP", "PARTIAL_OVERLAP"]
    overlap_fraction: float = Field(ge=0.0, le=1.0)
    provenance_status: Literal["MODELING_CHOICE"] = "MODELING_CHOICE"

    @model_validator(mode="after")
    def _policy_fraction(self) -> "HostOverlapSpec":
        if self.policy == "NO_OVERLAP" and self.overlap_fraction != 0.0:
            raise ValueError("NO_OVERLAP requires overlap_fraction=0")
        if self.policy == "FULL_OVERLAP" and self.overlap_fraction != 1.0:
            raise ValueError("FULL_OVERLAP requires overlap_fraction=1")
        return self


class CapacityAwareServingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: str
    workload: str
    usable_capacity_bytes: float
    weight_bytes: float
    runtime_fixed_bytes: float
    runtime_per_request_bytes: float
    kv_bytes_per_request: float
    max_resident_requests: int | None
    requested_requests: int
    local_resident_requests: int
    spilled_requests: int
    local_capacity_utilization: float | None
    host_read_bytes_per_step: float
    host_write_bytes_per_step: float
    host_transfer_bytes_per_step: float
    host_memory_bandwidth_GBps: float | None
    host_device_link_bandwidth_GBps: float | None
    host_offload_efficiency: float | None
    host_effective_bandwidth_bytes_per_second: float | None
    host_bandwidth_demand_bytes_per_second: float | None
    host_bandwidth_actual_bytes_per_second: float | None
    host_bandwidth_saturated: bool | None
    pcie_dynamic_power_W: float | None
    ddr_dynamic_power_W: float | None
    host_offload_dynamic_power_W: float | None
    host_power_model_status: str
    host_static_power_status: Literal["UNRESOLVED"]
    host_transfer_time_ms: float | None
    host_penalty_time_ms: float | None
    host_overlap_policy: str
    host_overlap_fraction: float
    gpu_performance_backend: str
    gpu_decode_step_time_ms: float | None
    total_decode_step_time_ms: float | None
    inter_token_latency_ms: float | None
    aggregate_tokens_per_s: float | None
    capacity_status: str
    evaluation_status: Literal[
        "EVALUATED",
        "WEIGHTS_NOT_LOCAL",
        "UNRESOLVED_HOST_BANDWIDTH",
    ]
    spill_policy: Literal["FULL_KV_REQUIRED_PER_DECODE_STEP"]
    serving_semantics_status: Literal[
        "ONE_TOKEN_PER_ACTIVE_SEQUENCE_PER_DECODE_STEP"
    ]
    capacity_source_status: str
    gpu_backend_status: str | None
    host_model_status: str
    runtime_capacity_semantics_status: str


class ServingOperatingPointResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: str
    workload: str
    requested_request_points: tuple[int, ...]
    rows: tuple[CapacityAwareServingResult, ...]
    optimal_requested_requests: int | None
    optimal_aggregate_tokens_per_s: float | None
    search_status: Literal["OPTIMUM_FOUND", "NO_EVALUATED_POINT"]


def evaluate_capacity_aware_serving(
    *,
    architecture: str,
    workload_id: str,
    workload: LLMDecodeInput,
    capacity: UsableCapacity,
    requested_requests: int,
    host_offload: HostOffloadSpec,
    overlap: HostOverlapSpec,
    gpu_model: GPUDecodePerformanceModel,
) -> CapacityAwareServingResult:
    """Evaluate one steady decode step; every active request emits one token."""
    metrics = evaluate_llm_decode(
        workload.model_copy(update={"batch_size": requested_requests})
    )
    residency = evaluate_capacity_residency(
        metrics, capacity, requested_requests=requested_requests
    )
    common = _common(
        architecture=architecture,
        workload_id=workload_id,
        residency=residency,
        overlap=overlap,
        host_offload=host_offload,
    )
    if residency.capacity_status == "WEIGHTS_NOT_RESIDENT":
        host_power = _host_power_fields(
            host_offload=host_offload,
            host_transfer_bytes_per_step=0.0,
            host_transfer_time_ms=None,
        )
        return CapacityAwareServingResult(
            **common,
            **host_power,
            host_read_bytes_per_step=0.0,
            host_write_bytes_per_step=0.0,
            host_transfer_bytes_per_step=0.0,
            host_transfer_time_ms=None,
            host_penalty_time_ms=None,
            gpu_performance_backend="NOT_EVALUATED",
            gpu_decode_step_time_ms=None,
            total_decode_step_time_ms=None,
            inter_token_latency_ms=None,
            aggregate_tokens_per_s=None,
            evaluation_status="WEIGHTS_NOT_LOCAL",
            gpu_backend_status=None,
        )

    gpu = gpu_model.evaluate(workload, batch_size=requested_requests)
    host_read = residency.spilled_requests * metrics.kv_read_bytes_per_token
    host_write = residency.spilled_requests * metrics.kv_write_bytes_per_token
    host_total = host_read + host_write
    effective = host_offload.effective_bandwidth_bytes_per_second
    if host_total == 0.0:
        host_time_ms = 0.0
    elif effective is None:
        host_time_ms = None
    else:
        host_time_ms = host_total / effective * 1e3

    host_power = _host_power_fields(
        host_offload=host_offload,
        host_transfer_bytes_per_step=host_total,
        host_transfer_time_ms=host_time_ms,
    )

    if host_time_ms is None:
        return CapacityAwareServingResult(
            **common,
            **host_power,
            host_read_bytes_per_step=host_read,
            host_write_bytes_per_step=host_write,
            host_transfer_bytes_per_step=host_total,
            host_transfer_time_ms=None,
            host_penalty_time_ms=None,
            gpu_performance_backend=gpu.backend,
            gpu_decode_step_time_ms=gpu.decode_step_time_ms,
            total_decode_step_time_ms=None,
            inter_token_latency_ms=None,
            aggregate_tokens_per_s=None,
            evaluation_status="UNRESOLVED_HOST_BANDWIDTH",
            gpu_backend_status=gpu.backend_status,
        )

    host_penalty_ms = (1.0 - overlap.overlap_fraction) * host_time_ms
    total_ms = gpu.decode_step_time_ms + host_penalty_ms
    return CapacityAwareServingResult(
        **common,
        **host_power,
        host_read_bytes_per_step=host_read,
        host_write_bytes_per_step=host_write,
        host_transfer_bytes_per_step=host_total,
        host_transfer_time_ms=host_time_ms,
        host_penalty_time_ms=host_penalty_ms,
        gpu_performance_backend=gpu.backend,
        gpu_decode_step_time_ms=gpu.decode_step_time_ms,
        total_decode_step_time_ms=total_ms,
        inter_token_latency_ms=total_ms,
        aggregate_tokens_per_s=requested_requests / (total_ms * 1e-3),
        evaluation_status="EVALUATED",
        gpu_backend_status=gpu.backend_status,
    )


def search_serving_operating_point(
    *,
    architecture: str,
    workload_id: str,
    workload: LLMDecodeInput,
    capacity: UsableCapacity,
    requested_request_points: Sequence[int],
    host_offload: HostOffloadSpec,
    overlap: HostOverlapSpec,
    gpu_model: GPUDecodePerformanceModel,
) -> ServingOperatingPointResult:
    points = tuple(requested_request_points)
    if not points or len(set(points)) != len(points) or any(
        isinstance(point, bool) or not isinstance(point, int) or point <= 0
        for point in points
    ):
        raise ValueError("requested request points must be unique positive ints")
    rows = tuple(
        evaluate_capacity_aware_serving(
            architecture=architecture,
            workload_id=workload_id,
            workload=workload,
            capacity=capacity,
            requested_requests=point,
            host_offload=host_offload,
            overlap=overlap,
            gpu_model=gpu_model,
        )
        for point in points
    )
    evaluated = [row for row in rows if row.aggregate_tokens_per_s is not None]
    if evaluated:
        optimal = max(
            evaluated,
            key=lambda row: (row.aggregate_tokens_per_s, -row.requested_requests),
        )
        optimal_requests = optimal.requested_requests
        optimal_throughput = optimal.aggregate_tokens_per_s
        status = "OPTIMUM_FOUND"
    else:
        optimal_requests = None
        optimal_throughput = None
        status = "NO_EVALUATED_POINT"
    return ServingOperatingPointResult(
        architecture=architecture,
        workload=workload_id,
        requested_request_points=points,
        rows=rows,
        optimal_requested_requests=optimal_requests,
        optimal_aggregate_tokens_per_s=optimal_throughput,
        search_status=status,
    )


def _common(
    *,
    architecture: str,
    workload_id: str,
    residency: CapacityResidencyResult,
    overlap: HostOverlapSpec,
    host_offload: HostOffloadSpec,
) -> dict[str, object]:
    return {
        "architecture": architecture,
        "workload": workload_id,
        "usable_capacity_bytes": residency.usable_capacity_bytes,
        "weight_bytes": residency.weight_bytes,
        "runtime_fixed_bytes": residency.runtime_fixed_bytes,
        "runtime_per_request_bytes": residency.runtime_per_request_bytes,
        "kv_bytes_per_request": residency.kv_bytes_per_request,
        "max_resident_requests": residency.max_resident_requests,
        "requested_requests": residency.requested_requests,
        "local_resident_requests": residency.local_resident_requests,
        "spilled_requests": residency.spilled_requests,
        "local_capacity_utilization": residency.local_capacity_utilization,
        "host_effective_bandwidth_bytes_per_second": (
            host_offload.effective_bandwidth_bytes_per_second),
        "host_memory_bandwidth_GBps": (
            host_offload.host_memory_bandwidth_GBps),
        "host_device_link_bandwidth_GBps": (
            host_offload.host_device_link_bandwidth_GBps),
        "host_offload_efficiency": host_offload.host_offload_efficiency,
        "host_overlap_policy": overlap.policy,
        "host_overlap_fraction": overlap.overlap_fraction,
        "capacity_status": residency.capacity_status,
        "spill_policy": "FULL_KV_REQUIRED_PER_DECODE_STEP",
        "serving_semantics_status": (
            "ONE_TOKEN_PER_ACTIVE_SEQUENCE_PER_DECODE_STEP"),
        "capacity_source_status": residency.capacity_source_status,
        "host_model_status": host_offload.status,
        "host_power_model_status": host_offload.power_model_status,
        "host_static_power_status": host_offload.host_static_power_status,
        "runtime_capacity_semantics_status": (
            residency.runtime_capacity_semantics_status),
    }


def _host_power_fields(
    *,
    host_offload: HostOffloadSpec,
    host_transfer_bytes_per_step: float,
    host_transfer_time_ms: float | None,
) -> dict[str, object]:
    """Resolve the active-transfer operating point without changing timing."""
    if host_offload.power_model_status == "UNRESOLVED":
        return {
            "host_bandwidth_demand_bytes_per_second": None,
            "host_bandwidth_actual_bytes_per_second": None,
            "host_bandwidth_saturated": None,
            "pcie_dynamic_power_W": None,
            "ddr_dynamic_power_W": None,
            "host_offload_dynamic_power_W": None,
        }
    effective = host_offload.effective_bandwidth_bytes_per_second
    assert effective is not None
    assert host_offload.e_pcie_dynamic_J_per_bit is not None
    assert host_offload.e_ddr_dynamic_J_per_bit is not None
    if host_transfer_bytes_per_step == 0.0:
        demand = 0.0
    else:
        assert host_transfer_time_ms is not None and host_transfer_time_ms > 0.0
        # Frozen serving timing is bytes / BW_eff, so an active transfer is
        # exactly at the transport boundary.  Assign the exact configured
        # value to avoid a floating round-trip changing saturation semantics.
        demand = effective
    point = resolve_host_offload_power(
        host_transfer_demand_bytes_per_second=demand,
        host_effective_bandwidth_bytes_per_second=effective,
        e_pcie_dynamic_J_per_bit=host_offload.e_pcie_dynamic_J_per_bit,
        e_ddr_dynamic_J_per_bit=host_offload.e_ddr_dynamic_J_per_bit,
    )
    return {
        "host_bandwidth_demand_bytes_per_second": (
            point.host_bandwidth_demand_bytes_per_second),
        "host_bandwidth_actual_bytes_per_second": (
            point.host_bandwidth_actual_bytes_per_second),
        "host_bandwidth_saturated": point.host_bandwidth_saturated,
        "pcie_dynamic_power_W": point.pcie_dynamic_power_W,
        "ddr_dynamic_power_W": point.ddr_dynamic_power_W,
        "host_offload_dynamic_power_W": point.host_offload_dynamic_power_W,
    }
