"""Canonical aggregate-batch FEOL-NMP dense Decode execution."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.placement.nmp_locality_e2e import NMP_BANK_TO_LOCAL_ROUTE_DELAY_NS
from om3dthermal.placement.nmp_load_balance import (
    build_performance_balanced_placement,
)
from om3dthermal.power import (
    calculate_memory_power,
    calculate_physical_access_latency,
    derive_architecture_bandwidth,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.power.nmp_die_activity import (
    canonical_nmp_hardware,
    evaluate_nmp_die_activity,
)
from om3dthermal.power.nmp_die_power import build_nmp_die_power_map
from om3dthermal.resident_pages import ResidentDataObject
from om3dthermal.workload import (
    LLMDecodeInput,
    build_m3d_only_workload_objects,
    build_m3d_workload_page_demand,
    evaluate_llm_decode,
)


NMP_BATCH_GENERALIZATION_STATUS = "RESOLVED_ANALYTICAL_BATCH_MODEL"


class NMPDecodeBatchResult(BaseModel):
    """One synchronized batch Decode step or a physical-capacity failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(gt=0)
    active_capacity_requests: int = Field(gt=0)
    batch_model_status: Literal["RESOLVED_ANALYTICAL_BATCH_MODEL"]
    capacity_status: Literal["FEASIBLE", "CAPACITY_INFEASIBLE"]
    evaluation_status: Literal["EVALUATED", "CAPACITY_INFEASIBLE"]
    weight_batch_reuse_status: Literal["PASS"] = "PASS"
    no_cross_request_handoff_alias: Literal["PASS"] | None = None
    qk_av_paired_request_ownership: Literal["PASS"] | None = None

    logical_required_capacity_GB: float
    physical_page_rounded_capacity_GB: float
    available_capacity_GB: float
    capacity_margin_GB: float
    capacity_utilization: float
    max_die_capacity_utilization: float | None = None
    capacity_violations: int | None = None

    matrix_weight_read_bytes_per_step: float | None = None
    embedding_weight_read_bytes_per_step: float | None = None
    weight_read_bytes_per_step: float | None = None
    weight_read_bytes_per_token: float | None = None
    kv_read_bytes_per_step: float | None = None
    kv_write_bytes_per_step: float | None = None
    qk_flops_per_step: float | None = None
    av_flops_per_step: float | None = None
    score_bytes_per_step: float | None = None
    probability_bytes_per_step: float | None = None
    partial_bytes_per_step: float | None = None
    local_memory_bytes_per_step: float | None = None
    boundary_bytes_per_step: float | None = None

    decode_step_time_ms: float | None = None
    aggregate_decode_tokens_per_s: float | None = None
    per_sequence_TPOT_ms: float | None = None
    memory_dynamic_J_per_step: float | None = None
    mac_dynamic_J_per_step: float | None = None
    residual_interface_J_per_step: float | None = None
    softmax_gpu_dynamic_J_per_step: float | None = None
    remaining_gpu_dynamic_J_per_step: float | None = None
    gpu_dynamic_J_per_step: float | None = None
    gpu_static_J_per_step: float | None = None
    refresh_J_per_step: float | None = None
    total_J_per_step: float | None = None
    J_per_token: float | None = None
    tokens_per_J: float | None = None

    @model_validator(mode="after")
    def _evaluated_closure(self) -> "NMPDecodeBatchResult":
        if self.evaluation_status == "CAPACITY_INFEASIBLE":
            if self.capacity_status != "CAPACITY_INFEASIBLE":
                raise ValueError("capacity failure statuses do not close")
            return self
        required = (
            self.decode_step_time_ms, self.aggregate_decode_tokens_per_s,
            self.per_sequence_TPOT_ms, self.memory_dynamic_J_per_step,
            self.mac_dynamic_J_per_step, self.residual_interface_J_per_step,
            self.softmax_gpu_dynamic_J_per_step,
            self.remaining_gpu_dynamic_J_per_step,
            self.gpu_dynamic_J_per_step, self.gpu_static_J_per_step,
            self.refresh_J_per_step, self.total_J_per_step, self.J_per_token,
            self.tokens_per_J,
        )
        if any(value is None for value in required):
            raise ValueError("evaluated NMP batch result is incomplete")
        step_s = float(self.decode_step_time_ms) * 1e-3
        if not math.isclose(
            float(self.aggregate_decode_tokens_per_s), self.batch_size / step_s,
            rel_tol=1e-12,
        ):
            raise ValueError("aggregate Decode throughput does not close")
        if self.per_sequence_TPOT_ms != self.decode_step_time_ms:
            raise ValueError("per-sequence TPOT must equal synchronized step time")
        if not math.isclose(
            float(self.gpu_dynamic_J_per_step),
            float(self.softmax_gpu_dynamic_J_per_step)
            + float(self.remaining_gpu_dynamic_J_per_step),
            rel_tol=1e-12,
        ):
            raise ValueError("GPU dynamic energy does not close")
        total = (
            float(self.memory_dynamic_J_per_step)
            + float(self.mac_dynamic_J_per_step)
            + float(self.residual_interface_J_per_step)
            + float(self.gpu_dynamic_J_per_step)
            + float(self.gpu_static_J_per_step)
            + float(self.refresh_J_per_step)
        )
        if not math.isclose(float(self.total_J_per_step), total, rel_tol=1e-12):
            raise ValueError("NMP step energy does not close")
        if not math.isclose(float(self.J_per_token), total / self.batch_size, rel_tol=1e-12):
            raise ValueError("per-token energy does not close")
        if not math.isclose(float(self.tokens_per_J), self.batch_size / total, rel_tol=1e-12):
            raise ValueError("tokens/J does not close")
        return self


@dataclass(frozen=True)
class _NMPArchitecture:
    case: object
    geometry: object
    memory: object
    topology: object
    feol: object
    physical_latency: object
    layout: object
    bandwidth: object
    gpu_compute_flops_per_s: float


@lru_cache(maxsize=2)
def _resolve_architecture(project_root: Path) -> _NMPArchitecture:
    from om3dthermal.experiment import load_experiment_spec

    case = load_case_config(project_root / "configs/cases/orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    memory = calculate_memory_power(
        case, read_bandwidth_gbps=case.workload.read_bandwidth_gbps,
        project_root=project_root, geometry=geometry)
    topology = calculate_m3d_subarray(case.architecture.m3d_subarray, geometry.m3d)
    feol = calculate_feol_route(case.architecture, topology)
    physical_latency = calculate_physical_access_latency(
        case.architecture.physical_access_latency,
        feol_route=feol,
        miv_length_per_layer_um=memory.diagnostics["miv_length_per_layer_um"],
        miv_delay_per_layer_ns=memory.diagnostics["miv_delay_per_layer_ns"],
        miv_status=memory.diagnostics["miv_latency_status"],
        miv_parameter_status=memory.diagnostics["miv_resistance_parameter_status"],
        miv_provenance=memory.diagnostics["miv_resistance_provenance"],
    )
    layout = memory.physical_capacity_layout
    bandwidth = memory.architecture_bandwidth_closure
    if layout is None or bandwidth is None:
        raise ValueError("canonical M3D capacity/bandwidth did not resolve")
    experiment = load_experiment_spec(
        project_root / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml",
        project_root=project_root)
    return _NMPArchitecture(
        case, geometry, memory, topology, feol, physical_latency, layout,
        bandwidth, experiment.scenario.effective_compute_flops_per_second)


def _rounded_capacity_bytes(objects: tuple[ResidentDataObject, ...], page_bytes: int) -> int:
    return sum(
        ((item.size_bytes + page_bytes - 1) // page_bytes) * page_bytes
        for item in objects)


def evaluate_nmp_decode_batch(
    workload: LLMDecodeInput, *, project_root: str | Path,
    active_capacity_requests: int | None = None,
) -> NMPDecodeBatchResult:
    """Run one aggregate placement/activity/power solve; never loops over B1."""
    root = Path(project_root).resolve()
    architecture = _resolve_architecture(root)
    active = workload.batch_size if active_capacity_requests is None else active_capacity_requests
    if active < workload.batch_size:
        raise ValueError("active capacity requests cannot be below Decode batch size")
    capacity_workload = workload.model_copy(update={"batch_size": active})
    capacity_metrics = evaluate_llm_decode(capacity_workload)
    capacity_objects = build_m3d_only_workload_objects(capacity_workload)
    rounded = _rounded_capacity_bytes(
        capacity_objects, architecture.layout.slot_capacity_bytes)
    available = architecture.layout.total_capacity_bytes
    capacity_common = dict(
        batch_size=workload.batch_size, active_capacity_requests=active,
        batch_model_status=NMP_BATCH_GENERALIZATION_STATUS,
        logical_required_capacity_GB=capacity_metrics.required_capacity_bytes / 1e9,
        physical_page_rounded_capacity_GB=rounded / 1e9,
        available_capacity_GB=available / 1e9,
        capacity_margin_GB=(available - rounded) / 1e9,
        capacity_utilization=rounded / available,
    )
    if rounded > available:
        return NMPDecodeBatchResult(
            **capacity_common, capacity_status="CAPACITY_INFEASIBLE",
            evaluation_status="CAPACITY_INFEASIBLE")

    local_access_latency_ns = statistics.fmean(
        location.mat_latency_ns + location.miv_latency_ns
        + NMP_BANK_TO_LOCAL_ROUTE_DELAY_NS
        for location in architecture.physical_latency.locations)
    hardware = canonical_nmp_hardware(architecture.layout.slab_count)
    bandwidth_per_die = (
        architecture.bandwidth.local_service_groups_per_slab
        * architecture.bandwidth.read_payload_bytes_per_service
        / (architecture.bandwidth.service_cycle_scale
           * local_access_latency_ns * 1e-9))
    capacity_demand = build_m3d_workload_page_demand(
        capacity_workload, architecture.layout)
    capacity_placement = build_performance_balanced_placement(
        capacity_workload, capacity_demand, architecture.layout,
        bandwidth_per_die_bytes_per_s=bandwidth_per_die,
        compute_per_die_flops_per_s=hardware.peak_flops_per_die)
    if capacity_placement.capacity_violations:
        raise RuntimeError("page-feasible workload produced die capacity violations")

    demand = (
        capacity_demand if active == workload.batch_size
        else build_m3d_workload_page_demand(workload, architecture.layout))
    placement = (
        capacity_placement if active == workload.batch_size
        else build_performance_balanced_placement(
            workload, demand, architecture.layout,
            bandwidth_per_die_bytes_per_s=bandwidth_per_die,
            compute_per_die_flops_per_s=hardware.peak_flops_per_die))
    if placement.capacity_violations:
        raise RuntimeError("capacity-feasible workload produced die capacity violations")
    activity = evaluate_nmp_die_activity(
        workload, demand, architecture.layout, architecture.bandwidth,
        local_access_latency_ns=local_access_latency_ns,
        bandwidth_demand_bytes_per_s=(
            architecture.bandwidth.coil_bandwidth_bytes_per_s),
        ownership=placement.ownership)
    power = build_nmp_die_power_map(
        architecture.case, architecture.memory, architecture.topology,
        architecture.feol, activity, placement)

    handoff_keys = [
        (row["producer"], row["consumer"], row["layer_id"], row["request_id"])
        for row in activity.handoffs]
    if any(key[3] is None for key in handoff_keys) or len(handoff_keys) != len(set(handoff_keys)):
        raise RuntimeError("NO_CROSS_REQUEST_HANDOFF_ALIAS gate failed")
    for row in activity.attention_layers.values():
        if row["qk_request_owners"] != row["av_request_owners"]:
            raise RuntimeError("QK/AV paired request ownership gate failed")

    loads = placement.unit_loads
    matrix_weight = sum(
        item.weight_read_bytes for item in loads
        if item.unit.operator_type != "TOKEN_EMBED_LOOKUP")
    embedding_weight = sum(
        item.weight_read_bytes for item in loads
        if item.unit.operator_type == "TOKEN_EMBED_LOOKUP")
    weight = matrix_weight + embedding_weight
    kv_read = sum(item.kv_read_bytes for item in loads)
    kv_write = sum(item.kv_write_bytes for item in loads)
    qk_flops = sum(
        item.nmp_flops for item in loads
        if item.unit.operator_type == "ATTENTION_QK")
    av_flops = sum(
        item.nmp_flops for item in loads
        if item.unit.operator_type == "ATTENTION_AV")
    interval_s = activity.decode_step_interval_ms * 1e-3
    memory_J = (
        power.aggregate_memory_read_dynamic_W
        + power.aggregate_memory_write_dynamic_W) * interval_s
    mac_J = power.aggregate_mac_dynamic_W * interval_s
    residual_J = power.aggregate_residual_external_W * interval_s
    softmax_J = activity.softmax_dynamic_energy_j
    remaining_gpu_J = activity.gpu_remaining_dynamic_energy_j
    gpu_dynamic_J = softmax_J + remaining_gpu_J
    static_J = activity.gpu_static_energy_j
    refresh_J = power.aggregate_refresh_W * interval_s
    total_J = memory_J + mac_J + residual_J + gpu_dynamic_J + static_J + refresh_J
    return NMPDecodeBatchResult(
        **capacity_common, capacity_status="FEASIBLE", evaluation_status="EVALUATED",
        no_cross_request_handoff_alias="PASS",
        qk_av_paired_request_ownership="PASS",
        max_die_capacity_utilization=(
            capacity_placement.max_capacity_utilization),
        capacity_violations=capacity_placement.capacity_violations,
        matrix_weight_read_bytes_per_step=matrix_weight,
        embedding_weight_read_bytes_per_step=embedding_weight,
        weight_read_bytes_per_step=weight,
        weight_read_bytes_per_token=weight / workload.batch_size,
        kv_read_bytes_per_step=kv_read, kv_write_bytes_per_step=kv_write,
        qk_flops_per_step=qk_flops, av_flops_per_step=av_flops,
        score_bytes_per_step=activity.score_bytes,
        probability_bytes_per_step=activity.probability_bytes,
        partial_bytes_per_step=activity.partial_bytes,
        local_memory_bytes_per_step=sum(
            item.total_local_memory_bytes for item in activity.activities),
        boundary_bytes_per_step=activity.residual_boundary_bytes,
        decode_step_time_ms=activity.decode_step_interval_ms,
        aggregate_decode_tokens_per_s=workload.batch_size / interval_s,
        per_sequence_TPOT_ms=activity.decode_step_interval_ms,
        memory_dynamic_J_per_step=memory_J,
        mac_dynamic_J_per_step=mac_J,
        residual_interface_J_per_step=residual_J,
        softmax_gpu_dynamic_J_per_step=softmax_J,
        remaining_gpu_dynamic_J_per_step=remaining_gpu_J,
        gpu_dynamic_J_per_step=gpu_dynamic_J,
        gpu_static_J_per_step=static_J, refresh_J_per_step=refresh_J,
        total_J_per_step=total_J, J_per_token=total_J / workload.batch_size,
        tokens_per_J=workload.batch_size / total_J,
    )
