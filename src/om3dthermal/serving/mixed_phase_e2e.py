"""Canonical non-thermal mixed Prefill/Decode E2E composition.

The module owns final serving aggregation.  Physical workload, power,
capacity, host-offload, and GPU primitives remain in their existing modules.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.architecture_capacity import resolve_architecture_capacity
from om3dthermal.platform import (
    HostOffloadSpec,
    load_platform_spec_file,
    resolve_gpu_bandwidth_service,
    resolve_local_memory_gpu_transfer,
    resolve_gpu_decode_power,
    resolve_gpu_prefill_compute_energy_calibration,
    resolve_host_offload_power,
)
from om3dthermal.power import (
    load_case_config,
    resolve_case_geometry,
    resolve_system_power,
)
from om3dthermal.power.memory_bandwidth import resolve_internal_service_bandwidth
from om3dthermal.power.nmp_die_power import resolve_orthogonal_m3d_write_energy_pj_per_bit
from om3dthermal.workload import (
    DenseLLMModelSpec,
    evaluate_gpu_prefill_roofline,
    evaluate_llm_decode,
    evaluate_llm_prefill,
    build_m3d_only_workload_objects,
)

from .gpu import AnalyticalRooflineGPUModel
from .nmp_decode import (
    resolve_m3d_architecture_backend,
    rounded_capacity_bytes,
)
from .residency import ServingCapacitySource, evaluate_capacity_residency


SystemId = Literal[
    "CONVENTIONAL_HBM_GPU",
    "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY",
    "IOM3D_FEOL_NMP",
]
CONVENTIONAL_HBM_WRITE_ENERGY_STATUS = "UNRESOLVED"


class SystemConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    system_id: SystemId
    comparison_role: Literal["BASELINE", "ABLATION", "PROPOSED"]
    memory: Literal["CONVENTIONAL_HBM3E", "ORTHOGONAL_M3D_IGZO"]
    prefill_executor: Literal["GPU"] = "GPU"
    decode_executor: Literal["GPU", "FEOL_NMP_GPU_HYBRID"]


SYSTEM_CONFIGURATIONS: dict[str, SystemConfiguration] = {
    item.system_id: item
    for item in (
        SystemConfiguration(
            system_id="CONVENTIONAL_HBM_GPU", comparison_role="BASELINE",
            memory="CONVENTIONAL_HBM3E", decode_executor="GPU"),
        SystemConfiguration(
            system_id="ORTHOGONAL_M3D_IGZO_MEMORY_ONLY",
            comparison_role="ABLATION", memory="ORTHOGONAL_M3D_IGZO",
            decode_executor="GPU"),
        SystemConfiguration(
            system_id="IOM3D_FEOL_NMP", comparison_role="PROPOSED",
            memory="ORTHOGONAL_M3D_IGZO",
            decode_executor="FEOL_NMP_GPU_HYBRID"),
    )
}


class MixedPhaseServingCase(BaseModel):
    """P:D is a request count, and the two phase batches are P and D."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    context_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    prefill_requests: int = Field(gt=0)
    decode_requests: int = Field(gt=0)

    @model_validator(mode="after")
    def _batch_closure(self) -> "MixedPhaseServingCase":
        if self.prefill_requests + self.decode_requests != self.batch_size:
            raise ValueError("prefill_requests + decode_requests must equal batch_size")
        return self


class FinalDenseE2EMatrixSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str
    model_registry_dir: str
    models: tuple[str, ...]
    context_length: int = Field(gt=0)
    mixed_points: tuple[dict[str, int], ...]
    systems: tuple[str, ...]
    phase_overlap_policy: Literal["NO_OVERLAP"]

    @model_validator(mode="after")
    def _matrix_closure(self) -> "FinalDenseE2EMatrixSpec":
        if len(set(self.models)) != len(self.models):
            raise ValueError("models must be unique")
        if set(self.systems) != set(SYSTEM_CONFIGURATIONS):
            raise ValueError("final matrix must contain exactly the three frozen systems")
        for point in self.mixed_points:
            if set(point) != {"batch_size", "prefill_requests", "decode_requests"}:
                raise ValueError("mixed point has unexpected fields")
            MixedPhaseServingCase(
                model_id=self.models[0], context_length=self.context_length,
                **point,
            )
        return self


def load_final_dense_e2e_matrix(path: str | Path) -> FinalDenseE2EMatrixSpec:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("final dense E2E matrix root must be a mapping")
    return FinalDenseE2EMatrixSpec.model_validate(raw)


class ConventionalHBMBackend(BaseModel):
    """Resolved canonical HBM/platform facts; it contains no fitted constants."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    architecture: str
    capacity_bytes: float
    peak_bandwidth_bytes_per_s: float
    sustained_bandwidth_bytes_per_s: float
    read_energy_pJ_per_bit: float
    write_energy_pJ_per_bit: float | None
    refresh_power_W: float
    hbm_read_energy_status: Literal["DREAMRAM_RD_ACCESS_RESOLVED"]
    hbm_write_energy_status: Literal["UNRESOLVED"]
    hbm_write_energy_reason: str
    capacity_source_status: str
    bandwidth_source_status: str
    host_offload: HostOffloadSpec


def resolve_conventional_hbm_backend(
    project_root: str | Path,
) -> ConventionalHBMBackend:
    """Resolve conventional HBM from the canonical case and H200 platform."""
    root = Path(project_root)
    case = load_case_config(root / "configs/cases/conventional_hbm_2x1.yaml")
    platform = load_platform_spec_file(
        root / "configs/platform/gpu_package_h200_reference.yaml")
    if (platform.gpu_decode_power is None or platform.host_offload is None):
        raise ValueError("canonical baseline requires GPU and host-offload platform data")
    geometry = resolve_case_geometry(case)
    gpu_spec = platform.gpu_decode_power
    idle_point = resolve_gpu_decode_power(
        static_power_W=gpu_spec.static_power_W,
        e_decode_J_per_bit=gpu_spec.e_decode_J_per_bit,
        bandwidth_demand_bytes_per_s=0.0,
        peak_bandwidth_bytes_per_s=gpu_spec.peak_memory_bandwidth_bytes_per_s,
    )
    system = resolve_system_power(
        case, project_root=root, geometry=geometry,
        gpu_operating_point=idle_point, transfer_operating_point=None,
    )
    capacity = resolve_architecture_capacity(case, geometry, system)
    if system.memory_access_energy_pJ_per_bit is None or system.refresh_power_W is None:
        raise ValueError("canonical HBM read energy and refresh must resolve")
    bandwidth = resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=gpu_spec.peak_memory_bandwidth_bytes_per_s,
        service_status=platform.gpu_bandwidth_service.service_status,
        provenance=platform.gpu_bandwidth_service.provenance,
    )
    return ConventionalHBMBackend(
        architecture=case.name,
        capacity_bytes=capacity.system_capacity_bytes,
        peak_bandwidth_bytes_per_s=gpu_spec.peak_memory_bandwidth_bytes_per_s,
        sustained_bandwidth_bytes_per_s=bandwidth.sustained_bandwidth_bytes_per_s,
        read_energy_pJ_per_bit=system.memory_access_energy_pJ_per_bit,
        write_energy_pJ_per_bit=None,
        refresh_power_W=system.refresh_power_W,
        hbm_read_energy_status="DREAMRAM_RD_ACCESS_RESOLVED",
        hbm_write_energy_status="UNRESOLVED",
        hbm_write_energy_reason=(
            "DreamRAM canonical backend exposes PRE/ACT/RD read-access energy; "
            "it has no WR command or documented symmetric read/write semantics"),
        capacity_source_status=capacity.source_status,
        bandwidth_source_status=bandwidth.service_status,
        host_offload=platform.host_offload,
    )


class MixedPhaseE2EResult(BaseModel):
    """Unified, non-thermal performance/energy/mechanism result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_id: SystemId
    comparison_role: str
    model_id: str
    context_length: int
    batch_size: int
    prefill_requests: int
    decode_requests: int
    phase_overlap_policy: Literal["NO_OVERLAP"] = "NO_OVERLAP"
    phase_batch_semantics: Literal["PREFILL_BATCH_P__DECODE_BATCH_D"] = (
        "PREFILL_BATCH_P__DECODE_BATCH_D")
    evaluation_status: str
    energy_status: str
    nmp_batch_generalization_status: str
    thermal: None = None

    TTFT_ms: float | None = None
    TPOT_ms: float | None = None
    prefill_service_time_ms: float | None = None
    decode_service_time_ms: float | None = None
    mixed_epoch_time_ms: float | None = None
    decode_tokens_per_s: float | None = None
    normalized_e2e_speed: None = None

    prefill_gpu_dynamic_J: float | None = None
    prefill_gpu_dynamic_J_min: float | None = None
    prefill_gpu_dynamic_J_max: float | None = None
    prefill_gpu_static_J: float | None = None
    prefill_memory_read_dynamic_J: float | None = None
    prefill_memory_write_dynamic_J: float | None = None
    prefill_memory_dynamic_J: float | None = None
    prefill_refresh_J: float | None = None
    prefill_host_ddr_J: float | None = None
    prefill_host_pcie_J: float | None = None
    prefill_total_J: float | None = None
    decode_gpu_dynamic_J: float | None = None
    decode_nmp_mac_dynamic_J: float = 0.0
    decode_residual_interface_J: float = 0.0
    decode_gpu_static_J: float | None = None
    decode_memory_read_dynamic_J: float | None = None
    decode_memory_write_dynamic_J: float | None = None
    decode_memory_dynamic_J: float | None = None
    decode_refresh_J: float | None = None
    decode_host_ddr_J: float | None = None
    decode_host_pcie_J: float | None = None
    decode_total_J: float | None = None
    decode_tokens_per_J: float | None = None
    mixed_total_energy_J: float | None = None
    mixed_service_tokens_per_J: float | None = None

    required_capacity_GB: float | None = None
    physical_page_rounded_capacity_GB: float | None = None
    capacity_margin_GB: float | None = None
    capacity_utilization: float | None = None
    max_die_capacity_utilization: float | None = None
    capacity_violations: int | None = None
    local_capacity_GB: float | None = None
    resident_requests: int | None = None
    spilled_requests: int | None = None
    resident_fraction: float | None = None
    resident_decode_requests: int | None = None
    spilled_decode_requests: int | None = None
    resident_prefill_requests: int | None = None
    spilled_prefill_requests: int | None = None
    local_KV_GB: float | None = None
    host_KV_GB: float | None = None
    host_read_GB: float | None = None
    host_write_GB: float | None = None
    capacity_status: str | None = None
    residency_policy: str | None = None
    residency_policy_status: str | None = None
    prefill_spill_semantics: str | None = None
    hbm_read_energy_pJ_per_bit: float | None = None
    hbm_write_energy_pJ_per_bit: float | None = None
    hbm_write_energy_status: str | None = None
    hbm_refresh_power_W: float | None = None

    @model_validator(mode="after")
    def _phase_closures(self) -> "MixedPhaseE2EResult":
        if self.mixed_epoch_time_ms is not None:
            expected = float(self.prefill_service_time_ms) + float(self.decode_service_time_ms)
            if not math.isclose(self.mixed_epoch_time_ms, expected, rel_tol=1e-12):
                raise ValueError("NO_OVERLAP mixed time does not close")
        if self.mixed_total_energy_J is not None:
            if self.prefill_total_J is None or self.decode_total_J is None:
                raise ValueError("complete mixed energy requires both phase totals")
            prefill_components = (
                self.prefill_gpu_dynamic_J, self.prefill_gpu_static_J,
                self.prefill_memory_dynamic_J, self.prefill_refresh_J,
                self.prefill_host_ddr_J, self.prefill_host_pcie_J,
            )
            decode_components = (
                self.decode_gpu_dynamic_J, self.decode_nmp_mac_dynamic_J,
                self.decode_residual_interface_J, self.decode_gpu_static_J,
                self.decode_memory_dynamic_J, self.decode_refresh_J,
                self.decode_host_ddr_J, self.decode_host_pcie_J,
            )
            if any(value is None for value in (*prefill_components, *decode_components)):
                raise ValueError("complete mixed energy requires every phase component")
            if not math.isclose(
                float(self.prefill_total_J), sum(float(x) for x in prefill_components),
                rel_tol=1e-12,
            ):
                raise ValueError("Prefill component energy does not close")
            if not math.isclose(
                float(self.decode_total_J), sum(float(x) for x in decode_components),
                rel_tol=1e-12,
            ):
                raise ValueError("Decode component energy does not close")
            expected = float(self.prefill_total_J) + float(self.decode_total_J)
            if not math.isclose(self.mixed_total_energy_J, expected, rel_tol=1e-12):
                raise ValueError("mixed energy does not close")
        return self


def _host_dynamic_energies(
    bytes_: float, host: HostOffloadSpec,
) -> tuple[float | None, float | None]:
    """Integrate the existing canonical host dynamic-power primitive."""
    effective = host.effective_bandwidth_bytes_per_second
    if (effective is None or host.host_link_dynamic_J_per_bit is None
            or host.host_memory_dynamic_J_per_bit is None):
        return None, None
    duration_s = 0.0 if bytes_ == 0.0 else bytes_ / effective
    point = resolve_host_offload_power(
        host_transfer_demand_bytes_per_second=(
            0.0 if bytes_ == 0.0 else effective),
        host_effective_bandwidth_bytes_per_second=effective,
        e_pcie_dynamic_J_per_bit=host.host_link_dynamic_J_per_bit,
        e_ddr_dynamic_J_per_bit=host.host_memory_dynamic_J_per_bit,
    )
    return point.ddr_dynamic_power_W * duration_s, point.pcie_dynamic_power_W * duration_s


def _m3d_prefill_memory_energies(
        architecture, prefill_metrics, duration_s: float,
) -> tuple[float, float, float, float]:
    """Resolve existing M3D read/write/refresh terms; GPU range stays separate."""
    read_J=(8.0*prefill_metrics.prefill_read_bytes
            * architecture.memory.E_access_total_pj_bit*1e-12)
    write_pj=resolve_orthogonal_m3d_write_energy_pj_per_bit(
        architecture.case,architecture.memory)
    write_J=8.0*prefill_metrics.prefill_write_bytes*write_pj*1e-12
    refresh_J=float(architecture.memory.P_refresh_W or 0.0)*duration_s
    return read_J,write_J,read_J+write_J,refresh_J


def evaluate_conventional_hbm_mixed_phase(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    case: MixedPhaseServingCase,
) -> MixedPhaseE2EResult:
    """Evaluate the capacity-aware conventional baseline without thermal."""
    if model.model_id != case.model_id:
        raise ValueError("model and mixed case model_id must match")
    root = Path(project_root)
    backend = resolve_conventional_hbm_backend(root)
    platform = load_platform_spec_file(
        root / "configs/platform/gpu_package_h200_reference.yaml")
    gpu_decode_spec = platform.gpu_decode_power
    gpu_compute = platform.gpu_compute_power
    gpu_prefill = platform.gpu_prefill_compute
    if gpu_decode_spec is None or gpu_compute is None or gpu_prefill is None:
        raise ValueError("canonical H200 GPU Prefill/Decode data is incomplete")

    # One shared capacity gate: weights once and KV for all P+D requests.
    all_decode = model.decode_input(
        batch_size=case.batch_size, context_length=case.context_length)
    all_metrics = evaluate_llm_decode(all_decode)
    capacity_source = ServingCapacitySource(
        architecture=backend.architecture,
        usable_capacity_bytes=backend.capacity_bytes,
        capacity_source_status=backend.capacity_source_status,
        provenance_status="CANONICAL_CONVENTIONAL_HBM_CASE",
    )
    residency = evaluate_capacity_residency(
        all_metrics, capacity_source, requested_requests=case.batch_size)
    resident_decode = min(case.decode_requests, residency.local_resident_requests)
    resident_prefill = min(
        case.prefill_requests,
        max(0, residency.local_resident_requests - resident_decode),
    )
    spilled_decode = case.decode_requests - resident_decode
    spilled_prefill = case.prefill_requests - resident_prefill

    prefill_input = model.prefill_input(
        batch_size=case.prefill_requests, prompt_length=case.context_length)
    prefill_metrics = evaluate_llm_prefill(prefill_input)
    calibration = resolve_gpu_prefill_compute_energy_calibration(
        gpu_compute, gpu_prefill)
    roofline = evaluate_gpu_prefill_roofline(
        prefill_metrics,
        peak_compute_flops_per_s=gpu_compute.peak_compute_BF16_dense_flops_per_s,
        large_gemm_effective_flops_per_s=gpu_prefill.large_gemm_effective_tflops * 1e12,
        causal_attention_effective_flops_per_s=gpu_prefill.causal_attention_effective_tflops * 1e12,
        sustained_memory_bandwidth_bytes_per_s=backend.sustained_bandwidth_bytes_per_s,
        static_power_W=gpu_compute.static_power_W,
        peak_reference_dynamic_J_per_FLOP_min=calibration.peak_reference_dynamic_J_per_FLOP_min,
        peak_reference_dynamic_J_per_FLOP_max=calibration.peak_reference_dynamic_J_per_FLOP_max,
        nominal_gemm_dynamic_J_per_FLOP_min=calibration.nominal_gemm_dynamic_J_per_FLOP_min,
        nominal_gemm_dynamic_J_per_FLOP_max=calibration.nominal_gemm_dynamic_J_per_FLOP_max,
        nominal_attention_dynamic_J_per_FLOP_min=calibration.nominal_attention_dynamic_J_per_FLOP_min,
        nominal_attention_dynamic_J_per_FLOP_max=calibration.nominal_attention_dynamic_J_per_FLOP_max,
        compute_bound_total_power_W_min=calibration.compute_bound_total_power_W_min,
        compute_bound_total_power_W_max=calibration.compute_bound_total_power_W_max,
    )

    host = backend.host_offload
    host_bw = host.effective_bandwidth_bytes_per_second
    if host_bw is None:
        raise ValueError("canonical host offload bandwidth is unresolved")
    host_prefill_write = spilled_prefill * all_metrics.kv_bytes_per_request
    prefill_host_s = host_prefill_write / host_bw
    prefill_s = roofline.nominal_prefill_latency_s + prefill_host_s

    decode_input = model.decode_input(
        batch_size=case.decode_requests, context_length=case.context_length)
    decode_metrics = evaluate_llm_decode(decode_input)
    gpu_model = AnalyticalRooflineGPUModel(
        matched_payload_bandwidth_bits_per_second=(
            8.0 * backend.sustained_bandwidth_bytes_per_s),
        effective_compute_flops_per_second=(
            gpu_compute.peak_compute_BF16_dense_flops_per_s),
    )
    gpu_decode = gpu_model.evaluate(decode_input, batch_size=case.decode_requests)
    host_decode_read = spilled_decode * decode_metrics.kv_bytes_per_request
    host_decode_write = spilled_decode * decode_metrics.kv_write_bytes_per_token
    decode_host_s = (host_decode_read + host_decode_write) / host_bw
    decode_s = gpu_decode.decode_step_time_ms * 1e-3 + decode_host_s
    mixed_s = prefill_s + decode_s

    e_read_J_bit = backend.read_energy_pJ_per_bit * 1e-12
    prefill_read_J = 8.0 * prefill_metrics.prefill_read_bytes * e_read_J_bit
    prefill_write_J = None
    prefill_refresh_J = backend.refresh_power_W * prefill_s
    prefill_ddr_J, prefill_pcie_J = _host_dynamic_energies(
        host_prefill_write, host)

    aggregate_decode_bytes = case.decode_requests * (
        decode_metrics.read_bytes_per_token + decode_metrics.write_bytes_per_token)
    decode_gpu_dynamic_J = (
        8.0 * aggregate_decode_bytes * gpu_decode_spec.e_decode_J_per_bit)
    local_decode_read = (
        decode_metrics.weight_active_per_step_bytes
        + resident_decode * decode_metrics.kv_bytes_per_request)
    local_decode_write = resident_decode * decode_metrics.kv_write_bytes_per_token
    decode_read_J = 8.0 * local_decode_read * e_read_J_bit
    decode_write_J = None if local_decode_write > 0.0 else 0.0
    decode_ddr_J, decode_pcie_J = _host_dynamic_energies(
        host_decode_read + host_decode_write, host)
    decode_refresh_J = backend.refresh_power_W * decode_s

    return MixedPhaseE2EResult(
        system_id="CONVENTIONAL_HBM_GPU", comparison_role="BASELINE",
        model_id=model.model_id, context_length=case.context_length,
        batch_size=case.batch_size, prefill_requests=case.prefill_requests,
        decode_requests=case.decode_requests,
        evaluation_status="PERFORMANCE_CAPACITY_AND_PARTIAL_ENERGY_EVALUATED",
        energy_status="INCOMPLETE_CONVENTIONAL_HBM_WRITE_ENERGY_UNRESOLVED",
        nmp_batch_generalization_status="NOT_APPLICABLE",
        TTFT_ms=prefill_s * 1e3, TPOT_ms=decode_s * 1e3,
        prefill_service_time_ms=prefill_s * 1e3,
        decode_service_time_ms=decode_s * 1e3,
        mixed_epoch_time_ms=mixed_s * 1e3,
        decode_tokens_per_s=case.decode_requests / decode_s,
        prefill_gpu_dynamic_J=None,
        prefill_gpu_dynamic_J_min=roofline.total_dynamic_energy_J_min,
        prefill_gpu_dynamic_J_max=roofline.total_dynamic_energy_J_max,
        prefill_gpu_static_J=gpu_compute.static_power_W * prefill_s,
        prefill_memory_read_dynamic_J=prefill_read_J,
        prefill_memory_write_dynamic_J=prefill_write_J,
        prefill_memory_dynamic_J=None,
        prefill_refresh_J=prefill_refresh_J,
        prefill_host_ddr_J=prefill_ddr_J,
        prefill_host_pcie_J=prefill_pcie_J,
        prefill_total_J=None,
        decode_gpu_dynamic_J=decode_gpu_dynamic_J,
        decode_gpu_static_J=gpu_decode_spec.static_power_W * decode_s,
        decode_memory_read_dynamic_J=decode_read_J,
        decode_memory_write_dynamic_J=decode_write_J,
        decode_memory_dynamic_J=None,
        decode_refresh_J=decode_refresh_J,
        decode_host_ddr_J=decode_ddr_J,
        decode_host_pcie_J=decode_pcie_J,
        decode_total_J=None, decode_tokens_per_J=None,
        mixed_total_energy_J=None, mixed_service_tokens_per_J=None,
        required_capacity_GB=all_metrics.required_capacity_bytes / 1e9,
        local_capacity_GB=backend.capacity_bytes / 1e9,
        resident_requests=residency.local_resident_requests,
        spilled_requests=residency.spilled_requests,
        resident_fraction=residency.local_resident_requests / case.batch_size,
        resident_decode_requests=resident_decode,
        spilled_decode_requests=spilled_decode,
        resident_prefill_requests=resident_prefill,
        spilled_prefill_requests=spilled_prefill,
        local_KV_GB=(residency.local_resident_requests
                     * all_metrics.kv_bytes_per_request / 1e9),
        host_KV_GB=(residency.spilled_requests
                    * all_metrics.kv_bytes_per_request / 1e9),
        host_read_GB=host_decode_read / 1e9,
        host_write_GB=(host_prefill_write + host_decode_write) / 1e9,
        capacity_status=residency.capacity_status,
        residency_policy="DECODE_FIRST_LOCAL_RESIDENCY",
        residency_policy_status="MODELING_CHOICE_CONSERVATIVE_FOR_BASELINE",
        prefill_spill_semantics=(
            "PREFILL_SPILLED_KV_FINAL_MATERIALIZATION_TO_HOST"),
        hbm_read_energy_pJ_per_bit=backend.read_energy_pJ_per_bit,
        hbm_write_energy_pJ_per_bit=None,
        hbm_write_energy_status=backend.hbm_write_energy_status,
        hbm_refresh_power_W=backend.refresh_power_W,
    )


def evaluate_orthogonal_m3d_igzo_memory_only_mixed_phase(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    case: MixedPhaseServingCase,
) -> MixedPhaseE2EResult:
    """Evaluate all-local Orthogonal M3D memory with GPU-only execution."""
    if model.model_id != case.model_id:
        raise ValueError("model and mixed case model_id must match")
    root = Path(project_root).resolve()
    architecture = resolve_m3d_architecture_backend(root)
    platform = load_platform_spec_file(
        root / "configs/platform/gpu_package_h200_reference.yaml")
    gpu_compute = platform.gpu_compute_power
    gpu_prefill = platform.gpu_prefill_compute
    gpu_decode_spec = platform.gpu_decode_power
    if gpu_compute is None or gpu_prefill is None or gpu_decode_spec is None:
        raise ValueError("canonical H200 GPU Prefill/Decode data is incomplete")

    capacity_input = model.decode_input(
        batch_size=case.batch_size, context_length=case.context_length)
    capacity_metrics = evaluate_llm_decode(capacity_input)
    rounded = rounded_capacity_bytes(
        build_m3d_only_workload_objects(capacity_input),
        architecture.layout.slot_capacity_bytes)
    available = architecture.layout.total_capacity_bytes
    capacity = dict(
        required_capacity_GB=capacity_metrics.required_capacity_bytes / 1e9,
        physical_page_rounded_capacity_GB=rounded / 1e9,
        local_capacity_GB=available / 1e9,
        capacity_margin_GB=(available - rounded) / 1e9,
        capacity_utilization=rounded / available,
    )
    common = dict(
        system_id="ORTHOGONAL_M3D_IGZO_MEMORY_ONLY", comparison_role="ABLATION",
        model_id=model.model_id, context_length=case.context_length,
        batch_size=case.batch_size, prefill_requests=case.prefill_requests,
        decode_requests=case.decode_requests,
        nmp_batch_generalization_status="NOT_APPLICABLE", **capacity)
    if rounded > available:
        return MixedPhaseE2EResult(
            **common, evaluation_status="CAPACITY_INFEASIBLE",
            energy_status="NOT_EVALUATED_CAPACITY_INFEASIBLE",
            capacity_status="CAPACITY_INFEASIBLE", capacity_violations=None,
            resident_requests=0, spilled_requests=case.batch_size,
            resident_fraction=0.0, hbm_write_energy_status="NOT_APPLICABLE")

    prefill_metrics = evaluate_llm_prefill(model.prefill_input(
        batch_size=case.prefill_requests, prompt_length=case.context_length))
    calibration = resolve_gpu_prefill_compute_energy_calibration(
        gpu_compute, gpu_prefill)
    gpu_service = resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=gpu_decode_spec.peak_memory_bandwidth_bytes_per_s,
        service_status=platform.gpu_bandwidth_service.service_status,
        provenance=platform.gpu_bandwidth_service.provenance)
    roofline = evaluate_gpu_prefill_roofline(
        prefill_metrics,
        peak_compute_flops_per_s=gpu_compute.peak_compute_BF16_dense_flops_per_s,
        large_gemm_effective_flops_per_s=gpu_prefill.large_gemm_effective_tflops * 1e12,
        causal_attention_effective_flops_per_s=gpu_prefill.causal_attention_effective_tflops * 1e12,
        sustained_memory_bandwidth_bytes_per_s=gpu_service.sustained_bandwidth_bytes_per_s,
        static_power_W=gpu_compute.static_power_W,
        peak_reference_dynamic_J_per_FLOP_min=calibration.peak_reference_dynamic_J_per_FLOP_min,
        peak_reference_dynamic_J_per_FLOP_max=calibration.peak_reference_dynamic_J_per_FLOP_max,
        nominal_gemm_dynamic_J_per_FLOP_min=calibration.nominal_gemm_dynamic_J_per_FLOP_min,
        nominal_gemm_dynamic_J_per_FLOP_max=calibration.nominal_gemm_dynamic_J_per_FLOP_max,
        nominal_attention_dynamic_J_per_FLOP_min=calibration.nominal_attention_dynamic_J_per_FLOP_min,
        nominal_attention_dynamic_J_per_FLOP_max=calibration.nominal_attention_dynamic_J_per_FLOP_max,
        compute_bound_total_power_W_min=calibration.compute_bound_total_power_W_min,
        compute_bound_total_power_W_max=calibration.compute_bound_total_power_W_max)

    decode_input = model.decode_input(
        batch_size=case.decode_requests, context_length=case.context_length)
    decode_metrics = evaluate_llm_decode(decode_input)
    local_latency_ns = statistics.fmean(
        item.mat_latency_ns + item.miv_latency_ns
        for item in architecture.physical_latency.locations)
    internal_bw = resolve_internal_service_bandwidth(
        architecture.bandwidth, local_latency_ns)
    boundary = resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=internal_bw,
        memory_capability_bytes_per_s=architecture.bandwidth.coil_bandwidth_bytes_per_s,
        gpu_peak_bandwidth_bytes_per_s=gpu_decode_spec.peak_memory_bandwidth_bytes_per_s)
    boundary_service = resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=boundary.bandwidth_actual_bytes_per_s,
        service_status=platform.gpu_bandwidth_service.service_status,
        provenance=platform.gpu_bandwidth_service.provenance)
    effective_bw = min(internal_bw, boundary_service.sustained_bandwidth_bytes_per_s)
    decode = AnalyticalRooflineGPUModel(
        matched_payload_bandwidth_bits_per_second=8.0 * effective_bw,
        effective_compute_flops_per_second=gpu_compute.peak_compute_BF16_dense_flops_per_s,
    ).evaluate(decode_input, batch_size=case.decode_requests)
    prefill_s = roofline.nominal_prefill_latency_s
    decode_s = decode.decode_step_time_ms * 1e-3
    aggregate_read = case.decode_requests * decode_metrics.read_bytes_per_token
    aggregate_write = case.decode_requests * decode_metrics.write_bytes_per_token
    read_J = 8.0 * aggregate_read * architecture.memory.E_access_total_pj_bit * 1e-12
    write_pj_bit = resolve_orthogonal_m3d_write_energy_pj_per_bit(
        architecture.case, architecture.memory)
    write_J = 8.0 * aggregate_write * write_pj_bit * 1e-12
    memory_J = read_J + write_J
    gpu_J = 8.0 * (aggregate_read + aggregate_write) * gpu_decode_spec.e_decode_J_per_bit
    refresh_J = float(architecture.memory.P_refresh_W or 0.0) * decode_s
    static_J = gpu_decode_spec.static_power_W * decode_s
    decode_total = memory_J + gpu_J + refresh_J + static_J
    prefill_read_J,prefill_write_J,prefill_memory_J,prefill_refresh_J=(
        _m3d_prefill_memory_energies(architecture,prefill_metrics,prefill_s))
    max_die = math.ceil(rounded / architecture.layout.slot_capacity_bytes
                        / architecture.layout.slab_count) * architecture.layout.slot_capacity_bytes
    max_die /= architecture.layout.capacity_per_slab_bytes
    prefill_ms = prefill_s * 1e3
    decode_ms = decode_s * 1e3
    return MixedPhaseE2EResult(
        **common, evaluation_status="EVALUATED",
        energy_status="PREFILL_GPU_DYNAMIC_RANGE_NO_SINGLE_NOMINAL",
        TTFT_ms=prefill_ms, TPOT_ms=decode_ms,
        prefill_service_time_ms=prefill_ms, decode_service_time_ms=decode_ms,
        mixed_epoch_time_ms=prefill_ms + decode_ms,
        decode_tokens_per_s=case.decode_requests / decode_s,
        prefill_gpu_dynamic_J_min=roofline.total_dynamic_energy_J_min,
        prefill_gpu_dynamic_J_max=roofline.total_dynamic_energy_J_max,
        prefill_gpu_static_J=gpu_compute.static_power_W * prefill_s,
        prefill_memory_read_dynamic_J=prefill_read_J,
        prefill_memory_write_dynamic_J=prefill_write_J,
        prefill_memory_dynamic_J=prefill_memory_J,
        prefill_refresh_J=prefill_refresh_J,
        prefill_host_ddr_J=0.0, prefill_host_pcie_J=0.0,
        decode_gpu_dynamic_J=gpu_J, decode_gpu_static_J=static_J,
        decode_memory_read_dynamic_J=read_J,
        decode_memory_write_dynamic_J=write_J,
        decode_memory_dynamic_J=memory_J, decode_refresh_J=refresh_J,
        decode_host_ddr_J=0.0, decode_host_pcie_J=0.0,
        decode_total_J=decode_total,
        decode_tokens_per_J=case.decode_requests / decode_total,
        max_die_capacity_utilization=max_die, capacity_violations=0,
        resident_requests=case.batch_size, spilled_requests=0,
        resident_fraction=1.0, resident_decode_requests=case.decode_requests,
        spilled_decode_requests=0, resident_prefill_requests=case.prefill_requests,
        spilled_prefill_requests=0,
        local_KV_GB=case.batch_size * capacity_metrics.kv_bytes_per_request / 1e9,
        host_KV_GB=0.0, host_read_GB=0.0, host_write_GB=0.0,
        capacity_status="FULLY_LOCAL",
        residency_policy="ALL_ACTIVE_REQUESTS_PHYSICALLY_PAGE_ROUNDED",
        residency_policy_status="CANONICAL_M3D_PHYSICAL_CAPACITY_GATE",
        prefill_spill_semantics="NOT_APPLICABLE_FULLY_LOCAL",
        hbm_write_energy_status="NOT_APPLICABLE")




def evaluate_mixed_phase_e2e(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    case: MixedPhaseServingCase, system_id: SystemId,
) -> MixedPhaseE2EResult:
    """Dispatch one final-system point without any thermal dependency."""
    if system_id == "CONVENTIONAL_HBM_GPU":
        return evaluate_conventional_hbm_mixed_phase(
            project_root=project_root, model=model, case=case)
    if system_id == "IOM3D_FEOL_NMP":
        raise ValueError("Die-only mixed NMP model retired; use the physical FEOL Decode comparison")
    return evaluate_orthogonal_m3d_igzo_memory_only_mixed_phase(
        project_root=project_root, model=model, case=case)


class MixedPhaseComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    system_id: SystemId
    e2e_speedup_vs_baseline: float | None
    energy_efficiency_gain_vs_baseline: float | None
    TPOT_improvement: float | None
    TTFT_improvement: float | None
    decode_throughput_speedup_vs_baseline: float | None
    spill_reduction: int | None


def compare_mixed_phase_results(
    baseline: MixedPhaseE2EResult,
    designs: tuple[MixedPhaseE2EResult, ...],
) -> tuple[MixedPhaseComparison, ...]:
    """Ratio-only comparison; no physical quantity is recomputed here."""
    if baseline.system_id != "CONVENTIONAL_HBM_GPU":
        raise ValueError("comparison baseline must be CONVENTIONAL_HBM_GPU")
    rows = []
    for design in designs:
        same = (
            design.model_id, design.context_length, design.batch_size,
            design.prefill_requests, design.decode_requests,
        ) == (
            baseline.model_id, baseline.context_length, baseline.batch_size,
            baseline.prefill_requests, baseline.decode_requests,
        )
        if not same:
            raise ValueError("comparison rows must describe the same workload")
        ratio = lambda numerator, denominator: (
            None if numerator is None or denominator is None else numerator / denominator)
        rows.append(MixedPhaseComparison(
            system_id=design.system_id,
            e2e_speedup_vs_baseline=ratio(
                baseline.mixed_epoch_time_ms, design.mixed_epoch_time_ms),
            energy_efficiency_gain_vs_baseline=ratio(
                baseline.mixed_total_energy_J, design.mixed_total_energy_J),
            TPOT_improvement=ratio(baseline.TPOT_ms, design.TPOT_ms),
            TTFT_improvement=ratio(baseline.TTFT_ms, design.TTFT_ms),
            decode_throughput_speedup_vs_baseline=ratio(
                design.decode_tokens_per_s, baseline.decode_tokens_per_s),
            spill_reduction=(
                None if design.evaluation_status == "CAPACITY_INFEASIBLE"
                or baseline.spilled_requests is None
                or design.spilled_requests is None
                else baseline.spilled_requests - design.spilled_requests),
        ))
    return tuple(rows)
