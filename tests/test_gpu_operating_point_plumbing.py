"""Audit tests for explicit GPU operating-point propagation."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from om3dthermal.adapters import resolve_architecture_spec
from om3dthermal.experiment.config import (
    load_architecture_spec,
    load_platform_spec,
)
from om3dthermal.platform import (
    resolve_gpu_bandwidth_service,
    resolve_gpu_compute_power,
    resolve_gpu_decode_power,
    resolve_local_memory_gpu_transfer,
)
from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    resolve_case_geometry,
    resolve_effective_bandwidth,
    resolve_system_power,
)


ROOT = Path(__file__).parents[1]
CASE_PATH = ROOT / "configs/cases/conventional_hbm_2x1.yaml"
ARCHITECTURE_PATH = ROOT / "configs/architecture/conventional_hbm_2x1.yaml"
M3D_CASE_PATH = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"
PLATFORM_PATH = ROOT / "configs/platform/gpu_package_h200_reference.yaml"


def test_power_and_architecture_signatures_require_resolved_point() -> None:
    for function in (resolve_system_power, resolve_architecture_spec):
        parameter = inspect.signature(function).parameters["gpu_operating_point"]
        assert parameter.default is inspect.Parameter.empty
        transfer = inspect.signature(function).parameters[
            "transfer_operating_point"]
        assert transfer.default is inspect.Parameter.empty


def test_decode_operating_point_propagates_unchanged() -> None:
    case = load_case_config(CASE_PATH)
    geometry = resolve_case_geometry(case)
    point = resolve_gpu_decode_power(
        static_power_W=74.0,
        e_decode_J_per_bit=7.645e-12,
        bandwidth_demand_bytes_per_s=2.4e12,
        peak_bandwidth_bytes_per_s=4.8e12,
    )
    system = resolve_system_power(
        case, project_root=ROOT, geometry=geometry,
        gpu_operating_point=point, transfer_operating_point=None)
    spec = load_architecture_spec(ARCHITECTURE_PATH, project_root=ROOT)
    resolved = resolve_architecture_spec(
        spec, project_root=ROOT, gpu_operating_point=point,
        transfer_operating_point=None)

    assert system.gpu_power_W == pytest.approx(point.gpu_power_W)
    assert resolved.system_power.gpu_power_W == pytest.approx(point.gpu_power_W)


def test_compute_operating_point_propagates_unchanged() -> None:
    point = resolve_gpu_compute_power(
        static_power_W=74.0,
        compute_energy_dynamic_J_per_FLOP=5.0e-13,
        compute_demand_FLOP_per_s=4.0e14,
        effective_compute_ceiling_FLOP_per_s=8.0e14,
    )
    spec = load_architecture_spec(ARCHITECTURE_PATH, project_root=ROOT)
    resolved = resolve_architecture_spec(
        spec, project_root=ROOT, gpu_operating_point=point,
        transfer_operating_point=None)

    assert resolved.system_power.gpu_power_W == pytest.approx(point.gpu_power_W)


def test_system_power_has_no_hidden_platform_or_gpu_resolution() -> None:
    source = inspect.getsource(resolve_system_power)

    assert "load_platform_spec_file" not in source
    assert "resolve_gpu_decode_power" not in source
    memory_source = inspect.getsource(calculate_memory_power)
    assert "config.workload.read_bandwidth_gbps" not in memory_source


def _resolve_m3d_system(
    *, peak_bytes_per_s: float = 4.8e12, links_per_slab: int = 50,
):
    case = load_case_config(M3D_CASE_PATH)
    if links_per_slab != 50:
        service = case.architecture.memory_service
        coil = service.coil.model_copy(
            update={"links_per_slab": links_per_slab})
        architecture = case.architecture.model_copy(update={
            "memory_service": service.model_copy(update={"coil": coil})})
        case = case.model_copy(update={"architecture": architecture})
    geometry = resolve_case_geometry(case)
    intrinsic = calculate_memory_power(
        case, project_root=ROOT, geometry=geometry,
        read_bandwidth_gbps=0.0)
    closure = intrinsic.architecture_bandwidth_closure
    assert closure is not None
    raw = resolve_effective_bandwidth(
        closure,
        closure.average_service_cycle_ns / closure.service_cycle_scale)
    demand = case.workload.read_bandwidth_gbps * 1e9 / 8.0
    transfer = resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=demand,
        memory_capability_bytes_per_s=raw.effective_bandwidth_bytes_per_s,
        gpu_peak_bandwidth_bytes_per_s=peak_bytes_per_s)
    spec = load_platform_spec(PLATFORM_PATH, project_root=ROOT).gpu_decode_power
    service_spec = load_platform_spec(
        PLATFORM_PATH, project_root=ROOT).gpu_bandwidth_service
    bandwidth_service = resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=transfer.bandwidth_actual_bytes_per_s,
        gpu_bandwidth_utilization=service_spec.nominal_utilization,
        utilization_status=service_spec.utilization_status,
        utilization_provenance=service_spec.provenance,
    )
    gpu = resolve_gpu_decode_power(
        static_power_W=spec.static_power_W,
        e_decode_J_per_bit=spec.e_decode_J_per_bit,
        bandwidth_demand_bytes_per_s=(
            bandwidth_service.sustained_bandwidth_bytes_per_s),
        peak_bandwidth_bytes_per_s=peak_bytes_per_s)
    system = resolve_system_power(
        case, project_root=ROOT, geometry=geometry,
        gpu_operating_point=gpu, transfer_operating_point=transfer,
        bandwidth_service_operating_point=bandwidth_service)
    return case, raw, transfer, gpu, system


def test_nominal_m3d_gpu_transfer_and_power_close() -> None:
    case, raw, transfer, gpu, system = _resolve_m3d_system()
    assert case.workload.read_bandwidth_gbps == 39_200
    assert raw.effective_bandwidth_bytes_per_s == pytest.approx(15.9e12)
    assert transfer.bandwidth_actual_bytes_per_s == 4.8e12
    assert transfer.bottleneck == "GPU"
    assert gpu.bandwidth_actual_bytes_per_s == pytest.approx(2.4e12)
    assert system.read_bandwidth_gbps == 19_200
    assert system.memory_result.E_access_total_pj_bit == pytest.approx(
        0.8552605756733209)
    assert system.memory_result.P_read_W == pytest.approx(
        16.421003052927762)
    assert gpu.gpu_power_W == pytest.approx(367.568)
    assert system.memory_dynamic_power_bandwidth_source == (
        "GPU_SUSTAINED_BANDWIDTH_SERVICE_OPERATING_POINT")


def test_gpu_peak_override_propagates_to_both_dynamic_powers() -> None:
    _, _, _, nominal_gpu, nominal_system = _resolve_m3d_system()
    _, _, lower_transfer, lower_gpu, lower_system = _resolve_m3d_system(
        peak_bytes_per_s=4.0e12)
    assert lower_transfer.bandwidth_actual_bytes_per_s == 4.0e12
    assert lower_gpu.bandwidth_actual_bytes_per_s == 2.0e12
    assert lower_system.read_bandwidth_gbps == 16_000
    assert lower_gpu.gpu_dynamic_power_W < nominal_gpu.gpu_dynamic_power_W
    assert lower_system.memory_result.P_read_W < nominal_system.memory_result.P_read_W


def test_m3d_link_capability_override_is_hidden_by_gpu_cap() -> None:
    _, _, _, nominal_gpu, nominal_system = _resolve_m3d_system()
    _, raw, transfer, gpu, system = _resolve_m3d_system(links_per_slab=40)
    assert raw.effective_bandwidth_bytes_per_s == pytest.approx(12.72e12)
    assert transfer.bottleneck == "GPU"
    assert transfer.bandwidth_actual_bytes_per_s == pytest.approx(4.8e12)
    assert gpu.bandwidth_actual_bytes_per_s == pytest.approx(2.4e12)
    assert gpu.gpu_dynamic_power_W == nominal_gpu.gpu_dynamic_power_W
    assert system.read_bandwidth_gbps == nominal_system.read_bandwidth_gbps
    # Fewer per-slab IO lanes do not lower the GPU-capped rate, but they do
    # change the explicit FEOL route-energy topology in this sensitivity.
    assert system.memory_result.P_read_W > nominal_system.memory_result.P_read_W


def test_bandwidth_bound_m3d_system_rejects_missing_transfer() -> None:
    case = load_case_config(M3D_CASE_PATH)
    geometry = resolve_case_geometry(case)
    gpu = resolve_gpu_decode_power(
        static_power_W=74.0,
        e_decode_J_per_bit=7.645e-12,
        bandwidth_demand_bytes_per_s=4.9e12,
        peak_bandwidth_bytes_per_s=4.8e12)
    with pytest.raises(ValueError, match="requires a shared"):
        resolve_system_power(
            case, project_root=ROOT, geometry=geometry,
            gpu_operating_point=gpu, transfer_operating_point=None)
