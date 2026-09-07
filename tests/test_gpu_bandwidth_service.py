"""GPU sustained-bandwidth service boundary and isolation tests."""

from pathlib import Path

import pytest

from om3dthermal.platform import (
    load_platform_spec_file,
    resolve_gpu_bandwidth_service,
    resolve_gpu_decode_power,
    resolve_local_memory_gpu_transfer,
)
from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    resolve_case_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
PLATFORM_PATH = ROOT / "configs/platform/gpu_package_h200_reference.yaml"
M3D_CASE_PATH = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"


def _service(*, utilization: float = 0.5, ceiling: float = 4.8e12):
    spec = load_platform_spec_file(PLATFORM_PATH).gpu_bandwidth_service
    return resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=ceiling,
        gpu_bandwidth_utilization=utilization,
        utilization_status=spec.utilization_status,
        utilization_provenance=spec.provenance,
    )


def test_canonical_nominal_utilization_loads_from_platform() -> None:
    spec = load_platform_spec_file(PLATFORM_PATH).gpu_bandwidth_service
    assert spec.nominal_utilization == pytest.approx(0.5)
    assert spec.utilization_status == (
        "MODELING_CHOICE_NOMINAL_GPU_BANDWIDTH_UTILIZATION")
    assert spec.provenance


def test_service_resolver_applies_utilization_after_transfer_ceiling() -> None:
    transfer = resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=4.9e12,
        memory_capability_bytes_per_s=5.3e12,
        gpu_peak_bandwidth_bytes_per_s=4.8e12,
    )
    service = _service(ceiling=transfer.bandwidth_actual_bytes_per_s)
    assert transfer.bandwidth_actual_bytes_per_s == pytest.approx(4.8e12)
    assert transfer.bottleneck == "GPU"
    assert service.transfer_ceiling_bytes_per_s == pytest.approx(4.8e12)
    assert service.sustained_bandwidth_bytes_per_s == pytest.approx(2.4e12)


@pytest.mark.parametrize("utilization", [0.0, -0.1, 1.01, float("nan"), float("inf"), True])
def test_service_resolver_rejects_invalid_utilization(utilization) -> None:
    with pytest.raises((TypeError, ValueError)):
        _service(utilization=utilization)


def test_service_layer_does_not_change_m3d_intrinsic_bandwidth_or_energy() -> None:
    case = load_case_config(M3D_CASE_PATH)
    before = calculate_memory_power(
        case,
        project_root=ROOT,
        read_bandwidth_gbps=case.workload.read_bandwidth_gbps,
        geometry=resolve_case_geometry(case),
    )
    _service()
    after = calculate_memory_power(
        case,
        project_root=ROOT,
        read_bandwidth_gbps=case.workload.read_bandwidth_gbps,
        geometry=resolve_case_geometry(case),
    )
    assert after.architecture_bandwidth_closure == before.architecture_bandwidth_closure
    assert after.E_access_total_pj_bit == before.E_access_total_pj_bit


def test_service_layer_is_not_a_gpu_power_model() -> None:
    platform = load_platform_spec_file(PLATFORM_PATH)
    service = _service()
    assert "power" not in " ".join(service.model_dump().keys()).lower()

    decode = platform.gpu_decode_power
    assert decode is not None
    achieved_bandwidth_model = resolve_gpu_decode_power(
        static_power_W=decode.static_power_W,
        e_decode_J_per_bit=decode.e_decode_J_per_bit,
        bandwidth_demand_bytes_per_s=service.sustained_bandwidth_bytes_per_s,
        peak_bandwidth_bytes_per_s=decode.peak_memory_bandwidth_bytes_per_s,
    )
    assert decode.e_decode_J_per_bit == pytest.approx(15.29e-12)
    assert achieved_bandwidth_model.gpu_power_W == pytest.approx(367.568)


def test_nmp_placement_does_not_consume_gpu_service_utilization() -> None:
    placement_root = ROOT / "src/om3dthermal/placement"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in placement_root.rglob("*.py")
    )
    assert "gpu_bandwidth_service" not in source
    assert "gpu_bandwidth_utilization" not in source
