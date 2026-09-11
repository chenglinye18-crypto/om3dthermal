"""Architecture-only gates for the nominal 100 um orthogonal M3D slab."""

from pathlib import Path

import pytest
import yaml

from om3dthermal.platform import (
    load_platform_spec_file,
    resolve_gpu_bandwidth_service,
    resolve_local_memory_gpu_transfer,
)
from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    resolve_case_geometry,
    resolve_effective_bandwidth,
)
from om3dthermal.power.config import CanonicalCaseConfig
from om3dthermal.thermal.nmp_die_mapping import physical_nmp_die_regions


ROOT = Path(__file__).parents[1]
CASE_PATH = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"
PLATFORM_PATH = ROOT / "configs/platform/gpu_package_h200_reference.yaml"


def _case_with_legacy_geometry() -> CanonicalCaseConfig:
    raw = yaml.safe_load(CASE_PATH.read_text(encoding="utf-8"))
    raw["geometry"]["orthogonal"].update({
        "slab_count": 106,
        "slab_pitch_x_um": 300.0,
    })
    raw["geometry"]["m3d_stack"]["si_substrate_um"] = 292.546
    return CanonicalCaseConfig.model_validate(raw)


@pytest.fixture(scope="module")
def revisions():
    old_case = _case_with_legacy_geometry()
    new_case = load_case_config(CASE_PATH)
    old_geometry = resolve_case_geometry(old_case)
    new_geometry = resolve_case_geometry(new_case)
    old_power = calculate_memory_power(
        old_case, read_bandwidth_gbps=0.0,
        project_root=ROOT, geometry=old_geometry)
    new_power = calculate_memory_power(
        new_case, read_bandwidth_gbps=0.0,
        project_root=ROOT, geometry=new_geometry)
    return old_case, new_case, old_geometry, new_geometry, old_power, new_power


def test_100um_slab_and_package_width_close_exactly(revisions) -> None:
    _, case, _, geometry, _, _ = revisions
    orth = case.geometry.orthogonal
    stack = case.geometry.m3d_stack
    assert orth is not None and stack is not None
    assert (
        stack.si_substrate_um + stack.feol_um
        + stack.bitcell_layers * stack.bitcell_layer_pitch_nm * 1e-3
        + stack.beol_interconnect_um + stack.daa_um
    ) == pytest.approx(100.0, abs=1e-12)
    assert stack.si_substrate_um == pytest.approx(92.546)
    assert stack.bitcell_layers == 8
    assert orth.slab_count == geometry.memory_region_count == 318
    assert orth.slab_count * orth.slab_pitch_x_um / 1000.0 == pytest.approx(31.8)


def test_capacity_scales_only_from_physical_slab_count(revisions) -> None:
    *_, old_power, new_power = revisions
    old = old_power.physical_capacity_layout
    new = new_power.physical_capacity_layout
    assert old is not None and new is not None
    assert old.capacity_per_slab_bytes == new.capacity_per_slab_bytes
    assert old.total_capacity_bytes == 497_947_770_880
    assert new.total_capacity_bytes == 1_493_843_312_640
    assert new.total_capacity_bytes / old.total_capacity_bytes == pytest.approx(3.0)


def test_per_slab_primitives_unchanged_and_aggregates_explicit(revisions) -> None:
    old_case, new_case, *_, old_power, new_power = revisions
    old = old_power.architecture_bandwidth_closure
    new = new_power.architecture_bandwidth_closure
    assert old is not None and new is not None
    assert old.links_per_slab == new.links_per_slab == 50
    assert old.rate_gbps_per_link == new.rate_gbps_per_link == 8.0
    assert old.coil_bandwidth_bytes_per_s == pytest.approx(5.3e12)
    assert new.coil_bandwidth_bytes_per_s == pytest.approx(15.9e12)
    assert len(physical_nmp_die_regions(old_case)) == 106
    assert len(physical_nmp_die_regions(new_case)) == 318


def test_gpu_backend_caps_raw_interface_growth(revisions) -> None:
    *_, new_power = revisions
    closure = new_power.architecture_bandwidth_closure
    assert closure is not None
    raw = resolve_effective_bandwidth(
        closure,
        closure.average_service_cycle_ns / closure.service_cycle_scale)
    platform = load_platform_spec_file(PLATFORM_PATH)
    gpu = platform.gpu_decode_power
    service = platform.gpu_bandwidth_service
    assert gpu is not None and service is not None
    transfer = resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=1e30,
        memory_capability_bytes_per_s=raw.effective_bandwidth_bytes_per_s,
        gpu_peak_bandwidth_bytes_per_s=gpu.peak_memory_bandwidth_bytes_per_s)
    sustained = resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=transfer.bandwidth_actual_bytes_per_s,
        service_status=service.service_status,
        provenance=service.provenance)
    assert raw.effective_bandwidth_bytes_per_s == pytest.approx(15.9e12)
    assert transfer.bottleneck == "GPU"
    assert transfer.bandwidth_actual_bytes_per_s == pytest.approx(4.8e12)
    assert sustained.sustained_bandwidth_bytes_per_s == pytest.approx(4.8e12)


def test_thinning_does_not_change_latency_or_energy_primitives(revisions) -> None:
    *_, old_power, new_power = revisions
    old_layout = old_power.physical_capacity_layout
    new_layout = new_power.physical_capacity_layout
    assert old_layout is not None and new_layout is not None
    assert tuple(item.physical_access_latency_ns for item in old_layout.slot_classes) == (
        tuple(item.physical_access_latency_ns for item in new_layout.slot_classes))
    assert old_power.E_access_total_pj_bit == new_power.E_access_total_pj_bit
    assert old_power.E_interface_pj_bit == new_power.E_interface_pj_bit


def test_100um_revision_provenance_is_explicit(revisions) -> None:
    _, case, *_ = revisions
    provenance = case.provenance
    assert provenance["platform_revision"] == "REV_V3_100UM_THINNED_SLAB"
    geometry = provenance["geometry"]
    assert geometry["total_slab_thickness_um"] == {
        "value": 100.0,
        "classification": "MODELING_CHOICE",
        "purpose": "THINNED_ORTHOGONAL_DIE_CAPACITY_SCALING",
    }
    assert geometry["si_substrate_um"]["classification"] == (
        "DERIVED_FROM_GEOMETRY_CLOSURE")
    assert geometry["slab_count"]["classification"] == (
        "DERIVED_FROM_PACKAGE_WIDTH_AND_SLAB_PITCH")
