"""Targeted tests for resident-object fixed-page decomposition."""

from dataclasses import replace
from pathlib import Path

import pytest

from om3dthermal.power import (
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.resident_pages import (
    CANONICAL_RESIDENT_SET_STATUS,
    ResidentCapacityExceededError,
    ResidentDataObject,
    ResidentDataPage,
    build_resident_page_layout,
)


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"
MIB = 2**20


@pytest.fixture(scope="module")
def canonical_hardware():
    case = load_case_config(CASE)
    geometry = resolve_case_geometry(case)
    power = calculate_memory_power(
        case, project_root=ROOT, geometry=geometry)
    assert geometry.m3d is not None
    topology = calculate_m3d_subarray(
        case.architecture.m3d_subarray, geometry.m3d)
    feol = calculate_feol_route(case.architecture.feol_route, topology)
    latency = calculate_physical_access_latency(
        case.architecture.physical_access_latency,
        feol_route=feol,
        miv_length_per_layer_um=power.diagnostics[
            "miv_length_per_layer_um"],
        miv_delay_per_layer_ns=power.diagnostics["miv_delay_per_layer_ns"],
        miv_status=power.diagnostics["miv_latency_status"],
        miv_parameter_status=power.diagnostics[
            "miv_resistance_parameter_status"],
        miv_provenance=power.diagnostics["miv_resistance_provenance"],
    )
    capacity = calculate_physical_capacity_layout(
        topology,
        latency,
        slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )
    return power, latency, capacity


def _small_layout(layout, *, slot_count: int, slot_size: int | None = None):
    capacity = layout.slot_capacity_bytes if slot_size is None else slot_size
    classes = tuple(
        replace(
            slot,
            capacity_bytes=capacity,
            capacity_mib=capacity / MIB,
            multiplicity=1,
        )
        for slot in layout.slot_classes[:slot_count]
    )
    return replace(
        layout,
        slab_count=1,
        clusters_per_slab=slot_count,
        layers_per_cluster=1,
        slot_capacity_bytes=capacity,
        slot_class_count=slot_count,
        physical_slot_count=slot_count,
        capacity_per_layer_per_slab_bytes=slot_count * capacity,
        capacity_per_slab_bytes=slot_count * capacity,
        total_capacity_bytes=slot_count * capacity,
        total_capacity_gib=slot_count * capacity / 2**30,
        slot_classes=classes,
    )


def test_exact_fit_uses_one_page_without_fragmentation(canonical_hardware):
    *_, hardware = canonical_hardware
    result = build_resident_page_layout((
        ResidentDataObject("weights.layer0", "WEIGHT", 2 * MIB),
    ), hardware)
    assert result.page_size_bytes == hardware.slot_capacity_bytes == 2 * MIB
    assert result.page_count == 1
    assert result.pages[0].size_bytes == 2 * MIB
    assert result.pages[0].capacity_bytes == 2 * MIB
    assert result.internal_fragmentation_bytes == 0
    assert result.internal_fragmentation_ratio == 0.0


def test_partial_final_page_has_full_slot_occupancy(canonical_hardware):
    *_, hardware = canonical_hardware
    result = build_resident_page_layout((
        ResidentDataObject("weights.layer0", "WEIGHT", 5 * MIB),
    ), hardware)
    assert result.page_count == 3
    assert tuple(page.page_id for page in result.pages) == (
        "weights.layer0:page:0",
        "weights.layer0:page:1",
        "weights.layer0:page:2",
    )
    assert tuple(page.size_bytes for page in result.pages) == (
        2 * MIB, 2 * MIB, 1 * MIB)
    assert all(page.capacity_bytes == 2 * MIB for page in result.pages)
    assert result.logical_resident_bytes == 5 * MIB
    assert result.allocated_page_bytes == 6 * MIB
    assert result.internal_fragmentation_bytes == 1 * MIB
    assert result.internal_fragmentation_ratio == pytest.approx(1 / 6)


def test_tiny_object_still_consumes_one_full_page(canonical_hardware):
    *_, hardware = canonical_hardware
    result = build_resident_page_layout((
        ResidentDataObject("metadata", "OTHER", 1),
    ), hardware)
    assert result.page_count == 1
    assert result.pages[0].size_bytes == 1
    assert result.pages[0].capacity_bytes == 2 * MIB
    assert result.internal_fragmentation_bytes == 2 * MIB - 1


def test_multiple_objects_never_share_tail_capacity(canonical_hardware):
    *_, hardware = canonical_hardware
    objects = (
        ResidentDataObject("weights", "WEIGHT", 1 * MIB),
        ResidentDataObject("kv.request0", "KV", 1 * MIB),
    )
    result = build_resident_page_layout(objects, hardware)
    assert result.page_count == 2
    assert result.allocated_page_bytes == 4 * MIB
    assert result.internal_fragmentation_bytes == 2 * MIB
    assert tuple(page.parent_object_id for page in result.pages) == (
        "weights", "kv.request0")
    assert result.page_sharing is False
    assert result.pages_by_object_type == {"WEIGHT": 1, "KV": 1, "OTHER": 0}


def test_page_ids_ordering_and_accounting_are_deterministic(canonical_hardware):
    *_, hardware = canonical_hardware
    objects = (
        ResidentDataObject("weights", "WEIGHT", 3 * MIB),
        ResidentDataObject("kv.request0", "KV", 5 * MIB),
    )
    first = build_resident_page_layout(objects, hardware)
    second = build_resident_page_layout(objects, hardware)
    assert first == second
    assert tuple(page.page_id for page in first.pages) == (
        "weights:page:0", "weights:page:1",
        "kv.request0:page:0", "kv.request0:page:1",
        "kv.request0:page:2",
    )
    assert first.ordering_semantics == (
        "INPUT_OBJECT_ORDER_THEN_ASCENDING_LOGICAL_PAGE_INDEX")
    assert first.page_to_physical_slot_mapping_included is False


def test_page_count_equal_to_slot_count_is_feasible(canonical_hardware):
    *_, canonical = canonical_hardware
    hardware = _small_layout(canonical, slot_count=3)
    result = build_resident_page_layout((
        ResidentDataObject("resident", "OTHER", 6 * MIB),
    ), hardware)
    assert result.page_count == result.physical_slot_count == 3
    assert result.remaining_slot_count == 0
    assert result.remaining_physical_capacity_bytes == 0
    assert result.capacity_feasible is True


def test_page_count_above_slot_count_fails_loudly(canonical_hardware):
    *_, canonical = canonical_hardware
    hardware = _small_layout(canonical, slot_count=3)
    with pytest.raises(
            ResidentCapacityExceededError,
            match="resident data set exceeds available local physical slots"):
        build_resident_page_layout((
            ResidentDataObject("resident", "OTHER", 7 * MIB),
        ), hardware)


def test_page_size_follows_dynamic_physical_slot_capacity(canonical_hardware):
    *_, canonical = canonical_hardware
    hardware = _small_layout(canonical, slot_count=4, slot_size=1 * MIB)
    result = build_resident_page_layout((
        ResidentDataObject("dynamic", "OTHER", 1536 * 2**10),
    ), hardware)
    assert result.page_size_bytes == 1 * MIB
    assert result.page_count == 2
    assert tuple(page.size_bytes for page in result.pages) == (
        1 * MIB, 512 * 2**10)
    assert all(page.capacity_bytes == hardware.slot_capacity_bytes
               for page in result.pages)


def test_page_creation_does_not_mutate_hardware_models(canonical_hardware):
    power, latency, hardware = canonical_hardware
    latency_before = latency
    hardware_before = hardware
    result = build_resident_page_layout((
        ResidentDataObject("resident", "OTHER", 5 * MIB),
    ), hardware)
    assert result.page_count == 3
    assert latency == latency_before
    assert hardware == hardware_before
    assert power.E_vertical_pj_bit == pytest.approx(0.002445862111816407)
    assert power.E_feol_route_pj_bit == pytest.approx(0.16705631334524151)


def test_canonical_resident_set_is_not_implicitly_invented():
    assert CANONICAL_RESIDENT_SET_STATUS == "NOT_YET_BOUND"
    forbidden_fields = {
        "hotness", "priority", "access_count", "assigned_slab",
        "assigned_cluster", "assigned_layer", "physical_address",
    }
    assert forbidden_fields.isdisjoint(ResidentDataPage.__dataclass_fields__)


def test_invalid_or_duplicate_objects_are_rejected(canonical_hardware):
    *_, hardware = canonical_hardware
    with pytest.raises(ValueError, match="must be unique"):
        build_resident_page_layout((
            ResidentDataObject("same", "WEIGHT", 1),
            ResidentDataObject("same", "KV", 1),
        ), hardware)
    with pytest.raises(ValueError, match="must be positive"):
        ResidentDataObject("empty", "OTHER", 0)
