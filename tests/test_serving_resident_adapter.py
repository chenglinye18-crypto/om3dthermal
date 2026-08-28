"""Targeted C-to-A resident-set adapter tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from om3dthermal.experiment import run_serving_experiment
from om3dthermal.power import (
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.serving import (
    CACapacitySemanticsMismatchError,
    CapacityResidencyResult,
    build_resident_objects_from_serving_residency,
    build_resident_pages_from_serving_residency,
)


ROOT = Path(__file__).parents[1]
SERVING_CONFIG = ROOT / "configs" / "experiment" / "capacity_aware_serving_v0.yaml"
M3D_CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"


def _residency(**overrides) -> CapacityResidencyResult:
    values = {
        "architecture": "synthetic",
        "usable_capacity_bytes": 1000.0,
        "weight_bytes": 100.0,
        "runtime_fixed_bytes": 0.0,
        "runtime_per_request_bytes": 0.0,
        "kv_bytes_per_request": 10.0,
        "resident_bytes_per_request": 10.0,
        "available_for_requests_bytes": 900.0,
        "max_resident_requests": 3,
        "requested_requests": 3,
        "local_resident_requests": 3,
        "spilled_requests": 0,
        "local_capacity_utilization": 0.13,
        "capacity_status": "FULLY_LOCAL",
        "capacity_source_status": "SYNTHETIC_TEST_ONLY",
        "runtime_capacity_semantics_status": "SYNTHETIC_TEST_ONLY",
        "residency_model_status": "ANALYTICAL_CAPACITY_RESIDENCY_V0",
    }
    values.update(overrides)
    return CapacityResidencyResult(**values)


@pytest.fixture(scope="module")
def canonical_physical_layout():
    case = load_case_config(M3D_CASE)
    geometry = resolve_case_geometry(case)
    power = calculate_memory_power(case, project_root=ROOT, geometry=geometry)
    assert geometry.m3d is not None
    topology = calculate_m3d_subarray(
        case.architecture.m3d_subarray, geometry.m3d)
    feol = calculate_feol_route(case.architecture.feol_route, topology)
    latency = calculate_physical_access_latency(
        case.architecture.physical_access_latency,
        feol_route=feol,
        miv_length_per_layer_um=power.diagnostics["miv_length_per_layer_um"],
        miv_delay_per_layer_ns=power.diagnostics["miv_delay_per_layer_ns"],
        miv_status=power.diagnostics["miv_latency_status"],
        miv_parameter_status=power.diagnostics[
            "miv_resistance_parameter_status"],
        miv_provenance=power.diagnostics["miv_resistance_provenance"],
    )
    return calculate_physical_capacity_layout(
        topology,
        latency,
        slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )


def test_weight_only_creates_one_coarse_weight_object() -> None:
    result = build_resident_objects_from_serving_residency(_residency(
        kv_bytes_per_request=0.0,
        resident_bytes_per_request=0.0,
        local_capacity_utilization=0.1,
        max_resident_requests=None,
        capacity_status="UNBOUNDED_PER_REQUEST_FOOTPRINT",
    ))
    assert [(obj.object_id, obj.object_type, obj.size_bytes)
            for obj in result.resident_objects] == [("weights", "WEIGHT", 100)]


def test_kv_only_uses_deterministic_accounting_ids() -> None:
    source = _residency(weight_bytes=0.0, local_capacity_utilization=0.03)
    first = build_resident_objects_from_serving_residency(source)
    second = build_resident_objects_from_serving_residency(source)
    assert first.resident_objects == second.resident_objects
    assert [obj.object_id for obj in first.resident_objects] == [
        "kv.request.0", "kv.request.1", "kv.request.2"]
    assert first.request_id_semantics == "DETERMINISTIC_ACCOUNTING_ID"


def test_weight_kv_runtime_byte_closure_and_no_policy_fields() -> None:
    result = build_resident_objects_from_serving_residency(_residency(
        runtime_fixed_bytes=7.0,
        runtime_per_request_bytes=3.0,
        resident_bytes_per_request=13.0,
    ))
    assert result.resident_weight_object_count == 1
    assert result.resident_kv_object_count == 3
    assert result.resident_runtime_object_count == 4
    assert result.total_resident_logical_bytes == 100 + 7 + 3 * (10 + 3)
    assert result.total_resident_logical_bytes == (
        result.c_reported_local_resident_bytes)
    assert result.byte_closure_error == 0
    forbidden = {"hotness", "access_count", "reuse", "priority",
                 "assigned_slab", "assigned_cluster", "assigned_layer"}
    assert forbidden.isdisjoint(result.__dataclass_fields__)
    assert all(forbidden.isdisjoint(obj.__dataclass_fields__)
               for obj in result.resident_objects)


def test_spilled_requests_are_excluded() -> None:
    result = build_resident_objects_from_serving_residency(_residency(
        max_resident_requests=2,
        requested_requests=5,
        local_resident_requests=2,
        spilled_requests=3,
        capacity_status="CAPACITY_PRESSURED",
    ))
    assert result.resident_kv_object_count == 2
    assert [obj.object_id for obj in result.resident_objects
            if obj.object_type == "KV"] == ["kv.request.0", "kv.request.1"]


def test_page_integration_is_deterministic_and_feasible(
        canonical_physical_layout) -> None:
    source = _residency()
    first = build_resident_pages_from_serving_residency(
        source, canonical_physical_layout)
    second = build_resident_pages_from_serving_residency(
        source, canonical_physical_layout)
    assert first.page_layout.capacity_feasible is True
    assert first.page_layout.pages == second.page_layout.pages
    assert first.page_layout.logical_resident_bytes == (
        first.resident_set.total_resident_logical_bytes)


def test_logical_pass_but_page_rounding_fail_is_loud(
        canonical_physical_layout) -> None:
    page_size = canonical_physical_layout.slot_capacity_bytes
    two_slots = replace(
        canonical_physical_layout,
        physical_slot_count=2,
        total_capacity_bytes=2 * page_size,
    )
    source = _residency(
        usable_capacity_bytes=2 * page_size,
        weight_bytes=float(page_size + 1),
        kv_bytes_per_request=float(page_size - 1),
        resident_bytes_per_request=float(page_size - 1),
        requested_requests=1,
        local_resident_requests=1,
        max_resident_requests=1,
        available_for_requests_bytes=float(page_size - 1),
        local_capacity_utilization=1.0,
    )
    with pytest.raises(
            CACapacitySemanticsMismatchError,
            match="LOGICAL_CAPACITY_PASS; PAGE_ALLOCATED_CAPACITY_FAIL"):
        build_resident_pages_from_serving_residency(source, two_slots)


def test_canonical_m3d_n16_c_to_object_to_page(
        canonical_physical_layout) -> None:
    experiment = run_serving_experiment(SERVING_CONFIG, project_root=ROOT)
    m3d = next(point for point in experiment.operating_points
               if point.architecture == "orthogonal_m3d_igzo")
    source = next(row for row in m3d.rows if row.requested_requests == 16)
    result = build_resident_pages_from_serving_residency(
        source, canonical_physical_layout)
    resident = result.resident_set
    pages = result.page_layout
    assert source.local_resident_requests == 16
    assert source.spilled_requests == 0
    assert resident.resident_weight_object_count == 1
    assert resident.resident_kv_object_count == 16
    assert resident.resident_object_count == 17
    assert resident.byte_closure_error == 0
    assert pages.capacity_feasible is True
    assert pages.page_to_physical_slot_mapping_included is False
