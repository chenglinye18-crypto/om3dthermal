"""Targeted tests for M3D-only workload page read demand."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import pytest

from om3dthermal.experiment.config import load_workload_spec
from om3dthermal.power import (
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.workload import (
    LLMDecodeInput,
    M3DOnlyCapacityError,
    build_m3d_only_workload_objects,
    build_m3d_workload_page_demand,
)
import om3dthermal.workload.m3d_page_demand as demand_module


ROOT = Path(__file__).parents[1]
M3D_CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"
WORKLOAD = ROOT / "configs" / "workload" / "llama31_8b_decode_b1_s131072.yaml"
MIB = 2**20


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


def _small_workload(weight_bytes: int = 3 * MIB) -> LLMDecodeInput:
    return LLMDecodeInput(
        n_param=weight_bytes,
        n_layers=1,
        n_heads_q=1,
        n_heads_kv=1,
        d_model=1,
        d_ff=1,
        vocab_size=1,
        batch_size=1,
        context_length=0,
        weight_bits=8,
        kv_bits=8,
        runtime_bytes=0,
    )


def test_tail_page_demand_is_proportional(canonical_physical_layout) -> None:
    result = build_m3d_workload_page_demand(
        _small_workload(), canonical_physical_layout)
    assert result.weight_page_count == 2
    assert [page.logical_size_bytes for page in result.page_demands] == [
        2 * MIB, MIB]
    assert result.page_demands[1].read_demand_bytes_per_decode_step == (
        0.5 * result.page_demands[0].read_demand_bytes_per_decode_step)


def test_weight_kv_and_total_traffic_close(canonical_physical_layout) -> None:
    spec = load_workload_spec(WORKLOAD, project_root=ROOT)
    workload = spec.decode.model_copy(update={"batch_size": 16})
    result = build_m3d_workload_page_demand(workload, canonical_physical_layout)
    weight_sum = sum(x.read_demand_bytes_per_decode_step
                     for x in result.page_demands
                     if x.object_type == "WEIGHT")
    kv_sum = sum(x.read_demand_bytes_per_decode_step
                 for x in result.page_demands if x.object_type == "KV")
    total = sum(x.read_demand_bytes_per_decode_step for x in result.page_demands)
    assert weight_sum == pytest.approx(
        result.total_weight_read_bytes_per_decode_step)
    assert kv_sum == pytest.approx(result.total_kv_read_bytes_per_decode_step)
    assert total == pytest.approx(result.total_read_bytes_per_decode_step)
    assert result.weight_traffic_closure_error_bytes == pytest.approx(0.0)
    assert result.kv_traffic_closure_error_bytes == pytest.approx(0.0)
    assert result.total_traffic_closure_error_bytes == pytest.approx(0.0)


def test_objects_pages_demands_and_order_are_deterministic(
        canonical_physical_layout) -> None:
    workload = _small_workload()
    assert build_m3d_only_workload_objects(workload) == (
        build_m3d_only_workload_objects(workload))
    first = build_m3d_workload_page_demand(workload, canonical_physical_layout)
    second = build_m3d_workload_page_demand(workload, canonical_physical_layout)
    assert first.resident_objects == second.resident_objects
    assert first.page_layout.pages == second.page_layout.pages
    assert first.page_demands == second.page_demands


def test_page_rounded_capacity_failure_does_not_spill(
        canonical_physical_layout) -> None:
    one_slot = replace(
        canonical_physical_layout,
        physical_slot_count=1,
        total_capacity_bytes=canonical_physical_layout.slot_capacity_bytes,
    )
    with pytest.raises(M3DOnlyCapacityError, match="M3D_ONLY_CAPACITY_FAIL"):
        build_m3d_workload_page_demand(_small_workload(), one_slot)


def test_builder_has_no_c_residency_dependency_or_policy_leakage(
        canonical_physical_layout) -> None:
    source = inspect.getsource(demand_module)
    assert "CapacityResidencyResult" not in source
    assert "local_resident_requests" not in source
    assert "spilled_requests" not in source
    result = build_m3d_workload_page_demand(
        _small_workload(), canonical_physical_layout)
    forbidden = {"assigned_slot", "assigned_cluster", "assigned_layer",
                 "assigned_slab", "hotness", "priority", "thermal_score"}
    assert forbidden.isdisjoint(result.__dataclass_fields__)
    assert forbidden.isdisjoint(
        result.page_demands[0].__dataclass_fields__)


def test_canonical_n16_capacity_and_model_regressions(
        canonical_physical_layout) -> None:
    spec = load_workload_spec(WORKLOAD, project_root=ROOT)
    result = build_m3d_workload_page_demand(
        spec.decode.model_copy(update={"batch_size": 16}),
        canonical_physical_layout,
    )
    assert result.page_layout.capacity_feasible is True
    assert result.page_layout.page_size_bytes == 2 * MIB
    assert canonical_physical_layout.total_capacity_gib == 428.75
    latencies = tuple(slot.physical_access_latency_ns
                      for slot in canonical_physical_layout.slot_classes)
    assert min(latencies) == pytest.approx(10.050912300102683)
    assert max(latencies) == pytest.approx(18.008616609416016)
