"""Hierarchical M3D internal/coil bandwidth and streaming service tests."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import pytest

from om3dthermal.experiment import load_workload_spec
from om3dthermal.experiment import load_moe_workload_spec
from om3dthermal.placement import (
    evaluate_hierarchical_placement_serving_timing,
    evaluate_placement_serving_timing,
    evaluate_published_moe_hierarchical_e2e,
    place_pages_on_slots,
)
from om3dthermal.power import (
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    derive_architecture_bandwidth,
    load_case_config,
    resolve_case_geometry,
    resolve_effective_bandwidth,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.workload import (
    build_m3d_workload_page_demand,
    evaluate_llm_decode,
    load_fiddler_published_profile,
)
import om3dthermal.evaluator.llm_decode_performance as performance_module


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"
WORKLOAD = ROOT / "configs/workload/llama31_8b_decode_b1_s131072.yaml"


@pytest.fixture(scope="module")
def canonical():
    case = load_case_config(CASE)
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
    layout = calculate_physical_capacity_layout(
        topology,
        latency,
        slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )
    spec = case.architecture.memory_service
    assert spec is not None
    closure = derive_architecture_bandwidth(
        spec, layout, topology,
        feol_io_channels=case.architecture.feol_route.io_channels)
    workload = load_workload_spec(WORKLOAD, project_root=ROOT).decode
    return case, geometry, topology, layout, closure, workload


def test_architecture_organization_and_die_count_reuse(canonical) -> None:
    _, geometry, topology, layout, closure, _ = canonical
    assert (topology.Nrow, topology.Ncol) == (512, 512)
    assert topology.subarrays_per_cluster == 8 * 8 == 64
    assert topology.clusters_per_layer == 280
    assert layout.layers_per_cluster == 8
    assert closure.num_m3d_dies == layout.slab_count == (
        geometry.memory_region_count) == 98
    assert closure.die_count_source == "GEOMETRY_MEMORY_REGION_COUNT"


def test_coil_bandwidth_unit_closure(canonical) -> None:
    _, _, _, _, closure, _ = canonical
    expected_bits = 98 * 50 * 8.0 * 1e9
    assert closure.coil_bandwidth_bits_per_s == expected_bits == 39.2e12
    assert closure.coil_bandwidth_bytes_per_s == expected_bits / 8
    assert closure.coil_bandwidth_bytes_per_s / 1e9 == 4900.0
    assert closure.coil_bandwidth_bytes_per_s / 1e12 == 4.9


@pytest.mark.parametrize(
    ("field", "value", "factor"),
    (("links_per_die", 25, 0.5),
     ("data_rate_gbps_per_link", 16.0, 2.0)),
)
def test_coil_parameters_scale_derived_bandwidth(
    canonical, field, value, factor
) -> None:
    case, _, topology, layout, nominal, _ = canonical
    spec = case.architecture.memory_service
    changed_coil = spec.coil.model_copy(update={field: value})
    changed_spec = spec.model_copy(update={"coil": changed_coil})
    changed = derive_architecture_bandwidth(
        changed_spec, layout, topology,
        feol_io_channels=case.architecture.feol_route.io_channels)
    assert changed.coil_bandwidth_bytes_per_s == pytest.approx(
        factor * nominal.coil_bandwidth_bytes_per_s)


def test_internal_bandwidth_is_spatial_and_prefix_monotonic(canonical) -> None:
    _, _, _, _, closure, _ = canonical
    assert closure.total_parallel_service_units == 98 * 50
    assert closure.clusters_per_service == 4
    assert closure.subarrays_per_service == 256
    assert closure.delivered_bits_per_service == 256
    assert closure.read_payload_bytes_per_service == 32
    assert closure.internal_bandwidth_fast_bytes_per_s >= (
        closure.internal_bandwidth_average_bytes_per_s)
    assert closure.internal_bandwidth_average_bytes_per_s >= (
        closure.internal_bandwidth_slow_bytes_per_s) > 0
    prefix = tuple(
        point.internal_bandwidth_bytes_per_s
        for point in closure.prefix_bandwidth)
    assert all(left >= right for left, right in zip(prefix, prefix[1:]))


def test_internal_parallelism_scaling_and_effective_min(canonical) -> None:
    _, _, _, _, closure, _ = canonical
    latency = (
        closure.average_service_cycle_ns / closure.service_cycle_scale)
    half = resolve_effective_bandwidth(
        closure, latency, internal_parallelism_scale=0.5)
    nominal = resolve_effective_bandwidth(closure, latency)
    double = resolve_effective_bandwidth(
        closure, latency, internal_parallelism_scale=2.0)
    assert half.internal_bandwidth_bytes_per_s == pytest.approx(
        0.5 * nominal.internal_bandwidth_bytes_per_s)
    assert double.internal_bandwidth_bytes_per_s == pytest.approx(
        2.0 * nominal.internal_bandwidth_bytes_per_s)
    assert nominal.effective_bandwidth_bytes_per_s == min(
        nominal.internal_bandwidth_bytes_per_s,
        nominal.coil_bandwidth_bytes_per_s,
    )
    assert nominal.bottleneck == "COIL_INTERFACE"
    assert nominal.gpu_internal_bandwidth_bytes_per_s is None


def test_service_cycle_scale_is_explicit_and_inverse_bandwidth(canonical) -> None:
    case, _, topology, layout, nominal, _ = canonical
    spec = case.architecture.memory_service
    changed_internal = spec.internal.model_copy(
        update={"service_cycle_scale": 2.0})
    changed_spec = spec.model_copy(update={"internal": changed_internal})
    changed = derive_architecture_bandwidth(
        changed_spec,
        layout,
        topology,
        feol_io_channels=case.architecture.feol_route.io_channels,
    )
    assert changed.average_service_cycle_ns == pytest.approx(
        2.0 * nominal.average_service_cycle_ns)
    assert changed.internal_bandwidth_average_bytes_per_s == pytest.approx(
        0.5 * nominal.internal_bandwidth_average_bytes_per_s)


def test_bottleneck_identity_includes_optional_gpu(canonical) -> None:
    _, _, _, _, closure, _ = canonical
    gpu_limited = replace(
        closure,
        gpu_internal_bandwidth_bytes_per_s=1e12,
        gpu_internal_status=(
            "NON_BINDING_NUMERICAL_CHOICE_NOT_HARDWARE_CAPABILITY"),
    )
    result = resolve_effective_bandwidth(
        gpu_limited,
        closure.average_service_cycle_ns / closure.service_cycle_scale,
    )
    assert result.effective_bandwidth_bytes_per_s == 1e12
    assert result.bottleneck == "GPU_INTERNAL"


def test_hierarchical_streaming_is_deterministic_and_not_page_serial(canonical) -> None:
    _, _, _, layout, closure, workload = canonical
    demand = build_m3d_workload_page_demand(workload, layout)
    placement = place_pages_on_slots(
        demand, layout, slot_policy="FASTEST",
        page_ordering="DEMAND_DESCENDING")
    metrics = evaluate_llm_decode(workload)
    kwargs = {
        "metrics": metrics,
        "demand": demand,
        "physical_layout": layout,
        "bandwidth_closure": closure,
        "requested_requests": workload.batch_size,
        "strategy": "TEST",
        "physical_access_latency_avg_ns": (
            placement.weighted_average_access_latency_ns),
        "physical_access_latency_max_ns": (
            placement.max_occupied_slot_latency_ns),
        "effective_compute_flops_per_second": 1e14,
    }
    first = evaluate_hierarchical_placement_serving_timing(**kwargs)
    assert first == evaluate_hierarchical_placement_serving_timing(**kwargs)
    assert first.startup_step_time_ms == pytest.approx(
        placement.weighted_average_access_latency_ns * 1e-6)
    assert first.memory_stage_step_time_ms == pytest.approx(
        first.startup_step_time_ms + first.streaming_transfer_step_time_ms)
    assert "RESIDENT_2MIB_PAGE_IS_PLACEMENT_ONLY" in (
        first.transaction_granularity_semantics)
    assert not hasattr(first, "read_page_equivalents_per_decode_step")


def test_legacy_model_remains_unchanged(canonical) -> None:
    _, _, _, layout, _, workload = canonical
    demand = build_m3d_workload_page_demand(workload, layout)
    result = evaluate_placement_serving_timing(
        workload,
        demand,
        layout,
        strategy="LEGACY",
        physical_access_latency_avg_ns=10.0,
        physical_access_latency_max_ns=10.0,
        matched_payload_bandwidth_bits_per_second=39.2e12,
        effective_compute_flops_per_second=1e14,
    )
    assert result.access_count_semantics == (
        "SERIAL_ONE_LATENCY_EXPOSURE_PER_2MIB_READ_PAGE_EQUIVALENT")
    assert result.read_page_equivalents_per_decode_step == pytest.approx(
        demand.total_read_bytes_per_decode_step / (2 * 2**20))


def test_no_canonical_4p9_tbps_magic_in_core_evaluator() -> None:
    source = inspect.getsource(performance_module)
    assert "4.9e12" not in source
    assert "4900000000000" not in source


def test_mixtral_hierarchical_integration_and_profile_regression(canonical) -> None:
    _, _, _, layout, closure, _ = canonical
    workload = load_moe_workload_spec(
        ROOT / "configs/workload/mixtral_8x7b_v01_decode_b1_s32768.yaml",
        project_root=ROOT,
    ).decode
    profile_path = (
        ROOT / "configs/workload/profiles"
        / "mixtral_8x7b_fiddler_iclr2025_sharegpt.csv")
    profile = load_fiddler_published_profile(
        profile_path, profile_path.with_suffix(".metadata.json"))
    result = evaluate_published_moe_hierarchical_e2e(
        profile,
        workload,
        layout,
        closure,
        legacy_matched_payload_bandwidth_bits_per_second=39.2e12,
        effective_compute_flops_per_second=1e14,
        random_seeds=(0, 1),
    )
    assert profile.extraction_gate == "PASS"
    assert result.legacy.total_throughput_gain > result.total_throughput_gain
    assert result.random_timing.bandwidth.bottleneck == "COIL_INTERFACE"
    assert result.fast_region_timing.bandwidth.bottleneck == "COIL_INTERFACE"
    assert result.popularity_aware_timing.bandwidth.bottleneck == (
        "COIL_INTERFACE")
    assert result.total_throughput_gain >= 0.0
