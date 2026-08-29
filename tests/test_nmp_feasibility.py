"""Targeted tests for the minimal Mixtral expert NMP feasibility model."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from om3dthermal.experiment import load_experiment_spec, load_moe_workload_spec
from om3dthermal.placement import (
    evaluate_nmp_feasibility,
    evaluate_published_moe_hierarchical_e2e,
    sweep_nmp_feasibility,
)
from om3dthermal.power import (
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    derive_architecture_bandwidth,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.workload import (
    build_published_moe_page_demand,
    evaluate_moe_decode,
    load_fiddler_published_profile,
)
import om3dthermal.placement.nmp_feasibility as nmp_module


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"
WORKLOAD = ROOT / "configs/workload/mixtral_8x7b_v01_decode_b1_s32768.yaml"
PROFILE = (
    ROOT / "configs/workload/profiles"
    / "mixtral_8x7b_fiddler_iclr2025_sharegpt.csv")
EXPERIMENT = (
    ROOT / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml")


@pytest.fixture(scope="module")
def architecture():
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
        miv_length_per_layer_um=power.diagnostics[
            "miv_length_per_layer_um"],
        miv_delay_per_layer_ns=power.diagnostics["miv_delay_per_layer_ns"],
        miv_status=power.diagnostics["miv_latency_status"],
        miv_parameter_status=power.diagnostics[
            "miv_resistance_parameter_status"],
        miv_provenance=power.diagnostics["miv_resistance_provenance"],
    )
    layout = calculate_physical_capacity_layout(
        topology, latency,
        slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )
    bandwidth = derive_architecture_bandwidth(
        case.architecture.memory_service, layout, topology,
        feol_io_channels=case.architecture.feol_route.io_channels)
    base = load_moe_workload_spec(WORKLOAD, project_root=ROOT).decode
    profile = load_fiddler_published_profile(
        PROFILE, PROFILE.with_suffix(".metadata.json"))
    experiment = load_experiment_spec(EXPERIMENT, project_root=ROOT)
    return (
        case, geometry, power, topology, feol, latency, layout, bandwidth,
        base, profile, experiment.scenario.effective_compute_flops_per_second,
        experiment.scenario.matched_payload_bandwidth_bits_per_second,
    )


def _evaluate(architecture, batch: int, tflops: float = 32.0, scale=1.0):
    (*_, layout, bandwidth, base, profile, gpu_compute, legacy_bw) = architecture
    workload = base.model_copy(update={"batch_size": batch})
    demand = build_published_moe_page_demand(profile, workload, layout)
    gpu_only = evaluate_published_moe_hierarchical_e2e(
        profile, workload, layout, bandwidth,
        legacy_matched_payload_bandwidth_bits_per_second=legacy_bw,
        effective_compute_flops_per_second=gpu_compute,
        random_seeds=(0, 1),
    )
    result = evaluate_nmp_feasibility(
        workload, demand, gpu_only,
        effective_gpu_compute_flops_per_second=gpu_compute,
        effective_nmp_tflops=tflops,
        internal_bandwidth_scale=scale,
    )
    return workload, demand, gpu_only, result


@pytest.fixture(scope="module")
def n1(architecture):
    return _evaluate(architecture, 1)


def test_tflops_unit_and_parameter_provenance(n1):
    *_, result = n1
    assert result.effective_nmp_tflops == 32.0
    assert result.effective_nmp_flops_per_second == 32.0e12
    assert result.nmp_parameter_classification == "MODELING_CHOICE"
    assert result.nmp_parameter_status == "NOT_HARDWARE_VALIDATED"


def test_expert_flops_and_bytes_reuse_existing_workload(n1):
    workload, demand, _, result = n1
    metrics = evaluate_moe_decode(workload)
    closure = result.workload
    assert closure.expert_flops_per_token == (
        metrics.active_expert_flops_per_token) == 22_548_578_304
    assert closure.expert_weight_bytes_per_decode_step == (
        demand.total_expert_read_bytes_per_decode_step
    ) == metrics.active_expert_weight_bytes_per_decode_step
    assert closure.expert_weight_bytes_per_decode_step == 21 * 2**30
    assert closure.expert_weight_reuse_semantics == metrics.weight_reuse_status


def test_arithmetic_intensity_and_balance_closure(n1):
    *_, result = n1
    closure = result.workload
    assert closure.expert_arithmetic_intensity_flop_per_byte == pytest.approx(
        closure.expert_flops_per_decode_step
        / closure.expert_weight_bytes_per_decode_step)
    assert closure.expert_arithmetic_intensity_flop_per_byte == 1.0
    for point in (result.p0, result.p1, result.p2):
        assert point.expert_balance_tflops == pytest.approx(
            point.internal_bandwidth_bytes_per_s
            * closure.expert_arithmetic_intensity_flop_per_byte / 1e12)


def test_roofline_max_and_bottleneck_classification(architecture):
    *_, low = _evaluate(architecture, 1, tflops=8.0)
    *_, high = _evaluate(architecture, 1, tflops=32.0)
    for point in (low.p0, low.p1, low.p2):
        assert point.nmp_expert_bottleneck == "COMPUTE"
        assert point.expert_nmp_time_ms == point.expert_compute_time_ms
    for point in (high.p0, high.p1, high.p2):
        assert point.nmp_expert_bottleneck == "MEMORY"
        assert point.expert_nmp_time_ms == point.expert_memory_time_ms
    for point in (low.p0, high.p2):
        assert point.expert_nmp_time_ms == max(
            point.expert_memory_time_ms, point.expert_compute_time_ms)


def test_activation_bytes_are_derived_from_dimensions(n1):
    workload, _, _, result = n1
    closure = result.workload
    expected = (
        2 * workload.num_experts_per_tok * workload.hidden_size
        * (workload.weight_bits // 8) * workload.num_hidden_layers)
    assert closure.activation_bytes_per_token == expected == 2**20
    assert closure.activation_bytes_per_decode_step == (
        workload.batch_size * expected)
    assert closure.activation_traffic_semantics.startswith(
        "CONSERVATIVE_NO_ACTIVATION_AGGREGATION_MODEL")
    source = inspect.getsource(nmp_module)
    assert "1048576" not in source
    assert "4.9e12" not in source
    assert "4900000000000" not in source


def test_coil_is_derived_and_weights_stay_local(n1, architecture):
    *_, bandwidth, _, _, _, _ = architecture
    _, _, _, result = n1
    assert bandwidth.coil_bandwidth_bytes_per_s == (
        bandwidth.num_m3d_dies * bandwidth.coil_links_per_die
        * bandwidth.coil_data_rate_gbps_per_link * 1e9 / 8)
    for point in (result.p0, result.p1, result.p2):
        assert point.coil_bandwidth_bytes_per_s == (
            bandwidth.coil_bandwidth_bytes_per_s)
        assert point.expert_weight_bytes_crossing_coil_per_decode_step == 0.0
        assert point.activation_bytes_crossing_coil_per_decode_step == 2**20
        assert point.expert_weights_residency == (
            "RESIDENT_IN_M3D_DO_NOT_CROSS_COIL")


def test_shared_and_kv_traffic_are_not_removed(n1):
    _, demand, _, result = n1
    expected = (
        demand.total_shared_weight_read_bytes_per_decode_step
        + demand.total_kv_read_bytes_per_decode_step
        + demand.kv_write_bytes_per_decode_step)
    assert result.workload.gpu_remaining_memory_bytes_per_decode_step == (
        expected) > 0.0
    for point in (result.p0, result.p1, result.p2):
        assert point.remaining_gpu_memory_bytes_crossing_coil_per_decode_step == (
            expected)
        assert point.gpu_remaining_memory_time_ms > 0.0
        assert point.serial_step_time_ms == pytest.approx(
            point.gpu_nonexpert_compute_time_ms
            + point.gpu_remaining_memory_time_ms
            + point.activation_coil_time_ms
            + point.expert_nmp_time_ms)


def test_gpu_only_baseline_is_reused_unchanged(n1):
    _, _, gpu_only, result = n1
    pairs = (
        (gpu_only.random_timing, result.gpu_only_p0),
        (gpu_only.fast_region_timing, result.gpu_only_p1),
        (gpu_only.popularity_aware_timing, result.gpu_only_p2),
    )
    for timing, baseline in pairs:
        assert baseline.current_total_step_time_ms == timing.total_step_time_ms
        assert baseline.current_tokens_per_s == timing.aggregate_tokens_per_s
        assert baseline.expert_weight_bytes_crossing_coil_per_decode_step == (
            21 * 2**30)
        assert baseline.activation_bytes_crossing_coil_per_decode_step == 0.0
    assert result.gpu_only_p0.current_total_step_time_ms == pytest.approx(
        6.1336760222277)
    assert result.gpu_only_p0.current_tokens_per_s == pytest.approx(
        163.034369010708)


def test_p0_p1_p2_internal_bandwidth_is_propagated(n1):
    _, _, gpu_only, result = n1
    pairs = (
        (gpu_only.random_timing, result.p0),
        (gpu_only.fast_region_timing, result.p1),
        (gpu_only.popularity_aware_timing, result.p2),
    )
    for timing, point in pairs:
        assert point.internal_bandwidth_bytes_per_s == (
            timing.bandwidth.internal_bandwidth_bytes_per_s)
    assert result.p0.internal_bandwidth_bytes_per_s < (
        result.p1.internal_bandwidth_bytes_per_s
    ) < result.p2.internal_bandwidth_bytes_per_s


def test_batch_sweep_regimes_and_determinism(architecture):
    summaries = []
    for batch, useful, saturating in (
            (1, 16.0, 16.0), (8, 128.0, 128.0), (16, None, None)):
        (*_, layout, bandwidth, base, profile,
         gpu_compute, legacy_bw) = architecture
        workload = base.model_copy(update={"batch_size": batch})
        demand = build_published_moe_page_demand(profile, workload, layout)
        gpu_only = evaluate_published_moe_hierarchical_e2e(
            profile, workload, layout, bandwidth,
            legacy_matched_payload_bandwidth_bits_per_second=legacy_bw,
            effective_compute_flops_per_second=gpu_compute,
            random_seeds=(0, 1),
        )
        first = sweep_nmp_feasibility(
            workload, demand, gpu_only,
            effective_gpu_compute_flops_per_second=gpu_compute)
        repeated = sweep_nmp_feasibility(
            workload, demand, gpu_only,
            effective_gpu_compute_flops_per_second=gpu_compute)
        assert first == repeated
        assert first.minimum_useful_nmp_tflops == useful
        assert first.memory_saturating_nmp_tflops == saturating
        assert tuple(point.effective_nmp_tflops for point in first.points) == (
            8.0, 16.0, 32.0, 64.0, 128.0)
        summaries.append(first)
    assert summaries[0].points[1].p2_over_p0_throughput_gain > 0.12
    assert summaries[1].points[-1].p2_over_p0_throughput_gain > 0.03
    assert summaries[2].points[-1].p2_over_p0_throughput_gain < 1e-6


def test_internal_bandwidth_sensitivity_scales_balance(architecture):
    _, _, _, half = _evaluate(architecture, 1, tflops=32.0, scale=0.5)
    _, _, _, nominal = _evaluate(architecture, 1, tflops=32.0, scale=1.0)
    _, _, _, double = _evaluate(architecture, 1, tflops=32.0, scale=2.0)
    for field in ("p0", "p1", "p2"):
        low = getattr(half, field)
        mid = getattr(nominal, field)
        high = getattr(double, field)
        assert low.internal_bandwidth_bytes_per_s == pytest.approx(
            0.5 * mid.internal_bandwidth_bytes_per_s)
        assert high.internal_bandwidth_bytes_per_s == pytest.approx(
            2.0 * mid.internal_bandwidth_bytes_per_s)
        assert low.expert_balance_tflops == pytest.approx(
            0.5 * mid.expert_balance_tflops)
        assert high.expert_balance_tflops == pytest.approx(
            2.0 * mid.expert_balance_tflops)


def test_model_does_not_mutate_latency_placement_or_thermal(architecture, n1):
    case, _, power, topology, feol, latency, *rest = architecture
    before = (
        case.model_dump(), power.as_dict(), topology.as_dict(),
        feol.as_dict(), latency.as_dict())
    _, _, gpu_only, _ = n1
    gpu_before = gpu_only
    _evaluate(architecture, 1, tflops=128.0)
    after = (
        case.model_dump(), power.as_dict(), topology.as_dict(),
        feol.as_dict(), latency.as_dict())
    assert after == before
    assert gpu_only == gpu_before
    source = inspect.getsource(nmp_module)
    assert "om3dthermal.thermal" not in source
    assert case.architecture.physical_access_latency.mat_latency_ns == 10.0
    assert case.architecture.vertical.miv_resistance_ohm_per_um == 10.0
    assert case.architecture.feol_route.wire.resistance_ohm_per_um == 2.0
