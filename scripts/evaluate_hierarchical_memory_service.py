"""Canonical legacy-versus-hierarchical M3D memory-service evaluation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from om3dthermal.experiment import (
    load_experiment_spec,
    load_moe_workload_spec,
    load_workload_spec,
)
from om3dthermal.placement import (
    compare_fast_region_placements,
    compare_placement_serving_performance,
    evaluate_hierarchical_placement_serving_timing,
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
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.workload import (
    build_m3d_workload_page_demand,
    build_published_moe_page_demand,
    evaluate_llm_decode,
    evaluate_moe_decode,
    load_fiddler_published_profile,
)
from om3dthermal.placement.moe_published_e2e import (
    MoEPlacementPerformanceInput,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    case, topology, layout, bandwidth = _architecture(root)
    experiment = load_experiment_spec(
        root / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml",
        project_root=root)
    scenario = experiment.scenario
    dense_base = load_workload_spec(
        root / "configs/workload/llama31_8b_decode_b1_s131072.yaml",
        project_root=root).decode
    moe_base = load_moe_workload_spec(
        root / "configs/workload/mixtral_8x7b_v01_decode_b1_s32768.yaml",
        project_root=root).decode
    profile_path = (
        root / "configs/workload/profiles"
        / "mixtral_8x7b_fiddler_iclr2025_sharegpt.csv")
    profile = load_fiddler_published_profile(
        profile_path, profile_path.with_suffix(".metadata.json"))
    seeds = tuple(range(20))
    dense_rows = []
    moe_rows = []
    moe_one = None
    for requests in (1, 8, 16):
        dense = dense_base.model_copy(update={"batch_size": requests})
        dense_rows.append(_dense_case(
            dense, layout, bandwidth,
            legacy_bandwidth_bits_per_s=(
                scenario.matched_payload_bandwidth_bits_per_second),
            compute_flops_per_s=scenario.effective_compute_flops_per_second,
            seeds=seeds,
        ))
        moe = moe_base.model_copy(update={"batch_size": requests})
        evaluated = evaluate_published_moe_hierarchical_e2e(
            profile,
            moe,
            layout,
            bandwidth,
            legacy_matched_payload_bandwidth_bits_per_second=(
                scenario.matched_payload_bandwidth_bits_per_second),
            effective_compute_flops_per_second=(
                scenario.effective_compute_flops_per_second),
            random_seeds=seeds,
        )
        moe_rows.append(asdict(evaluated))
        if requests == 1:
            moe_one = (moe, evaluated)
    assert moe_one is not None
    sensitivity = _sensitivities(
        case,
        topology,
        layout,
        bandwidth,
        profile,
        moe_one[0],
        moe_one[1],
        scenario.effective_compute_flops_per_second,
    )
    output = {
        "architecture_bandwidth_closure": bandwidth.as_dict(),
        "scenario": {
            "legacy_fixed_bandwidth_bits_per_s": (
                scenario.matched_payload_bandwidth_bits_per_second),
            "legacy_bandwidth_status": scenario.bandwidth_status,
            "effective_compute_flops_per_s": (
                scenario.effective_compute_flops_per_second),
            "compute_status": scenario.compute_status,
            "random_seeds": list(seeds),
        },
        "dense": dense_rows,
        "mixtral_fiddler": moe_rows,
        "sensitivity": sensitivity,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _dense_case(
    workload,
    layout,
    bandwidth,
    *,
    legacy_bandwidth_bits_per_s: float,
    compute_flops_per_s: float,
    seeds,
):
    demand = build_m3d_workload_page_demand(workload, layout)
    placement = compare_fast_region_placements(
        demand, layout, random_seeds=seeds)
    legacy = compare_placement_serving_performance(
        workload,
        demand,
        placement,
        layout,
        matched_payload_bandwidth_bits_per_second=(
            legacy_bandwidth_bits_per_s),
        effective_compute_flops_per_second=compute_flops_per_s,
    )
    p1_placement = place_pages_on_slots(
        demand, layout, slot_policy="FASTEST", page_ordering="CANONICAL")
    metrics = evaluate_llm_decode(workload)
    common = {
        "metrics": metrics,
        "demand": demand,
        "physical_layout": layout,
        "bandwidth_closure": bandwidth,
        "requested_requests": workload.batch_size,
        "effective_compute_flops_per_second": compute_flops_per_s,
    }
    p0 = evaluate_hierarchical_placement_serving_timing(
        **common,
        strategy="P0_LATENCY_OBLIVIOUS_RANDOM_MEAN",
        physical_access_latency_avg_ns=(
            placement.random.mean_average_access_latency_ns),
        physical_access_latency_max_ns=(
            placement.random.mean_max_occupied_latency_ns),
    )
    p1 = evaluate_hierarchical_placement_serving_timing(
        **common,
        strategy="P1_FAST_REGION_ONLY",
        physical_access_latency_avg_ns=(
            p1_placement.weighted_average_access_latency_ns),
        physical_access_latency_max_ns=(
            p1_placement.max_occupied_slot_latency_ns),
    )
    p2 = evaluate_hierarchical_placement_serving_timing(
        **common,
        strategy="P2_DEMAND_AWARE_FAST_REGION",
        physical_access_latency_avg_ns=(
            placement.fast_pack.weighted_average_access_latency_ns),
        physical_access_latency_max_ns=(
            placement.fast_pack.max_occupied_slot_latency_ns),
    )
    return {
        "requested_requests": workload.batch_size,
        "legacy": asdict(legacy),
        "hierarchical": {
            "random": asdict(p0),
            "fast_region": asdict(p1),
            "demand_aware": asdict(p2),
            "total_latency_gain": 1.0 - p2.total_step_time_ms / p0.total_step_time_ms,
            "total_throughput_gain": (
                p2.aggregate_tokens_per_s / p0.aggregate_tokens_per_s - 1.0),
        },
    }


def _sensitivities(
    case,
    topology,
    layout,
    nominal_bandwidth,
    profile,
    workload,
    canonical,
    compute_flops_per_s,
):
    demand = build_published_moe_page_demand(profile, workload, layout)
    metrics = evaluate_moe_decode(workload)
    performance_input = MoEPlacementPerformanceInput(
        required_capacity_bytes=metrics.required_capacity_bytes,
        read_bytes_per_token=(
            demand.total_read_bytes_per_decode_step / workload.batch_size),
        write_bytes_per_token=(
            demand.kv_write_bytes_per_decode_step / workload.batch_size),
        flops_per_token=metrics.flops_per_token,
    )
    physical = canonical.legacy.all_read_physical

    def run(closure, scale):
        common = {
            "metrics": performance_input,
            "demand": demand,
            "physical_layout": layout,
            "bandwidth_closure": closure,
            "requested_requests": workload.batch_size,
            "effective_compute_flops_per_second": compute_flops_per_s,
            "internal_parallelism_scale": scale,
        }
        p0 = evaluate_hierarchical_placement_serving_timing(
            **common,
            strategy="P0",
            physical_access_latency_avg_ns=(
                physical.random.weighted_average_latency_ns),
            physical_access_latency_max_ns=(
                physical.random.max_occupied_slot_latency_ns),
        )
        p2 = evaluate_hierarchical_placement_serving_timing(
            **common,
            strategy="P2",
            physical_access_latency_avg_ns=(
                physical.popularity_aware_fast_region.weighted_average_latency_ns),
            physical_access_latency_max_ns=(
                physical.popularity_aware_fast_region.max_occupied_slot_latency_ns),
        )
        return {
            "p0_internal_bytes_per_s": p0.bandwidth.internal_bandwidth_bytes_per_s,
            "p2_internal_bytes_per_s": p2.bandwidth.internal_bandwidth_bytes_per_s,
            "p0_effective_bytes_per_s": p0.bandwidth.effective_bandwidth_bytes_per_s,
            "p2_effective_bytes_per_s": p2.bandwidth.effective_bandwidth_bytes_per_s,
            "p0_bottleneck": p0.bandwidth.bottleneck,
            "p2_bottleneck": p2.bandwidth.bottleneck,
            "placement_throughput_gain": (
                p2.aggregate_tokens_per_s / p0.aggregate_tokens_per_s - 1.0),
        }

    coil_rows = []
    spec = case.architecture.memory_service
    for rate in (4.0, 8.0, 16.0, 32.0):
        changed_coil = spec.coil.model_copy(
            update={"data_rate_gbps_per_link": rate})
        changed_spec = spec.model_copy(update={"coil": changed_coil})
        closure = derive_architecture_bandwidth(
            changed_spec,
            layout,
            topology,
            feol_io_channels=case.architecture.feol_route.io_channels,
        )
        coil_rows.append({
            "coil_data_rate_gbps_per_link": rate,
            "coil_bandwidth_bytes_per_s": closure.coil_bandwidth_bytes_per_s,
            **run(closure, 1.0),
        })
    parallel_rows = []
    for scale in (0.5, 1.0, 2.0):
        parallel_rows.append({
            "internal_parallelism_scale": scale,
            **run(nominal_bandwidth, scale),
        })
    cycle_rows = []
    for scale in (0.5, 1.0, 2.0):
        changed_internal = spec.internal.model_copy(
            update={"service_cycle_scale": scale})
        changed_spec = spec.model_copy(update={"internal": changed_internal})
        closure = derive_architecture_bandwidth(
            changed_spec,
            layout,
            topology,
            feol_io_channels=case.architecture.feol_route.io_channels,
        )
        cycle_rows.append({
            "service_cycle_scale": scale,
            **run(closure, 1.0),
        })
    return {
        "coil_data_rate": coil_rows,
        "internal_parallelism": parallel_rows,
        "service_cycle_scale": cycle_rows,
    }


def _architecture(root: Path):
    case = load_case_config(root / "configs/cases/orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    power = calculate_memory_power(case, project_root=root, geometry=geometry)
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
    bandwidth = derive_architecture_bandwidth(
        case.architecture.memory_service,
        layout,
        topology,
        feol_io_channels=case.architecture.feol_route.io_channels,
    )
    return case, topology, layout, bandwidth


if __name__ == "__main__":
    raise SystemExit(main())
