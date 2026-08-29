"""Print canonical Fiddler-Mixtral placement/E2E results as JSON."""

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
    evaluate_published_moe_placement_e2e,
)
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
    build_m3d_workload_page_demand,
    load_fiddler_published_profile,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    layout = _physical_layout(root)
    experiment = load_experiment_spec(
        root / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml",
        project_root=root,
    )
    scenario = experiment.scenario
    mixtral = load_moe_workload_spec(
        root / "configs/workload/mixtral_8x7b_v01_decode_b1_s32768.yaml",
        project_root=root,
    ).decode
    dense = load_workload_spec(
        root / "configs/workload/llama31_8b_decode_b1_s131072.yaml",
        project_root=root,
    ).decode
    profile_path = (
        root / "configs/workload/profiles"
        / "mixtral_8x7b_fiddler_iclr2025_sharegpt.csv")
    profile = load_fiddler_published_profile(
        profile_path, profile_path.with_suffix(".metadata.json"))
    seeds = tuple(range(20))
    mixtral_rows = []
    dense_rows = []
    for requests in (1, 8, 16):
        moe_workload = mixtral.model_copy(update={"batch_size": requests})
        result = evaluate_published_moe_placement_e2e(
            profile,
            moe_workload,
            layout,
            matched_payload_bandwidth_bits_per_second=(
                scenario.matched_payload_bandwidth_bits_per_second),
            effective_compute_flops_per_second=(
                scenario.effective_compute_flops_per_second),
            random_seeds=seeds,
        )
        mixtral_rows.append(asdict(result))

        dense_workload = dense.model_copy(update={"batch_size": requests})
        dense_demand = build_m3d_workload_page_demand(dense_workload, layout)
        dense_placement = compare_fast_region_placements(
            dense_demand, layout, random_seeds=seeds)
        dense_e2e = compare_placement_serving_performance(
            dense_workload,
            dense_demand,
            dense_placement,
            layout,
            matched_payload_bandwidth_bits_per_second=(
                scenario.matched_payload_bandwidth_bits_per_second),
            effective_compute_flops_per_second=(
                scenario.effective_compute_flops_per_second),
        )
        dense_rows.append({
            "requested_requests": requests,
            "workload_id": "Llama-3.1-8B-BF16-S131072",
            "random_weighted_latency_ns": (
                dense_placement.random.mean_average_access_latency_ns),
            "popularity_aware_fast_latency_ns": (
                dense_placement.fast_pack.weighted_average_access_latency_ns),
            "total_physical_gain": dense_placement.slot_selection_gain_vs_random,
            "random_total_step_time_ms": dense_e2e.random_mean.total_step_time_ms,
            "fast_total_step_time_ms": dense_e2e.fast_pack.total_step_time_ms,
            "total_e2e_latency_gain": dense_e2e.end_to_end_latency_gain_vs_random,
            "total_throughput_gain": dense_e2e.tokens_per_s_gain_vs_random,
        })
    output = {
        "scenario": {
            "matched_payload_bandwidth_bits_per_second": (
                scenario.matched_payload_bandwidth_bits_per_second),
            "bandwidth_status": scenario.bandwidth_status,
            "effective_compute_flops_per_second": (
                scenario.effective_compute_flops_per_second),
            "compute_status": scenario.compute_status,
            "random_seeds": list(seeds),
            "physical_capacity_gib": layout.total_capacity_gib,
        },
        "mixtral_fiddler": mixtral_rows,
        "dense_canonical_comparison": dense_rows,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _physical_layout(root: Path):
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
    return calculate_physical_capacity_layout(
        topology,
        latency,
        slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
