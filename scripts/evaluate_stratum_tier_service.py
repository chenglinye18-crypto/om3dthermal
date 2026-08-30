"""Evaluate Stratum-style tier-dependent local M3D service abstraction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from om3dthermal.experiment import load_experiment_spec, load_workload_spec
from om3dthermal.placement import (
    compare_fast_region_placements,
    compare_placement_serving_performance,
    sweep_local_service_fraction,
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
from om3dthermal.workload import build_m3d_workload_page_demand


ROOT = Path(__file__).parents[1]


def _architecture():
    case = load_case_config(
        ROOT / "configs/cases/orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    power = calculate_memory_power(case, project_root=ROOT, geometry=geometry)
    if geometry.m3d is None:
        raise ValueError("tier-service evaluation requires M3D geometry")
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
    return case, layout, bandwidth


def _first_fraction(rows: list[dict[str, object]], target: float):
    return next((row["local_service_fraction"] for row in rows
                 if row["end_to_end_speedup"] >= target), None)


def run(output_dir: Path) -> dict[str, object]:
    case, layout, bandwidth = _architecture()
    experiment = load_experiment_spec(
        ROOT
        / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml",
        project_root=ROOT,
    )
    scenario = experiment.scenario
    base = load_workload_spec(
        ROOT / "configs/workload/llama31_8b_decode_b1_s131072.yaml",
        project_root=ROOT,
    ).decode
    rows = []
    negative_control = []
    for requests in (1, 8, 16):
        workload = base.model_copy(update={"batch_size": requests})
        demand = build_m3d_workload_page_demand(workload, layout)
        placement = compare_fast_region_placements(
            demand, layout, random_seeds=tuple(range(20)))
        sweep = sweep_local_service_fraction(
            workload, demand, layout, bandwidth, placement.fast_pack,
            matched_external_bandwidth_bits_per_second=(
                scenario.matched_payload_bandwidth_bits_per_second),
            effective_compute_flops_per_second=(
                scenario.effective_compute_flops_per_second),
        )
        serialized = [item.as_dict() for item in sweep]
        rows.append({
            "requests": requests,
            "occupancy_fraction": placement.occupancy_fraction,
            "t_no_tier_ns": serialized[0]["no_tier"]["physical_service_latency_ns"],
            "t_fast_ns": serialized[0]["tier_aware_fast_pack"]["physical_service_latency_ns"],
            "service_rate_speedup": serialized[0]["tier_aware_fast_pack"]["service_rate_speedup"],
            "sweep": serialized,
            "first_fraction_reaching_1p1x": _first_fraction(serialized, 1.1),
            "first_fraction_reaching_1p2x": _first_fraction(serialized, 1.2),
            "first_fraction_reaching_1p3x": _first_fraction(serialized, 1.3),
        })
        external = compare_placement_serving_performance(
            workload, demand, placement, layout,
            matched_payload_bandwidth_bits_per_second=(
                scenario.matched_payload_bandwidth_bits_per_second),
            effective_compute_flops_per_second=(
                scenario.effective_compute_flops_per_second),
        )
        negative_control.append({
            "requests": requests,
            "model": "EXTERNAL_GPU_STREAMING_DIAGNOSTIC",
            "fast_pack_total_step_ms": external.fast_pack.total_step_time_ms,
            "conventional_total_step_ms": external.conventional.total_step_time_ms,
            "fast_pack_tokens_per_s": external.fast_pack.aggregate_tokens_per_s,
            "conventional_tokens_per_s": external.conventional.aggregate_tokens_per_s,
            "fast_over_conventional_throughput_gain": (
                external.fast_pack.aggregate_tokens_per_s
                / external.conventional.aggregate_tokens_per_s - 1.0),
        })
    payload = {
        "model": "STRATUM_STYLE_TIER_DEPENDENT_INTERNAL_SERVICE",
        "scope": (
            "ABSTRACTION_ONLY__NOT_STRATUM_MOE_NMP_MICROARCHITECTURE_"
            "REPRODUCTION"),
        "architecture": {
            "coil_bandwidth_bytes_per_s": bandwidth.coil_bandwidth_bytes_per_s,
            "coil_interface_retained_for_external_fraction": True,
            "canonical_case_unchanged": True,
            "tmat_ns": case.architecture.physical_access_latency.mat_latency_ns,
        },
        "canonical": rows,
        "external_only_negative_control": negative_control,
        "STRATUM_ABSTRACTION_CLOSURE": {
            "stratum_no_tiering": "GLOBAL_WORST_MEMORY_SERVICE_TIMING",
            "stratum_tiering": "PLACEMENT_DEPENDENT_MEMORY_SERVICE_TIMING",
            "our_no_tiering": "GLOBAL_WORST_PHYSICAL_SLOT_LATENCY",
            "our_tiering": "ACTUAL_FAST_PACK_WEIGHTED_SLOT_LATENCY",
            "our_effect": "INTERNAL_SERVICE_RATE_SCALES_AS_T_NO_TIER_OVER_T_EFF",
            "comparison_status": "ABSTRACTION_MATCH_ONLY_NOT_MICROARCHITECTURE_REPRODUCTION",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stratum_tier_service.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs/stratum_tier_service")
    args = parser.parse_args()
    payload = run(args.output_dir)
    for row in payload["canonical"]:
        full_local = row["sweep"][-1]
        print(
            f"N={row['requests']} t_no/t_fast="
            f"{row['t_no_tier_ns']:.6f}/{row['t_fast_ns']:.6f} ns, "
            f"rate={row['service_rate_speedup']:.4f}x, f=1 E2E="
            f"{full_local['end_to_end_speedup']:.4f}x")


if __name__ == "__main__":
    main()
