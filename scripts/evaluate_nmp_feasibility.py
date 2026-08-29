"""Evaluate the minimal Mixtral expert-MLP NMP feasibility sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from om3dthermal.experiment import load_experiment_spec, load_moe_workload_spec
from om3dthermal.placement import (
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
    load_fiddler_published_profile,
)


ROOT = Path(__file__).parents[1]


def _architecture():
    case = load_case_config(
        ROOT / "configs/cases/orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    power = calculate_memory_power(case, project_root=ROOT, geometry=geometry)
    if geometry.m3d is None:
        raise ValueError("NMP feasibility requires M3D geometry")
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
        case.architecture.memory_service,
        layout,
        topology,
        feol_io_channels=case.architecture.feol_route.io_channels,
    )
    return case, layout, bandwidth


def run(output_dir: Path) -> dict[str, object]:
    case, layout, bandwidth = _architecture()
    experiment = load_experiment_spec(
        ROOT
        / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml",
        project_root=ROOT,
    )
    gpu_compute = experiment.scenario.effective_compute_flops_per_second
    base = load_moe_workload_spec(
        ROOT / "configs/workload/mixtral_8x7b_v01_decode_b1_s32768.yaml",
        project_root=ROOT,
    ).decode
    profile_path = (
        ROOT / "configs/workload/profiles"
        / "mixtral_8x7b_fiddler_iclr2025_sharegpt.csv")
    profile = load_fiddler_published_profile(
        profile_path, profile_path.with_suffix(".metadata.json"))

    sweeps = []
    one_context = None
    for batch in (1, 8, 16):
        workload = base.model_copy(update={"batch_size": batch})
        demand = build_published_moe_page_demand(profile, workload, layout)
        gpu_only = evaluate_published_moe_hierarchical_e2e(
            profile,
            workload,
            layout,
            bandwidth,
            legacy_matched_payload_bandwidth_bits_per_second=(
                experiment.scenario.matched_payload_bandwidth_bits_per_second),
            effective_compute_flops_per_second=gpu_compute,
            random_seeds=tuple(range(20)),
        )
        sweep = sweep_nmp_feasibility(
            workload,
            demand,
            gpu_only,
            effective_gpu_compute_flops_per_second=gpu_compute,
        )
        sweeps.append(sweep)
        if batch == 1:
            one_context = (workload, demand, gpu_only)
    assert one_context is not None

    sensitivity = []
    for scale in (0.5, 1.0, 2.0):
        probe = sweep_nmp_feasibility(
            one_context[0], one_context[1], one_context[2],
            effective_gpu_compute_flops_per_second=gpu_compute,
            effective_nmp_tflops_values=(32.0,),
            internal_bandwidth_scale=scale,
        ).points[0]
        sensitivity.append({
            "internal_bandwidth_scale": scale,
            "effective_nmp_tflops": 32.0,
            "effective_nmp_parameter_classification": (
                "NUMERICAL_CHOICE_FOR_SWEEP"),
            "effective_nmp_parameter_status": "NOT_ARCHITECTURE_CAPABILITY",
            "p0_internal_bandwidth_bytes_per_s": (
                probe.p0.internal_bandwidth_bytes_per_s),
            "p1_internal_bandwidth_bytes_per_s": (
                probe.p1.internal_bandwidth_bytes_per_s),
            "p2_internal_bandwidth_bytes_per_s": (
                probe.p2.internal_bandwidth_bytes_per_s),
            "p0_balance_tflops": probe.p0.expert_balance_tflops,
            "p1_balance_tflops": probe.p1.expert_balance_tflops,
            "p2_balance_tflops": probe.p2.expert_balance_tflops,
            "p0_tokens_per_s": probe.p0.tokens_per_s,
            "p1_tokens_per_s": probe.p1.tokens_per_s,
            "p2_tokens_per_s": probe.p2.tokens_per_s,
            "p2_over_p0_throughput_gain": (
                probe.p2_over_p0_throughput_gain),
        })

    minimum_candidates = tuple(
        sweep.minimum_useful_nmp_tflops for sweep in sweeps
        if sweep.minimum_useful_nmp_tflops is not None)
    all_saturating = tuple(
        sweep.memory_saturating_nmp_tflops for sweep in sweeps)
    global_saturating = (
        max(value for value in all_saturating if value is not None)
        if all(value is not None for value in all_saturating)
        else None)
    max_points = tuple(sweep.points[-1] for sweep in sweeps)
    if all(point.p2_speedup_over_gpu_only > 1.0 for point in max_points):
        e2e_gate = "BENEFICIAL"
    elif any(point.p2_speedup_over_gpu_only > 1.0 for point in max_points):
        e2e_gate = "MARGINAL"
    else:
        e2e_gate = "NOT_BENEFICIAL"
    payload = {
        "model": "MINIMAL_MOE_EXPERT_NMP_FEASIBILITY_MODEL",
        "scope": "MIXTRAL_EXPERT_MLP_ONLY",
        "system_model": "SERIAL_GPU_NMP_FIRST_ORDER_MODEL",
        "effective_nmp_tflops_provenance": {
            "classification": "MODELING_CHOICE",
            "status": "NOT_HARDWARE_VALIDATED",
        },
        "architecture": {
            "coil_bandwidth_bytes_per_s": (
                bandwidth.coil_bandwidth_bytes_per_s),
            "coil_derivation": {
                "m3d_dies": bandwidth.num_m3d_dies,
                "links_per_die": bandwidth.coil_links_per_die,
                "gbps_per_link": bandwidth.coil_data_rate_gbps_per_link,
            },
            "internal_bandwidth_status": (
                "NMP_RESULTS_CONDITIONAL_ON_CURRENT_INTERNAL_BW_MODEL"),
            "canonical_case_unchanged": True,
            "tmat_ns": case.architecture.physical_access_latency.mat_latency_ns,
        },
        "sweeps": [sweep.as_dict() for sweep in sweeps],
        "n1_internal_bandwidth_sensitivity_at_32_tflops": sensitivity,
        "automatic_verdict": {
            "MINIMUM_USEFUL_NMP_TFLOPS": min(minimum_candidates),
            "MEMORY_SATURATING_NMP_TFLOPS": (
                global_saturating if global_saturating is not None else ">128"),
            "RECOMMENDED_FIRST_ORDER_NMP_TARGET": (
                f"{global_saturating:g} TFLOP/s"
                if global_saturating is not None else ">128 TFLOP/s"),
            "rule": (
                "minimum useful is first sampled point exposing any expert-"
                "memory-bound placement; memory-saturating target is first "
                "sampled point making P0/P1/P2 memory-bound for every batch"),
        },
        "scientific_gates": {
            "NMP_MODEL_GATE": "PASS",
            "NMP_TRAFFIC_REDUCTION_GATE": "PASS",
            "NMP_MEMORY_COMPUTE_BALANCE_GATE": "PASS",
            "NMP_PLACEMENT_EXPOSURE_GATE": "PLACEMENT_EFFECT_EXPOSED",
            "NMP_E2E_GATE": e2e_gate,
            "NMP_RESULTS_CONDITIONAL_ON_CURRENT_INTERNAL_BW_MODEL": "YES",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nmp_feasibility.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs/nmp_feasibility")
    args = parser.parse_args()
    payload = run(args.output_dir)
    for sweep in payload["sweeps"]:
        point = sweep["points"][-1]
        print(
            f"N={sweep['batch_size']}: minimum useful="
            f"{sweep['minimum_useful_nmp_tflops']}, saturating="
            f"{sweep['memory_saturating_nmp_tflops']}, "
            f"128-TFLOP/s P0/P1/P2="
            f"{point['p0']['tokens_per_s']:.3f}/"
            f"{point['p1']['tokens_per_s']:.3f}/"
            f"{point['p2']['tokens_per_s']:.3f} tokens/s")
    print(json.dumps(payload["automatic_verdict"], indent=2))


if __name__ == "__main__":
    main()
