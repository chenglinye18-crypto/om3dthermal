"""Run the frozen A placement through the canonical steady-state GPU-PCG solver."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from om3dthermal.case_runner import run_steady_pipeline
from om3dthermal.experiment import load_experiment_spec, load_workload_spec
from om3dthermal.placement import evaluate_nmp_locality_case
from om3dthermal.placement.nmp_load_balance import (
    build_locality_only_placement, build_performance_balanced_placement,
    remaining_external_bytes_for_ownership,
)
from om3dthermal.power import (calculate_memory_power,
    calculate_physical_access_latency, load_case_config, resolve_case_geometry,
    resolve_system_power)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.power.nmp_die_activity import canonical_nmp_hardware, evaluate_nmp_die_activity
from om3dthermal.power.nmp_die_power import build_nmp_die_power_map
from om3dthermal.thermal.nmp_die_mapping import (
    analyze_nmp_die_thermal_pipeline, compile_nmp_die_thermal_config)
from om3dthermal.workload import build_m3d_workload_page_demand

try:
    from evaluate_die_local_placement import ROOT, _architecture
except ModuleNotFoundError:
    from scripts.evaluate_die_local_placement import ROOT, _architecture


def _frozen_case_inputs(requests: int):
    layout, bandwidth = _architecture()
    case_path = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"
    case = load_case_config(case_path)
    geometry = resolve_case_geometry(case)
    memory = calculate_memory_power(case, project_root=ROOT, geometry=geometry)
    topology = calculate_m3d_subarray(case.architecture.m3d_subarray, geometry.m3d)
    feol = calculate_feol_route(case.architecture.feol_route, topology)
    physical = calculate_physical_access_latency(
        case.architecture.physical_access_latency, feol_route=feol,
        miv_length_per_layer_um=memory.diagnostics["miv_length_per_layer_um"],
        miv_delay_per_layer_ns=memory.diagnostics["miv_delay_per_layer_ns"],
        miv_status=memory.diagnostics["miv_latency_status"],
        miv_parameter_status=memory.diagnostics["miv_resistance_parameter_status"],
        miv_provenance=memory.diagnostics["miv_resistance_provenance"])
    base = load_workload_spec(
        ROOT / "configs/workload/llama31_8b_decode_b1_s131072.yaml",
        project_root=ROOT).decode
    workload = base.model_copy(update={"batch_size": requests})
    demand = build_m3d_workload_page_demand(workload, layout)
    gpu_flops = load_experiment_spec(
        ROOT / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml",
        project_root=ROOT).scenario.effective_compute_flops_per_second
    baseline = evaluate_nmp_locality_case(
        workload, demand, layout, physical, bandwidth, case="NON_NMP_GPU",
        nmp_aggregate_tflops=None, gpu_compute_flops_per_s=gpu_flops)
    hardware = canonical_nmp_hardware(layout.slab_count)
    canonical = evaluate_nmp_locality_case(
        workload, demand, layout, physical, bandwidth,
        case="NMP_LOCALITY_AWARE_PLACEMENT",
        nmp_aggregate_tflops=hardware.aggregate_peak_flops / 1e12,
        gpu_compute_flops_per_s=gpu_flops)
    bandwidth_per_die = (
        bandwidth.local_service_groups_per_die * bandwidth.read_payload_bytes_per_service
        / (bandwidth.service_cycle_scale * canonical.placement.local_access_latency_ns * 1e-9))
    locality = build_locality_only_placement(
        workload, demand, layout, bandwidth_per_die_bytes_per_s=bandwidth_per_die,
        compute_per_die_flops_per_s=hardware.peak_flops_per_die)
    external_bytes = remaining_external_bytes_for_ownership(locality.unit_loads, locality.ownership)
    external_ms = external_bytes / bandwidth.coil_bandwidth_bytes_per_s * 1e3
    placement = build_performance_balanced_placement(
        workload, demand, layout, bandwidth_per_die_bytes_per_s=bandwidth_per_die,
        compute_per_die_flops_per_s=hardware.peak_flops_per_die)
    activity = evaluate_nmp_die_activity(
        workload, demand, layout, bandwidth,
        local_access_latency_ns=canonical.placement.local_access_latency_ns,
        external_boundary_time_ms=external_ms, ownership=placement.ownership)
    power_map = build_nmp_die_power_map(case, memory, topology, feol, activity, placement)
    gain = requests / (activity.decode_step_interval_ms * 1e-3) / baseline.timing.tokens_per_s
    system = resolve_system_power(case, project_root=ROOT, geometry=geometry)
    return case, system, power_map, gain, placement


def _summary(baseline) -> dict:
    power = np.array([row.total_power_W for row in baseline.dies])
    temp = np.array([row.die_temperature_degC for row in baseline.dies])
    return {
        "requests": baseline.requests,
        "power_min_W": float(power.min()), "power_mean_W": float(power.mean()),
        "power_p90_W": float(np.percentile(power, 90)), "power_max_W": float(power.max()),
        "power_max_over_mean": float(power.max() / power.mean()),
        "temperature_min_degC": float(temp.min()), "temperature_mean_degC": float(temp.mean()),
        "temperature_p90_degC": float(np.percentile(temp, 90)),
        "temperature_max_degC": float(temp.max()),
        "temperature_spread_degC": float(temp.max() - temp.min()),
        "power_temperature_correlation": baseline.power_temperature_correlation,
        "memory_power_temperature_correlation": baseline.memory_power_temperature_correlation,
        "nmp_power_temperature_correlation": baseline.nmp_power_temperature_correlation,
        "max_power_die_id": baseline.max_power_die_id,
        "hottest_m3d_die_id": baseline.hottest_m3d_die_id,
        "global_Tmax_degC": baseline.global_Tmax_degC,
        "global_Tmax_region": baseline.global_Tmax_region,
    }


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for requests in (1, 8, 16):
        case, system, power_map, gain, placement = _frozen_case_inputs(requests)
        if any(row.nmp_logic_overhead_factor != 1.0 for row in power_map.die_powers):
            raise ValueError("gamma_NMP drifted from one")
        # The frozen ownership model has no inter-die transfer path; all
        # non-local bytes remain on the existing external boundary.
        direct_die_to_die_bytes = 0
        thermal_config, regions = compile_nmp_die_thermal_config(case, system, power_map)
        pipeline = run_steady_pipeline(
            thermal_config, backend="gpu_pcg", alpha=0.7,
            rtol=float(case.thermal["solver"]["rtol"]), max_iterations=100_000,
            initial_temperature_K=293.15)
        if not pipeline.result.converged:
            raise RuntimeError(f"N={requests} GPU-PCG thermal solve did not converge")
        baseline = analyze_nmp_die_thermal_pipeline(
            requests=requests, power_map=power_map, regions=regions,
            pipeline=pipeline, solver_backend="gpu_pcg")
        cases.append({
            "A_canonical_gain": gain,
            "aggregate_m3d_power_W": power_map.aggregate_total_W,
            "gpu_power_W": system.gpu_power_W,
            "direct_die_to_die_bytes": direct_die_to_die_bytes,
            "package_source_power_W": pipeline.power.total_power_W,
            "summary": _summary(baseline),
            "baseline": baseline.as_dict(),
        })
        (output_dir / f"nmp_thermal_N{requests}.json").write_text(
            json.dumps(cases[-1], indent=2), encoding="utf-8")
    payload = {
        "model": "FROZEN_A_PERFORMANCE_BALANCED_NMP_STEADY_STATE_THERMAL_BASELINE",
        "solver": "FP64_MATRIX_FREE_GPU_PCG_JACOBI_STEADY_STATE_ONLY",
        "physical_die_count": 98,
        "thermal_power_mapping_closure": "PASS",
        "residual_external_mapping_status": "RESIDUAL_EXTERNAL_THERMAL_MAPPING_APPROXIMATION",
        "cases": cases,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/nmp_thermal_baseline")
    payload = run(parser.parse_args().output_dir)
    for case in payload["cases"]:
        row = case["summary"]
        print(f"N={row['requests']} Pmax/Pmean={row['power_max_over_mean']:.6f} "
              f"dT={row['temperature_spread_degC']:.6f} C "
              f"corr={row['power_temperature_correlation']:.6f} "
              f"power_die={row['max_power_die_id']} hot_die={row['hottest_m3d_die_id']}")


if __name__ == "__main__":
    main()
