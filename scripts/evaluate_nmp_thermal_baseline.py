"""Run the frozen A placement through the canonical steady-state GPU-PCG solver."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from om3dthermal.case_runner import run_steady_pipeline
from om3dthermal.architecture_comparison import _resolve_case_power_operating_points
from om3dthermal.experiment import load_workload_spec
from om3dthermal.power import resolve_system_power
from om3dthermal.serving import evaluate_nmp_decode_batch
from om3dthermal.thermal.nmp_die_mapping import (
    analyze_nmp_die_thermal_pipeline, compile_nmp_die_thermal_config)

try:
    from evaluate_die_local_placement import ROOT
except ModuleNotFoundError:
    from scripts.evaluate_die_local_placement import ROOT


def _frozen_case_inputs(requests: int):
    base = load_workload_spec(
        ROOT / "configs/workload/llama31_8b_decode_b1_s131072.yaml",
        project_root=ROOT).decode
    workload = base.model_copy(update={"batch_size": requests})
    result = evaluate_nmp_decode_batch(workload, project_root=ROOT)
    trace = result.execution_trace
    if trace is None:
        raise ValueError("thermal carrier construction requires a feasible Decode")
    architecture = trace.architecture
    case = architecture.case
    geometry = architecture.geometry
    placement = trace.placement
    activity = trace.activity
    power_map = trace.power_map
    baseline_seconds = (
        float(result.matrix_weight_read_bytes_per_step)
        + 8.0 * (
            float(result.kv_read_bytes_per_step)
            + float(result.kv_write_bytes_per_step))) / 2.4e12
    gain = baseline_seconds / (activity.decode_step_interval_ms * 1e-3)
    gpu_point, transfer_point, service_point = _resolve_case_power_operating_points(
        case, ROOT)
    system = resolve_system_power(
        case, project_root=ROOT, geometry=geometry,
        gpu_operating_point=gpu_point,
        transfer_operating_point=transfer_point,
        bandwidth_service_operating_point=service_point)
    # NMP GPU processing is Softmax only; memory carriers come from power_map.
    # Keep the solver and spatial mapping untouched.
    interval_s=activity.decode_step_interval_ms*1e-3
    system=replace(system,
        gpu_power_W=(activity.gpu_static_energy_j+activity.softmax_dynamic_energy_j)/interval_s,
        diagnostics={**system.diagnostics,"nmp_gpu_energy_boundary":"SOFTMAX_LOCAL_PLUS_STATIC_ONCE",
                     "nmp_activity":activity.as_dict()})
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
