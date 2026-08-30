"""Run the strict no-direct-die-to-die dense LLaMA placement sweep."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from om3dthermal.experiment import load_experiment_spec, load_workload_spec
from om3dthermal.placement import (evaluate_die_local_placement,
    independent_domain_semantics, minimum_active_domains)
from om3dthermal.power import (calculate_memory_power, calculate_physical_access_latency,
    calculate_physical_capacity_layout, derive_architecture_bandwidth, load_case_config,
    resolve_case_geometry)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.workload import build_m3d_workload_page_demand

ROOT = Path(__file__).parents[1]

def _architecture():
    case = load_case_config(ROOT / "configs/cases/orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    power = calculate_memory_power(case, project_root=ROOT, geometry=geometry)
    topology = calculate_m3d_subarray(case.architecture.m3d_subarray, geometry.m3d)
    feol = calculate_feol_route(case.architecture.feol_route, topology)
    latency = calculate_physical_access_latency(case.architecture.physical_access_latency,
        feol_route=feol, miv_length_per_layer_um=power.diagnostics["miv_length_per_layer_um"],
        miv_delay_per_layer_ns=power.diagnostics["miv_delay_per_layer_ns"],
        miv_status=power.diagnostics["miv_latency_status"],
        miv_parameter_status=power.diagnostics["miv_resistance_parameter_status"],
        miv_provenance=power.diagnostics["miv_resistance_provenance"])
    layout = calculate_physical_capacity_layout(topology, latency, slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"])
    return layout, derive_architecture_bandwidth(case.architecture.memory_service, layout, topology,
        feol_io_channels=case.architecture.feol_route.io_channels)

def run(output_dir: Path) -> dict[str, object]:
    layout, bandwidth = _architecture()
    scenario = load_experiment_spec(ROOT / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml", project_root=ROOT).scenario
    base = load_workload_spec(ROOT / "configs/workload/llama31_8b_decode_b1_s131072.yaml", project_root=ROOT).decode
    domains = independent_domain_semantics(layout)
    rows = []
    for requests in (1, 8, 16):
        workload = base.model_copy(update={"batch_size": requests})
        demand = build_m3d_workload_page_demand(workload, layout)
        dmin = minimum_active_domains(demand, domains)
        points = sorted({dmin, max(dmin, (domains.independent_memory_domain_count + 3)//4),
                         max(dmin, (domains.independent_memory_domain_count + 1)//2),
                         max(dmin, (3*domains.independent_memory_domain_count + 3)//4),
                         domains.independent_memory_domain_count})
        models = {}
        for model in ("FUSED_DIE_LOCAL", "CONSERVATIVE_GPU_BOUNDARY"):
            result = [evaluate_die_local_placement(workload, demand, layout, bandwidth,
                active_domains=d, communication_model=model,
                gpu_compute_flops_per_s=scenario.effective_compute_flops_per_second).as_dict() for d in points]
            best = min(result, key=lambda x: x["fast_no_overlap"]["total_step_ms"])
            models[model] = {"points": result, "D_opt": best["active_domain_count"], "D_opt_result": best}
        rows.append({"requests": requests, "working_set_bytes": demand.allocated_page_bytes,
            "D_min_fit": dmin, "D_max": domains.independent_memory_domain_count, "models": models})
    payload = {"model": "NO_DIRECT_DIE_TO_DIE_DENSE_DECODE_PLACEMENT", "DIRECT_DIE_TO_DIE_COMMUNICATION": "FORBIDDEN",
        "independent_domain_semantics": asdict(domains), "fixed_external_bandwidth_bytes_per_s": bandwidth.coil_bandwidth_bytes_per_s,
        "canonical_overlap": "NO_OVERLAP", "rows": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "die_local_placement.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/die_local_placement")
    payload = run(parser.parse_args().output_dir)
    for row in payload["rows"]:
        for model, result in row["models"].items():
            opt = result["D_opt_result"]
            print(f"N={row['requests']} {model} Dopt={result['D_opt']} E2E={opt['e2e_speedup_no_overlap']:.4f}x locality={opt['traffic']['derived_local_byte_fraction']:.6f}")

if __name__ == "__main__": main()
