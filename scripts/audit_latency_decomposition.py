"""DreamRAM-vs-M3D latency decomposition and modeling-confidence audit.

Read-only: prints the unified taxonomy mapping, both native latency
decompositions, the FEOL R' diagnostic sensitivity, parameter-confidence
table, modeling-risk ranking, and scientific gates.  No workload E2E, no
placement optimization, no bandwidth or thermal evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from om3dthermal.power import (
    audit_dream_latency_decomposition,
    build_m3d_latency_decomposition,
    build_risk_ranking,
    build_unified_taxonomy,
    calculate_memory_power,
    calculate_physical_access_latency,
    classify_gates,
    load_case_config,
    resolve_case_geometry,
    run_feol_resistance_sensitivity,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray


def _m3d_inputs(root: Path):
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
    miv_delays = power.diagnostics["miv_delay_per_layer_ns"]
    return case, topology, latency, miv_delays


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    dream = audit_dream_latency_decomposition(root)
    case, topology, latency, miv_delays = _m3d_inputs(root)
    m3d = build_m3d_latency_decomposition(case, latency)
    taxonomy = build_unified_taxonomy()
    sensitivity = run_feol_resistance_sensitivity(
        case, topology, miv_delays)
    risks = build_risk_ranking(dream, m3d)
    gates = classify_gates(dream, m3d)

    report = {
        "unified_taxonomy": [row.as_dict() for row in taxonomy],
        "dream_decomposition": dream.as_dict(),
        "m3d_decomposition": m3d.as_dict(),
        "feol_resistance_sensitivity": [row.as_dict() for row in sensitivity],
        "risk_ranking": [item.as_dict() for item in risks],
        "gates": gates.as_dict(),
        "answers": _answers(dream, m3d),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    _print_text_report(dream, m3d, taxonomy, sensitivity, risks, gates)
    return 0


def _answers(dream, m3d):
    return {
        "Q1_dream_largest_term": {
            "term": "tCL",
            "value_ns": dream.tcl_ns,
            "share_of_first_access": dream.tcl_ns / dream.first_access_ns,
            "dominant_subterm": "lateral bus transport",
            "lateral_ns": dream.tcl_lateral_bus_ns,
        },
        "Q2_m3d_largest_term": {
            "term": "tMAT",
            "value_ns": m3d.mat_latency_ns,
            "share_of_near_access": m3d.near_mat_share,
            "share_of_far_access": m3d.far_mat_share,
        },
        "Q3_apples_to_apples": (
            "NO: Dream 64.17 ns is a row-conflict worst case dominated by "
            "system-level bus/TSV/DQ transport, while M3D 10-18 ns is a "
            "clean-access path dominated by a lumped MAT placeholder; "
            "PARTIALLY_MATCHED"),
        "Q4_most_likely_m3d_underestimate": (
            "tMAT lumped placeholder scope (activation/sensing/decoder/"
            "reset not validated) and the 0 ns interface placeholder; "
            "tMAT dominates because it is the largest term"),
        "Q5_spatial_ordering_robust_to": (
            "tMAT and tInterface (position-independent constants) and "
            "MIV R' (ps-level contribution); spatial ordering is "
            "topology-determined by FEOL route lengths"),
        "Q6_highest_priority_model_fix": (
            "MAT read-timing model: physically decompose and validate "
            "the 10 ns tMAT against the 2T0C IGZO array read path "
            "(activation, bitline development, sensing, decoder); "
            "second: coil interface startup latency"),
    }


def _print_text_report(dream, m3d, taxonomy, sensitivity, risks, gates
                       ) -> None:
    print()
    print("=== Table: unified stage mapping ===")
    for row in taxonomy:
        print(f"{row.stage:<34} D:{row.dream_status:<28} "
              f"M:{row.m3d_status:<28} comparable={row.comparable}")
    print()
    print("=== DreamRAM first-access decomposition ===")
    print(f"tRP   = {dream.trp_ns:.3f} ns  (aliased to tRCD by reference "
          f"assumption: {dream.trp_aliased_to_trcd})")
    print(f"tRCD  = {dream.trcd_ns:.3f} ns = signal "
          f"{dream.trcd_signal_ns:.3f} + sensing {dream.trcd_sensing_ns:.3f}")
    print(f"tCL   = {dream.tcl_ns:.3f} ns = lateral "
          f"{dream.tcl_lateral_bus_ns:.3f} + TSV {dream.tcl_tsv_vertical_ns:.3f}"
          f" + DQ window {dream.tcl_dq_window_ns:.3f}")
    print(f"first = {dream.first_access_ns:.3f} ns  "
          f"access_case={dream.access_case}")
    print(f"row-hit {dream.row_hit_ns:.3f} / row-miss "
          f"{dream.row_miss_ns:.3f} / row-conflict {dream.row_conflict_ns:.3f}"
          f" ns ({dream.row_state_source})")
    print()
    print("=== M3D first-access decomposition ===")
    print(f"tMAT       = {m3d.mat_latency_ns:.3f} ns  "
          f"({m3d.parameter_provenance['tMAT']})")
    print(f"tMIV       = {m3d.miv_min_ns:.4f} .. {m3d.miv_max_ns:.4f} ns")
    print(f"tFEOL      = {m3d.feol_min_ns:.3f} .. {m3d.feol_max_ns:.3f} ns "
          f"(median {m3d.feol_median_ns:.3f}, p90 {m3d.feol_p90_ns:.3f})")
    print(f"tInterface = {m3d.interface_latency_ns:.3f} ns  "
          f"({m3d.parameter_provenance['tInterface']})")
    print(f"near total = {m3d.near_total_ns:.3f} ns "
          f"(tMAT share {100.0 * m3d.near_mat_share:.1f}%)")
    print(f"far  total = {m3d.far_total_ns:.3f} ns "
          f"(tMAT {100.0 * m3d.far_mat_share:.1f}% / "
          f"tFEOL {100.0 * m3d.far_feol_share:.1f}% / "
          f"tMIV {100.0 * m3d.miv_share_of_far_total:.2f}%)")
    print()
    print("=== FEOL R' diagnostic sensitivity (canonical unchanged) ===")
    for row in sensitivity:
        print(f"R'={row.resistance_ohm_per_um:.1f} ohm/um: total "
              f"{row.total_min_ns:.3f} .. {row.total_max_ns:.3f} ns, "
              f"FEOL share of far {100.0 * row.feol_share_of_far_total:.1f}%, "
              f"argmax cluster unchanged={row.argmax_cluster_unchanged}")
    print()
    print("=== LATENCY_MODEL_RISK_RANKING ===")
    for item in risks:
        print(f"{item.rank}. {item.item} [{item.impact}]")
    print()
    print(f"DREAM_LATENCY_DECOMPOSITION_GATE = "
          f"{gates.dream_latency_decomposition_gate}")
    print(f"M3D_LATENCY_DECOMPOSITION_GATE   = "
          f"{gates.m3d_latency_decomposition_gate}")
    print(f"LATENCY_SEMANTIC_MATCH_GATE      = "
          f"{gates.latency_semantic_match_gate}")
    print(f"M3D_ABSOLUTE_LATENCY_CONFIDENCE  = "
          f"{gates.m3d_absolute_latency_confidence}")
    print(f"M3D_SPATIAL_LATENCY_RANKING_CONFIDENCE = "
          f"{gates.m3d_spatial_latency_ranking_confidence}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
