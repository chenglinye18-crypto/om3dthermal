"""Export the read-only MAT-output to coil-TX-input topology audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from om3dthermal.power import (
    audit_dream_latency_decomposition,
    calculate_hierarchical_mat_to_coil,
    calculate_normalized_single_path_delay,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray


ROOT = Path(__file__).parents[1]
DEFAULT_CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("CSV export requires at least one row")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(case_path: Path, output_dir: Path) -> dict[str, object]:
    case = load_case_config(case_path)
    geometry = resolve_case_geometry(case)
    if geometry.m3d is None:
        raise ValueError("MAT-to-coil audit requires M3D geometry")
    topology = calculate_m3d_subarray(
        case.architecture.m3d_subarray, geometry.m3d)
    feol = calculate_feol_route(case.architecture.feol_route, topology)
    audit = calculate_hierarchical_mat_to_coil(feol)
    dream = audit_dream_latency_decomposition(ROOT)
    normalized = calculate_normalized_single_path_delay(feol)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = audit.as_dict()
    payload["dream_lateral_ns"] = dream.tcl_lateral_bus_ns
    payload["normalized_11mm_single_path"] = normalized.as_dict()
    payload["normalized_length_experiment"] = {
        "D1_same_m3d_rc_current_direct_geometry": (
            audit.legacy_direct_latency.as_dict()),
        "D2_same_m3d_rc_hierarchical_geometry": (
            audit.hierarchical_latency.as_dict()),
        "D3_same_m3d_rc_dream_like_11mm_single_path_ns": (
            normalized.delay_ns),
        "D3_status": "SANITY_DIAGNOSTIC_NOT_CANONICAL",
    }
    payload["quantitative_comparison"] = {
        "hierarchical_over_direct_median": (
            audit.hierarchical_latency.median_ns
            / audit.legacy_direct_latency.median_ns),
        "hierarchical_over_direct_p90": (
            audit.hierarchical_latency.p90_ns
            / audit.legacy_direct_latency.p90_ns),
        "hierarchical_over_direct_max": (
            audit.hierarchical_latency.max_ns
            / audit.legacy_direct_latency.max_ns),
        "dream_over_direct_median": (
            dream.tcl_lateral_bus_ns
            / audit.legacy_direct_latency.median_ns),
        "dream_over_hierarchical_median": (
            dream.tcl_lateral_bus_ns
            / audit.hierarchical_latency.median_ns),
        "dream_minus_direct_median_ns": (
            dream.tcl_lateral_bus_ns
            - audit.legacy_direct_latency.median_ns),
        "dream_minus_hierarchical_median_ns": (
            dream.tcl_lateral_bus_ns
            - audit.hierarchical_latency.median_ns),
    }
    payload["comparison_gates"] = {
        "MAT_TO_COIL_TOPOLOGY_GATE": "PASS",
        "DIRECT_FEOL_MODEL_GATE": "TOPOLOGY_INCOMPLETE",
        "HIERARCHICAL_FEOL_MODEL_GATE": "PARTIAL",
        "M3D_LATERAL_ABSOLUTE_LATENCY_CONFIDENCE": "LOW",
        "M3D_LATERAL_SPATIAL_ORDERING_CONFIDENCE": "MEDIUM",
        "DREAM_M3D_LATERAL_COMPARISON_GATE": "PARTIAL",
    }
    payload["OLD_0_TO_8NS_FEOL_VERDICT"] = "VALID_LOWER_BOUND_ONLY"
    (output_dir / "mat_to_coil_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(
        output_dir / "mat_to_coil_paths.csv",
        [path.as_dict() for path in audit.paths])
    _write_csv(
        output_dir / "mat_to_coil_ports.csv",
        [port.as_dict() for port in audit.ports])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs" / "mat_to_coil_audit")
    args = parser.parse_args()
    payload = run(args.case, args.output_dir)
    legacy = payload["legacy_direct_latency"]
    hierarchical = payload["hierarchical_latency"]
    print("MAT -> coil lateral comparison (ns)")
    print("                         median       p90        max")
    print(f"DreamRAM lateral              -          -  "
          f"{payload['dream_lateral_ns']:10.6f}")
    print(f"M3D legacy direct    {legacy['median_ns']:10.6f} "
          f"{legacy['p90_ns']:10.6f} {legacy['max_ns']:10.6f}")
    print(f"M3D hierarchical     {hierarchical['median_ns']:10.6f} "
          f"{hierarchical['p90_ns']:10.6f} "
          f"{hierarchical['max_ns']:10.6f}")
    print(f"fan-in all ports min/median/mean/max = "
          f"{payload['fan_in_min']}/{payload['fan_in_median']}/"
          f"{payload['fan_in_mean']}/{payload['fan_in_max']}")
    print(f"active ports = {payload['active_port_count']}/"
          f"{payload['port_count']}; active fan-in = "
          f"{payload['active_fan_in_min']}/"
          f"{payload['active_fan_in_median']}/"
          f"{payload['active_fan_in_mean']}/"
          f"{payload['active_fan_in_max']}")
    print(f"11 mm M3D-RC diagnostic = "
          f"{payload['normalized_11mm_single_path']['delay_ns']:.6f} ns")


if __name__ == "__main__":
    main()
