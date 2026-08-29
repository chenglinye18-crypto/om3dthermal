"""DreamRAM/reference-baseline memory-service audit with M3D side-by-side.

Runs the read-only DreamRAM reference audit and the formal M3D
hierarchical memory-service closure, then prints the latency /
internal-bandwidth / interface-bandwidth / bottleneck comparison with
explicit gates.  No E2E workload, placement, or thermal evaluation is
performed.
"""

from __future__ import annotations

import json
from pathlib import Path

from om3dthermal.power import (
    ArchitectureBandwidthClosure,
    DreamReferenceServiceAudit,
    audit_dream_reference_service,
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    derive_architecture_bandwidth,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray


def _m3d_architecture(root: Path):
    """Re-derive the canonical M3D closure from the formal evaluator."""
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
    closure = derive_architecture_bandwidth(
        case.architecture.memory_service,
        layout,
        topology,
        feol_io_channels=case.architecture.feol_route.io_channels,
    )
    return case, latency, layout, closure


def _tbps(bytes_per_s: float) -> float:
    return bytes_per_s / 1e12


def _comparison_gate(
        dream: DreamReferenceServiceAudit,
        m3d_bottleneck: str,
        ) -> str:
    gates = (
        dream.latency_gate,
        dream.internal_bandwidth_gate,
        dream.interface_bandwidth_gate,
    )
    if any(gate != "PASS" for gate in gates):
        return "INSUFFICIENT_INFORMATION"
    dream_interface_bound = dream.bottleneck == "EXTERNAL_INTERFACE"
    m3d_interface_bound = m3d_bottleneck == "COIL_INTERFACE"
    if dream_interface_bound and m3d_interface_bound:
        return "SAME_INTERFACE_BOUND_REGIME"
    return "DIFFERENT_BOTTLENECK_REGIME"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    dream = audit_dream_reference_service(root)
    _, m3d_latency, _, closure = _m3d_architecture(root)

    m3d_coil = closure.coil_bandwidth_bytes_per_s
    m3d_internal_avg = closure.internal_bandwidth_average_bytes_per_s
    m3d_internal_fast = closure.internal_bandwidth_fast_bytes_per_s
    m3d_effective = min(m3d_internal_avg, m3d_coil)
    m3d_bottleneck = (
        "COIL_INTERFACE" if m3d_coil <= m3d_internal_avg else "INTERNAL")
    m3d_r_avg = m3d_internal_avg / m3d_coil
    m3d_r_fast = m3d_internal_fast / m3d_coil
    m3d_first_min = m3d_latency.min_total_latency_ns
    m3d_first_max = m3d_latency.max_total_latency_ns

    gate = _comparison_gate(dream, m3d_bottleneck)

    report = {
        "DREAM_BASELINE": {
            "first_access_latency_ns": dream.latency.first_access_latency_ns,
            "latency_decomposition_ns": {
                "trp": dream.latency.trp_ns,
                "trcd": dream.latency.trcd_ns,
                "tcl": dream.latency.tcl_ns,
            },
            "repeated_service_cycle_ns": (
                dream.latency.repeated_service_cycle_ns),
            "dq_atom_window_ns": dream.latency.dq_atom_window_ns,
            "internal_bandwidth_TBps": _tbps(
                dream.internal_bandwidth_bytes_per_s),
            "internal_binding_stages": list(dream.internal_binding_stages),
            "interface_num_links": dream.interface_num_links,
            "interface_rate_gbps_per_link": (
                dream.interface_rate_gbps_per_link),
            "interface_bandwidth_TBps": _tbps(
                dream.interface_bandwidth_bytes_per_s),
            "effective_bandwidth_TBps": _tbps(
                dream.effective_bandwidth_bytes_per_s),
            "bottleneck": dream.bottleneck,
            "internal_over_interface_ratio": (
                dream.ratio_internal_over_interface),
            "latency_gate": dream.latency_gate,
            "internal_bandwidth_gate": dream.internal_bandwidth_gate,
            "interface_bandwidth_gate": dream.interface_bandwidth_gate,
            "provenance": dream.provenance,
        },
        "ORTHOGONAL_M3D": {
            "first_access_latency_ns_range": [m3d_first_min, m3d_first_max],
            "average_service_cycle_ns": closure.average_service_cycle_ns,
            "fast_service_cycle_ns": closure.fast_service_cycle_ns,
            "internal_bandwidth_average_TBps": _tbps(m3d_internal_avg),
            "internal_bandwidth_fast_TBps": _tbps(m3d_internal_fast),
            "coil_bandwidth_TBps": _tbps(m3d_coil),
            "coil_derivation": {
                "num_m3d_dies": closure.num_m3d_dies,
                "coil_links_per_die": closure.coil_links_per_die,
                "coil_data_rate_gbps_per_link": (
                    closure.coil_data_rate_gbps_per_link),
            },
            "effective_bandwidth_TBps": _tbps(m3d_effective),
            "bottleneck": m3d_bottleneck,
            "internal_over_interface_ratio_average": m3d_r_avg,
            "internal_over_interface_ratio_fast": m3d_r_fast,
        },
        "BASELINE_BOTTLENECK_COMPARISON_GATE": gate,
        "answers": _answers(dream, closure, m3d_bottleneck, gate,
                            m3d_first_min, m3d_first_max),
        "dream_audit": dream.as_dict(),
        "m3d_closure": closure.as_dict(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    _print_text_report(dream, closure, m3d_bottleneck, gate,
                       m3d_first_min, m3d_first_max)
    return 0


def _answers(dream, closure, m3d_bottleneck, gate, m3d_first_min,
             m3d_first_max):
    dream_lat = dream.latency
    m3d_coil = closure.coil_bandwidth_bytes_per_s
    m3d_internal_avg = closure.internal_bandwidth_average_bytes_per_s
    m3d_effective = min(m3d_internal_avg, m3d_coil)
    dream_interface_bound = dream.bottleneck == "EXTERNAL_INTERFACE"
    q6 = dream.internal_bandwidth_bytes_per_s > (
        dream.interface_bandwidth_bytes_per_s)
    q7 = (
        "YES"
        if (q6 and dream_interface_bound
            and m3d_bottleneck == "COIL_INTERFACE")
        else "NO")
    return {
        "Q1_first_access_latency_ns": dream_lat.first_access_latency_ns,
        "Q2_repeated_service_cycle_ns": dream_lat.repeated_service_cycle_ns,
        "Q3_internal_bandwidth_TBps": _tbps(
            dream.internal_bandwidth_bytes_per_s),
        "Q4_interface_bandwidth_TBps": _tbps(
            dream.interface_bandwidth_bytes_per_s),
        "Q5_dream_bottleneck": dream.bottleneck,
        "Q6_dream_internal_exceeds_interface": q6,
        "Q7_m3d_coil_bottleneck_is_common_memory_rich_phenomenon": q7,
        "Q8_m3d_vs_dream": {
            "first_access_latency_speedup_range": [
                dream_lat.first_access_latency_ns / m3d_first_max,
                dream_lat.first_access_latency_ns / m3d_first_min,
            ],
            "internal_bandwidth_improvement": (
                m3d_internal_avg / dream.internal_bandwidth_bytes_per_s),
            "external_bandwidth_difference_TBps": (
                _tbps(m3d_coil)
                - _tbps(dream.interface_bandwidth_bytes_per_s)),
            "external_bandwidth_ratio": (
                m3d_coil / dream.interface_bandwidth_bytes_per_s),
            "effective_bandwidth_difference_TBps": (
                _tbps(m3d_effective)
                - _tbps(dream.effective_bandwidth_bytes_per_s)),
            "effective_bandwidth_ratio": (
                m3d_effective / dream.effective_bandwidth_bytes_per_s),
        },
    }


def _print_text_report(dream, closure, m3d_bottleneck, gate,
                       m3d_first_min, m3d_first_max) -> None:
    m3d_coil = closure.coil_bandwidth_bytes_per_s
    m3d_internal_avg = closure.internal_bandwidth_average_bytes_per_s
    lines = [
        "",
        "================ DREAM_BASELINE ================",
        f"first-access latency   = "
        f"{dream.latency.first_access_latency_ns:.3f} ns "
        f"(tRP {dream.latency.trp_ns:.3f} + tRCD {dream.latency.trcd_ns:.3f}"
        f" + tCL {dream.latency.tcl_ns:.3f})",
        f"service cycle          = "
        f"{dream.latency.repeated_service_cycle_ns:.4f} ns "
        f"(DQ atom window {dream.latency.dq_atom_window_ns:.4f} ns)",
        f"internal bandwidth     = "
        f"{_tbps(dream.internal_bandwidth_bytes_per_s):.4f} TB/s "
        f"(binding: {', '.join(dream.internal_binding_stages)})",
        f"interface bandwidth    = "
        f"{_tbps(dream.interface_bandwidth_bytes_per_s):.4f} TB/s "
        f"({dream.interface_num_links} DQ x "
        f"{dream.interface_rate_gbps_per_link:.3f} Gbps / "
        f"{dream.interface_payload_ecc_factor:.4f} ECC)",
        f"effective bandwidth    = "
        f"{_tbps(dream.effective_bandwidth_bytes_per_s):.4f} TB/s",
        f"bottleneck             = {dream.bottleneck}",
        f"internal/interface     = "
        f"{dream.ratio_internal_over_interface:.6f}x",
        "",
        "================ ORTHOGONAL_M3D ================",
        f"first-access latency   = {m3d_first_min:.2f} .. "
        f"{m3d_first_max:.2f} ns (spatial)",
        f"service cycle          = "
        f"{closure.average_service_cycle_ns:.3f} ns avg / "
        f"{closure.fast_service_cycle_ns:.3f} ns fast",
        f"internal bandwidth     = {_tbps(m3d_internal_avg):.2f} TB/s avg / "
        f"{_tbps(closure.internal_bandwidth_fast_bytes_per_s):.2f} TB/s fast",
        f"coil bandwidth         = {_tbps(m3d_coil):.2f} TB/s "
        f"({closure.num_m3d_dies} dies x {closure.coil_links_per_die} links"
        f" x {closure.coil_data_rate_gbps_per_link:.1f} Gbps)",
        f"effective bandwidth    = "
        f"{_tbps(min(m3d_internal_avg, m3d_coil)):.2f} TB/s",
        f"bottleneck             = {m3d_bottleneck}",
        f"internal/interface     = {m3d_internal_avg / m3d_coil:.2f}x avg / "
        f"{closure.internal_bandwidth_fast_bytes_per_s / m3d_coil:.2f}x fast",
        "",
        f"BASELINE_BOTTLENECK_COMPARISON_GATE = {gate}",
        "",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
