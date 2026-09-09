"""Audit the canonical 100 um orthogonal M3D slab revision.

This is an architecture-only audit.  It constructs canonical geometry and
analytical capacity/resource closures; it never invokes a thermal solver or a
serving/workload benchmark.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

from om3dthermal.platform import (
    load_platform_spec_file,
    resolve_gpu_bandwidth_service,
    resolve_local_memory_gpu_transfer,
)
from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    resolve_case_geometry,
    resolve_effective_bandwidth,
)
from om3dthermal.power.config import CanonicalCaseConfig
from om3dthermal.power.nmp_die_activity import canonical_nmp_hardware
from om3dthermal.thermal.nmp_die_mapping import physical_nmp_die_regions


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"
PLATFORM_PATH = ROOT / "configs/platform/gpu_package_h200_reference.yaml"
DEFAULT_OUTPUT = ROOT / "runs/m3d_100um_slab_capacity_audit"
LEGACY = {"slab_count": 106, "slab_pitch_x_um": 300.0,
          "si_substrate_um": 292.546}


def _legacy_case() -> CanonicalCaseConfig:
    raw = yaml.safe_load(CASE_PATH.read_text(encoding="utf-8"))
    raw["geometry"]["orthogonal"].update({
        "slab_count": LEGACY["slab_count"],
        "slab_pitch_x_um": LEGACY["slab_pitch_x_um"],
    })
    raw["geometry"]["m3d_stack"]["si_substrate_um"] = (
        LEGACY["si_substrate_um"])
    return CanonicalCaseConfig.model_validate(raw)


def _resolve(case: CanonicalCaseConfig) -> dict[str, object]:
    geometry = resolve_case_geometry(case)
    power = calculate_memory_power(
        case,
        read_bandwidth_gbps=case.workload.read_bandwidth_gbps,
        project_root=ROOT,
        geometry=geometry,
    )
    layout = power.physical_capacity_layout
    closure = power.architecture_bandwidth_closure
    assert layout is not None and closure is not None
    raw = resolve_effective_bandwidth(
        closure,
        closure.average_service_cycle_ns / closure.service_cycle_scale,
    )
    platform = load_platform_spec_file(PLATFORM_PATH)
    gpu = platform.gpu_decode_power
    service = platform.gpu_bandwidth_service
    assert gpu is not None and service is not None
    transfer = resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=1.0e30,
        memory_capability_bytes_per_s=raw.effective_bandwidth_bytes_per_s,
        gpu_peak_bandwidth_bytes_per_s=(
            gpu.peak_memory_bandwidth_bytes_per_s),
    )
    sustained = resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=transfer.bandwidth_actual_bytes_per_s,
        gpu_bandwidth_utilization=service.nominal_utilization,
        utilization_status=service.utilization_status,
        utilization_provenance=service.provenance,
    )
    nmp = canonical_nmp_hardware(layout.slab_count)
    thermal_regions = physical_nmp_die_regions(case)
    orth = case.geometry.orthogonal
    stack = case.geometry.m3d_stack
    assert orth is not None and stack is not None
    return {
        "case": case,
        "orthogonal": orth,
        "stack": stack,
        "geometry": geometry,
        "power": power,
        "layout": layout,
        "closure": closure,
        "raw": raw,
        "transfer": transfer,
        "sustained": sustained,
        "nmp": nmp,
        "thermal_regions": thermal_regions,
    }


def _ratio(old: float, new: float) -> float:
    return new / old


def build_audit() -> dict[str, object]:
    old = _resolve(_legacy_case())
    new = _resolve(load_case_config(CASE_PATH))
    oo, no = old["orthogonal"], new["orthogonal"]
    os, ns = old["stack"], new["stack"]
    ol, nl = old["layout"], new["layout"]
    ob, nb = old["closure"], new["closure"]
    on, nn = old["nmp"], new["nmp"]
    op, np = old["power"], new["power"]
    ot, nt = old["transfer"], new["transfer"]
    og, ng = old["sustained"], new["sustained"]

    old_non_si = (os.feol_um + os.bitcell_layers
                  * os.bitcell_layer_pitch_nm * 1e-3
                  + os.beol_interconnect_um + os.daa_um)
    new_non_si = (ns.feol_um + ns.bitcell_layers
                  * ns.bitcell_layer_pitch_nm * 1e-3
                  + ns.beol_interconnect_um + ns.daa_um)
    old_total = os.si_substrate_um + old_non_si
    new_total = ns.si_substrate_um + new_non_si

    capacity = {
        "old_capacity_bytes": ol.total_capacity_bytes,
        "new_capacity_bytes": nl.total_capacity_bytes,
        "old_capacity_GB_decimal": ol.total_capacity_bytes / 1e9,
        "new_capacity_GB_decimal": nl.total_capacity_bytes / 1e9,
        "old_capacity_GiB": ol.total_capacity_gib,
        "new_capacity_GiB": nl.total_capacity_gib,
        "old_capacity_TB_decimal": ol.total_capacity_bytes / 1e12,
        "new_capacity_TB_decimal": nl.total_capacity_bytes / 1e12,
        "capacity_scaling": _ratio(
            ol.total_capacity_bytes, nl.total_capacity_bytes),
    }

    def row(quantity: str, old_value: object, new_value: object,
            scales: str, reason: str, status: str, location: str) -> dict[str, object]:
        ratio = ""
        if isinstance(old_value, (int, float)) and old_value != 0:
            ratio = float(new_value) / float(old_value)
        return {
            "quantity": quantity,
            "old_value": old_value,
            "new_value": new_value,
            "new_over_old": ratio,
            "scales_with_slab_count": scales,
            "physical_reason": reason,
            "status": status,
            "code_location": location,
        }

    resources = [
        row("physical_capacity_bytes", ol.total_capacity_bytes,
            nl.total_capacity_bytes, "YES",
            "identical per-slab organization instantiated once per physical slab",
            "PHYSICAL_CAPACITY_MODEL_CLOSED",
            "src/om3dthermal/power/physical_capacity.py"),
        row("memory_region_count", old["geometry"].memory_region_count,
            new["geometry"].memory_region_count, "YES",
            "orthogonal slab_count is the canonical physical memory-region count",
            "CANONICAL_GEOMETRY_DERIVED",
            "src/om3dthermal/power/geometry.py"),
        row("links_per_slab", ob.links_per_slab, nb.links_per_slab, "NO",
            "per-slab contactless primitive is frozen",
            "PER_SLAB_RESOURCE_UNCHANGED",
            "configs/cases/orthogonal_m3d_igzo.yaml"),
        row("aggregate_contactless_links",
            ob.slab_count * ob.links_per_slab,
            nb.slab_count * nb.links_per_slab, "YES",
            "architecture assigns the same contactless-link set to every slab",
            "AGGREGATE_INTERFACE_SCALING_CONDITIONAL",
            "src/om3dthermal/power/memory_bandwidth.py"),
        row("raw_contactless_capability_bytes_per_s",
            ob.coil_bandwidth_bytes_per_s, nb.coil_bandwidth_bytes_per_s, "YES",
            "slab_count times links_per_slab times fixed link rate",
            "CONDITIONAL_UPPER_BOUND_GPU_RECEIVER_NOT_PROVEN",
            "src/om3dthermal/power/memory_bandwidth.py"),
        row("effective_transfer_ceiling_bytes_per_s",
            ot.bandwidth_actual_bytes_per_s, nt.bandwidth_actual_bytes_per_s, "NO",
            "H200 GPU peak interface ceiling is 4.8 TB/s",
            "GPU_LIMITED_MODELED_TRANSFER_CEILING",
            "src/om3dthermal/platform/transfer.py"),
        row("nominal_sustained_external_bandwidth_bytes_per_s",
            og.sustained_bandwidth_bytes_per_s,
            ng.sustained_bandwidth_bytes_per_s, "NO",
            "0.5 nominal GPU service utilization applies after the 4.8 TB/s ceiling",
            "MODELING_CHOICE_GPU_LIMITED",
            "src/om3dthermal/platform/transfer.py"),
        row("internal_parallel_service_units",
            ob.total_parallel_service_units, nb.total_parallel_service_units, "YES",
            "one unchanged FEOL IO service-lane set per physical slab",
            "ARCHITECTURE_MODEL_DERIVED_CONDITIONAL",
            "src/om3dthermal/power/memory_bandwidth.py"),
        row("nmp_macs_per_slab", on.macs_per_die, nn.macs_per_die, "NO",
            "canonical NMP primitive remains 512 MAC per physical slab/die",
            "PER_SLAB_RESOURCE_UNCHANGED",
            "src/om3dthermal/power/nmp_die_activity.py"),
        row("nmp_physical_die_count", on.physical_die_count,
            nn.physical_die_count, "YES",
            "current architecture defines one FEOL NMP resource set per slab",
            "ARCHITECTURE_MODEL_DERIVED",
            "src/om3dthermal/power/nmp_die_activity.py"),
        row("total_nmp_macs", on.macs_per_die * on.physical_die_count,
            nn.macs_per_die * nn.physical_die_count, "YES",
            "fixed MACs per slab multiplied by physical slab count",
            "CONDITIONAL_NOT_PHYSICALLY_SYNTHESIZED",
            "src/om3dthermal/power/nmp_die_activity.py"),
        row("aggregate_nmp_peak_flops_per_s", on.aggregate_peak_flops,
            nn.aggregate_peak_flops, "YES",
            "fixed 512 MAC/slab, 1 GHz, and 2 FLOP/MAC",
            "CONDITIONAL_NOT_PHYSICALLY_SYNTHESIZED",
            "src/om3dthermal/power/nmp_die_activity.py"),
        row("thermal_source_geometry_count", len(old["thermal_regions"]),
            len(new["thermal_regions"]), "YES",
            "thermal source geometry maps one memory/FEOL carrier pair per slab",
            "GEOMETRY_BUILD_ONLY_THERMAL_NOT_RUN",
            "src/om3dthermal/thermal/nmp_die_mapping.py"),
        row("refresh_power_W", op.P_refresh_W, np.P_refresh_W, "YES",
            "existing refresh model scales with total stored bits",
            "EXISTING_MODEL_NATURAL_PROPAGATION",
            "src/om3dthermal/power/refresh.py"),
        row("total_memory_power_W", "UNRESOLVED", "UNRESOLVED", "PARTIAL",
            "fixed requested dynamic traffic is unchanged and refresh grows, but logic background remains unresolved",
            "UNRESOLVED_LOGIC_BACKGROUND",
            "src/om3dthermal/power/model.py"),
        row("read_energy_pj_per_bit", op.E_access_total_pj_bit,
            np.E_access_total_pj_bit, "NO",
            "slab thinning does not alter any per-access electrical primitive",
            "PER_OPERATION_PRIMITIVE_UNCHANGED",
            "src/om3dthermal/power/model.py"),
        row("interface_energy_pj_per_bit", op.E_interface_pj_bit,
            np.E_interface_pj_bit, "NO",
            "contactless per-bit primitive is independent of slab count",
            "PER_OPERATION_PRIMITIVE_UNCHANGED",
            "configs/cases/orthogonal_m3d_igzo.yaml"),
        row("physical_access_latency_max_ns",
            max(item.physical_access_latency_ns for item in ol.slot_classes),
            max(item.physical_access_latency_ns for item in nl.slot_classes), "NO",
            "substrate thickness is not in the current MAT/MIV/FEOL electrical path",
            "PRIMITIVE_UNCHANGED_NO_SILENT_THINNING_BENEFIT",
            "src/om3dthermal/power/physical_latency.py"),
        row("page_allocation_slot_count", ol.physical_slot_count,
            nl.physical_slot_count, "YES",
            "page slots are expanded across every physical slab",
            "CAPACITY_ALLOCATION_DERIVED",
            "src/om3dthermal/power/physical_capacity.py"),
        row("nmp_placement_die_domain_count", ol.slab_count,
            nl.slab_count, "YES",
            "placement and routing consume layout.slab_count as die domains",
            "ARCHITECTURE_MODEL_DERIVED",
            "src/om3dthermal/placement/nmp_load_balance.py"),
    ]

    checks = {
        "slab_thickness_closes_to_100um": abs(new_total - 100.0) < 1e-12,
        "non_si_stack_unchanged": abs(old_non_si - new_non_si) < 1e-12,
        "orthogonal_width_closes_to_31_8mm": abs(
            no.slab_count * no.slab_pitch_x_um / 1000.0 - 31.8) < 1e-12,
        "bitcell_layers_remain_8": ns.bitcell_layers == os.bitcell_layers == 8,
        "capacity_scales_3x": abs(capacity["capacity_scaling"] - 3.0) < 1e-12,
        "per_slab_interface_primitives_unchanged": (
            ob.links_per_slab == nb.links_per_slab == 50
            and ob.rate_gbps_per_link == nb.rate_gbps_per_link == 8.0),
        "per_slab_nmp_primitives_unchanged": (
            on.macs_per_die == nn.macs_per_die == 512
            and on.clock_hz == nn.clock_hz == 1.0e9),
        "latency_primitive_unchanged": (
            max(item.physical_access_latency_ns for item in ol.slot_classes)
            == max(item.physical_access_latency_ns for item in nl.slot_classes)),
        "geometry_builds_318_regions": len(new["thermal_regions"]) == 318,
        "thermal_simulation_run_is_no": True,
    }
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_revision": "REV_V3_100UM_THINNED_SLAB",
        "thermal_simulation_run": "NO",
        "geometry": {
            "old": {
                "total_slab_thickness_um": old_total,
                "si_substrate_um": os.si_substrate_um,
                "slab_count": oo.slab_count,
                "bitcell_layers": os.bitcell_layers,
                "orthogonal_width_mm": oo.slab_count * oo.slab_pitch_x_um / 1000.0,
            },
            "new": {
                "total_slab_thickness_um": new_total,
                "si_substrate_um": ns.si_substrate_um,
                "slab_count": no.slab_count,
                "bitcell_layers": ns.bitcell_layers,
                "orthogonal_width_mm": no.slab_count * no.slab_pitch_x_um / 1000.0,
            },
            "non_si_thickness_um": new_non_si,
        },
        "capacity": capacity,
        "resources": resources,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def write_audit(audit: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    capacity = audit["capacity"]
    with (output / "capacity_audit.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=capacity.keys())
        writer.writeheader()
        writer.writerow(capacity)
    resources = audit["resources"]
    with (output / "resource_scaling_audit.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=resources[0].keys())
        writer.writeheader()
        writer.writerows(resources)
    (output / "geometry_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    geometry = audit["geometry"]
    lines = [
        "# M3D 100 um slab capacity/resource audit",
        "",
        f"- Commit: `{audit['commit']}`",
        f"- Architecture: `{audit['architecture_revision']}`",
        f"- All checks pass: `{audit['all_checks_pass']}`",
        f"- Thermal simulation run: `{audit['thermal_simulation_run']}`",
        "",
        "## Geometry",
        "",
        "| State | Slab thickness (um) | Si (um) | Slabs | Layers | Width (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for state in ("old", "new"):
        item = geometry[state]
        lines.append(
            f"| {state} | {item['total_slab_thickness_um']:.3f} | "
            f"{item['si_substrate_um']:.3f} | {item['slab_count']} | "
            f"{item['bitcell_layers']} | {item['orthogonal_width_mm']:.3f} |")
    lines += [
        "",
        "## Capacity",
        "",
        f"- Old: {capacity['old_capacity_GB_decimal']:.6f} GB",
        f"- New: {capacity['new_capacity_GB_decimal']:.6f} GB "
        f"({capacity['new_capacity_TB_decimal']:.9f} TB)",
        f"- Scaling: {capacity['capacity_scaling']:.6f}x",
        "",
        "## Resource-scaling status",
        "",
        "Raw contactless and aggregate NMP capabilities scale with the modeled "
        "physical slab count. They remain conditional architectural upper "
        "bounds. The H200 transfer backend caps memory-to-GPU transfer at "
        "4.8 TB/s and applies the existing 0.5 nominal service utilization.",
    ]
    (output / "geometry_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = build_audit()
    write_audit(audit, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "all_checks_pass": audit["all_checks_pass"]}))
    if not audit["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
