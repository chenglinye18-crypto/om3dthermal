"""Generate the derived E2E bandwidth/energy research ledger.

This module is presentation-only.  Runtime authority remains the canonical
platform/case YAML documents and their resolvers; the generated CSV is never
an evaluation input.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
import math
from pathlib import Path

from om3dthermal.platform.models import PlatformSpec, load_platform_spec_file
from om3dthermal.platform.host_offload_power import resolve_host_offload_power
from om3dthermal.platform.transfer import (
    resolve_gpu_bandwidth_service,
    resolve_local_memory_gpu_transfer,
)
from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    resolve_case_geometry,
    resolve_effective_bandwidth,
)
from om3dthermal.power.result import MemoryPowerResult


LEDGER_FILENAME = "e2e_bandwidth_energy_ledger_2026-09-07.csv"


@dataclass(frozen=True)
class LedgerRow:
    component_id: str
    system_layer: str
    physical_boundary: str
    bandwidth_nominal_GBps: float | None = None
    bandwidth_min_GBps: float | None = None
    bandwidth_max_GBps: float | None = None
    bandwidth_demand_GBps: float | None = None
    bandwidth_efficiency: float | None = None
    bandwidth_semantics: str = ""
    bandwidth_status: str = ""
    bandwidth_provenance: str = ""
    energy_nominal_pJ_per_bit: float | None = None
    energy_min_pJ_per_bit: float | None = None
    energy_max_pJ_per_bit: float | None = None
    energy_semantics: str = ""
    energy_status: str = ""
    energy_provenance: str = ""
    included_in_system_bandwidth_min: bool = False
    included_in_system_energy: bool = False
    parent_aggregate: str = ""
    double_counting_note: str = ""
    runtime_source: str = ""
    source_reference: str = ""
    notes: str = ""


def _record(spec: object, record_id: str):
    for item in getattr(spec, "provenance"):
        if item.record_id == record_id:
            return item
    raise ValueError(f"missing provenance record {record_id!r}")


def _gpu_reference_range(project_root: Path, nominal: float) -> tuple[float, float, str]:
    """Read sensitivity endpoints from the existing provenance-only ledger."""
    path = project_root / "docs/research/gpu_platform_table_2026-09-06.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [
        row for row in rows
        if row["category"] == "gpu_energy" and row["name"] == "H200 SXM"
    ]
    if len(matches) != 1:
        raise ValueError("GPU provenance ledger must contain one H200 SXM row")
    row = matches[0]
    minimum = float(row["e_decode_dynamic_pJ_per_bit_min"])
    ledger_nominal = float(row["e_decode_dynamic_pJ_per_bit_nominal"])
    maximum = float(row["e_decode_dynamic_pJ_per_bit_max"])
    if not math.isclose(ledger_nominal, nominal, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("GPU provenance nominal disagrees with canonical platform YAML")
    return (
        minimum,
        maximum,
        f"docs/research/gpu_platform_table_2026-09-06.csv; {row['source']}",
    )


def _resolve_memory(project_root: Path, relative_path: str) -> tuple[object, MemoryPowerResult]:
    path = project_root / relative_path
    case = load_case_config(path)
    result = calculate_memory_power(
        case,
        project_root=project_root,
        read_bandwidth_gbps=case.workload.read_bandwidth_gbps,
        geometry=resolve_case_geometry(case),
    )
    return case, result


def _validate_rows(rows: tuple[LedgerRow, ...]) -> None:
    by_id = {row.component_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("ledger component_id values must be unique")

    m3d_components = (
        "M3D_MAT_LOCAL_READ",
        "M3D_GLOBAL_ROUTING",
        "M3D_MIV",
        "M3D_FEOL_ROUTE",
        "M3D_CONTACTLESS_INTERFACE",
    )
    component_sum = sum(
        float(by_id[name].energy_nominal_pJ_per_bit) for name in m3d_components
    )
    if not math.isclose(
        component_sum,
        float(by_id["M3D_TOTAL_READ"].energy_nominal_pJ_per_bit),
        rel_tol=1e-13,
        abs_tol=1e-15,
    ):
        raise ValueError("M3D component energy does not close to total read energy")
    if not math.isclose(
        float(by_id["M3D_TOTAL_READ"].bandwidth_nominal_GBps),
        min(
            float(by_id["M3D_INTERNAL"].bandwidth_nominal_GBps),
            float(by_id["M3D_CONTACTLESS_INTERFACE"].bandwidth_nominal_GBps),
        ),
        rel_tol=1e-13,
    ):
        raise ValueError("M3D raw bandwidth does not close")
    if not math.isclose(
        float(by_id["M3D_GPU_SHARED_TRANSFER"].bandwidth_nominal_GBps),
        min(
            float(by_id["M3D_GPU_SHARED_TRANSFER"].bandwidth_demand_GBps),
            float(by_id["M3D_TOTAL_READ"].bandwidth_nominal_GBps),
            float(by_id["GPU_MEMORY_INTERFACE"].bandwidth_nominal_GBps),
        ),
        rel_tol=1e-13,
    ):
        raise ValueError("shared M3D-to-GPU transfer bandwidth does not close")
    if not math.isclose(
        float(by_id["GPU_SUSTAINED_BANDWIDTH_SERVICE"].bandwidth_nominal_GBps),
        float(by_id["GPU_SUSTAINED_BANDWIDTH_SERVICE"].bandwidth_max_GBps)
        * float(by_id["GPU_SUSTAINED_BANDWIDTH_SERVICE"].bandwidth_efficiency),
        rel_tol=1e-13,
    ):
        raise ValueError("GPU sustained service bandwidth does not close")
    if float(by_id["HOST_OFFLOAD_EFFECTIVE"].bandwidth_nominal_GBps) > min(
        float(by_id["HOST_MEMORY_SUBSYSTEM"].bandwidth_nominal_GBps),
        float(by_id["HOST_COHERENT_LINK"].bandwidth_nominal_GBps),
    ):
        raise ValueError("GH200 effective bandwidth exceeds a capability bound")
    if not math.isclose(
        float(by_id["HOST_OFFLOAD_DYNAMIC_PATH"].energy_nominal_pJ_per_bit),
        float(by_id["HOST_MEMORY_SUBSYSTEM"].energy_nominal_pJ_per_bit)
        + float(by_id["HOST_COHERENT_LINK"].energy_nominal_pJ_per_bit),
        rel_tol=1e-13,
    ):
        raise ValueError("host dynamic aggregate energy does not close")


def build_ledger_rows(project_root: Path) -> tuple[LedgerRow, ...]:
    """Resolve every ledger value from canonical sources."""
    root = project_root.resolve()
    platform_path = root / "configs/platform/gpu_package_h200_reference.yaml"
    platform: PlatformSpec = load_platform_spec_file(platform_path)
    if platform.host_offload is None or platform.gpu_decode_power is None:
        raise ValueError("canonical platform must resolve host and GPU decode inputs")
    host = platform.host_offload
    gpu = platform.gpu_decode_power
    if (
        host.effective_bandwidth_bytes_per_second is None
        or host.e_host_offload_dynamic_J_per_bit is None
        or host.memory_bandwidth_upper_bound_bytes_per_second is None
        or host.link_bandwidth_upper_bound_bytes_per_second is None
        or host.host_link_dynamic_J_per_bit is None
        or host.host_memory_dynamic_J_per_bit is None
    ):
        raise ValueError("canonical host offload parameters are unresolved")
    host_point = resolve_host_offload_power(
        host_transfer_demand_bytes_per_second=(
            host.effective_bandwidth_bytes_per_second),
        host_effective_bandwidth_bytes_per_second=(
            host.effective_bandwidth_bytes_per_second),
        e_pcie_dynamic_J_per_bit=host.host_link_dynamic_J_per_bit,
        e_ddr_dynamic_J_per_bit=host.host_memory_dynamic_J_per_bit,
    )
    host_actual_bit_rate = 8.0 * host_point.host_bandwidth_actual_bytes_per_second
    resolved_pcie_pJ_per_bit = (
        host_point.pcie_dynamic_power_W / host_actual_bit_rate * 1e12)
    resolved_ddr_pJ_per_bit = (
        host_point.ddr_dynamic_power_W / host_actual_bit_rate * 1e12)
    resolved_host_pJ_per_bit = (
        host_point.host_offload_dynamic_power_W / host_actual_bit_rate * 1e12)

    conventional_case, conventional = _resolve_memory(
        root, "configs/cases/conventional_hbm_2x1.yaml")
    m3d_case, m3d = _resolve_memory(
        root, "configs/cases/orthogonal_m3d_igzo.yaml")
    closure = m3d.architecture_bandwidth_closure
    if closure is None:
        raise ValueError("canonical M3D architecture bandwidth is unresolved")
    m3d_raw = resolve_effective_bandwidth(
        closure, closure.average_service_cycle_ns)
    transfer = resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=(
            m3d_case.workload.read_bandwidth_gbps * 1e9 / 8.0),
        memory_capability_bytes_per_s=m3d_raw.effective_bandwidth_bytes_per_s,
        gpu_peak_bandwidth_bytes_per_s=gpu.peak_memory_bandwidth_bytes_per_s,
    )
    bandwidth_service = resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=transfer.bandwidth_actual_bytes_per_s,
        gpu_bandwidth_utilization=(
            platform.gpu_bandwidth_service.nominal_utilization),
        utilization_status=(
            platform.gpu_bandwidth_service.utilization_status),
        utilization_provenance=platform.gpu_bandwidth_service.provenance,
    )

    host_memory = _record(host, "grace_lpddr5x_subsystem_energy_v0")
    host_link = _record(host, "nvlink_c2c_energy_per_bit_v0")
    host_link_bw = _record(host, "nvlink_c2c_sensitivity_v0")
    host_eff = _record(host, "gh200_grace_hopper_h2d_416_34_gbps_v0")
    host_total = _record(host, "gh200_host_path_dynamic_energy_v0")
    gpu_min, gpu_max, gpu_range_source = _gpu_reference_range(
        root, gpu.e_decode_J_per_bit * 1e12)
    replacements = m3d.diagnostics["replacement_components_pj_bit"]
    mat = float(replacements["zhu_scaled_local_operation"])
    global_route = float(replacements["tang_global_control_routing"])
    platform_runtime = (
        "configs/platform/gpu_package_h200_reference.yaml -> "
        "load_platform_spec_file"
    )
    m3d_runtime = (
        "configs/cases/orthogonal_m3d_igzo.yaml -> calculate_memory_power / "
        "derive_architecture_bandwidth"
    )
    component_note = (
        "Already included in M3D_TOTAL_READ; do not add when the aggregate "
        "read-access coefficient is used."
    )
    rows = (
        LedgerRow(
            "HOST_MEMORY_SUBSYSTEM", "host", "Grace LPDDR5X memory subsystem",
            host.memory_bandwidth_upper_bound_bytes_per_second/1e9,
            bandwidth_semantics="Grace memory subsystem capability upper bound",
            bandwidth_status=host_memory.status,
            bandwidth_provenance=host_memory.classification,
            energy_nominal_pJ_per_bit=resolved_ddr_pJ_per_bit,
            energy_semantics="first-order average Grace memory-subsystem energy per transferred bit",
            energy_status=host_memory.status,
            energy_provenance=host_memory.classification,
            included_in_system_bandwidth_min=True,
            parent_aggregate="HOST_OFFLOAD_DYNAMIC_PATH",
            double_counting_note="Use this component only when not using HOST_OFFLOAD_DYNAMIC_PATH.",
            runtime_source=platform_runtime,
            source_reference=host_memory.source,
            notes="Derived from 16 W / (500 GB/s x 8); not isolated LPDDR dynamic-read energy.",
        ),
        LedgerRow(
            "HOST_COHERENT_LINK", "host", "one-direction NVLink-C2C link",
            host.link_bandwidth_upper_bound_bytes_per_second/1e9,
            bandwidth_semantics="ideal one-direction NVLink-C2C upper bound",
            bandwidth_status=host_link_bw.status,
            bandwidth_provenance=host_link_bw.classification,
            energy_nominal_pJ_per_bit=resolved_pcie_pJ_per_bit,
            energy_semantics="vendor-reported NVLink-C2C link energy",
            energy_status=host_link.status,
            energy_provenance=host_link.classification,
            included_in_system_bandwidth_min=True,
            parent_aggregate="HOST_OFFLOAD_DYNAMIC_PATH",
            double_counting_note="Use this component only when not using HOST_OFFLOAD_DYNAMIC_PATH.",
            runtime_source=platform_runtime,
            source_reference=host_link.source,
        ),
        LedgerRow(
            "HOST_OFFLOAD_EFFECTIVE", "host", "end-to-end host-to-device transport ceiling",
            host.effective_bandwidth_bytes_per_second / 1e9,
            bandwidth_semantics="direct measured Grace-memory-to-Hopper H2D effective bandwidth",
            bandwidth_status=host_eff.status,
            bandwidth_provenance=host_eff.classification,
            included_in_system_bandwidth_min=True,
            runtime_source=platform_runtime,
            source_reference=host_eff.source,
            notes="Rate aggregate only; its energy field is intentionally blank.",
        ),
        LedgerRow(
            "HOST_OFFLOAD_DYNAMIC_PATH", "host", "additive Grace memory plus NVLink-C2C dynamic path",
            energy_nominal_pJ_per_bit=resolved_host_pJ_per_bit,
            energy_semantics="additive dynamic aggregate = HOST_MEMORY_SUBSYSTEM + HOST_COHERENT_LINK",
            energy_status=host_total.status,
            energy_provenance=host_total.classification,
            included_in_system_energy=True,
            double_counting_note="When this aggregate is used, do not add host memory or coherent-link components again.",
            runtime_source=platform_runtime,
            source_reference=host_total.source,
        ),
        LedgerRow(
            "CONVENTIONAL_HBM_READ", "memory", "GPU-local HBM service / platform interface boundary",
            gpu.peak_memory_bandwidth_bytes_per_s / 1e9,
            bandwidth_semantics="GPU-local HBM service/platform interface boundary; no separate HBM capability is modeled",
            bandwidth_status=gpu.bandwidth_status,
            bandwidth_provenance="CANONICAL_GPU_PLATFORM_BOUNDARY",
            energy_nominal_pJ_per_bit=conventional.E_access_total_pj_bit,
            energy_semantics="complete canonical Conventional-HBM read access",
            energy_status="SOFTWARE_DERIVED_FROM_CANONICAL_RESOLVER",
            energy_provenance=str(conventional_case.provenance["access_energy_organization"]),
            included_in_system_bandwidth_min=True,
            included_in_system_energy=True,
            runtime_source="configs/cases/conventional_hbm_2x1.yaml -> calculate_memory_power; " + platform_runtime,
            source_reference="DreamRAM HBM3 analytical reference plus canonical H200 platform boundary",
            notes="Resolved decomposition: memory_internal + vertical + base_route + interface.",
        ),
        LedgerRow(
            "M3D_INTERNAL", "memory", "internal FEOL-aligned M3D service capability",
            closure.internal_bandwidth_average_bytes_per_s / 1e9,
            closure.internal_bandwidth_slow_bytes_per_s / 1e9,
            closure.internal_bandwidth_fast_bytes_per_s / 1e9,
            bandwidth_semantics="slow / spatial-average / fast internal service capability",
            bandwidth_status=closure.internal_classification,
            bandwidth_provenance=closure.physical_latency_status,
            included_in_system_bandwidth_min=True,
            parent_aggregate="M3D_TOTAL_READ",
            runtime_source=m3d_runtime,
            source_reference=closure.access_topology_provenance,
            notes="Internal service capability only; no complete energy boundary is assigned.",
        ),
        LedgerRow(
            "M3D_MAT_LOCAL_READ", "memory", "M3D MAT-local read operation",
            energy_nominal_pJ_per_bit=mat,
            energy_semantics="scaled MAT-local read primitive",
            energy_status="SOFTWARE_DERIVED_FROM_CANONICAL_RESOLVER",
            energy_provenance=str(m3d.diagnostics["zhu_reference_energy_provenance"]),
            parent_aggregate="M3D_TOTAL_READ", double_counting_note=component_note,
            runtime_source=m3d_runtime,
            source_reference=str(m3d.diagnostics["operation_energy_provenance"]),
        ),
        LedgerRow(
            "M3D_GLOBAL_ROUTING", "memory", "M3D global control routing",
            energy_nominal_pJ_per_bit=global_route,
            energy_semantics="global word/write/control routing contribution",
            energy_status="SOFTWARE_DERIVED_FROM_CANONICAL_RESOLVER",
            energy_provenance="MODELING_CHOICE_ELECTRICAL_ROUTING",
            parent_aggregate="M3D_TOTAL_READ", double_counting_note=component_note,
            runtime_source=m3d_runtime,
            source_reference=closure.access_topology_provenance,
        ),
        LedgerRow(
            "M3D_MIV", "memory", "vertical MIV transport",
            energy_nominal_pJ_per_bit=m3d.E_vertical_pj_bit,
            energy_semantics="MIV contribution per delivered read bit",
            energy_status=str(m3d.diagnostics["miv_energy_status"]),
            energy_provenance=str(m3d.diagnostics["miv_vertical_capacitance_classification"]),
            parent_aggregate="M3D_TOTAL_READ", double_counting_note=component_note,
            runtime_source=m3d_runtime,
            source_reference=str(m3d.diagnostics["miv_vertical_capacitance_source"]),
        ),
        LedgerRow(
            "M3D_FEOL_ROUTE", "memory", "nearest-edge FEOL route",
            energy_nominal_pJ_per_bit=m3d.E_feol_route_pj_bit,
            energy_semantics="FEOL route contribution per delivered read bit",
            energy_status=str(m3d.diagnostics["feol_status"]),
            energy_provenance=str(m3d.diagnostics["feol_provenance"]),
            parent_aggregate="M3D_TOTAL_READ", double_counting_note=component_note,
            runtime_source=m3d_runtime,
            source_reference=str(m3d.diagnostics["feol_route_topology_provenance"]),
        ),
        LedgerRow(
            "M3D_CONTACTLESS_INTERFACE", "memory", "aggregate contactless / inductive-coupling interface",
            closure.coil_bandwidth_bytes_per_s / 1e9,
            bandwidth_semantics="slab count * links/slab * link rate",
            bandwidth_status=closure.coil_classification,
            bandwidth_provenance=closure.coil_parameter_classification,
            energy_nominal_pJ_per_bit=m3d.E_interface_pj_bit,
            energy_semantics="contactless TX/RX/coil link contribution",
            energy_status=str(m3d.diagnostics["interface_energy_status"]),
            energy_provenance="PAPER_REPORTED",
            included_in_system_bandwidth_min=True,
            parent_aggregate="M3D_TOTAL_READ", double_counting_note=component_note,
            runtime_source=m3d_runtime,
            source_reference=str(m3d.diagnostics["interface_source_boundary"]),
        ),
        LedgerRow(
            "M3D_TOTAL_READ", "memory", "complete nominal M3D read-access path",
            m3d_raw.effective_bandwidth_bytes_per_s / 1e9,
            bandwidth_semantics="raw M3D capability = min(spatial-average internal, contactless interface)",
            bandwidth_status=m3d_raw.bottleneck,
            bandwidth_provenance="CANONICAL_EFFECTIVE_BANDWIDTH_RESOLVER",
            energy_nominal_pJ_per_bit=m3d.E_access_total_pj_bit,
            energy_semantics="MAT + global routing + MIV + FEOL + contactless",
            energy_status="SOFTWARE_DERIVED_FROM_CANONICAL_RESOLVER",
            energy_provenance="COMPLETE_NOMINAL_ANALYTICAL_M3D_READ_ACCESS",
            included_in_system_bandwidth_min=True,
            included_in_system_energy=True,
            double_counting_note="Use this aggregate for normal system accounting; do not add its component rows.",
            runtime_source=m3d_runtime + " -> resolve_effective_bandwidth",
            source_reference="canonical orthogonal_m3d_igzo case and formal memory-power resolver",
        ),
        LedgerRow(
            "GPU_MEMORY_INTERFACE", "gpu", "H200 peak HBM3e memory interface",
            gpu.peak_memory_bandwidth_bytes_per_s / 1e9,
            bandwidth_semantics="vendor peak HBM3e bandwidth",
            bandwidth_status=gpu.bandwidth_status,
            bandwidth_provenance="MATCHED_REFERENCE",
            included_in_system_bandwidth_min=True,
            runtime_source=platform_runtime,
            source_reference=_record(gpu, "h200_sxm_platform_spec_v0").source,
            notes="Energy intentionally blank: the GPU decode coefficient is not interface energy.",
        ),
        LedgerRow(
            "GPU_SUSTAINED_BANDWIDTH_SERVICE", "gpu",
            "nominal GPU-side sustained service after transfer ceiling",
            bandwidth_service.sustained_bandwidth_bytes_per_s / 1e9,
            bandwidth_max_GBps=(
                bandwidth_service.transfer_ceiling_bytes_per_s / 1e9),
            bandwidth_efficiency=(
                bandwidth_service.gpu_bandwidth_utilization),
            bandwidth_semantics=(
                "sustained actual = eta_gpu_bandwidth * transfer ceiling"),
            bandwidth_status=bandwidth_service.utilization_status,
            bandwidth_provenance="MODELING_CHOICE",
            included_in_system_bandwidth_min=True,
            runtime_source=(
                platform_runtime + " -> resolve_gpu_bandwidth_service"),
            source_reference=(
                bandwidth_service.utilization_provenance[0].source),
            notes=(
                "Formal roofline performance, M3D dynamic read power, and GPU "
                "bandwidth-bound dynamic power consume this sustained rate."),
        ),
        LedgerRow(
            "GPU_DECODE_DYNAMIC", "gpu", "GPU-only bandwidth-bound decode dynamic accounting",
            bandwidth_service.sustained_bandwidth_bytes_per_s / 1e9,
            bandwidth_max_GBps=gpu.peak_memory_bandwidth_bytes_per_s / 1e9,
            bandwidth_efficiency=bandwidth_service.gpu_bandwidth_utilization,
            bandwidth_semantics="GPU dynamic power uses nominal sustained actual bandwidth",
            bandwidth_status=gpu.bandwidth_status,
            bandwidth_provenance="CANONICAL_GPU_PLATFORM_BOUNDARY",
            energy_nominal_pJ_per_bit=gpu.e_decode_J_per_bit * 1e12,
            energy_min_pJ_per_bit=gpu_min,
            energy_max_pJ_per_bit=gpu_max,
            energy_semantics="GPU decode dynamic energy normalized per actual/sustained transferred bit",
            energy_status=gpu.coefficient_nominal_status,
            energy_provenance=gpu.coefficient_range_status,
            included_in_system_bandwidth_min=True,
            included_in_system_energy=True,
            double_counting_note="Independent of memory read energy and may be added to M3D_TOTAL_READ; no memory-energy subtraction is applied.",
            runtime_source=platform_runtime,
            source_reference=gpu_range_source,
            notes="Range endpoints are provenance/sensitivity values; runtime nominal comes only from platform YAML and is normalized by eta_bw * peak bandwidth.",
        ),
        LedgerRow(
            "M3D_GPU_SHARED_TRANSFER", "system", "shared M3D-to-GPU transfer operating point",
            transfer.bandwidth_actual_bytes_per_s / 1e9,
            bandwidth_demand_GBps=transfer.bandwidth_demand_bytes_per_s / 1e9,
            bandwidth_semantics="actual = min(demand, raw M3D capability, GPU peak)",
            bandwidth_status=transfer.bottleneck,
            bandwidth_provenance="CANONICAL_LOCAL_MEMORY_GPU_TRANSFER_RESOLVER",
            included_in_system_bandwidth_min=True,
            runtime_source=m3d_runtime + "; " + platform_runtime + " -> resolve_local_memory_gpu_transfer",
            source_reference="canonical M3D workload demand, M3D bandwidth resolver, and H200 platform spec",
            notes="Physical transfer ceiling only. Formal dynamic power and performance consume GPU_SUSTAINED_BANDWIDTH_SERVICE.",
        ),
    )
    _validate_rows(rows)
    return rows


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def write_ledger(project_root: Path, output_path: Path | None = None) -> Path:
    """Generate the deterministic CSV and return its path."""
    root = project_root.resolve()
    target = output_path or root / "docs/research" / LEDGER_FILENAME
    rows = build_ledger_rows(root)
    fieldnames = [item.name for item in fields(LedgerRow)]
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: _format_value(getattr(row, name)) for name in fieldnames
            })
    return target


__all__ = ["LEDGER_FILENAME", "LedgerRow", "build_ledger_rows", "write_ledger"]


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    print(write_ledger(repository_root).relative_to(repository_root))
