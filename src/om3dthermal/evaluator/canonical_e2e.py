"""Pure integration of the frozen canonical NMP and thermal result paths."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CanonicalE2ECaseResult:
    requests: int
    context_tokens: int
    model: str
    total_physical_capacity_bytes: int
    logical_working_set_bytes: int
    allocated_working_set_bytes: int
    resident_fraction: float
    capacity_headroom_bytes: int
    capacity_feasible: bool
    total_weight_read_bytes: float
    total_kv_read_bytes: float
    total_kv_write_bytes: float
    local_nmp_bytes: float
    residual_external_bytes: float
    non_nmp_external_bytes: float
    external_traffic_reduction: float
    direct_die_to_die_bytes: float
    non_nmp_step_ms: float
    locality_only_step_ms: float
    balanced_step_ms: float
    ideal_step_ms: float
    non_nmp_tokens_per_s: float
    locality_only_tokens_per_s: float
    balanced_tokens_per_s: float
    ideal_tokens_per_s: float
    locality_only_gain: float
    balanced_gain: float
    ideal_gain: float
    placement_latency_ratio: float
    placement_incremental_speedup: float
    max_mean_service: float
    max_capacity_utilization: float
    max_power_die: int
    hottest_die: int
    aggregate_m3d_nmp_power_W: float
    read_W: float
    write_W: float
    mac_W: float
    refresh_W: float
    residual_external_W: float
    energy_per_decode_step_J: float
    energy_per_token_J: float
    m3d_Tmin_degC: float
    m3d_Tmean_degC: float
    m3d_Tmax_degC: float
    m3d_delta_T_degC: float
    global_Tmax_degC: float
    global_Tmax_region: str
    gamma_NMP: float
    capacity_gate: str
    traffic_gate: str
    performance_gate: str
    power_gate: str
    thermal_gate: str
    canonical_gate: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalE2EResult:
    cases: tuple[CanonicalE2ECaseResult, ...]
    model: str
    architecture_status: str
    full_system_energy_comparison_status: str
    residual_external_thermal_mapping_status: str
    gates: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def integrate_canonical_e2e(*, nmp_payload: dict[str, Any],
                            thermal_payload: dict[str, Any],
                            total_capacity_bytes: int,
                            context_tokens: int = 131072) -> CanonicalE2EResult:
    """Join already-resolved models without changing their equations."""
    nmp_rows = {int(row["requests"]): row for row in nmp_payload["rows"]}
    thermal_rows = {
        int(row["summary"]["requests"]): row for row in thermal_payload["cases"]}
    if set(nmp_rows) != {1, 8, 16} or set(thermal_rows) != {1, 8, 16}:
        raise ValueError("canonical E2E requires exactly N=1,8,16")
    results = []
    for requests in (1, 8, 16):
        row = nmp_rows[requests]
        thermal = thermal_rows[requests]
        power = row["B_PREP_DIE_POWER_MAP"]
        balanced = row["A_FINAL_CANONICAL_GAIN"]
        locality = row["LOCALITY_ONLY_BASELINE"]
        ideal = row["IDEAL_AGGREGATE_UPPER_BOUND"]
        non_nmp = row["non_nmp_gpu"]
        activities = balanced["activity"]["activities"]

        weight = sum(float(item["weight_read_bytes"]) for item in activities)
        kv_read = sum(float(item["kv_read_bytes"]) for item in activities)
        kv_write = sum(float(item["kv_write_bytes"]) for item in activities)
        local_bytes = sum(float(item["total_local_memory_bytes"]) for item in activities)
        if not math.isclose(local_bytes, weight + kv_read + kv_write,
                            rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError("local NMP traffic byte closure failed")
        residual = float(balanced["remaining_external_bytes"])
        non_external = float(non_nmp["traffic"]["external_interface_bytes"])
        reduction = 1.0 - residual / non_external
        direct = float(non_nmp["traffic"]["direct_die_to_die_bytes"])
        if not (0.0 <= reduction <= 1.0) or residual >= non_external or direct != 0.0:
            raise ValueError("external traffic boundary failed")

        allocated = int(row["working_set_bytes"])
        logical = int(row["logical_working_set_bytes"])
        capacity_feasible = allocated <= total_capacity_bytes
        resident_fraction = 1.0 if capacity_feasible else total_capacity_bytes / allocated
        if not capacity_feasible or resident_fraction != 1.0:
            raise ValueError("canonical workload is not fully resident")

        non_ms = float(non_nmp["timing"]["total_step_ms"])
        local_ms = float(locality["activity"]["decode_step_interval_ms"])
        balanced_ms = float(balanced["nmp_step_ms"])
        ideal_ms = float(ideal["nmp_step_ms"])
        throughput = tuple(requests / (value * 1e-3)
                           for value in (non_ms, local_ms, balanced_ms, ideal_ms))
        gains = tuple(value / throughput[0] for value in throughput[1:])
        if not math.isclose(gains[1], float(balanced["combined_A_gain"]),
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("balanced A gain changed during integration")

        service = tuple(float(value) for value in
                        balanced["placement"]["service_time_ms_per_die"])
        mean_service = sum(service) / len(service)
        power_total = float(power["aggregate_total_W"])
        power_components = sum(float(power[name]) for name in (
            "aggregate_memory_read_dynamic_W", "aggregate_memory_write_dynamic_W",
            "aggregate_mac_dynamic_W", "aggregate_refresh_W",
            "aggregate_residual_external_W"))
        if not math.isclose(power_total, power_components,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("M3D+NMP aggregate power component closure failed")
        if not math.isclose(power_total, float(thermal["aggregate_m3d_power_W"]),
                            rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("performance and thermal power payloads differ")
        energy_step = power_total * balanced_ms * 1e-3
        thermal_summary = thermal["summary"]
        thermal_baseline = thermal["baseline"]
        gamma = float(power["primitives"]["nmp_logic_overhead_factor"])
        if gamma != 1.0:
            raise ValueError("gamma_NMP changed during integration")
        thermal_gate = (
            "PASS" if thermal_baseline["converged"]
            and thermal_baseline["thermal_power_mapping_closure"] == "PASS"
            else "FAIL")
        values = (non_ms, local_ms, balanced_ms, ideal_ms, *throughput, *gains,
                  power_total, energy_step, float(thermal_summary["temperature_min_degC"]),
                  float(thermal_summary["temperature_mean_degC"]),
                  float(thermal_summary["temperature_max_degC"]),
                  float(thermal_summary["global_Tmax_degC"]))
        if not _finite(values):
            raise ValueError("canonical E2E metric is non-finite")
        results.append(CanonicalE2ECaseResult(
            requests=requests, context_tokens=context_tokens,
            model="LLAMA_3_1_8B_CLASS_DENSE_DECODE_FP16",
            total_physical_capacity_bytes=total_capacity_bytes,
            logical_working_set_bytes=logical,
            allocated_working_set_bytes=allocated,
            resident_fraction=resident_fraction,
            capacity_headroom_bytes=total_capacity_bytes - allocated,
            capacity_feasible=capacity_feasible,
            total_weight_read_bytes=weight, total_kv_read_bytes=kv_read,
            total_kv_write_bytes=kv_write, local_nmp_bytes=local_bytes,
            residual_external_bytes=residual,
            non_nmp_external_bytes=non_external,
            external_traffic_reduction=reduction,
            direct_die_to_die_bytes=direct,
            non_nmp_step_ms=non_ms, locality_only_step_ms=local_ms,
            balanced_step_ms=balanced_ms, ideal_step_ms=ideal_ms,
            non_nmp_tokens_per_s=throughput[0],
            locality_only_tokens_per_s=throughput[1],
            balanced_tokens_per_s=throughput[2], ideal_tokens_per_s=throughput[3],
            locality_only_gain=gains[0], balanced_gain=gains[1], ideal_gain=gains[2],
            placement_latency_ratio=balanced_ms / local_ms,
            placement_incremental_speedup=local_ms / balanced_ms,
            max_mean_service=max(service) / mean_service,
            max_capacity_utilization=float(balanced["placement"]["max_capacity_utilization"]),
            max_power_die=int(thermal_baseline["max_power_die_id"]),
            hottest_die=int(thermal_baseline["hottest_m3d_die_id"]),
            aggregate_m3d_nmp_power_W=power_total,
            read_W=float(power["aggregate_memory_read_dynamic_W"]),
            write_W=float(power["aggregate_memory_write_dynamic_W"]),
            mac_W=float(power["aggregate_mac_dynamic_W"]),
            refresh_W=float(power["aggregate_refresh_W"]),
            residual_external_W=float(power["aggregate_residual_external_W"]),
            energy_per_decode_step_J=energy_step,
            energy_per_token_J=energy_step / requests,
            m3d_Tmin_degC=float(thermal_summary["temperature_min_degC"]),
            m3d_Tmean_degC=float(thermal_summary["temperature_mean_degC"]),
            m3d_Tmax_degC=float(thermal_summary["temperature_max_degC"]),
            m3d_delta_T_degC=float(thermal_summary["temperature_spread_degC"]),
            global_Tmax_degC=float(thermal_summary["global_Tmax_degC"]),
            global_Tmax_region=str(thermal_summary["global_Tmax_region"]),
            gamma_NMP=gamma, capacity_gate="PASS", traffic_gate="PASS",
            performance_gate="PASS", power_gate="PASS", thermal_gate=thermal_gate,
            canonical_gate=("CONDITIONAL_PASS" if thermal_gate == "PASS" else "FAIL"),
        ))
    thermal_pass = all(row.thermal_gate == "PASS" for row in results)
    gates = {
        "E2E_CAPACITY_GATE": "PASS",
        "E2E_TRAFFIC_GATE": "PASS",
        "E2E_PERFORMANCE_GATE": "PASS",
        "E2E_POWER_GATE": "PASS",
        "E2E_THERMAL_GATE": "PASS" if thermal_pass else "FAIL",
        "E2E_CANONICAL_GATE": "CONDITIONAL_PASS" if thermal_pass else "FAIL",
    }
    return CanonicalE2EResult(
        cases=tuple(results), model="CANONICAL_FROZEN_M3D_NMP_E2E",
        architecture_status="THERMALLY_FEASIBLE_UNDER_CANONICAL_STEADY_STATE_MODEL",
        full_system_energy_comparison_status=(
            "NOT_CLAIMED__PROPOSED_M3D_NMP_ENERGY_PER_TOKEN_ONLY"),
        residual_external_thermal_mapping_status=(
            "RESIDUAL_EXTERNAL_THERMAL_MAPPING_APPROXIMATION"), gates=gates)


def write_canonical_e2e_artifacts(result: CanonicalE2EResult,
                                  output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    rows = [row.as_dict() for row in result.cases]
    for row in result.cases:
        (output_dir / f"case_n{row.requests}.json").write_text(
            json.dumps(row.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    headline_fields = [
        "requests", "allocated_working_set_bytes", "non_nmp_step_ms",
        "balanced_step_ms", "balanced_gain", "balanced_tokens_per_s",
        "external_traffic_reduction", "aggregate_m3d_nmp_power_W",
        "energy_per_token_J", "m3d_Tmax_degC", "global_Tmax_degC"]
    _write_csv(output_dir / "summary.csv", rows, headline_fields)
    performance_fields = [
        "requests", "non_nmp_step_ms", "non_nmp_tokens_per_s",
        "locality_only_step_ms", "locality_only_tokens_per_s", "locality_only_gain",
        "balanced_step_ms", "balanced_tokens_per_s", "balanced_gain",
        "ideal_step_ms", "ideal_tokens_per_s", "ideal_gain"]
    _write_csv(output_dir / "architecture_comparison.csv", rows, performance_fields)
    placement_fields = [
        "requests", "locality_only_step_ms", "balanced_step_ms",
        "placement_latency_ratio", "placement_incremental_speedup",
        "max_mean_service", "max_capacity_utilization"]
    _write_csv(output_dir / "placement_comparison.csv", rows, placement_fields)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
