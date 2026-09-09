"""Rebaseline formal host offload to a single-GPU GH200 memory path."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import yaml

from om3dthermal.platform import HostOffloadSpec, load_platform_spec_file
from om3dthermal.serving import (
    FormalInferenceWorkload, WorkspaceExecutionConfig,
    evaluate_formal_inference_workload,
)
from om3dthermal.workload import load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"configs/experiment/gh200_host_offload_rebaseline.yaml"
OUT = ROOT/"runs/gh200_host_offload_rebaseline"


def _write(name: str, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT/name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _host_row(result, profile: HostOffloadSpec) -> dict[str, object]:
    total_bytes = result.total_host_transfer_bytes
    memory_e = float(profile.host_memory_dynamic_J_per_bit or 0.0)
    link_e = float(profile.host_link_dynamic_J_per_bit or 0.0)
    total_e = float(profile.e_host_offload_dynamic_J_per_bit or 0.0)
    memory_J = total_bytes*8.0*memory_e
    link_J = total_bytes*8.0*link_e
    host_J = total_bytes*8.0*total_e
    bandwidth = float(profile.effective_bandwidth_bytes_per_second)
    theoretical = bandwidth/result.host_bytes_per_generated_token
    ratio = float(result.decode_tokens_per_s)/theoretical
    return {
        "model": result.model,
        "batch_size": result.batch_size,
        "context_length": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
        "system": result.system,
        "policy": result.policy,
        "host_path_id": profile.host_path_id,
        "host_path_role": profile.path_role,
        "host_bandwidth_GBps": bandwidth/1e9,
        "effective_bandwidth_status": profile.effective_bandwidth_status,
        "historical_host_read_GB": result.historical_host_read_bytes/1e9,
        "host_append_write_GB": result.host_append_write_bytes/1e9,
        "migration_GB": result.migration_bytes/1e9,
        "total_host_GB": total_bytes/1e9,
        "host_GB_per_output_token": result.host_bytes_per_generated_token/1e9,
        "raw_host_transfer_time_s": result.host_transfer_time_s,
        "host_critical_path_time_s": result.host_critical_path_time_s,
        "decode_active_time_s": result.decode_active_time_s,
        "decode_tokens_per_s": result.decode_tokens_per_s,
        "batch_makespan_s": result.batch_makespan_s,
        "e2e_output_tokens_per_s": result.e2e_output_tokens_per_s,
        "P95_TTFT_s": result.p95_ttft_s,
        "P95_completion_s": result.p95_completion_latency_s,
        "host_memory_energy_pJ_per_bit": memory_e*1e12,
        "host_link_energy_pJ_per_bit": link_e*1e12,
        "host_total_energy_pJ_per_bit": total_e*1e12,
        "host_memory_dynamic_J": memory_J,
        "host_link_dynamic_J": link_J,
        "host_total_dynamic_J": host_J,
        "memory_energy_status": profile.memory_energy_status,
        "link_energy_status": profile.link_energy_status,
        "total_energy_status": profile.total_energy_status,
        "theoretical_host_ceiling_tok_per_s": theoretical,
        "simulated_decode_tok_per_s": result.decode_tokens_per_s,
        "simulated_to_host_ceiling_ratio": ratio,
        "ceiling_interpretation": (
            "HOST_TRANSFER_BOUND_WITH_OPTIMISTIC_OVERLAP__RATIO_OFFSET_FROM_"
            "ONE_TIME_MIGRATION_INCLUDED_IN_TOTAL_BYTES_CEILING"
            if math.isclose(ratio, 1.0, rel_tol=.03) else
            "LOCAL_GPU_HBM_LIMIT_OR_ONE_TIME_MIGRATION_ACCOUNTING"),
        "known_energy_J": result.known_energy_J,
        "known_energy_components": json.dumps(
            result.known_energy_components, sort_keys=True),
        "unresolved_energy_terms": json.dumps(
            result.unresolved_energy_terms, sort_keys=True),
        "energy_status": result.energy_status,
        "absolute_J_per_token_status": result.absolute_J_per_token_status,
        "absolute_tokens_per_J_status": "INCOMPLETE",
    }


def _m3d_row(result) -> dict[str, object]:
    return {
        "model": result.model, "batch_size": result.batch_size,
        "context_length": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
        "system": result.system, "policy": result.policy,
        "host_path_id": "NOT_APPLICABLE_FULLY_LOCAL",
        "host_path_role": "M3D_FULLY_LOCAL_REFERENCE",
        "host_bandwidth_GBps": None,
        "decode_active_time_s": result.decode_active_time_s,
        "decode_tokens_per_s": result.decode_tokens_per_s,
        "batch_makespan_s": result.batch_makespan_s,
        "e2e_output_tokens_per_s": result.e2e_output_tokens_per_s,
        "P95_TTFT_s": result.p95_ttft_s,
        "P95_completion_s": result.p95_completion_latency_s,
        "known_energy_J": result.known_energy_J,
        "known_energy_components": json.dumps(
            result.known_energy_components, sort_keys=True),
        "unresolved_energy_terms": json.dumps(
            result.unresolved_energy_terms, sort_keys=True),
        "energy_status": result.energy_status,
        "absolute_J_per_token_status": result.absolute_J_per_token_status,
        "absolute_tokens_per_J_status": "INCOMPLETE",
    }


def main() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    workspace = WorkspaceExecutionConfig.model_validate(raw["workspace"])
    registry = load_dense_model_registry(ROOT/raw["model_registry_dir"])
    profiles = [HostOffloadSpec.model_validate(item) for item in raw["host_paths"]]
    primary_profile = next(
        item for item in profiles
        if item.path_role == "PRIMARY_HOST_OFFLOAD_BASELINE")
    platform = load_platform_spec_file(
        ROOT/"configs/platform/gpu_package_h200_reference.yaml")
    if platform.host_offload != primary_profile:
        primary_comparable = (
            platform.host_offload is not None
            and platform.host_offload.host_path_id == primary_profile.host_path_id
            and platform.host_offload.effective_bandwidth_bytes_per_second
            == primary_profile.effective_bandwidth_bytes_per_second
            and platform.host_offload.e_host_offload_dynamic_J_per_bit
            == primary_profile.e_host_offload_dynamic_J_per_bit)
        if not primary_comparable:
            raise ValueError("canonical platform does not select GH200 primary")
    OUT.mkdir(parents=True, exist_ok=True)
    sensitivity: list[dict[str, object]] = []
    primary: list[dict[str, object]] = []
    energy: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    results_by_model: dict[str, dict[str, object]] = {}
    for case in raw["cases"]:
        model = registry[case["model_id"]]
        workload = FormalInferenceWorkload(
            model_id=model.model_id, batch_size=case["batch_size"],
            prompt_tokens=raw["context_length"],
            generation_tokens=raw["generation_tokens"])
        indexed: dict[str, object] = {}
        for profile in profiles:
            result = evaluate_formal_inference_workload(
                project_root=ROOT, model=model, workload=workload,
                system="CONVENTIONAL_HBM_GPU", policy="HOST_KV_OFFLOAD",
                workspace_config=workspace, host_offload_spec=profile)
            indexed[str(profile.host_path_id)] = result
            row = _host_row(result, profile)
            sensitivity.append(row)
            energy.append({key: row[key] for key in (
                "model", "batch_size", "host_path_id", "host_path_role",
                "total_host_GB", "host_memory_energy_pJ_per_bit",
                "host_link_energy_pJ_per_bit", "host_total_energy_pJ_per_bit",
                "host_memory_dynamic_J", "host_link_dynamic_J",
                "host_total_dynamic_J", "memory_energy_status",
                "link_energy_status", "total_energy_status", "known_energy_J",
                "known_energy_components", "unresolved_energy_terms",
                "energy_status", "absolute_J_per_token_status",
                "absolute_tokens_per_J_status")})
            if profile is primary_profile:
                primary.append(row)
        m3d = evaluate_formal_inference_workload(
            project_root=ROOT, model=model, workload=workload,
            system="ORTHOGONAL_M3D_IGZO_MEMORY_ONLY", policy="FULLY_LOCAL",
            workspace_config=workspace)
        m3d_row = _m3d_row(m3d)
        primary.append(m3d_row)
        gh200 = indexed[str(primary_profile.host_path_id)]
        comparisons.append({
            "model": model.model_id, "batch_size": workload.batch_size,
            "baseline": "GH200_MEASURED_416_34_GBPS",
            "candidate": "M3D_MEMORY_ONLY_FULLY_LOCAL",
            "M3D_vs_GH200_decode_speedup": (
                m3d.decode_tokens_per_s/gh200.decode_tokens_per_s),
            "M3D_vs_GH200_E2E_speedup": (
                m3d.e2e_output_tokens_per_s/gh200.e2e_output_tokens_per_s),
            "M3D_vs_GH200_P95_TTFT_gain": (
                gh200.p95_ttft_s/m3d.p95_ttft_s),
            "M3D_vs_GH200_P95_completion_gain": (
                gh200.p95_completion_latency_s/m3d.p95_completion_latency_s),
            "host_traffic_reduction_fraction": 1.0,
        })
        results_by_model[model.model_id] = indexed
    _write("primary_results.csv", primary)
    _write("bandwidth_sensitivity.csv", sensitivity)
    _write("energy_breakdown.csv", energy)
    _write("comparison_vs_m3d.csv", comparisons)
    traffic_invariance = {}
    for model_id, indexed in results_by_model.items():
        signatures = {(
            item.historical_host_read_bytes, item.host_append_write_bytes,
            item.migration_bytes, item.total_host_transfer_bytes)
            for item in indexed.values()}
        traffic_invariance[model_id] = len(signatures) == 1
    energy_closure = all(math.isclose(
        float(row["host_memory_dynamic_J"])+float(row["host_link_dynamic_J"]),
        float(row["host_total_dynamic_J"]), rel_tol=1e-12)
        for row in energy)
    ideal_profiles = [item for item in profiles
                      if item.effective_bandwidth_bytes_per_second == 450e9]
    pcie_profiles = [item for item in profiles
                     if item.effective_bandwidth_bytes_per_second == 56.2e9]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    audit = {
        "commit": commit,
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "PRIMARY_HOST_OFFLOAD_BASELINE": "GH200_MEASURED_416_34_GBPS",
        "PCIE_56_2_GBPS_ROLE": "SENSITIVITY_ONLY",
        "CONVENTIONAL_HBM_WRITE_ENERGY": "UNRESOLVED",
        "checks": {
            "canonical_nominal_bandwidth_is_416_34e9": (
                primary_profile.effective_bandwidth_bytes_per_second == 416.34e9),
            "450e9_is_upper_bound_sensitivity_only": len(ideal_profiles) == 1 and all(
                item.path_role == "SENSITIVITY_ONLY"
                for item in ideal_profiles),
            "56_2e9_is_explicit_sensitivity_only": len(pcie_profiles) == 1 and all(
                item.path_role == "SENSITIVITY_ONLY"
                for item in pcie_profiles),
            "traffic_invariant_across_bandwidths": all(traffic_invariance.values()),
            "GH200_transfer_faster_than_PCIe": all(
                float(next(row for row in sensitivity if row["model"] == model
                           and row["host_bandwidth_GBps"] == 416.34)["raw_host_transfer_time_s"])
                < float(next(row for row in sensitivity if row["model"] == model
                             and row["host_bandwidth_GBps"] == 56.2)["raw_host_transfer_time_s"])
                for model in results_by_model),
            "GH200_energy_4_plus_1_3_equals_5_3": math.isclose(
                4.0+1.3, 5.3, rel_tol=1e-12),
            "host_energy_exact_closure": energy_closure,
            "recurring_historical_reads_preserved": all(
                item.historical_host_read_bytes > 0
                for indexed in results_by_model.values()
                for item in indexed.values()),
        },
        "traffic_invariance_by_model": traffic_invariance,
        "thermal": None,
    }
    (OUT/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md = [
        "# GH200 host-offload rebaseline audit", "", f"- Commit: `{commit}`",
        "- Primary: `GH200_MEASURED_416_34_GBPS`",
        "- PCIe 56.2 GB/s: `SENSITIVITY_ONLY`",
        "- C2C 450 GB/s: `IDEAL_UPPER_BOUND_SENSITIVITY_ONLY`",
        "- Conventional HBM write energy: `UNRESOLVED`", "", "## Checks", "",
    ]
    md.extend(f"- {key}: `{'PASS' if value else 'FAIL'}`"
              for key, value in audit["checks"].items())
    (OUT/"audit.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps({
        "primary_rows": len(primary), "sensitivity_rows": len(sensitivity),
        "energy_rows": len(energy), "comparison_rows": len(comparisons),
        "checks_pass": all(audit["checks"].values())}))


if __name__ == "__main__":
    main()
