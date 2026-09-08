"""Single formal B x S x G non-thermal benchmark entry point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from om3dthermal.serving.formal_workload import (
    FormalInferenceWorkload,
    evaluate_formal_inference_workload,
    formal_capacity_limits,
)
from om3dthermal.serving.workspace import WorkspaceExecutionConfig
from om3dthermal.workload import load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"configs/experiment/formal_e2e_benchmark.yaml"
OUT = ROOT/"runs/formal_e2e_benchmark"
LINES = (
    ("CONVENTIONAL_HBM_GPU", "RESIDENT_ONLY_QUEUE_TO_FIT"),
    ("CONVENTIONAL_HBM_GPU", "HOST_KV_OFFLOAD"),
    ("ORTHOGONAL_M3D_IGZO_MEMORY_ONLY", "FULLY_LOCAL"),
    ("IOM3D_FEOL_NMP", "FULLY_LOCAL"),
)


def _load():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    workspace = WorkspaceExecutionConfig.model_validate(raw["workspace"])
    registry = load_dense_model_registry(ROOT/raw["model_registry_dir"])
    return raw, workspace, registry


def _row(result) -> dict[str, object]:
    row = result.model_dump(exclude={
        "known_energy_components", "unresolved_energy_terms", "thermal"})
    row["known_energy_components"] = json.dumps(
        result.known_energy_components, sort_keys=True)
    row["unresolved_energy_terms"] = json.dumps(
        result.unresolved_energy_terms, sort_keys=True)
    return row


def _evaluate(model, B, S, G, workspace):
    workload = FormalInferenceWorkload(
        model_id=model.model_id, batch_size=B,
        prompt_tokens=S, generation_tokens=G)
    return [evaluate_formal_inference_workload(
        project_root=ROOT, model=model, workload=workload,
        system=system, policy=policy, workspace_config=workspace)
        for system, policy in LINES]


def _add_speedups(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[object, ...], dict[tuple[str, str], dict[str, object]]] = {}
    for row in rows:
        key = (row["model"], row["batch_size"], row["prompt_tokens"],
               row["generation_tokens"])
        grouped.setdefault(key, {})[(row["system"], row["policy"])] = row
    for group in grouped.values():
        queue = group[("CONVENTIONAL_HBM_GPU", "RESIDENT_ONLY_QUEUE_TO_FIT")]
        offload = group[("CONVENTIONAL_HBM_GPU", "HOST_KV_OFFLOAD")]
        m3d = group[("ORTHOGONAL_M3D_IGZO_MEMORY_ONLY", "FULLY_LOCAL")]
        nmp = group[("IOM3D_FEOL_NMP", "FULLY_LOCAL")]
        comparisons = {
            "capacity_gain_vs_resident_only": (queue, m3d),
            "capacity_gain_vs_host_offload": (offload, m3d),
            "pure_nmp_gain_vs_m3d_only": (m3d, nmp),
            "total_gain_vs_resident_only__capacity_plus_nmp": (queue, nmp),
            "total_gain_vs_host_offload__capacity_plus_nmp": (offload, nmp),
        }
        for label, (baseline, candidate) in comparisons.items():
            metrics = {
                f"{label}__makespan_speedup": _ratio(
                    baseline["batch_makespan_s"], candidate["batch_makespan_s"]),
                f"{label}__e2e_throughput_speedup": _ratio(
                    candidate["e2e_output_tokens_per_s"],
                    baseline["e2e_output_tokens_per_s"]),
                f"{label}__decode_throughput_speedup": _ratio(
                    candidate["decode_tokens_per_s"], baseline["decode_tokens_per_s"]),
                f"{label}__p95_completion_speedup": _ratio(
                    baseline["p95_completion_latency_s"],
                    candidate["p95_completion_latency_s"]),
                f"{label}__p95_ttft_speedup": _ratio(
                    baseline["p95_ttft_s"], candidate["p95_ttft_s"]),
            }
            if label == "capacity_gain_vs_resident_only":
                metrics[f"{label}__resident_concurrency_gain"] = _ratio(
                    candidate["safe_resident_batch"],
                    baseline["safe_resident_batch"])
            if label == "capacity_gain_vs_host_offload":
                baseline_host = float(baseline["total_host_transfer_bytes"])
                candidate_host = float(candidate["total_host_transfer_bytes"])
                metrics[f"{label}__host_traffic_reduction_fraction"] = (
                    None if baseline_host == 0 else
                    1.0-candidate_host/baseline_host)
            for row in group.values():
                row.update(metrics)


def _ratio(numerator, denominator) -> float | None:
    if numerator in (None, "") or denominator in (None, "", 0, "0"):
        return None
    return float(numerator)/float(denominator)


def _write(name: str, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT/name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _summaries(rows):
    latency = [{key: row.get(key) for key in (
        "model", "system", "policy", "batch_size", "prompt_tokens",
        "generation_tokens", "mean_ttft_s", "p95_ttft_s", "max_ttft_s",
        "mean_completion_latency_s", "p95_completion_latency_s",
        "max_completion_latency_s", "mean_tpot_s", "p95_tpot_s")}
        for row in rows]
    host = [{key: row.get(key) for key in (
        "model", "system", "policy", "batch_size", "prompt_tokens",
        "generation_tokens", "historical_host_read_bytes",
        "host_append_write_bytes", "migration_bytes",
        "total_host_transfer_bytes", "host_bytes_per_generated_token",
        "host_transfer_time_s", "host_transfer_time_fraction")}
        for row in rows]
    energy = [{key: row.get(key) for key in (
        "model", "system", "policy", "batch_size", "prompt_tokens",
        "generation_tokens", "known_energy_J", "known_energy_components",
        "energy_status", "unresolved_energy_terms",
        "absolute_J_per_token_status")}
        for row in rows]
    return latency, host, energy


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _refresh_commit() -> None:
    path = OUT/"benchmark_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit["commit"] = _commit()
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md = OUT/"benchmark_audit.md"
    lines = [f"- Commit: `{_commit()}`" if line.startswith("- Commit:") else line
             for line in md.read_text(encoding="utf-8").splitlines()]
    md.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--generation-tokens", type=int)
    parser.add_argument("--system", choices=[item[0] for item in LINES])
    parser.add_argument("--policy", choices=[item[1] for item in LINES])
    parser.add_argument("--primary-matrix", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--bs-sweep", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--refresh-commit", action="store_true")
    args = parser.parse_args()
    if args.refresh_commit:
        _refresh_commit(); return
    raw, workspace, registry = _load()
    if args.model:
        if None in (args.batch_size, args.prompt_tokens, args.generation_tokens,
                    args.system, args.policy):
            parser.error("single case requires B, S, G, system, and policy")
        workload = FormalInferenceWorkload(
            model_id=args.model, batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
            generation_tokens=args.generation_tokens)
        result = evaluate_formal_inference_workload(
            project_root=ROOT, model=registry[args.model], workload=workload,
            system=args.system, policy=args.policy, workspace_config=workspace)
        print(result.model_dump_json(indent=2)); return
    if not any((args.primary_matrix, args.sensitivity, args.bs_sweep, args.all)):
        parser.error("select --all or one matrix mode")
    OUT.mkdir(parents=True, exist_ok=True)
    primary: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    sweep: list[dict[str, object]] = []
    capacity: list[dict[str, object]] = []
    limits_by_model = {}
    for model_id in raw["models"]:
        limits = formal_capacity_limits(
            project_root=ROOT, model=registry[model_id],
            S=raw["primary"]["prompt_tokens"],
            G=raw["primary"]["generation_tokens"],
            workspace_config=workspace)
        limits["B_stress"] = min(28, limits["B_max_M3D_safe"])
        if limits["B_stress"] < 1:
            raise RuntimeError(f"{model_id} has no feasible M3D batch")
        limits_by_model[model_id] = limits
        capacity.append({"model": model_id, **limits,
                         "prompt_tokens": 131072, "generation_tokens": 512})
    if args.bs_sweep or args.all:
        boundary = limits_by_model["llama31_8b"]["B_max_M3D_safe"]
        for B in range(1, boundary+2):
            sweep.extend(_row(item) for item in _evaluate(
                registry["llama31_8b"], B, 131072, 512, workspace))
        _add_speedups(sweep)
        _write("llama_bs_sweep.csv", sweep)
    if args.primary_matrix or args.all:
        for model_id in raw["models"]:
            batches = list(raw["primary"]["batch_sizes"])
            stress = limits_by_model[model_id]["B_stress"]
            if stress not in batches:
                batches.append(stress)
            for B in batches:
                primary.extend(_row(item) for item in _evaluate(
                    registry[model_id], B, raw["primary"]["prompt_tokens"],
                    raw["primary"]["generation_tokens"], workspace))
        _add_speedups(primary)
        _write("primary_results.csv", primary)
    if args.sensitivity or args.all:
        for model_id in raw["models"]:
            for point in raw["sensitivities"]:
                for item in _evaluate(
                    registry[model_id], point["batch_size"],
                    point["prompt_tokens"], point["generation_tokens"], workspace):
                    row = _row(item); row["sensitivity"] = point["name"]
                    sensitivity.append(row)
        _add_speedups(sensitivity)
        _write("sensitivity_results.csv", sensitivity)
    all_rows = primary+sensitivity+sweep
    latency, host, energy = _summaries(all_rows)
    _write("capacity_summary.csv", capacity)
    _write("request_latency_summary.csv", latency)
    _write("host_traffic_summary.csv", host)
    _write("energy_summary.csv", energy)
    b1 = [row for row in sweep if row["batch_size"] == 1]
    knee = limits_by_model["llama31_8b"]["B_max_HBM_safe"]
    recurring = [row for row in sweep if (
        row["system"] == "CONVENTIONAL_HBM_GPU"
        and row["policy"] == "HOST_KV_OFFLOAD" and row["batch_size"] > knee)]
    b1_queue = next((row for row in b1 if row["policy"] ==
                     "RESIDENT_ONLY_QUEUE_TO_FIT"), None)
    b1_offload = next((row for row in b1 if row["policy"] ==
                       "HOST_KV_OFFLOAD"), None)
    b1_equal = bool(b1_queue and b1_offload) and all(
        b1_queue[field] == b1_offload[field]
        for field in ("batch_makespan_s", "e2e_output_tokens_per_s",
                      "mean_ttft_s", "mean_completion_latency_s"))
    audit = {
        "commit": _commit(), "formal_workload": "B_REQUESTS_EACH_PREFILL_ONCE_THEN_G_DECODE",
        "arrival_policy": "ALL_REQUESTS_AT_T0", "primary": "S131072_G512",
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "capacity_limits": limits_by_model,
        "checks": {
            "formal_has_no_P_D_fields": all("P" not in row and "D" not in row for row in all_rows),
            "B1_policy_equivalence": (len(b1) == 4 and b1_equal and all(
                row["historical_host_read_bytes"] == 0 for row in b1)),
            "recurring_host_after_hbm_knee": bool(recurring) and all(
                row["historical_host_read_bytes"] > 0 for row in recurring),
            "m3d_boundary_followed_by_infeasible_row": any(
                row["batch_size"] == limits_by_model["llama31_8b"]["B_max_M3D_safe"]+1
                and row["system"] == "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY"
                and row["status"] == "CAPACITY_INFEASIBLE" for row in sweep)},
        "CONVENTIONAL_HBM_WRITE_ENERGY": "UNRESOLVED",
        "HOST_KV_OVERFLOW_MODEL": "CLOSED", "thermal": None}
    (OUT/"benchmark_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Formal E2E benchmark audit", "", f"- Commit: `{audit['commit']}`",
        "- Workload: `B requests x (one S-token Prefill + G Decode steps)`",
        "- Arrival: `ALL_REQUESTS_AT_T0`", "- Primary: `S=131072, G=512`",
        "- Conventional HBM write energy: `UNRESOLVED`", "",
        "| Model | HBM safe B | M3D safe B | NMP safe B | B_stress |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id, limits in limits_by_model.items():
        lines.append(f"| {model_id} | {limits['B_max_HBM_safe']} | "
                     f"{limits['B_max_M3D_safe']} | {limits['B_max_NMP_safe']} | "
                     f"{limits['B_stress']} |")
    (OUT/"benchmark_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"primary_rows": len(primary), "sensitivity_rows": len(sensitivity),
                      "sweep_rows": len(sweep), "capacity_limits": limits_by_model}))


if __name__ == "__main__":
    main()
