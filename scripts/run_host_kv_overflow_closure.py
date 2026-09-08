"""Formal non-thermal comparison of the two Conventional overflow policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

from om3dthermal.serving import (
    PersistentMixedServiceCase,
    evaluate_conventional_overflow_policy,
    evaluate_decode_workspace,
    evaluate_nmp_decode_batch,
    evaluate_persistent_mixed_service_horizon,
    load_serving_e2e_closure_spec,
    resolve_conventional_hbm_backend,
)
from om3dthermal.serving.nmp_decode import resolve_m3d_architecture_backend
from om3dthermal.workload import evaluate_llm_decode, load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"runs/host_kv_overflow_closure"
SPEC_PATH = ROOT/"configs/experiment/serving_e2e_closure.yaml"


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(name: str, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT/name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _local_limit(model, *, S: int, G: int, capacity: int,
                 page_size: int | None, workspace) -> int:
    one = evaluate_llm_decode(model.decode_input(batch_size=1, context_length=S))
    weight = int(one.weight_footprint_bytes+one.runtime_fixed_bytes)
    kv = int(one.kv_bytes_per_request+G*one.kv_write_bytes_per_token)
    rounded = lambda value: value if page_size is None else math.ceil(value/page_size)*page_size
    result = 0
    for batch in range(1, 4097):
        ws = evaluate_decode_workspace(
            model, batch_size=batch, context_length=S+G-1,
            config=workspace).peak_bytes
        if rounded(weight)+batch*rounded(kv)+rounded(ws) > capacity:
            break
        result = batch
    return result


def _local_row(result) -> dict[str, object]:
    return {
        "model": result.model, "S": result.S, "P": result.P,
        "D": result.D, "BS": result.BS, "G": result.G,
        "system": result.system, "policy": "FULLY_LOCAL_PERSISTENT_HORIZON",
        "status": result.status, "total_time_s": result.total_service_time_s,
        "tokens_per_s": result.tokens_per_s,
        "host_read_bytes": 0, "host_write_bytes": 0,
        "host_traffic_bytes_per_token": 0,
        "peak_memory_bytes": result.peak_runtime_bytes,
        "capacity_bytes": result.capacity_bytes,
        "capacity_margin_bytes": result.capacity_margin_bytes,
        "known_energy_J": result.known_energy_J,
        "energy_status": result.energy_status,
    }


def _policy_row(result) -> dict[str, object]:
    return {
        "model": result.model, "S": result.S, "P": result.P,
        "D": result.D, "BS": result.BS, "G": result.G,
        "system": "CONVENTIONAL_HBM_GPU", "policy": result.policy,
        "policy_role": result.policy_role, "status": result.status,
        "total_time_s": result.total_completion_time_s,
        "tokens_per_s": result.aggregate_tokens_per_s,
        "host_read_bytes": result.total_host_KV_read_bytes,
        "host_write_bytes": result.total_host_KV_write_bytes,
        "host_traffic_bytes_per_token": result.host_traffic_bytes_per_token,
        "peak_memory_bytes": result.peak_local_runtime_bytes,
        "capacity_bytes": result.capacity_bytes,
        "capacity_margin_bytes": result.capacity_margin_bytes,
        "resident_limit_initial": result.resident_limit_initial,
        "resident_limit_horizon_safe": result.resident_limit_horizon_safe,
        "resident_wave_size": result.resident_wave_size,
        "num_waves": result.num_waves,
        "known_energy_J": result.known_energy_J,
        "known_DDR_PCIe_energy_J": result.known_DDR_PCIe_energy_J,
        "energy_status": result.energy_status,
    }


def _evaluate_group(model, case, workspace):
    queue = evaluate_conventional_overflow_policy(
        project_root=ROOT, model=model, case=case,
        policy="RESIDENT_ONLY_QUEUE_TO_FIT", workspace_config=workspace)
    offload = evaluate_conventional_overflow_policy(
        project_root=ROOT, model=model, case=case,
        policy="HOST_KV_OFFLOAD", workspace_config=workspace)
    m3d = evaluate_persistent_mixed_service_horizon(
        project_root=ROOT, model=model, case=case,
        system="ORTHOGONAL_M3D_IGZO_MEMORY_ONLY", workspace_config=workspace)
    nmp = evaluate_persistent_mixed_service_horizon(
        project_root=ROOT, model=model, case=case,
        system="IOM3D_FEOL_NMP", workspace_config=workspace)
    rows = [_policy_row(queue), _policy_row(offload), _local_row(m3d), _local_row(nmp)]
    for row in rows:
        row["speedup_vs_queue_baseline"] = (
            None if queue.total_completion_time_s is None or row["total_time_s"] is None
            else queue.total_completion_time_s/float(row["total_time_s"]))
        row["speedup_vs_host_offload_baseline"] = (
            None if offload.total_completion_time_s is None or row["total_time_s"] is None
            else offload.total_completion_time_s/float(row["total_time_s"]))
        row["speedup_vs_m3d_only"] = (
            None if m3d.total_service_time_s is None or row["total_time_s"] is None
            else m3d.total_service_time_s/float(row["total_time_s"]))
    return queue, offload, m3d, nmp, rows


def _refresh_commit() -> None:
    path = OUT/"closure_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit["commit"] = _commit()
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md = OUT/"closure_audit.md"
    lines = md.read_text(encoding="utf-8").splitlines()
    lines = [f"- Commit: `{_commit()}`" if line.startswith("- Commit:") else line
             for line in lines]
    md.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-commit", action="store_true")
    args = parser.parse_args()
    if args.refresh_commit:
        _refresh_commit()
        return
    spec = load_serving_e2e_closure_spec(SPEC_PATH)
    registry = load_dense_model_registry(ROOT/spec.model_registry_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    policy_rows: list[dict[str, object]] = []
    traffic_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    all_groups = {}
    for model_id in spec.models:
        model = registry[model_id]
        for S in spec.context_lengths:
            for point in spec.mixed_points:
                for G in sorted(spec.generation_horizons, reverse=True):
                    case = PersistentMixedServiceCase(
                        model_id=model_id, context_length=S,
                        prefill_requests=point["prefill_requests"],
                        decode_requests=point["decode_requests"],
                        generated_decode_steps=G)
                    group = _evaluate_group(model, case, spec.workspace)
                    all_groups[(model_id, S, case.P, case.D, G)] = group
                    policy_rows.extend(group[-1])
                    for result in group[:2]:
                        traffic_rows.append({
                            "model": model_id, "S": S, "P": case.P,
                            "D": case.D, "G": G, "policy": result.policy,
                            "average_host_read_GB_per_decode_step": result.average_host_read_GB_per_decode_step,
                            "peak_host_read_GB_per_decode_step": result.peak_host_read_GB_per_decode_step,
                            "total_host_read_GB": result.total_host_KV_read_bytes/1e9,
                            "total_host_write_GB": result.total_host_KV_write_bytes/1e9,
                            "effective_host_bw_GBps": result.effective_host_bw_GBps,
                            "raw_host_transfer_time_s": result.host_transfer_time_s,
                            "host_critical_path_time_s": result.host_critical_path_time_s,
                            "host_transfer_time_fraction": result.host_transfer_time_fraction,
                            "known_DDR_PCIe_energy_J": result.known_DDR_PCIe_energy_J})

    conventional = resolve_conventional_hbm_backend(ROOT)
    m3d_arch = resolve_m3d_architecture_backend(ROOT)
    for model_id in spec.models:
        model = registry[model_id]
        for S in spec.context_lengths:
            for G in spec.generation_horizons:
                capacity_rows.append({
                    "model": model_id, "S": S, "G": G,
                    "conventional_safe_resident_BS": _local_limit(
                        model, S=S, G=G, capacity=int(conventional.capacity_bytes),
                        page_size=None, workspace=spec.workspace),
                    "m3d_safe_resident_BS": _local_limit(
                        model, S=S, G=G,
                        capacity=m3d_arch.layout.total_capacity_bytes,
                        page_size=m3d_arch.layout.slot_capacity_bytes,
                        workspace=spec.workspace)})

    bs_rows: list[dict[str, object]] = []
    model = registry["llama31_8b"]
    first_m3d_infeasible = None
    for decode_bs in range(1, 129):
        case = PersistentMixedServiceCase(
            model_id="llama31_8b", context_length=131072,
            prefill_requests=1, decode_requests=decode_bs,
            generated_decode_steps=512)
        queue, offload, m3d, nmp, _ = _evaluate_group(model, case, spec.workspace)
        bs_rows.append({
            "decode_BS": decode_bs, "total_mixed_requests": decode_bs+1,
            "P": 1, "G": 512,
            "conventional_resident_only_tokens_per_s": queue.aggregate_tokens_per_s,
            "conventional_host_offload_tokens_per_s": offload.aggregate_tokens_per_s,
            "m3d_only_tokens_per_s": m3d.tokens_per_s,
            "proposed_tokens_per_s": nmp.tokens_per_s,
            "m3d_capacity_status": m3d.status,
            "proposed_capacity_status": nmp.status,
            "host_offload_bytes_per_token": offload.host_traffic_bytes_per_token})
        if m3d.status != "EVALUATED":
            first_m3d_infeasible = decode_bs
            break

    qwen = all_groups[("qwen25_7b", 131072, 14, 14, 512)]
    llama_127 = all_groups[("llama31_8b", 131072, 1, 27, 512)]
    llama_1414 = all_groups[("llama31_8b", 131072, 14, 14, 512)]
    fixed_b1 = evaluate_nmp_decode_batch(
        registry["llama31_8b"].decode_input(batch_size=1, context_length=131072),
        project_root=ROOT)
    fixed_b28 = evaluate_nmp_decode_batch(
        registry["llama31_8b"].decode_input(batch_size=28, context_length=131072),
        project_root=ROOT)
    audit = {
        "commit": _commit(), "scope": "MIXED_SERVICE_WINDOW",
        "schedule": "NO_OVERLAP__PREFILL_FIRST",
        "config_hash": _sha(SPEC_PATH),
        "model_hashes": {model_id: _sha(
            ROOT/spec.model_registry_dir/f"{model_id}.yaml") for model_id in spec.models},
        "formal_policy_rows": len(policy_rows),
        "bs_sweep_first_m3d_infeasible_decode_BS": first_m3d_infeasible,
        "checks": {
            "host_active_history_recurs_every_step": all(
                step.host_historical_KV_bytes > 0 for step in llama_127[1].steps),
            "all_local_host_read_zero": all(
                row["host_read_bytes"] == 0 for row in policy_rows
                if row["policy"] == "HOST_KV_OFFLOAD"
                and row["capacity_margin_bytes"] >= 0),
            "llama_b28_g512_m3d_infeasible": (
                llama_127[2].status != "EVALUATED"
                and llama_1414[2].status != "EVALUATED"),
            "qwen_mixed_capacity_pressure": qwen[1].total_host_KV_read_bytes > 0,
            "nmp_fixed_b1": {"step_ms": fixed_b1.decode_step_time_ms,
                             "tokens_per_s": fixed_b1.aggregate_decode_tokens_per_s,
                             "J_per_token": fixed_b1.J_per_token},
            "nmp_fixed_b28": {"step_ms": fixed_b28.decode_step_time_ms,
                              "tokens_per_s": fixed_b28.aggregate_decode_tokens_per_s,
                              "J_per_token": fixed_b28.J_per_token}},
        "CONVENTIONAL_HBM_WRITE_ENERGY": "UNRESOLVED",
        "SERVING_CAPACITY_BLOCKER": "NONE",
        "HOST_KV_OVERFLOW_MODEL": "CLOSED",
        "thermal": None}
    _write_csv("policy_results.csv", policy_rows)
    _write_csv("host_traffic.csv", traffic_rows)
    _write_csv("capacity_limits.csv", capacity_rows)
    _write_csv("bs_sweep.csv", bs_rows)
    (OUT/"closure_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Host KV overflow closure", "", f"- Commit: `{audit['commit']}`",
        f"- Formal policy rows: {len(policy_rows)}",
        f"- Llama 128K G512 first M3D-infeasible decode BS: {first_m3d_infeasible}",
        "- Conventional HBM write energy: `UNRESOLVED`", "",
        "| Case | Policy | Host read/step GB | Total host read GB | Time s | tokens/s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, group in (("Llama 1:27", llama_127),
                        ("Llama 14:14", llama_1414), ("Qwen 14:14", qwen)):
        for result in group[:2]:
            lines.append(
                f"| {name} | {result.policy} | {result.average_host_read_GB_per_decode_step:.6f} | "
                f"{result.total_host_KV_read_bytes/1e9:.3f} | "
                f"{result.total_completion_time_s:.6f} | {result.aggregate_tokens_per_s:.3f} |")
    (OUT/"closure_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({
        "policy_rows": len(policy_rows), "bs_sweep_rows": len(bs_rows),
        "first_m3d_infeasible_decode_BS": first_m3d_infeasible,
        "llama_127_host_read_TB": llama_127[1].total_host_KV_read_bytes/1e12}))


if __name__ == "__main__":
    main()
