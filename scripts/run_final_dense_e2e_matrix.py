"""Run the frozen 36-point final dense mixed-phase non-thermal matrix."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from om3dthermal.serving import (
    MixedPhaseServingCase,
    compare_mixed_phase_results,
    evaluate_mixed_phase_e2e,
    evaluate_nmp_decode_batch,
    load_final_dense_e2e_matrix,
)
from om3dthermal.workload import (
    evaluate_llm_decode,
    evaluate_llm_prefill,
    load_dense_model_registry,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/final_dense_e2e_matrix"
B28_STATUS = "ANALYTICAL_MAXIMUM_FEASIBLE_CAPACITY_POINT"


def _audit(model, context: int) -> dict[str, object]:
    decode = model.decode_input(batch_size=28, context_length=context)
    dm = evaluate_llm_decode(decode)
    pm = evaluate_llm_prefill(model.prefill_input(
        batch_size=1, prompt_length=context))
    return {
        "model_id": model.model_id,
        "L": decode.n_layers,
        "d_model": decode.d_model,
        "d_ff": decode.d_ff,
        "Hq": decode.n_heads_q,
        "Hkv": decode.n_heads_kv,
        "d_head": decode.d_head,
        "vocab": decode.vocab_size,
        "weight_footprint_GB": dm.weight_footprint_bytes / 1e9,
        "KV_GB_per_request_at_131072": dm.kv_bytes_per_request / 1e9,
        "prefill_FLOPs_P1": pm.total_flops,
        "decode_FLOPs_per_token_B28": dm.flops_per_token,
        "B28_required_capacity_GB": dm.required_capacity_bytes / 1e9,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object) -> str:
    if value is None:
        return "UNRESOLVED"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    spec = load_final_dense_e2e_matrix(
        ROOT / "configs/experiment/final_dense_e2e_matrix.yaml")
    registry = load_dense_model_registry(ROOT / spec.model_registry_dir)

    gate_input = registry["llama31_8b"].decode_input(
        batch_size=28, context_length=spec.context_length)
    gate = evaluate_nmp_decode_batch(gate_input, project_root=ROOT)
    if not (gate.capacity_status == "FEASIBLE"
            and gate.evaluation_status == "EVALUATED"
            and gate.capacity_violations == 0):
        raise RuntimeError("D=28 pre-run gate failed; final matrix was not run")

    rows = []
    by_key = {}
    for model_id in spec.models:
        model = registry[model_id]
        for point in spec.mixed_points:
            case = MixedPhaseServingCase(
                model_id=model_id, context_length=spec.context_length, **point)
            key = (model_id, case.context_length, case.batch_size,
                   case.prefill_requests, case.decode_requests)
            by_key[key] = []
            for system_id in spec.systems:
                result = evaluate_mixed_phase_e2e(
                    project_root=ROOT, model=model, case=case,
                    system_id=system_id)
                row = result.model_dump(mode="json")
                row["B28_status"] = B28_STATUS if case.batch_size == 28 else None
                rows.append(row)
                by_key[key].append(result)

    if len(rows) != 36 or len(by_key) != 12:
        raise RuntimeError("matrix cardinality closure failed")
    for key, results in by_key.items():
        if len(results) != 3 or {r.system_id for r in results} != set(spec.systems):
            raise RuntimeError(f"system closure failed for {key}")
        for result in results:
            if result.evaluation_status != "CAPACITY_INFEASIBLE":
                if not math.isclose(
                    float(result.mixed_epoch_time_ms),
                    float(result.prefill_service_time_ms)
                    + float(result.decode_service_time_ms), rel_tol=1e-12):
                    raise RuntimeError("NO_OVERLAP closure failed")
                expected = result.decode_requests / (
                    float(result.decode_service_time_ms) * 1e-3)
                if not math.isclose(float(result.decode_tokens_per_s), expected,
                                    rel_tol=1e-12):
                    raise RuntimeError("Decode throughput closure failed")
            if (result.system_id == "IOM3D_FEOL_NMP"
                    and result.evaluation_status != "CAPACITY_INFEASIBLE"):
                if result.capacity_violations != 0:
                    raise RuntimeError("feasible proposed row has capacity violations")
                if result.nmp_batch_generalization_status != (
                        "RESOLVED_ANALYTICAL_BATCH_MODEL"):
                    raise RuntimeError("NMP batch status closure failed")

    comparisons = []
    for key, results in by_key.items():
        indexed = {r.system_id: r for r in results}
        compared = compare_mixed_phase_results(
            indexed["CONVENTIONAL_HBM_GPU"],
            (indexed["ORTHOGONAL_M3D_IGZO_MEMORY_ONLY"],
             indexed["IOM3D_FEOL_NMP"]))
        for item in compared:
            comparisons.append({
                "model_id": key[0], "context_length": key[1],
                "batch_size": key[2], "prefill_requests": key[3],
                "decode_requests": key[4], **item.model_dump(mode="json")})

    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT / "final_dense_e2e_matrix.csv", rows)
    _write_csv(OUT / "final_dense_e2e_comparison.csv", comparisons)
    audits = [_audit(registry[model_id], spec.context_length)
              for model_id in spec.models]
    payload = {
        "experiment_id": spec.experiment_id,
        "phase_overlap_policy": spec.phase_overlap_policy,
        "B28_STATUS": B28_STATUS,
        "pure_D28_gate": gate.model_dump(mode="json"),
        "model_audit": audits,
        "closure": {"ROW_COUNT": len(rows), "workload_count": len(by_key),
                    "system_count": len(spec.systems), "comparison_count": len(comparisons)},
        "rows": rows, "comparisons": comparisons,
    }
    (OUT / "final_dense_e2e_matrix.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    b28 = [row for row in rows if row["batch_size"] == 28]
    lines = [
        "# Final dense mixed-phase E2E matrix", "",
        "NON-THERMAL; phase overlap policy: `NO_OVERLAP`.", "",
        f"- ROW_COUNT: {len(rows)}", f"- B28_STATUS: `{B28_STATUS}`",
        f"- Pure D=28 gate: `{gate.evaluation_status}` / `{gate.capacity_status}` / violations={gate.capacity_violations}",
        "", "## Model audit", "",
        "| model | L | d_model | d_ff | Hq | Hkv | d_head | vocab | weights GB | KV GB/request | B28 required GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in audits:
        lines.append("| " + " | ".join(_fmt(a[k]) for k in (
            "model_id", "L", "d_model", "d_ff", "Hq", "Hkv", "d_head",
            "vocab", "weight_footprint_GB", "KV_GB_per_request_at_131072",
            "B28_required_capacity_GB")) + " |")
    lines += ["", "## B=28 headline", "",
              "| model | P:D | system | status | spill | TTFT ms | TPOT ms | epoch ms | decode tok/s |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in b28:
        lines.append("| " + " | ".join(_fmt(x) for x in (
            row["model_id"], f'{row["prefill_requests"]}:{row["decode_requests"]}',
            row["system_id"], row["evaluation_status"], row["spilled_requests"],
            row["TTFT_ms"], row["TPOT_ms"], row["mixed_epoch_time_ms"],
            row["decode_tokens_per_s"])) + " |")
    (OUT / "final_dense_e2e_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["closure"]))


if __name__ == "__main__":
    main()
