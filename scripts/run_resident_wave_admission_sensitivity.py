"""Historical fixed-context Decode-only admission sensitivity runner.

This runner intentionally makes no cross-system comparison.  Growing-KV or
mixed-window results require separately matched evaluation semantics.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from om3dthermal.serving import (
    MixedPhaseServingCase,
    evaluate_conventional_hbm_resident_wave_admission,
    load_final_dense_e2e_matrix,
)
from om3dthermal.workload import load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs/final_dense_e2e_matrix"
OUTPUT_LENGTHS = (128, 256, 512, 1024, 2048)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value) if isinstance(value, (list, tuple)) else value
                for key, value in row.items()
            })


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(x) for x in value) + "]"
    return str(value)


def main() -> None:
    spec = load_final_dense_e2e_matrix(
        ROOT / "configs/experiment/final_dense_e2e_matrix.yaml")
    registry = load_dense_model_registry(ROOT / spec.model_registry_dir)
    final = json.loads((RUN_DIR / "final_dense_e2e_matrix.json").read_text(
        encoding="utf-8"))
    optimistic = json.loads((
        RUN_DIR / "final_dense_e2e_resident_wave_sensitivity.json").read_text(
            encoding="utf-8"))
    if len(final["rows"]) != 36 or len(optimistic["rows"]) != 12:
        raise RuntimeError("frozen predecessor artifact closure failed")

    final_by_key = {}
    for item in final["rows"]:
        key = (item["model_id"], item["context_length"], item["batch_size"],
               item["prefill_requests"], item["decode_requests"])
        final_by_key.setdefault(key, {})[item["system_id"]] = item
    optimistic_by_key = {
        (item["model_id"], item["context_length"], item["batch_size"],
         item["prefill_requests"], item["requested_decode_requests"]): item
        for item in optimistic["rows"]
    }

    rows: list[dict[str, object]] = []
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for model_id in spec.models:
        model = registry[model_id]
        for point in spec.mixed_points:
            case = MixedPhaseServingCase(
                model_id=model_id, context_length=spec.context_length, **point)
            key = (model_id, spec.context_length, case.batch_size,
                   case.prefill_requests, case.decode_requests)
            systems = final_by_key[key]
            old = optimistic_by_key[key]
            grouped[key] = []
            for generated in OUTPUT_LENGTHS:
                result = evaluate_conventional_hbm_resident_wave_admission(
                    project_root=ROOT, model=model, case=case,
                    generated_output_tokens_per_request=generated)
                if result.optimistic_resident_wave_tokens_per_s != (
                        old["aggregate_decode_tokens_per_s"]):
                    raise RuntimeError("optimistic resident-wave regression changed")
                row = result.model_dump(mode="json")
                row.update({
                    "comparison_scope_status": (
                        "NO_CROSS_SYSTEM_RATIO__FIXED_CONTEXT_DECODE_ONLY"),
                })
                rows.append(row)
                grouped[key].append(row)

    if len(rows) != 60 or len(grouped) != 12:
        raise RuntimeError("admission-aware 60-row closure failed")
    for key, sensitivity in grouped.items():
        if [row["generated_output_tokens_per_request"]
                for row in sensitivity] != list(OUTPUT_LENGTHS):
            raise RuntimeError(f"output-length grid closure failed: {key}")
        if any(row["admission_aware_tokens_per_s"] >
               row["optimistic_resident_wave_tokens_per_s"]
               for row in sensitivity):
            raise RuntimeError(f"fixed-context result exceeds optimistic bound: {key}")

    low_impact_G = {}
    for key, sensitivity in grouped.items():
        low = next((row["generated_output_tokens_per_request"]
                    for row in sensitivity
                    if row["admission_impact_classification"] == "LOW_IMPACT"),
                   None)
        low_impact_G["|".join(str(x) for x in key)] = low
    payload = {
        "experiment_id": "final_dense_e2e_resident_wave_admission_sensitivity",
        "output_length_sensitivity": list(OUTPUT_LENGTHS),
        "sensitivity_row_count": len(rows),
        "workload_key_count": len(grouped),
        "final_matrix_row_count_unchanged": len(final["rows"]),
        "optimistic_resident_wave_row_count_unchanged": len(optimistic["rows"]),
        "new_physical_parameters": False,
        "new_workload_parameter": "generated_output_tokens_per_request",
        "admission_policy": "SERIAL_HOST_TO_HBM_ADMISSION_BETWEEN_WAVES",
        "first_wave_admission": "ZERO_ALREADY_RESIDENT",
        "scheduler_software_overhead_included": False,
        "evaluation_scope": "DECODE_SERVICE_ONLY",
        "context_evolution": "FIXED_CONTEXT_SNAPSHOT",
        "cross_system_comparison_status": (
            "NOT_COMPUTED__REQUIRES_MATCHED_GROWING_KV_OR_MIXED_WINDOW_SCOPE"),
        "first_LOW_IMPACT_G_by_workload": low_impact_G,
        "impact_classification_counts": dict(Counter(
            row["admission_impact_classification"] for row in rows)),
        "rows": rows,
    }
    _write_csv(
        RUN_DIR / "final_dense_e2e_resident_wave_admission_sensitivity.csv",
        rows)
    (RUN_DIR / "final_dense_e2e_resident_wave_admission_sensitivity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Admission-aware resident-wave sensitivity", "",
        "Scheduler robustness analysis only; the final 36-point matrix and the 12-row optimistic resident-wave sensitivity are unchanged.", "",
        "- Output lengths: `128, 256, 512, 1024, 2048` tokens/request",
        "- Admission: `SERIAL_HOST_TO_HBM_ADMISSION_BETWEEN_WAVES`",
        "- First wave: `ZERO_ALREADY_RESIDENT`",
        "- Scheduler software overhead: `NOT_INCLUDED`", "",
        "## B=28 sensitivity", "",
        "| model | P:D | G | limit | waves | optimistic tok/s | aware tok/s | retention | admission GB | admission s | impact |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["batch_size"] != 28:
            continue
        values = (
            row["model_id"],
            f'{row["prefill_requests"]}:{row["requested_decode_requests"]}',
            row["generated_output_tokens_per_request"],
            row["resident_batch_limit"], row["wave_sizes"],
            row["optimistic_resident_wave_tokens_per_s"],
            row["admission_aware_tokens_per_s"],
            row["throughput_retention_vs_optimistic"],
            row["total_admission_GB"], row["total_admission_time_ms"] / 1e3,
            row["admission_impact_classification"],
        )
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    lines += [
        "", "## Interpretation", "",
        "These rows are historical fixed-context Decode-only snapshots. They do not model S+j KV growth and must not be divided into mixed-window or growing-KV M3D/NMP rates. Single-wave workloads have zero admission by their declared initial state.",
    ]
    (RUN_DIR / "final_dense_e2e_resident_wave_admission_sensitivity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "workload_keys": len(grouped),
                      "output_lengths": len(OUTPUT_LENGTHS)}))


if __name__ == "__main__":
    main()
