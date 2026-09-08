"""Run the 12-row resident-wave scheduler sensitivity without thermal."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from om3dthermal.serving import (
    MixedPhaseServingCase,
    evaluate_conventional_hbm_resident_wave_decode,
    load_final_dense_e2e_matrix,
)
from om3dthermal.workload import load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs/final_dense_e2e_matrix"
FINAL_JSON = RUN_DIR / "final_dense_e2e_matrix.json"


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
    with FINAL_JSON.open("r", encoding="utf-8") as stream:
        final = json.load(stream)
    final_rows = final["rows"]
    if len(final_rows) != 36:
        raise RuntimeError("existing final matrix must remain exactly 36 rows")

    rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    keys: set[tuple[object, ...]] = set()
    for model_id in spec.models:
        model = registry[model_id]
        for point in spec.mixed_points:
            case = MixedPhaseServingCase(
                model_id=model_id, context_length=spec.context_length, **point)
            key = (model_id, spec.context_length, case.batch_size,
                   case.prefill_requests, case.decode_requests)
            if key in keys:
                raise RuntimeError(f"duplicate sensitivity workload key: {key}")
            keys.add(key)
            result = evaluate_conventional_hbm_resident_wave_decode(
                project_root=ROOT, model=model, case=case)
            row = result.model_dump(mode="json")
            rows.append(row)

            indexed = {
                item["system_id"]: item for item in final_rows
                if (item["model_id"], item["context_length"], item["batch_size"],
                    item["prefill_requests"], item["decode_requests"]) == key
            }
            if len(indexed) != 3:
                raise RuntimeError(f"existing final matrix key is incomplete: {key}")
            wave_rate = result.aggregate_decode_tokens_per_s
            memory_rate = indexed[
                "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY"]["decode_tokens_per_s"]
            nmp_rate = indexed["IOM3D_FEOL_NMP"]["decode_tokens_per_s"]
            comparisons.append({
                "model_id": model_id,
                "context_length": spec.context_length,
                "batch_size": case.batch_size,
                "prefill_requests": case.prefill_requests,
                "decode_requests": case.decode_requests,
                "host_offload_decode_tokens_per_s": indexed[
                    "CONVENTIONAL_HBM_GPU"]["decode_tokens_per_s"],
                "resident_wave_decode_tokens_per_s": wave_rate,
                "m3d_memory_only_decode_tokens_per_s": memory_rate,
                "iom3d_feol_nmp_decode_tokens_per_s": nmp_rate,
                "m3d_memory_only_speedup_vs_resident_wave": (
                    None if memory_rate is None else memory_rate / wave_rate),
                "iom3d_feol_nmp_speedup_vs_resident_wave": (
                    None if nmp_rate is None else nmp_rate / wave_rate),
            })

    if len(rows) != 12 or len(keys) != 12:
        raise RuntimeError("resident-wave 12-row closure failed")
    payload = {
        "experiment_id": "final_dense_e2e_resident_wave_sensitivity",
        "final_matrix_row_count_unchanged": len(final_rows),
        "sensitivity_row_count": len(rows),
        "scheduler_sensitivity_not_architecture": True,
        "resident_wave_model": (
            "CAPACITY_CONSTRAINED_RUN_TO_COMPLETION_OPTIMISTIC_BOUND"),
        "new_physical_parameters": False,
        "output_tokens_parameter_added": False,
        "output_length_cancels_in_equal_length_wave_throughput": True,
        "swap_admission_overhead_included": False,
        "rows": rows,
        "comparisons": comparisons,
    }
    _write_csv(
        RUN_DIR / "final_dense_e2e_resident_wave_sensitivity.csv", rows)
    (RUN_DIR / "final_dense_e2e_resident_wave_sensitivity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Resident-wave scheduler sensitivity", "",
        "This is a scheduler sensitivity, not a fourth architecture.", "",
        "- Sensitivity rows: 12",
        "- Existing final matrix rows: 36 (unchanged)",
        "- `OUTPUT_LENGTH_CANCELS_IN_EQUAL_LENGTH_WAVE_THROUGHPUT = YES`",
        "- `SWAP_ADMISSION_OVERHEAD_INCLUDED = NO`", "",
        "## B=28 throughput comparison", "",
        "| model | P:D | resident limit | waves | host-offload tok/s | resident-wave tok/s | M3D memory-only tok/s | NMP tok/s | M3D / wave | NMP / wave |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_key = {
        (row["model_id"], row["batch_size"], row["prefill_requests"]): row
        for row in rows
    }
    for comparison in comparisons:
        if comparison["batch_size"] != 28:
            continue
        row = by_key[(comparison["model_id"], 28,
                      comparison["prefill_requests"])]
        values = (
            comparison["model_id"],
            f'{comparison["prefill_requests"]}:{comparison["decode_requests"]}',
            row["resident_batch_limit"], row["wave_sizes"],
            comparison["host_offload_decode_tokens_per_s"],
            comparison["resident_wave_decode_tokens_per_s"],
            comparison["m3d_memory_only_decode_tokens_per_s"],
            comparison["iom3d_feol_nmp_decode_tokens_per_s"],
            comparison["m3d_memory_only_speedup_vs_resident_wave"],
            comparison["iom3d_feol_nmp_speedup_vs_resident_wave"],
        )
        lines.append("| " + " | ".join(_fmt(x) for x in values) + " |")
    lines += [
        "", "### Capacity pressure has two possible conventional responses", "",
        "- **HOST_OFFLOAD:** maintain concurrency, pay host traffic.",
        "- **RESIDENT_WAVE:** avoid per-step host traffic, pay reduced resident concurrency and serialized waves.",
        "- **M3D:** maintain high concurrency while keeping all KV local.",
        "- **NMP:** additionally avoid bulk local-memory-to-GPU traffic.", "",
        "Insufficient local capacity forces a trade-off between host-offload traffic and reduced resident concurrency. Resident-wave results exclude KV swap/admission, scheduler overhead, and queue waiting and are therefore an optimistic Conventional-HBM bound.",
    ]
    (RUN_DIR / "final_dense_e2e_resident_wave_sensitivity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"sensitivity_rows": len(rows),
                      "final_matrix_rows": len(final_rows)}))


if __name__ == "__main__":
    main()
