"""Run the independent cached-prefix long-context workload diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from om3dthermal.serving import (
    CachedPrefixDiagnosticWorkload, WorkspaceExecutionConfig,
    evaluate_cached_prefix_diagnostic,
)
from om3dthermal.workload import load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"configs/experiment/cached_prefix_long_context_diagnostic.yaml"
OUT = ROOT/"runs/cached125k_prefill1k_decode2k_diagnostic"
LABELS = {
    "CONVENTIONAL_HBM_GPU": "Conventional HBM",
    "ORTHOGONAL_M3D_IGZO_MEMORY_ONLY": "M3D-only",
}
COLORS = {"admission": "#8C8C8C", "prefill": "#D89C3D", "decode": "#3976AF"}


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


def _save(fig, stem: str) -> None:
    fig.savefig(OUT/f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT/f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _draw_phase(result) -> None:
    phases = {item.system: item for item in result.phases
              if item.active_batch == next(
                  cap.safe_active_batch for cap in result.capacities
                  if cap.system == item.system)}
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    systems = list(LABELS)
    bottoms = [0.0, 0.0]
    values = {
        "admission": [phases[s].cached_kv_admission_time_s for s in systems],
        "prefill": [phases[s].incremental_prefill_time_s for s in systems],
        "decode": [phases[s].decode_time_s for s in systems],
    }
    for key in ("admission", "prefill", "decode"):
        ax.bar(range(2), values[key], bottom=bottoms, color=COLORS[key],
               label={"admission": "Cached KV admission",
                      "prefill": "Incremental Prefill", "decode": "Decode"}[key])
        bottoms = [bottoms[i]+values[key][i] for i in range(2)]
    for index, system in enumerate(systems):
        point = phases[system]
        ax.text(index, bottoms[index]*1.015,
                f"B={point.active_batch}\nDecode {100*point.decode_fraction:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(bottoms)*1.18)
    ax.set_xticks(range(2), [LABELS[s] for s in systems])
    ax.set_ylabel("Single-wave time (s)")
    ax.set_title("Cached-prefix workload phase breakdown")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.16))
    fig.subplots_adjust(bottom=.27)
    _save(fig, "phase_breakdown")


def _draw_wave_timeline(result) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for y, system in enumerate(LABELS):
        for wave in (item for item in result.waves if item.system == system):
            phases = (
                (wave.admission_start_s,
                 wave.admission_end_s-wave.admission_start_s, "admission"),
                (wave.incremental_prefill_start_s,
                 wave.incremental_prefill_end_s-wave.incremental_prefill_start_s,
                 "prefill"),
                (wave.decode_start_s, wave.decode_end_s-wave.decode_start_s, "decode"),
            )
            for start, width, key in phases:
                ax.broken_barh([(start, width)], (y-.28, .56),
                               facecolors=COLORS[key])
            ax.text((wave.wave_start_s+wave.wave_end_s)/2, y,
                    f"B{wave.batch_size}", ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")
    ax.set_yticks(range(len(LABELS)), list(LABELS.values()))
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_title("Resident-wave serving timeline: 128 requests at t=0")
    ax.grid(axis="x", alpha=.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[key])
               for key in ("admission", "prefill", "decode")]
    ax.legend(handles, ["Cached KV admission", "Incremental Prefill", "Decode"],
              frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.17))
    fig.subplots_adjust(bottom=.25)
    _save(fig, "wave_timeline")


def _draw_completion(result) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    colors = ["#B24A3B", "#3976AF"]
    for color, system in zip(colors, LABELS):
        points = [item for item in result.requests if item.system == system]
        summary = next(item for item in result.summaries if item.system == system)
        ax.step([item.request_id for item in points],
                [item.completion_latency_s for item in points], where="post",
                color=color, linewidth=1.8, label=LABELS[system])
        ax.axhline(summary.p95_completion_latency_s, color=color,
                   linestyle="--", linewidth=1.0, alpha=.75,
                   label=f"{LABELS[system]} P95")
    ax.set_xlim(1, 128)
    ax.set_xlabel("Request ID")
    ax.set_ylabel("Completion latency (s)")
    ax.set_title("Request completion latency under resident-wave queueing")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=2)
    _save(fig, "request_completion_timeline")


def main() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    workspace = WorkspaceExecutionConfig.model_validate(raw["workspace"])
    registry = load_dense_model_registry(ROOT/raw["model_registry_dir"])
    workload = CachedPrefixDiagnosticWorkload.model_validate({
        key: raw[key] for key in (
            "model_id", "cached_history_tokens", "new_prompt_tokens",
            "generation_tokens", "final_context_tokens", "total_requests",
            "arrival_policy", "cached_kv_backing_store")})
    result = evaluate_cached_prefix_diagnostic(
        project_root=ROOT, model=registry[workload.model_id],
        workload=workload, workspace_config=workspace)
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = [item.model_dump() for item in result.summaries]
    for row in summary_rows:
        row["wave_sizes"] = json.dumps(row["wave_sizes"])
        row.update({
            "workload_phase_gate": result.workload_phase_gate,
            "resident_concurrency_gain": result.resident_concurrency_gain,
            "p95_ttft_gain": result.p95_ttft_gain,
            "p95_completion_gain": result.p95_completion_gain,
            "wave_count_reduction": result.wave_count_reduction,
        })
    _write_csv("summary.csv", summary_rows)
    _write_csv("phase_breakdown.csv", [item.model_dump() for item in result.phases])
    _write_csv("capacity.csv", [item.model_dump() for item in result.capacities])
    _write_csv("wave_timeline.csv", [item.model_dump() for item in result.waves])
    _write_csv("request_latency.csv", [item.model_dump() for item in result.requests])
    metrics = result.incremental_prefill_metrics_b1
    checks = {
        "final_context_exactly_128K": (
            workload.cached_history_tokens+workload.new_prompt_tokens
            + workload.generation_tokens == 131072),
        "cached_history_not_recomputed": (
            metrics.linear_ffn_tokens_computed_per_request
            == workload.new_prompt_tokens),
        "new_queries_attend_cached_history": (
            metrics.historical_kv_tokens_attended_per_query
            == workload.cached_history_tokens),
        "incremental_attention_pair_count_exact": (
            metrics.attention_pairs_per_request
            == workload.new_prompt_tokens*workload.cached_history_tokens
            + workload.new_prompt_tokens*(workload.new_prompt_tokens+1)//2),
        "incremental_prefill_only_appends_new_KV": (
            metrics.final_kv_cache_bytes
            == metrics.cached_kv_bytes_before+metrics.new_kv_write_bytes
            and not metrics.cached_kv_rewritten),
        "decode_start_context_129024": (
            workload.decode_start_context_tokens == 129024),
        "decode_exactly_2048_steps_to_131072": (
            workload.decode_start_context_tokens+workload.generation_tokens
            == workload.final_context_tokens),
        "safe_batch_uses_final_context_and_workspace": all(
            item.limiting_phase == "DECODE_FINAL" for item in result.capacities),
        "queued_requests_do_not_consume_local_capacity": all(
            wave.batch_size <= next(cap.safe_active_batch for cap in result.capacities
                                    if cap.system == wave.system)
            for wave in result.waves),
        "NMP_disabled": not result.nmp_enabled,
        "recurring_host_offload_disabled": not result.recurring_host_offload_enabled,
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    audit = {
        "commit": commit,
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "workload": workload.model_dump(),
        "incremental_prefill_b1": metrics.model_dump(),
        "admission_semantics": result.admission_semantics,
        "workload_phase_gate": result.workload_phase_gate,
        "checks": checks,
        "thermal": None,
    }
    (OUT/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    capacity = {item.system: item for item in result.capacities}
    md = [
        "# Cached-prefix workload diagnostic audit", "",
        f"- Commit: `{commit}`",
        "- Workload: `128000 cached + 1024 incremental Prefill + 2048 Decode = 131072`",
        f"- Phase gate: `{result.workload_phase_gate}`",
        f"- Admission: `{result.admission_semantics}`", "",
        "| System | Safe active B | Capacity margin (GiB) | Limiting phase |",
        "|---|---:|---:|---|",
    ]
    for system in LABELS:
        item = capacity[system]
        md.append(f"| {LABELS[system]} | {item.safe_active_batch} | "
                  f"{item.capacity_margin_bytes/2**30:.2f} | {item.limiting_phase} |")
    md.extend(["", "## Checks", ""])
    md.extend(f"- {name}: `{'PASS' if passed else 'FAIL'}`"
              for name, passed in checks.items())
    (OUT/"audit.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    _draw_phase(result)
    if result.workload_phase_gate == "PASS":
        _draw_wave_timeline(result)
        _draw_completion(result)
    print(json.dumps({
        "phase_gate": result.workload_phase_gate,
        "safe_batches": {item.system: item.safe_active_batch
                         for item in result.capacities},
        "p95_ttft_gain": result.p95_ttft_gain,
        "p95_completion_gain": result.p95_completion_gain,
        "wave_count_reduction": result.wave_count_reduction,
    }))


if __name__ == "__main__":
    main()
