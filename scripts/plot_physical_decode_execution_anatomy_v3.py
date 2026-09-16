"""Compare one frozen Decode layer across HBM and the three IOM3D paths."""

from __future__ import annotations

import csv
import gc
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_physical_decode_execution_anatomy_v2 as v2  # noqa: E402
from run_formal_long_context_v3 import POINTS, inputs  # noqa: E402
from om3dthermal.serving.decode_policy import DecodePolicyModel  # noqa: E402
from om3dthermal.serving.workload_matrix import setup  # noqa: E402
from om3dthermal.workload import evaluate_llm_decode  # noqa: E402


FORMAL = ROOT / "runs" / "formal_long_context_v3"
OUT = ROOT / "runs" / "physical_decode_execution_anatomy_v3"
MODEL = "Qwen2.5-32B"
CONTEXT = "LC64K"
BATCH = 8
DECODE_CONTEXT = 64128
LAYER = 0
TAG = f"{MODEL}_{CONTEXT}_B{BATCH}"

PATHS = (
    ("HBM_BEST", "HBM-GPU"),
    ("M3D_GPU", "M3D-GPU"),
    ("M3D_NMP_UNIFORM", "DNS"),
    ("M3D_NMP_CPA", "CPA"),
)

PATH_LABEL = dict(PATHS)
PLACEMENTS = {
    "M3D_GPU": "GPU_PORT_BALANCED",
    "M3D_NMP_UNIFORM": "UNIFORM_STRIPING",
}
POLICIES = {
    "M3D_GPU": "NO_NMP",
    "M3D_NMP_UNIFORM": "MAC_NMP",
}

OPERATORS_GPU = tuple(operator for operator in v2.OPERATORS if operator != "AV_REDUCTION")
SHORT_LABELS = {
    "ATTENTION_QK": "QK",
    "SOFTMAX": "S",
    "ATTENTION_AV": "AV",
    "AV_REDUCTION": "R",
    "FFN_GATE": "G",
    "FFN_UP": "U",
    "FFN_DOWN": "D",
}

TIMELINE_CATEGORY = {
    "ARRAY": "HBM/Array",
    "HBM_MEMORY": "HBM/Array",
    "MAC": "MAC",
    "LOCAL_FABRIC": "Fabric",
    "INTER_REGION_NOC": "NoC",
    "EXTERNAL_BOUNDARY": "Boundary",
    "GPU_COMPUTE": "GPU",
}
CATEGORY_COLORS = {
    "HBM/Array": "#5C87B2",
    "MAC": "#C77B47",
    "Fabric": "#8A70A8",
    "NoC": "#A789BD",
    "Boundary": "#D0A53E",
    "GPU": "#4D9584",
}


def candidate(path: str) -> dict:
    return json.loads(
        (FORMAL / "candidates" / f"{TAG}_{path}.json").read_text(encoding="utf-8")
    )


def checkpoint(path: str) -> tuple[Path, dict]:
    source = FORMAL / "checkpoints" / f"{TAG}_{path}.jsonl"
    with source.open(encoding="utf-8") as stream:
        return source, json.loads(next(stream))


def validate_step(actual: dict, expected: dict) -> None:
    for key in ("context", "latency_s", "boundary_bytes", "local_array_bytes"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-15)
    for key, value in expected["component_sums"].items():
        np.testing.assert_allclose(actual["component_sums"][key], value, rtol=1e-12, atol=1e-15)


def extract_exact_timeline(stages: list[dict], path: str) -> list[dict]:
    allowed = OPERATORS_GPU if path == "M3D_GPU" else v2.OPERATORS
    elapsed = 0.0
    selected = []
    for stage_index, stage in enumerate(stages):
        start = elapsed
        elapsed += float(stage["latency_s"])
        if stage["layer"] != LAYER or stage["operator"] not in allowed:
            continue
        components = {
            resource: float(stage["components"].get(resource, 0.0))
            for resource in v2.RESOURCES
        }
        dominant = max(v2.RESOURCES, key=components.get)
        selected.append(
            {
                "path": path,
                "path_label": PATH_LABEL[path],
                "operator": stage["operator"],
                "display_operator": SHORT_LABELS.get(stage["operator"], stage["operator"]),
                "stage_index": stage_index,
                "executor": stage["executor"],
                "start_s": start,
                "end_s": elapsed,
                "duration_s": float(stage["latency_s"]),
                "dominant_resource": dominant,
                "dominant_category": TIMELINE_CATEGORY[dominant],
                "exact_or_reconstructed": "exact",
            }
        )
    assert tuple(row["operator"] for row in selected) == allowed
    origin = selected[0]["start_s"]
    for row in selected:
        row["start_s"] -= origin
        row["end_s"] -= origin
    return selected


def replay_iom3d(path: str) -> tuple[list[dict], dict]:
    index = POINTS.index((MODEL, CONTEXT, BATCH))
    _, _, _, spec, workload = inputs(index)
    external_cap = setup(ROOT)[4] if path == "M3D_GPU" else None
    engine = DecodePolicyModel(
        spec.decode_input(
            batch_size=BATCH,
            context_length=workload.history + workload.prompt + workload.generated,
        ),
        project_root=ROOT,
        placement_policy=PLACEMENTS[path],
        record_energy=False,
        decode_start_context=workload.history + workload.prompt,
        external_bandwidth_cap=external_cap,
    )
    source, expected = checkpoint(path)
    actual = engine.step(DECODE_CONTEXT, POLICIES[path], include_stages=True)
    validate_step(actual, expected)
    timeline = extract_exact_timeline(actual["stages"], path)
    provenance = {
        "source_checkpoint": str(source.relative_to(ROOT)),
        "replay_used": True,
        "validation_status": "PASS",
        "optimizer_runs": 0,
        "placement": PLACEMENTS[path],
    }
    del engine, actual
    gc.collect()
    return timeline, provenance


def replay_cpa() -> tuple[list[dict], dict, list[dict]]:
    replay = v2.replay_case(MODEL, CONTEXT, BATCH)
    timeline = []
    for stage in replay["stages"]:
        timeline.append(
            {
                "path": "M3D_NMP_CPA",
                "path_label": "CPA",
                "operator": stage["operator"],
                "display_operator": SHORT_LABELS.get(stage["operator"], stage["operator"]),
                "stage_index": stage["stage_index"],
                "executor": stage["executor"],
                "start_s": stage["start_s"],
                "end_s": stage["end_s"],
                "duration_s": stage["latency_s"],
                "dominant_resource": stage["dominant_resource"],
                "dominant_category": TIMELINE_CATEGORY[stage["dominant_resource"]],
                "exact_or_reconstructed": "exact",
            }
        )
    origin = timeline[0]["start_s"]
    for row in timeline:
        row["start_s"] -= origin
        row["end_s"] -= origin
    source = FORMAL / "checkpoints" / f"{TAG}_M3D_NMP_CPA.jsonl"
    provenance = {
        "source_checkpoint": str(source.relative_to(ROOT)),
        "replay_used": True,
        "validation_status": "PASS",
        "optimizer_runs": 0,
        "placement": "CRITICAL_PATH_AWARE_FROZEN_CACHE",
    }
    matrix = v2.matrix_rows(replay)
    return timeline, provenance, matrix


def hbm_reference() -> tuple[list[dict], dict, dict]:
    result = candidate("HBM_BEST")
    assert result["selected_HBM_policy"] == "HBM_RESIDENT_WAVE"
    assert result["wave_sizes"] == [4, 4]
    active_batch = result["wave_sizes"][0]
    active_step_s = float(result["waves"][0]["steps"][0]["latency_s"])

    index = POINTS.index((MODEL, CONTEXT, BATCH))
    _, _, _, spec, _ = inputs(index)
    _, _, platform, hbm, _ = setup(ROOT)
    with (ROOT / "runs/no_nmp_geometry_sensitivity_v2/thermal_limits.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        bandwidth = float(
            next(
                row
                for row in csv.DictReader(stream)
                if row["architecture"] == "conventional_hbm_2x1"
            )["Bthermal_TBps"]
        ) * 1e12
    metrics = evaluate_llm_decode(
        spec.decode_input(batch_size=active_batch, context_length=DECODE_CONTEXT)
    )
    compute_s = (
        metrics.flops_per_token
        * active_batch
        / platform.gpu_compute_power.peak_compute_BF16_dense_flops_per_s
    )
    memory_s = (
        (metrics.read_bytes_per_token + metrics.write_bytes_per_token)
        * active_batch
        / bandwidth
    )
    np.testing.assert_allclose(max(compute_s, memory_s), active_step_s, rtol=1e-12, atol=1e-15)
    dominant = "HBM_MEMORY" if memory_s >= compute_s else "GPU_COMPUTE"

    mean_layer_s = active_step_s / spec.n_layers
    timeline = [
        {
            "path": "HBM_BEST",
            "path_label": "HBM-GPU",
            "operator": "AGGREGATE_LAYER_REFERENCE",
            "display_operator": "Aggregate active-wave layer",
            "stage_index": "",
            "executor": "GPU",
            "start_s": 0.0,
            "end_s": mean_layer_s,
            "duration_s": mean_layer_s,
            "dominant_resource": dominant,
            "dominant_category": TIMELINE_CATEGORY[dominant],
            "exact_or_reconstructed": "aggregate-only",
        }
    ]
    provenance = {
        "source_checkpoint": str(
            (FORMAL / "candidates" / f"{TAG}_HBM_BEST.json").relative_to(ROOT)
        ),
        "replay_used": False,
        "validation_status": "PASS_AGGREGATE_STEP_CLOSURE",
        "optimizer_runs": 0,
        "placement": "NOT_APPLICABLE",
    }
    details = {
        "selected_HBM_policy": result["selected_HBM_policy"],
        "safe_resident_batch": result["safe_resident_batch"],
        "wave_sizes": result["wave_sizes"],
        "active_wave_batch": active_batch,
        "active_decode_step_s": active_step_s,
        "aggregate_mean_layer_s": mean_layer_s,
        "aggregate_compute_s": compute_s,
        "aggregate_memory_s": memory_s,
        "aggregate_dominant_resource": dominant,
    }
    return timeline, provenance, details


def provenance_rows(timelines: dict[str, list[dict]], sources: dict[str, dict]) -> list[dict]:
    rows = []
    for path, _ in PATHS:
        source = sources[path]
        for item in timelines[path]:
            rows.append(
                {
                    "path": path,
                    "operator": item["operator"],
                    "start_us": item["start_s"] * 1e6,
                    "end_us": item["end_s"] * 1e6,
                    "duration_us": item["duration_s"] * 1e6,
                    "exact_or_reconstructed": item["exact_or_reconstructed"],
                    "source_checkpoint": source["source_checkpoint"],
                    "replay_used": source["replay_used"],
                    "validation_status": source["validation_status"],
                }
            )
    return rows


def flattened_timeline(timelines: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for path, _ in PATHS:
        for order, item in enumerate(timelines[path]):
            rows.append(
                {
                    "path": path,
                    "path_label": item["path_label"],
                    "order": order,
                    "operator": item["operator"],
                    "display_operator": item["display_operator"],
                    "executor": item["executor"],
                    "start_us": item["start_s"] * 1e6,
                    "end_us": item["end_s"] * 1e6,
                    "duration_us": item["duration_s"] * 1e6,
                    "dominant_resource": item["dominant_resource"],
                    "dominant_category": item["dominant_category"],
                    "timeline_semantics": item["exact_or_reconstructed"],
                }
            )
    return rows


def latency_summary(timelines: dict[str, list[dict]], hbm: dict) -> list[dict]:
    latencies = {
        path: rows[-1]["end_s"] - rows[0]["start_s"]
        for path, rows in timelines.items()
    }
    reference = latencies["HBM_BEST"]
    return [
        {
            "path": path,
            "path_label": label,
            "layer_latency_us": latencies[path] * 1e6,
            "speedup_vs_HBM_layer": reference / latencies[path],
            "selected_HBM_policy": hbm["selected_HBM_policy"],
            "timeline_semantics": timelines[path][0]["exact_or_reconstructed"],
        }
        for path, label in PATHS
    ]


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": v2.TNR.get_name(),
            "font.size": 8.2,
            "axes.linewidth": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "svg.fonttype": "none",
            "svg.hashsalt": "physical-decode-execution-anatomy-v3",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def apply_fonts(axis) -> None:
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        size = tick.get_fontsize()
        tick.set_fontproperties(v2.TNR)
        tick.set_fontsize(size)


def plot(timelines: dict[str, list[dict]], summary: list[dict], matrix: list[dict]) -> None:
    configure_plot()
    fig = plt.figure(figsize=(7.25, 3.35))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.72, 1.0), wspace=0.31)
    axis = fig.add_subplot(grid[0, 0])
    heat = fig.add_subplot(grid[0, 1])
    y_positions = {path: 3 - index for index, (path, _) in enumerate(PATHS)}
    latency_lookup = {row["path"]: float(row["layer_latency_us"]) for row in summary}
    max_latency = max(latency_lookup.values())

    for path, _ in PATHS:
        y = y_positions[path]
        for row in timelines[path]:
            start_us = row["start_s"] * 1e6
            width_us = row["duration_s"] * 1e6
            axis.broken_barh(
                [(start_us, width_us)],
                (y - 0.29, 0.58),
                facecolor=CATEGORY_COLORS[row["dominant_category"]],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            if path == "HBM_BEST":
                label = "Aggregate"
            else:
                label = row["display_operator"]
            if row["operator"] not in {"Q", "K", "V"} and width_us >= 7:
                axis.text(
                    start_us + width_us / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    color="white",
                    fontproperties=v2.font(5.2, bold=True),
                    clip_on=True,
                    zorder=4,
                )

        if path != "HBM_BEST":
            qkv = timelines[path][:3]
            assert [row["operator"] for row in qkv] == ["Q", "K", "V"]
            qkv_start = qkv[0]["start_s"] * 1e6
            qkv_end = qkv[-1]["end_s"] * 1e6
            axis.text(
                (qkv_start + qkv_end) / 2,
                y,
                "QKV",
                ha="center",
                va="center",
                color="white",
                fontproperties=v2.font(4.9, bold=True),
                clip_on=True,
                zorder=4,
            )

        if path != "HBM_BEST":
            for left, right in zip(timelines[path], timelines[path][1:]):
                if left["executor"] != right["executor"]:
                    x = right["start_s"] * 1e6
                    axis.plot(
                        (x, x),
                        (y - 0.34, y + 0.34),
                        color="#333333",
                        linestyle=(0, (2, 2)),
                        linewidth=0.55,
                        zorder=2,
                    )
        axis.text(
            latency_lookup[path] + max_latency * 0.018,
            y,
            f"{latency_lookup[path]:.1f} µs",
            ha="left",
            va="center",
            fontproperties=v2.font(7.0, bold=True),
        )

    axis.set_yticks(
        [y_positions[path] for path, _ in PATHS],
        [label for _, label in PATHS],
    )
    axis.set_xlim(0, max_latency * 1.18)
    axis.set_ylim(-0.58, 3.58)
    axis.set_xlabel("Time from layer start (µs)", fontproperties=v2.font(8.4, bold=True))
    axis.set_title("(a) Cross-configuration Decode execution", fontproperties=v2.font(8.8, bold=True), pad=7)
    axis.grid(axis="x", color="#D0D0D0", linestyle=(0, (3, 2)), linewidth=0.45, zorder=0)
    axis.text(
        max_latency * 0.01,
        -0.48,
        "S: Softmax   R: AV Reduce   G/U/D: Gate/Up/Down",
        ha="left",
        va="center",
        fontproperties=v2.font(6.2),
    )
    apply_fonts(axis)

    handles = [
        Patch(facecolor=color, edgecolor="#303030", linewidth=0.4, label=category)
        for category, color in CATEGORY_COLORS.items()
    ]
    handles.append(
        Line2D([0], [0], color="#333333", linestyle=(0, (2, 2)), linewidth=0.6,
               label="GPU–NMP handoff")
    )
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        frameon=False,
        prop=v2.font(6.8, bold=True),
        handlelength=1.25,
        columnspacing=0.75,
    )

    cpa = timelines["M3D_NMP_CPA"]
    lookup = {(int(row["order"]), row["resource"]): row for row in matrix}
    values = np.asarray(
        [
            [float(lookup[(index, resource)]["normalized_service"]) for resource in v2.RESOURCES]
            for index in range(len(cpa))
        ]
    )
    image = heat.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto", interpolation="nearest")
    heat.set_yticks(range(len(cpa)), [v2.DISPLAY_NAMES.get(row["operator"], row["operator"]) for row in cpa])
    heat.set_xticks(range(6), ["Array+MIV", "MAC", "Fabric", "NoC", "Boundary", "GPU"],
                    rotation=32, ha="right", rotation_mode="anchor")
    heat.tick_params(length=0, labelsize=7.1)
    for index, row in enumerate(cpa):
        column = v2.RESOURCES.index(row["dominant_resource"])
        heat.add_patch(Rectangle((column - 0.49, index - 0.49), 0.98, 0.98,
                                 fill=False, edgecolor="black", linewidth=0.9))
    heat.set_title("(b) CPA resource-service decomposition", fontproperties=v2.font(8.8, bold=True), pad=7)
    apply_fonts(heat)
    colorbar = fig.colorbar(image, ax=heat, fraction=0.046, pad=0.035, ticks=(0, 0.5, 1))
    colorbar.set_label("Normalized service", fontproperties=v2.font(7.8, bold=True), labelpad=3)
    apply_fonts(colorbar.ax)

    fig.subplots_adjust(left=0.075, right=0.94, top=0.89, bottom=0.25)
    for extension in ("svg", "pdf"):
        metadata = {"Date": None} if extension == "svg" else {"CreationDate": None, "ModDate": None}
        fig.savefig(OUT / f"figure.{extension}", metadata=metadata)
    svg = OUT / "figure.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    plt.close(fig)


def validate(timelines: dict[str, list[dict]], summary: list[dict], matrix: list[dict]) -> None:
    assert set(timelines) == {path for path, _ in PATHS}
    for path, rows in timelines.items():
        assert rows[0]["start_s"] == 0.0
        assert all(left["start_s"] < right["start_s"] for left, right in zip(rows, rows[1:]))
        latency = rows[-1]["end_s"] - rows[0]["start_s"]
        recorded = next(row for row in summary if row["path"] == path)
        assert math.isclose(latency * 1e6, float(recorded["layer_latency_us"]), rel_tol=1e-12)
    grouped = {}
    for row in matrix:
        grouped.setdefault(row["operator"], []).append(row)
    for rows in grouped.values():
        maximum = max(float(row["normalized_service"]) for row in rows)
        dominant = [row for row in rows if row["dominant"]]
        assert math.isclose(maximum, 1.0, rel_tol=0, abs_tol=1e-15)
        assert len(dominant) == 1
        assert math.isclose(float(dominant[0]["normalized_service"]), 1.0,
                            rel_tol=0, abs_tol=1e-15)


def write_docs(hbm: dict, summary: list[dict], sources: dict[str, dict]) -> None:
    caption = (
        "Physical Decode execution anatomy for Qwen2.5-32B at 64K context and B=8. "
        "(a) Layer-level operator critical-path comparison across HBM-GPU, M3D-GPU, DNS, "
        "and CPA using a shared absolute-time axis. The HBM bar is an aggregate-only "
        "mean-layer reference from the canonical active-wave Decode step; the other rows "
        "use exact operator intervals. Blocks are colored by their dominant execution/service "
        "resource, and dashed markers indicate GPU–NMP dependency handoffs. The HBM row shows "
        "active-wave B=4 layer execution; inter-wave queuing is accounted for in E2E latency "
        "but is not shown here. (b) CPA per-operator physical-resource service normalized as "
        "T_hat(o,r)=T(o,r)/max_r' T(o,r'); black boxes indicate the dominant resource."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    latency_lines = "\n".join(
        f"- {row['path_label']}: {float(row['layer_latency_us']):.3f} us, "
        f"{float(row['speedup_vs_HBM_layer']):.3f}x vs the HBM aggregate mean-layer reference."
        for row in summary
    )
    source_lines = "\n".join(
        f"- {PATH_LABEL[path]}: {('aggregate-only' if path == 'HBM_BEST' else 'exact')}; "
        f"validation `{source['validation_status']}`; optimizer runs {source['optimizer_runs']}."
        for path, source in sources.items()
    )
    readme = f"""# Physical Decode Execution Anatomy v3

Fixed workload: Qwen2.5-32B, LC64K, B=8, Decode step 0 (context 64128), layer 0, nominal 1.0 GHz CPA.

## HBM policy and provenance

`HBM_BEST` selects `{hbm['selected_HBM_policy']}` with safe resident batch {hbm['safe_resident_batch']} and waves `{hbm['wave_sizes']}`. The HBM evaluator has no operator scheduler. Its row is therefore an aggregate-only active-wave reference: the canonical first Decode-step latency for B={hbm['active_wave_batch']} divided by 64 layers. It is not split into invented operator intervals. Request-level wave queuing remains represented only in the formal E2E metrics.

{source_lines}

M3D-GPU and DNS are deterministic single-step replays using their canonical placement policies and are checked against their frozen checkpoints. CPA is loaded from the existing nominal-frequency serialized cache; the CPA optimizer is not rerun. All four rows align their layer reference to x=0 and retain absolute microsecond durations without per-row normalization.

## Representative layer latency

{latency_lines}

These ratios are single-layer anatomy ratios, not end-to-end throughput speedups.

The right panel retains the v2 CPA matrix exactly: `T_norm(o,r)=T(o,r)/max_r T(o,r)`. Each row maximum is 1 and the black outline marks `argmax_r`. `AV_REDUCTION` remains `AV Reduce` because it is an independently scheduled GPU stage.

No formal benchmark, thermal solve, frequency sweep, or placement optimizer is run, and no canonical formal result is modified.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = v2.formal_hashes()

    timelines = {}
    sources = {}
    timelines["HBM_BEST"], sources["HBM_BEST"], hbm = hbm_reference()
    for path in ("M3D_GPU", "M3D_NMP_UNIFORM"):
        timelines[path], sources[path] = replay_iom3d(path)
    timelines["M3D_NMP_CPA"], sources["M3D_NMP_CPA"], matrix = replay_cpa()

    summary = latency_summary(timelines, hbm)
    validate(timelines, summary, matrix)
    v2.write_csv(OUT / "timeline_all_paths.csv", flattened_timeline(timelines))
    v2.write_csv(OUT / "timeline_provenance.csv", provenance_rows(timelines, sources))
    v2.write_csv(OUT / "layer_latency_summary.csv", summary)
    v2.write_csv(OUT / "operator_resource_matrix_cpa.csv", matrix)
    v2.write_json(
        OUT / "selected_case.json",
        {
            "model": MODEL,
            "context": CONTEXT,
            "batch": BATCH,
            "decode_step_zero_based": 0,
            "decode_context": DECODE_CONTEXT,
            "layer_zero_based": LAYER,
            "CPA_frequency_GHz": 1.0,
            "selected_HBM_policy": hbm["selected_HBM_policy"],
            "HBM_timeline_semantics": "aggregate-only",
            "M3D_timeline_semantics": "exact",
            "per_row_time_normalization": False,
            "benchmark_runs": 0,
            "thermal_runs": 0,
            "optimizer_runs": 0,
            "HBM_details": hbm,
        },
    )
    write_docs(hbm, summary, sources)
    plot(timelines, summary, matrix)

    after = v2.formal_hashes()
    assert before == after
    v2.write_json(
        OUT / "formal_v3_preservation.json",
        {"status": "BYTE_IDENTICAL", "file_count": len(before), "sha256": before},
    )
    print(json.dumps({"HBM": hbm, "latencies": summary}, indent=2))
    print(f"formal_long_context_v3 byte-identical: {len(before)} files")


if __name__ == "__main__":
    main()
