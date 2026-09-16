"""Build the v2 Physical Decode Execution Anatomy figure from frozen CPA state."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import formal_parallel_runtime as shared  # noqa: E402


FORMAL = ROOT / "runs" / "formal_long_context_v3"
OUT = ROOT / "runs" / "physical_decode_execution_anatomy_v2"
CACHE = Path("F:/om3dthermal_cache/cpa_feol_frequency_sweep_v3")

CANDIDATES = (
    ("Qwen2.5-32B", "LC64K", 8),
    ("Qwen2.5-32B", "LC64K", 32),
    ("Qwen2.5-32B", "LC126K", 8),
)

OPERATORS = (
    "Q",
    "K",
    "V",
    "ATTENTION_QK",
    "SOFTMAX",
    "ATTENTION_AV",
    "AV_REDUCTION",
    "O",
    "FFN_GATE",
    "FFN_UP",
    "FFN_DOWN",
)

DISPLAY_NAMES = {
    "ATTENTION_QK": "QK",
    "ATTENTION_AV": "AV",
    "AV_REDUCTION": "AV Reduce",
    "FFN_GATE": "Gate",
    "FFN_UP": "Up",
    "FFN_DOWN": "Down",
}

RESOURCES = (
    "ARRAY",
    "MAC",
    "LOCAL_FABRIC",
    "INTER_REGION_NOC",
    "EXTERNAL_BOUNDARY",
    "GPU_COMPUTE",
)

RESOURCE_LABELS = {
    "ARRAY": "Array+MIV",
    "MAC": "MAC",
    "LOCAL_FABRIC": "Fabric",
    "INTER_REGION_NOC": "NoC",
    "EXTERNAL_BOUNDARY": "Boundary",
    "GPU_COMPUTE": "GPU",
}

TIMELINE_CATEGORIES = {
    "ARRAY": "Array+MIV",
    "MAC": "MAC",
    "LOCAL_FABRIC": "Fabric/NoC",
    "INTER_REGION_NOC": "Fabric/NoC",
    "EXTERNAL_BOUNDARY": "Boundary",
    "GPU_COMPUTE": "GPU",
}

CATEGORY_COLORS = {
    "Array+MIV": "#5C87B2",
    "MAC": "#C77B47",
    "Fabric/NoC": "#8A70A8",
    "Boundary": "#D0A53E",
    "GPU": "#4D9584",
}

TNR_PATH = Path(r"C:\Windows\Fonts\times.ttf")
TNR_BOLD_PATH = Path(r"C:\Windows\Fonts\timesbd.ttf")
if not TNR_PATH.is_file() or not TNR_BOLD_PATH.is_file():
    raise FileNotFoundError("Windows Times New Roman font files are required")
TNR = FontProperties(fname=str(TNR_PATH))
TNR_BOLD = FontProperties(fname=str(TNR_BOLD_PATH))


def font(size: float, *, bold: bool = False) -> FontProperties:
    value = (TNR_BOLD if bold else TNR).copy()
    value.set_size(size)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def formal_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(FORMAL.rglob("*"))
        if path.is_file()
    }


def case_tag(model: str, context: str, batch: int) -> str:
    return f"{model}_{context}_B{batch}"


def read_checkpoint(tag: str) -> dict:
    path = FORMAL / "checkpoints" / f"{tag}_M3D_NMP_CPA.jsonl"
    with path.open(encoding="utf-8") as stream:
        return json.loads(next(stream))


def replay_case(model: str, context: str, batch: int) -> dict:
    """Replay one frozen step with stages; never place or optimize."""
    tag = case_tag(model, context, batch)
    cache = CACHE / tag
    ready = json.loads((cache / "ready.json").read_text(encoding="utf-8"))
    expected = read_checkpoint(tag)

    shared.initialize(cache)
    engine = shared.ENGINE
    assert math.isclose(engine.floorplan.config["clock_hz"], 1.0e9)
    actual = engine.step(expected["context"], "MAC_NMP", include_stages=True)

    for key in ("context", "latency_s", "boundary_bytes", "local_array_bytes"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-15)
    for key, value in expected["component_sums"].items():
        np.testing.assert_allclose(actual["component_sums"][key], value, rtol=1e-12, atol=1e-15)

    stages = []
    elapsed = 0.0
    layer_origin = None
    seen = set()
    for stage_index, stage in enumerate(actual["stages"]):
        start = elapsed
        elapsed += stage["latency_s"]
        if stage["layer"] == 0 and layer_origin is None:
            layer_origin = start
        if stage["layer"] != 0 or stage["operator"] not in OPERATORS:
            continue
        operator = stage["operator"]
        if operator in seen:
            raise AssertionError(f"Duplicate selected operator in layer 0: {operator}")
        seen.add(operator)
        components = {name: float(stage["components"].get(name, 0.0)) for name in RESOURCES}
        dominant = max(RESOURCES, key=components.get)
        stages.append(
            {
                "stage_index": stage_index,
                "operator": operator,
                "display_operator": DISPLAY_NAMES.get(operator, operator),
                "executor": stage["executor"],
                "start_s": start - layer_origin,
                "end_s": elapsed - layer_origin,
                "latency_s": stage["latency_s"],
                "components": components,
                "dominant_resource": dominant,
                "dominant_category": TIMELINE_CATEGORIES[dominant],
            }
        )

    assert tuple(stage["operator"] for stage in stages) == OPERATORS
    assert math.isclose(elapsed, actual["latency_s"], rel_tol=1e-12)

    counts = {resource: 0 for resource in RESOURCES}
    for stage in stages:
        counts[stage["dominant_resource"]] += 1
    diversity = sum(count > 0 for count in counts.values())
    memory_present = counts["ARRAY"] + counts["EXTERNAL_BOUNDARY"] > 0
    feol_present = (
        counts["MAC"] + counts["LOCAL_FABRIC"] + counts["INTER_REGION_NOC"] > 0
    )
    gpu_present = counts["GPU_COMPUTE"] > 0
    summary = {
        "model": model,
        "context": context,
        "batch": batch,
        "decode_step_zero_based": 0,
        "decode_context": expected["context"],
        "layer_zero_based": 0,
        "frequency_GHz": engine.floorplan.config["clock_hz"] / 1e9,
        "dominant_resource_types": diversity,
        **{f"dominant_{resource}_operators": counts[resource] for resource in RESOURCES},
        "boundary_service_operators": sum(
            stage["components"]["EXTERNAL_BOUNDARY"] > 0 for stage in stages
        ),
        "boundary_dominant_operators": counts["EXTERNAL_BOUNDARY"],
        "GPU_is_dominant": gpu_present,
        "memory_FEOL_GPU_classes_present": memory_present and feol_present and gpu_present,
        "checkpoint_match": "PASS",
        "optimizer_runs": 0,
    }

    result = {
        "tag": tag,
        "ready": ready,
        "checkpoint": expected,
        "stages": stages,
        "summary": summary,
        "cache_ready_sha256": sha256(cache / "ready.json"),
        "cache_engine_sha256": sha256(cache / "engine.pkl"),
        "checkpoint_sha256": sha256(
            FORMAL / "checkpoints" / f"{tag}_M3D_NMP_CPA.jsonl"
        ),
    }

    shared.ENGINE = None
    shared.MAPPING = None
    del engine, actual
    gc.collect()
    return result


def choose_case(replays: list[dict]) -> dict:
    first = replays[0]
    summary = first["summary"]
    if summary["dominant_resource_types"] >= 3 and summary["memory_FEOL_GPU_classes_present"]:
        return first
    return max(
        replays,
        key=lambda item: (
            item["summary"]["memory_FEOL_GPU_classes_present"],
            item["summary"]["dominant_resource_types"],
            -CANDIDATES.index(
                (
                    item["summary"]["model"],
                    item["summary"]["context"],
                    item["summary"]["batch"],
                )
            ),
        ),
    )


def timeline_rows(selected: dict) -> list[dict]:
    return [
        {
            "order": order,
            "stage_index": stage["stage_index"],
            "operator": stage["operator"],
            "display_operator": stage["display_operator"],
            "executor": stage["executor"],
            "start_s": stage["start_s"],
            "end_s": stage["end_s"],
            "latency_s": stage["latency_s"],
            "dominant_resource": stage["dominant_resource"],
            "dominant_category": stage["dominant_category"],
            "timeline_semantics": "EXACT_SIMULATED_START_END",
        }
        for order, stage in enumerate(selected["stages"])
    ]


def matrix_rows(selected: dict) -> list[dict]:
    result = []
    for order, stage in enumerate(selected["stages"]):
        maximum = max(stage["components"].values())
        assert maximum > 0
        for resource in RESOURCES:
            service = stage["components"][resource]
            normalized = service / maximum
            result.append(
                {
                    "order": order,
                    "stage_index": stage["stage_index"],
                    "operator": stage["operator"],
                    "display_operator": stage["display_operator"],
                    "resource": resource,
                    "resource_label": RESOURCE_LABELS[resource],
                    "service_s": service,
                    "normalized_service": normalized,
                    "dominant": resource == stage["dominant_resource"],
                }
            )
    return result


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": TNR.get_name(),
            "font.size": 8.2,
            "axes.linewidth": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "svg.fonttype": "none",
            "svg.hashsalt": "physical-decode-execution-anatomy-v2",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def apply_fonts(ax) -> None:
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        size = tick.get_fontsize()
        tick.set_fontproperties(TNR)
        tick.set_fontsize(size)


def save_figure(fig) -> None:
    for extension in ("svg", "pdf"):
        path = OUT / f"figure.{extension}"
        metadata = {"Date": None} if extension == "svg" else {
            "CreationDate": None,
            "ModDate": None,
        }
        fig.savefig(path, metadata=metadata)
    svg = OUT / "figure.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def plot(timeline: list[dict], matrix: list[dict]) -> None:
    configure_plotting()
    fig = plt.figure(figsize=(7.2, 3.05))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.42, 1.0), wspace=0.34)
    axis = fig.add_subplot(grid[0, 0])
    heat = fig.add_subplot(grid[0, 1])

    for row in timeline:
        start_us = float(row["start_s"]) * 1e6
        width_us = float(row["latency_s"]) * 1e6
        color = CATEGORY_COLORS[row["dominant_category"]]
        narrow_label = row["operator"] in {"SOFTMAX", "AV_REDUCTION"}
        axis.broken_barh(
            [(start_us, width_us)],
            (0.18, 0.64),
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        axis.text(
            start_us + width_us / 2,
            0.5,
            row["display_operator"],
            ha="center",
            va="center",
            color="white",
            fontproperties=font(6.3 if narrow_label else 7.1, bold=True),
            rotation=90 if narrow_label else 0,
            clip_on=True,
            zorder=4,
        )

    handoffs = []
    for left, right in zip(timeline, timeline[1:]):
        if left["executor"] != right["executor"]:
            location = float(right["start_s"]) * 1e6
            handoffs.append(location)
            axis.axvline(
                location,
                color="#333333",
                linestyle=(0, (2, 2)),
                linewidth=0.65,
                ymin=0.12,
                ymax=0.88,
                zorder=2,
            )

    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xlim(0, max(float(row["end_s"]) for row in timeline) * 1e6 * 1.01)
    axis.set_xlabel("Time from layer start (µs)", fontproperties=font(8.4, bold=True))
    axis.set_title("(a) Operator critical-path timeline", fontproperties=font(8.8, bold=True), pad=7)
    axis.grid(axis="x", color="#D0D0D0", linestyle=(0, (3, 2)), linewidth=0.45, zorder=0)
    apply_fonts(axis)

    legend_handles = [
        Patch(facecolor=color, edgecolor="#303030", linewidth=0.45, label=category)
        for category, color in CATEGORY_COLORS.items()
    ]
    legend_handles.append(
        Line2D([0], [0], color="#333333", linestyle=(0, (2, 2)), linewidth=0.7,
               label="GPU–NMP handoff")
    )
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        prop=font(7.1, bold=True),
        handlelength=1.35,
        columnspacing=0.9,
    )

    lookup = {(int(row["order"]), row["resource"]): row for row in matrix}
    values = np.asarray(
        [
            [float(lookup[(index, resource)]["normalized_service"]) for resource in RESOURCES]
            for index in range(len(timeline))
        ]
    )
    image = heat.imshow(
        values,
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        aspect="auto",
        interpolation="nearest",
    )
    heat.set_yticks(
        range(len(timeline)),
        [row["display_operator"] for row in timeline],
    )
    heat.set_xticks(
        range(len(RESOURCES)),
        ["Array\n+MIV", "MAC", "Fabric", "NoC", "Boundary", "GPU"],
        rotation=0,
    )
    heat.tick_params(length=0, labelsize=7.2)
    for index, row in enumerate(timeline):
        dominant_index = RESOURCES.index(row["dominant_resource"])
        heat.add_patch(
            Rectangle(
                (dominant_index - 0.49, index - 0.49),
                0.98,
                0.98,
                fill=False,
                edgecolor="black",
                linewidth=0.9,
            )
        )
    heat.set_title("(b) Normalized resource service", fontproperties=font(8.8, bold=True), pad=7)
    apply_fonts(heat)
    colorbar = fig.colorbar(image, ax=heat, fraction=0.046, pad=0.035, ticks=(0, 0.5, 1))
    colorbar.set_label("Normalized service", fontproperties=font(7.8, bold=True), labelpad=3)
    colorbar.ax.tick_params(labelsize=7)
    apply_fonts(colorbar.ax)

    fig.subplots_adjust(left=0.055, right=0.94, top=0.89, bottom=0.25)
    save_figure(fig)
    plt.close(fig)


def write_documentation(selected: dict, summaries: list[dict]) -> None:
    summary = selected["summary"]
    caption = (
        f"Physical Decode execution anatomy for {summary['model']} at "
        f"{summary['context'][2:]}, B={summary['batch']}, using CPA at "
        f"{summary['frequency_GHz']:.0f} GHz. (a) Exact dependency-ordered operator "
        "critical-path intervals, colored by the dominant physical service resource; "
        "dashed lines mark GPU–NMP dependency handoffs. (b) Per-operator physical-resource "
        "service normalized to the maximum service time of that operator; black boxes "
        "indicate the dominant resource. AV Reduction is retained because it is an "
        "independently scheduled GPU stage. The trace is derived from the same physical "
        "execution model used for end-to-end performance and energy evaluation."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")

    candidate_lines = "\n".join(
        f"- {row['model']} / {row['context']} / B{row['batch']}: "
        f"{row['dominant_resource_types']} dominant-resource types; "
        f"Boundary-dominant={row['dominant_EXTERNAL_BOUNDARY_operators']}, "
        f"GPU-dominant={row['dominant_GPU_COMPUTE_operators']}."
        for row in summaries
    )
    readme = f"""# Physical Decode Execution Anatomy v2

This directory is derived evidence from frozen `formal_long_context_v3` CPA results. It does not modify any formal result.

## Candidate audit

{candidate_lines}

The selected case is **{summary['model']} / {summary['context']} / B{summary['batch']}**. The preferred non-extreme case already exposes memory-side, FEOL, and GPU-side bottlenecks with {summary['dominant_resource_types']} distinct dominant resources, so no speedup- or temperature-based selection was used.

## Trace provenance

- Decode step: 0 (context {summary['decode_context']})
- Transformer layer: 0
- CPA frequency: {summary['frequency_GHz']:.1f} GHz
- Replay: deterministic, read-only replay of the existing serialized CPA engine
- Optimizer executions: 0
- Timeline: exact simulated start/end intervals from the sequential scheduler, measured relative to layer 0 start
- Checkpoint comparison: PASS for context, total latency, traffic, and every recorded component sum

The heatmap metric is `T_norm(o,r) = T(o,r) / max_r T(o,r)`. Every row therefore has a maximum of exactly 1. The dominant resource is `argmax_r T(o,r)` and is outlined in black. `Array` includes MIV service. `NoC` contains the modeled collective/reduction traffic where applicable.

`AV_REDUCTION` is retained as **AV Reduce** because the canonical Decode scheduler emits it as a distinct GPU stage with its own latency. It is not presented as an additional Transformer matrix operator.

The source checkpoint does not contain stage-level intervals. Stage detail is recovered by deterministic replay from the already placed nominal-frequency engine cache; the replayed step is then checked against the frozen checkpoint. No benchmark, placement optimization, frequency sweep, or thermal simulation is run.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def validate(timeline: list[dict], matrix: list[dict]) -> None:
    assert tuple(row["operator"] for row in timeline) == OPERATORS
    assert all(float(row["start_s"]) < float(row["end_s"]) for row in timeline)
    assert all(
        float(left["start_s"]) < float(right["start_s"])
        for left, right in zip(timeline, timeline[1:])
    )
    by_operator = {}
    for row in matrix:
        by_operator.setdefault(row["operator"], []).append(row)
    for operator, rows in by_operator.items():
        maximum = max(float(row["normalized_service"]) for row in rows)
        assert math.isclose(maximum, 1.0, rel_tol=0, abs_tol=1e-15), operator
        dominant = [row for row in rows if row["dominant"]]
        assert len(dominant) == 1
        assert math.isclose(float(dominant[0]["normalized_service"]), 1.0, rel_tol=0, abs_tol=1e-15)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = formal_hashes()
    replays = [replay_case(*case) for case in CANDIDATES]
    selected = choose_case(replays)

    summaries = []
    for replay in replays:
        row = dict(replay["summary"])
        row["selected"] = replay is selected
        summaries.append(row)
    timeline = timeline_rows(selected)
    matrix = matrix_rows(selected)
    validate(timeline, matrix)

    write_csv(OUT / "candidate_resource_diversity.csv", summaries)
    write_csv(OUT / "resource_diversity_summary.csv", summaries)
    write_csv(OUT / "timeline.csv", timeline)
    write_csv(OUT / "operator_resource_matrix.csv", matrix)
    write_json(
        OUT / "selected_case.json",
        {
            **selected["summary"],
            "path": "M3D_NMP_CPA",
            "placement": "CRITICAL_PATH_AWARE",
            "selection_rule": "PREFERRED_NON_EXTREME_CASE_WITH_AT_LEAST_THREE_RESOURCE_TYPES_AND_MEMORY_FEOL_GPU_COVERAGE",
            "timeline_semantics": "EXACT_SIMULATED_START_END",
            "deterministic_replay": True,
            "optimizer_runs": 0,
            "cache_ready_sha256": selected["cache_ready_sha256"],
            "cache_engine_sha256": selected["cache_engine_sha256"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "source_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )
    write_documentation(selected, summaries)
    plot(timeline, matrix)

    after = formal_hashes()
    assert before == after, "formal_long_context_v3 changed"
    write_json(
        OUT / "formal_v3_preservation.json",
        {"status": "BYTE_IDENTICAL", "file_count": len(before), "sha256": before},
    )
    print(json.dumps(selected["summary"], indent=2))
    print(f"formal_long_context_v3 byte-identical: {len(before)} files")


if __name__ == "__main__":
    main()
