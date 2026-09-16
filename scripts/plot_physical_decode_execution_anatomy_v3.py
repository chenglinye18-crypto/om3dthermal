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
from matplotlib.patches import Patch
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
M3D_PATHS = PATHS[1:]
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
    "ARRAY": "Array+MIV",
    "HBM_MEMORY": "Array+MIV",
    "MAC": "MAC",
    "LOCAL_FABRIC": "Fabric",
    "INTER_REGION_NOC": "NoC",
    "EXTERNAL_BOUNDARY": "Boundary",
    "GPU_COMPUTE": "GPU",
}
CATEGORY_COLORS = {
    "Array+MIV": "#5C87B2",
    "MAC": "#C77B47",
    "Fabric": "#8A70A8",
    "NoC": "#A789BD",
    "Boundary": "#D0A53E",
    "GPU": "#4D9584",
}

PHASE_COLORS = {
    "Queue": "#E1E3E5",
    "Admission": "#D29A62",
    "Prefill": "#7398BE",
    "Decode": "#5B9A85",
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


def decode_layer_rows(timelines: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for path, _ in M3D_PATHS:
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


def latency_summary(timelines: dict[str, list[dict]]) -> list[dict]:
    latencies = {
        path: timelines[path][-1]["end_s"] - timelines[path][0]["start_s"]
        for path, _ in M3D_PATHS
    }
    gpu = latencies["M3D_GPU"]
    dns = latencies["M3D_NMP_UNIFORM"]
    return [
        {
            "path": path,
            "path_label": label,
            "layer_latency_us": latencies[path] * 1e6,
            "relative_to_M3D_GPU": gpu / latencies[path],
            "relative_to_DNS": dns / latencies[path],
            "timeline_semantics": "exact",
        }
        for path, label in M3D_PATHS
    ]


def e2e_timeline() -> list[dict]:
    rows = []
    hbm = candidate("HBM_BEST")
    assert hbm["selected_HBM_policy"] == "HBM_RESIDENT_WAVE"
    assert sum(hbm["wave_sizes"]) == BATCH
    for wave in hbm["waves"]:
        assert len(wave["steps"]) == 32
        cursor = 0.0
        phases = (
            ("Queue", float(wave["queue_delay"])),
            ("Admission", float(wave["admission_s"])),
            ("Prefill", float(wave["prefill_s"])),
            ("Decode", float(wave["decode_s"])),
        )
        for phase, duration in phases:
            if duration <= 0:
                continue
            rows.append(
                {
                    "path": "HBM_BEST",
                    "path_label": "HBM",
                    "wave": wave["wave"],
                    "wave_batch": wave["batch"],
                    "phase": phase,
                    "start_s": cursor,
                    "end_s": cursor + duration,
                    "duration_s": duration,
                    "source": f"runs/formal_long_context_v3/candidates/{TAG}_HBM_BEST.json",
                    "timing_semantics": "EXACT_CANONICAL_SCHEDULER",
                }
            )
            cursor += duration
        assert math.isclose(cursor, float(wave["queue_delay"]) + float(wave["admission_s"])
                            + float(wave["prefill_s"]) + float(wave["decode_s"]), rel_tol=1e-12)
    assert math.isclose(max(row["end_s"] for row in rows if row["path"] == "HBM_BEST"),
                        float(hbm["E2E_s"]), rel_tol=1e-12)

    for path, label in M3D_PATHS:
        result = candidate(path)
        prefill = float(result["prefill"]["latency_s"])
        decode = float(result["decode_s"])
        assert math.isclose(prefill + decode, float(result["E2E_s"]), rel_tol=1e-12)
        for phase, start, end in (
            ("Prefill", 0.0, prefill),
            ("Decode", prefill, prefill + decode),
        ):
            rows.append(
                {
                    "path": path,
                    "path_label": label,
                    "wave": 0,
                    "wave_batch": BATCH,
                    "phase": phase,
                    "start_s": start,
                    "end_s": end,
                    "duration_s": end - start,
                    "source": f"runs/formal_long_context_v3/candidates/{TAG}_{path}.json",
                    "timing_semantics": "EXACT_CANONICAL_PHASE_CLOSURE",
                }
            )
    return rows


def configure_plot() -> None:
    # Explicit family metadata is required for editable SVG text.  The font
    # file itself keeps PDF embedding tied to the native Windows TNR faces.
    v2.TNR.set_family("Times New Roman")
    v2.TNR.set_weight("normal")
    v2.TNR_BOLD.set_family("Times New Roman")
    v2.TNR_BOLD.set_weight("bold")
    plt.rcParams.update(
        {
            "font.family": v2.TNR.get_name(),
            "font.size": 9.2,
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


def save_figure(fig, stem: str) -> None:
    """Save independent vector outputs without letting a Windows lock stop the run."""
    for extension in ("svg", "pdf"):
        metadata = {"Date": None} if extension == "svg" else {
            "CreationDate": None,
            "ModDate": None,
        }
        target = OUT / f"{stem}.{extension}"
        try:
            fig.savefig(target, metadata=metadata)
        except PermissionError:
            target = OUT / f"{stem}_new.{extension}"
            print(f"WARNING: locked output; saving {target}", flush=True)
            fig.savefig(target, metadata=metadata)
        if extension == "svg":
            target.write_text(
                "\n".join(
                    line.rstrip()
                    for line in target.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )


def plot_e2e(e2e: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(7.25, 2.30))
    row_y = {"HBM_BEST": 3.3, "M3D_GPU": 2.1, "M3D_NMP_UNIFORM": 1.05, "M3D_NMP_CPA": 0.0}
    e2e_lookup = {path: float(candidate(path)["E2E_s"]) for path, _ in PATHS}
    for row in e2e:
        path = row["path"]
        y = row_y[path]
        height = 0.30 if path == "HBM_BEST" else 0.52
        if path == "HBM_BEST":
            y += 0.18 if int(row["wave"]) == 0 else -0.18
        start = float(row["start_s"])
        width = float(row["duration_s"])
        axis.broken_barh(
            [(start, width)],
            (y - height / 2, height),
            facecolor=PHASE_COLORS[row["phase"]],
            edgecolor="#404040",
            linewidth=0.35,
            zorder=3,
        )
        if width >= 0.14:
            label = {"Queue": "Wait", "Admission": "Adm."}.get(row["phase"], row["phase"])
            axis.text(
                start + width / 2,
                y,
                label,
                ha="center",
                va="center",
                color="#222222" if row["phase"] == "Queue" else "white",
                fontproperties=v2.font(7.5, bold=True),
                clip_on=True,
                zorder=4,
            )

    maximum_e2e = max(e2e_lookup.values())
    for path, _ in PATHS:
        axis.text(
            e2e_lookup[path] + maximum_e2e * 0.018,
            row_y[path],
            f"E2E = {e2e_lookup[path]:.2f} s",
            ha="left",
            va="center",
            fontproperties=v2.font(8.4, bold=True),
        )
    axis.set_yticks(
        [row_y[path] for path, _ in PATHS],
        ["HoG", "M3D-GPU", "DNS", "CPA"],
    )
    axis.set_xlim(0, maximum_e2e * 1.18)
    axis.set_ylim(-0.48, 3.82)
    axis.set_xlabel("Time from request arrival (s)", fontproperties=v2.font(9.6, bold=True))
    axis.grid(axis="x", color="#D0D0D0", linestyle=(0, (3, 2)), linewidth=0.45, zorder=0)
    apply_fonts(axis)
    axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="#404040", linewidth=0.35, label=phase)
            for phase, color in PHASE_COLORS.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        prop=v2.font(8.6, bold=True),
        handlelength=1.35,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.80, bottom=0.23)
    save_figure(fig, "e2e_timeline")
    plt.close(fig)


def plot_decode(timelines: dict[str, list[dict]], summary: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(7.25, 2.35))
    latency_lookup = {row["path"]: float(row["layer_latency_us"]) for row in summary}
    max_latency = max(latency_lookup.values())
    row_y = {"M3D_GPU": 2, "M3D_NMP_UNIFORM": 1, "M3D_NMP_CPA": 0}
    for path, _ in M3D_PATHS:
        y = row_y[path]
        for row in timelines[path]:
            start_us = row["start_s"] * 1e6
            width_us = row["duration_s"] * 1e6
            axis.broken_barh(
                [(start_us, width_us)],
                (y - 0.28, 0.56),
                facecolor=CATEGORY_COLORS[row["dominant_category"]],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            if row["operator"] not in {"Q", "K", "V"} and width_us >= 7:
                axis.text(
                    start_us + width_us / 2,
                    y,
                    row["display_operator"],
                    ha="center",
                    va="center",
                    color="white",
                    fontproperties=v2.font(6.4, bold=True),
                    clip_on=True,
                    zorder=4,
                )
        qkv = timelines[path][:3]
        axis.text(
            (qkv[0]["start_s"] + qkv[-1]["end_s"]) * 0.5e6,
            y,
            "QKV",
            ha="center",
            va="center",
            color="white",
            fontproperties=v2.font(6.0, bold=True),
            clip_on=True,
            zorder=4,
        )
        axis.text(
            latency_lookup[path] + max_latency * 0.018,
            y,
            f"{latency_lookup[path]:.1f} µs",
            ha="left",
            va="center",
            fontproperties=v2.font(8.4, bold=True),
        )

    axis.set_yticks(
        [row_y[path] for path, _ in M3D_PATHS],
        [label for _, label in M3D_PATHS],
    )
    axis.set_xlim(0, max_latency * 1.18)
    axis.set_ylim(-0.58, 2.58)
    axis.set_xlabel("Time from layer start (µs)", fontproperties=v2.font(9.6, bold=True))
    axis.grid(axis="x", color="#D0D0D0", linestyle=(0, (3, 2)), linewidth=0.45, zorder=0)
    apply_fonts(axis)
    axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="#303030", linewidth=0.4, label=category)
            for category, color in CATEGORY_COLORS.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(CATEGORY_COLORS),
        frameon=False,
        prop=v2.font(8.2, bold=True),
        handlelength=1.15,
        columnspacing=0.82,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.79, bottom=0.23)
    save_figure(fig, "decode_execution_timeline")
    plt.close(fig)


def plot(e2e: list[dict], timelines: dict[str, list[dict]], summary: list[dict]) -> None:
    configure_plot()
    plot_e2e(e2e)
    plot_decode(timelines, summary)


def validate(e2e: list[dict], timelines: dict[str, list[dict]],
             summary: list[dict], matrix: list[dict]) -> None:
    assert set(timelines) == {path for path, _ in PATHS}
    for path, _ in M3D_PATHS:
        rows = timelines[path]
        assert rows[0]["start_s"] == 0.0
        assert all(left["start_s"] < right["start_s"] for left, right in zip(rows, rows[1:]))
        latency = rows[-1]["end_s"] - rows[0]["start_s"]
        recorded = next(row for row in summary if row["path"] == path)
        assert math.isclose(latency * 1e6, float(recorded["layer_latency_us"]), rel_tol=1e-12)
    for path, _ in PATHS:
        expected = float(candidate(path)["E2E_s"])
        actual = max(float(row["end_s"]) for row in e2e if row["path"] == path)
        assert math.isclose(actual, expected, rel_tol=1e-12)
    hbm = candidate("HBM_BEST")
    assert sum(hbm["wave_sizes"]) == BATCH
    assert all(len(wave["steps"]) == 32 for wave in hbm["waves"])
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
        "Separate E2E serving and representative-layer Decode execution timelines for "
        "Qwen2.5-32B with 64K cached context and B=8. The E2E timeline uses the selected "
        "conventional-HBM policy and canonical system timings. The Decode timeline uses exact "
        "operator intervals at Decode step 0, layer 0, and nominal 1.0 GHz, colored by dominant "
        "physical service resource."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    latency_lines = "\n".join(
        f"- {row['path_label']}: {float(row['layer_latency_us']):.3f} us, "
        f"{float(row['relative_to_M3D_GPU']):.3f}x relative to M3D-GPU and "
        f"{float(row['relative_to_DNS']):.3f}x relative to DNS."
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

`e2e_timeline.pdf` and `e2e_timeline.svg` are clean system-level plots from the formal candidate rows. `HBM_BEST` selects `{hbm['selected_HBM_policy']}` with safe resident batch {hbm['safe_resident_batch']} and waves `{hbm['wave_sizes']}`. Wave 1 starts with resident historical KV. Wave 2 waits for Wave 1, admits historical KV once, then performs the same cached-history incremental Prefill and 32-step growing Decode. Historical H=64K is never recomputed as a full Prefill. The M3D rows use their exact canonical `prefill.latency_s`, `decode_s`, and `E2E_s` phase closure. Each row retains its canonical total E2E duration at the right edge.

{source_lines}

`decode_execution_timeline.pdf` and `decode_execution_timeline.svg` are a separate single-layer physical execution plot at Decode step 0, context 64128, layer 0, active B=8, and nominal 1.0 GHz. M3D-GPU and DNS are deterministic single-step replays using their canonical placement policies and are checked against frozen checkpoints. CPA is loaded from the existing serialized cache; the CPA optimizer is not rerun. The three rows align layer start to x=0 and retain absolute microsecond durations without per-row normalization. Each row reports `last operator end - first operator start` at the right edge.

## Representative layer latency

{latency_lines}

The two formal plots are intentionally separate and contain no title, panel label, connector, handoff marker, or explanatory annotation. They use Times New Roman vector text. The prior combined figure and CPA operator-resource matrix remain in this directory as legacy/provenance artifacts but are not regenerated or included in the formal plots.

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

    e2e = e2e_timeline()
    summary = latency_summary(timelines)
    validate(e2e, timelines, summary, matrix)
    v2.write_csv(OUT / "e2e_timeline.csv", e2e)
    v2.write_csv(OUT / "decode_layer_timeline.csv", decode_layer_rows(timelines))
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
            "panel_a_semantics": "SYSTEM_LEVEL_EXACT_CANONICAL_E2E",
            "panel_b_semantics": "SINGLE_LAYER_EXACT_SIMULATED_START_END",
            "incremental_prefill_tokens": 128,
            "generated_tokens": 32,
            "per_row_time_normalization": False,
            "benchmark_runs": 0,
            "thermal_runs": 0,
            "optimizer_runs": 0,
            "HBM_details": hbm,
        },
    )
    write_docs(hbm, summary, sources)
    plot(e2e, timelines, summary)
    (OUT / "resource_diversity_summary.csv").write_bytes(
        (ROOT / "runs/physical_decode_execution_anatomy_v2/resource_diversity_summary.csv").read_bytes()
    )

    after = v2.formal_hashes()
    assert before == after
    v2.write_json(
        OUT / "formal_v3_preservation.json",
        {"status": "BYTE_IDENTICAL", "file_count": len(before), "sha256": before},
    )
    print(json.dumps({"HBM": hbm, "E2E": {
        path: candidate(path)["E2E_s"] for path, _ in PATHS
    }, "layer_latencies": summary}, indent=2))
    print(f"formal_long_context_v3 byte-identical: {len(before)} files")


if __name__ == "__main__":
    main()
