"""Checks for cross-configuration Decode anatomy evidence; no simulation."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "physical_decode_execution_anatomy_v3"


def rows(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_fixed_workload_and_hbm_policy():
    selected = json.loads((OUT / "selected_case.json").read_text())
    assert (selected["model"], selected["context"], selected["batch"]) == (
        "Qwen2.5-32B", "LC64K", 8
    )
    assert selected["decode_context"] == 64128
    assert selected["selected_HBM_policy"] == "HBM_RESIDENT_WAVE"
    assert selected["HBM_details"]["wave_sizes"] == [4, 4]
    assert selected["HBM_timeline_semantics"] == "aggregate-only"
    assert selected["optimizer_runs"] == 0


def test_shared_absolute_axis_and_latency_closure():
    timeline = rows("timeline_all_paths.csv")
    summary = {row["path"]: row for row in rows("layer_latency_summary.csv")}
    grouped = {}
    for row in timeline:
        grouped.setdefault(row["path"], []).append(row)
    assert set(grouped) == {"HBM_BEST", "M3D_GPU", "M3D_NMP_UNIFORM", "M3D_NMP_CPA"}
    for path, values in grouped.items():
        assert float(values[0]["start_us"]) == 0.0
        latency = float(values[-1]["end_us"]) - float(values[0]["start_us"])
        assert math.isclose(latency, float(summary[path]["layer_latency_us"]), rel_tol=1e-12)
    assert grouped["HBM_BEST"][0]["operator"] == "AGGREGATE_LAYER_REFERENCE"
    assert grouped["HBM_BEST"][0]["timeline_semantics"] == "aggregate-only"


def test_checkpoint_provenance_and_cpa_heatmap():
    provenance = rows("timeline_provenance.csv")
    for path in ("M3D_GPU", "M3D_NMP_UNIFORM", "M3D_NMP_CPA"):
        path_rows = [row for row in provenance if row["path"] == path]
        assert path_rows
        assert all(row["exact_or_reconstructed"] == "exact" for row in path_rows)
        assert all(row["validation_status"] == "PASS" for row in path_rows)
    matrix = rows("operator_resource_matrix_cpa.csv")
    grouped = {}
    for row in matrix:
        grouped.setdefault(row["operator"], []).append(row)
    for values in grouped.values():
        assert math.isclose(max(float(row["normalized_service"]) for row in values), 1.0,
                            rel_tol=0, abs_tol=1e-15)
        dominant = [row for row in values if row["dominant"] == "True"]
        assert len(dominant) == 1
        assert float(dominant[0]["normalized_service"]) == 1.0


def test_formal_v3_is_preserved():
    value = json.loads((OUT / "formal_v3_preservation.json").read_text())
    assert value["status"] == "BYTE_IDENTICAL"
    assert value["file_count"] > 0
