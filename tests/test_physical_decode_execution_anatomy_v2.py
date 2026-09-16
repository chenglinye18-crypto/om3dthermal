"""Checks for the derived v2 Decode anatomy evidence; no simulation."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "physical_decode_execution_anatomy_v2"


def read_csv(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_representative_case_follows_preferred_case_rule():
    rows = read_csv("candidate_resource_diversity.csv")
    assert len(rows) == 3
    assert rows[0]["model"] == "Qwen2.5-32B"
    assert rows[0]["context"] == "LC64K"
    assert rows[0]["batch"] == "8"
    assert rows[0]["selected"] == "True"
    assert int(rows[0]["dominant_resource_types"]) >= 3
    assert rows[0]["memory_FEOL_GPU_classes_present"] == "True"
    assert all(row["checkpoint_match"] == "PASS" for row in rows)


def test_heatmap_rows_are_normalized_and_boxes_match_argmax():
    rows = read_csv("operator_resource_matrix.csv")
    operators = {}
    for row in rows:
        operators.setdefault(row["operator"], []).append(row)
    assert len(operators) == 11
    for values in operators.values():
        maximum = max(float(row["normalized_service"]) for row in values)
        dominant = [row for row in values if row["dominant"] == "True"]
        assert math.isclose(maximum, 1.0, rel_tol=0, abs_tol=1e-15)
        assert len(dominant) == 1
        assert math.isclose(
            float(dominant[0]["normalized_service"]), 1.0, rel_tol=0, abs_tol=1e-15
        )


def test_timeline_order_and_formal_preservation():
    timeline = read_csv("timeline.csv")
    assert [row["operator"] for row in timeline] == [
        "Q", "K", "V", "ATTENTION_QK", "SOFTMAX", "ATTENTION_AV",
        "AV_REDUCTION", "O", "FFN_GATE", "FFN_UP", "FFN_DOWN",
    ]
    assert all(
        float(left["start_s"]) < float(right["start_s"])
        for left, right in zip(timeline, timeline[1:])
    )
    assert all(row["timeline_semantics"] == "EXACT_SIMULATED_START_END" for row in timeline)
    preservation = json.loads((OUT / "formal_v3_preservation.json").read_text())
    assert preservation["status"] == "BYTE_IDENTICAL"
    assert preservation["file_count"] > 0
