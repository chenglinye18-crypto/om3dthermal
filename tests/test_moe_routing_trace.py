"""Synthetic-tensor unit tests for real-routing infrastructure only."""

from __future__ import annotations

import csv
import builtins
import inspect
import json
from pathlib import Path

import pytest

from om3dthermal.experiment import load_moe_workload_spec
from om3dthermal.workload.moe_routing_trace import (
    CANONICAL_MODEL_ID,
    REAL_DATASET_ID,
    TEST_MEASUREMENT_STATUS,
    PromptRoutingSelections,
    build_expert_object_read_demand,
    build_routing_profile,
    load_sharegpt_prompts,
    run_real_mixtral_routing_profile,
    write_routing_artifacts,
)


ROOT = Path(__file__).parents[1]
MIXTRAL = (
    ROOT / "configs" / "workload"
    / "mixtral_8x7b_v01_decode_b1_s32768.yaml"
)


def _token(offset: int = 0) -> tuple[tuple[int, ...], ...]:
    return tuple(
        ((layer + offset) % 8, (layer + offset + 1) % 8)
        for layer in range(32)
    )


def _traces() -> tuple[PromptRoutingSelections, ...]:
    return tuple(
        PromptRoutingSelections(
            sample_id=f"prompt-{index}",
            selected_experts=(_token(0), _token(2), _token(0)),
        )
        for index in range(4)
    )


def _profile():
    return build_routing_profile(
        _traces(),
        model_id=CANONICAL_MODEL_ID,
        dataset_id="SYNTHETIC_ROUTER_TENSOR_TEST_ONLY",
        seed=17,
        num_layers=32,
        num_experts=8,
        top_k=2,
        measurement_status=TEST_MEASUREMENT_STATUS,
    )


def test_shape_and_exact_event_closures() -> None:
    profile = _profile()
    assert (profile.num_layers, profile.num_experts, profile.top_k) == (32, 8, 2)
    assert profile.num_prompts == 4
    assert profile.num_decode_tokens == 12
    assert len(profile.selection_counts) == 32
    assert all(len(row) == 8 for row in profile.selection_counts)
    assert profile.total_selection_events == 12 * 32 * 2
    assert sum(sum(row) for row in profile.selection_counts) == (
        profile.total_selection_events)
    assert all(sum(row) == 2 * profile.num_decode_tokens
               for row in profile.selection_counts)
    assert all(sum(row) == pytest.approx(2.0)
               for row in profile.selection_probability)
    assert all(sum(row) == pytest.approx(1.0)
               for row in profile.normalized_share)
    assert profile.routing_scope == "DECODE_ROUTING_ONLY"
    assert profile.prefill_routing_included is False


def test_aggregation_and_split_are_deterministic_and_stable() -> None:
    first = _profile()
    second = _profile()
    assert first == second
    stability = first.split_stability
    assert stability.split_a_prompt_ids == ("prompt-0", "prompt-1")
    assert stability.split_b_prompt_ids == ("prompt-2", "prompt-3")
    assert stability.split_a_decode_tokens == stability.split_b_decode_tokens == 6
    assert stability.pearson_correlation == pytest.approx(1.0)
    assert stability.spearman_rank_correlation == pytest.approx(1.0)
    assert stability.top_10_percent_expert_overlap == 1.0
    assert stability.top_25_percent_expert_overlap == 1.0


def test_layer_expert_identity_is_not_collapsed() -> None:
    profile = _profile()
    assert profile.selection_counts[0][0] == 8
    assert profile.selection_counts[10][0] == 0
    assert profile.selection_counts[10][2] == 8
    assert len(profile.selection_counts) * len(profile.selection_counts[0]) == 256


def test_invalid_topk_tensor_is_rejected() -> None:
    duplicate = tuple((0, 0) for _ in range(32))
    traces = (PromptRoutingSelections(
        sample_id="bad", selected_experts=(duplicate,)),)
    with pytest.raises(ValueError, match="distinct top-k"):
        build_routing_profile(
            traces,
            model_id=CANONICAL_MODEL_ID,
            dataset_id="SYNTHETIC_ROUTER_TENSOR_TEST_ONLY",
            seed=17,
            num_layers=32,
            num_experts=8,
            top_k=2,
            measurement_status=TEST_MEASUREMENT_STATUS,
        )


def test_real_measurement_label_requires_real_source_identity() -> None:
    with pytest.raises(ValueError, match="MEASURED_REAL_ROUTING"):
        build_routing_profile(
            _traces(),
            model_id=CANONICAL_MODEL_ID,
            dataset_id="SYNTHETIC_ROUTER_TENSOR_TEST_ONLY",
            seed=17,
            num_layers=32,
            num_experts=8,
            top_k=2,
            measurement_status="MEASURED_REAL_ROUTING",
        )


def test_routing_to_existing_expert_objects_and_21_gib_closure() -> None:
    workload = load_moe_workload_spec(MIXTRAL, project_root=ROOT).decode
    demand = build_expert_object_read_demand(_profile(), workload)
    assert len(demand.expert_objects) == 256
    assert demand.expert_objects[0].object_id == "expert.layer.00.expert.00"
    assert demand.expert_objects[-1].object_id == "expert.layer.31.expert.07"
    assert demand.expert_footprint_bytes == 352_321_536
    assert demand.total_expert_read_bytes_per_token == pytest.approx(21 * 2**30)
    assert demand.expected_active_expert_read_bytes_per_token == 21 * 2**30
    assert demand.closure_error_bytes == pytest.approx(0.0, abs=1e-6)
    assert demand.shared_nonexpert_traffic_modified is False
    assert demand.kv_traffic_modified is False


def test_batch_greater_than_one_is_explicitly_rejected() -> None:
    workload = load_moe_workload_spec(MIXTRAL, project_root=ROOT).decode
    with pytest.raises(ValueError, match="batch_size=1"):
        build_expert_object_read_demand(
            _profile(), workload.model_copy(update={"batch_size": 2}))


def test_sharegpt_loader_uses_real_text_and_deterministic_ids(tmp_path) -> None:
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text(json.dumps([
        {"id": "c", "conversations": [
            {"from": "human", "value": "third real prompt"}]},
        {"id": "a", "conversations": [
            {"from": "human", "value": "first real prompt"}]},
        {"id": "b", "conversations": [
            {"from": "human", "value": "second real prompt"}]},
    ]), encoding="utf-8")
    first = load_sharegpt_prompts(dataset, num_prompts=2, seed=17)
    second = load_sharegpt_prompts(dataset, num_prompts=2, seed=17)
    assert first == second
    assert len(first) == 2
    assert all("real prompt" in item.prompt for item in first)


def test_machine_readable_artifacts_are_complete(tmp_path) -> None:
    output = tmp_path / "profile"
    profile = _profile()
    write_routing_artifacts(
        profile,
        output,
        resolved_config={"seed": 17},
        dataset_metadata={"sample_ids": list(profile.sample_ids)},
    )
    assert {path.name for path in output.iterdir()} == {
        "routing_profile.json",
        "routing_counts.csv",
        "routing_probability.csv",
        "resolved_config.json",
        "dataset_metadata.json",
        "summary.json",
    }
    stored = json.loads((output / "routing_profile.json").read_text("utf-8"))
    assert stored["measurement_status"] == TEST_MEASUREMENT_STATUS
    with (output / "routing_counts.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 256
    assert rows[0]["layer"] == "0" and rows[-1]["layer"] == "31"


def test_production_capture_uses_model_router_logits_not_synthetic_popularity() -> None:
    source = inspect.getsource(run_real_mixtral_routing_profile)
    assert "output_router_logits=True" in source
    assert "decoded.router_logits" in source
    assert "torch.topk" in source
    assert "random.Random" not in source
    for forbidden in ("zipf", "gaussian", "hot_expert", "priority"):
        assert forbidden not in source.lower()


def test_missing_real_inference_dependencies_fail_loudly(
    tmp_path, monkeypatch
) -> None:
    real_import = builtins.__import__

    def blocked_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("synthetic missing-dependency test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_torch_import)
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="REAL_ROUTING_TRACE_BLOCKED"):
        run_real_mixtral_routing_profile(
            model_id=CANONICAL_MODEL_ID,
            dataset_path=dataset,
            dataset_source="unit-test-not-a-formal-dataset",
            num_prompts=1,
            max_new_tokens=1,
            seed=17,
            output_dir=tmp_path / "out",
        )
