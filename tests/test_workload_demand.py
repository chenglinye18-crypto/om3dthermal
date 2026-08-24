from pathlib import Path

import pytest

from om3dthermal.experiment.config import load_workload_spec
from om3dthermal.evaluation import evaluate_capacity_feasibility
from om3dthermal.workload import (
    evaluate_llm_decode,
    resolve_llm_decode_demand,
)


ROOT = Path(__file__).parents[1]
WORKLOAD_PATH = (
    ROOT / "configs" / "workload" / "llama31_8b_decode_b1_s131072.yaml"
)


def test_llm_decode_demand_is_field_exact_with_validated_metrics() -> None:
    spec = load_workload_spec(WORKLOAD_PATH, project_root=ROOT)
    metrics = evaluate_llm_decode(spec.decode)

    demand = resolve_llm_decode_demand(spec, metrics)

    assert demand.required_capacity_bytes == metrics.required_capacity_bytes
    assert demand.weight_footprint_bytes == metrics.weight_footprint_bytes
    assert demand.persistent_state_footprint_bytes == metrics.kv_footprint_bytes
    assert demand.read_bytes_per_output == metrics.read_bytes_per_token
    assert demand.write_bytes_per_output == metrics.write_bytes_per_token
    assert demand.flops_per_output == metrics.flops_per_token


def test_workload_demand_keeps_footprint_and_traffic_semantically_separate() -> None:
    spec = load_workload_spec(WORKLOAD_PATH, project_root=ROOT)
    demand = resolve_llm_decode_demand(spec, evaluate_llm_decode(spec.decode))

    # The frozen B=1/runtime=0 case happens to make these values numerically
    # equal.  Separate fields and statuses preserve their different meanings.
    assert "required_capacity_bytes" in type(demand).model_fields
    assert "read_bytes_per_output" in type(demand).model_fields
    assert demand.footprint_scope_status == (
        "ANALYTICAL_BYTE_EQUIVALENT_NOT_PHYSICAL_ALLOCATION"
    )
    assert demand.traffic_scope_status == (
        "ALGORITHMIC_WORKLOAD_TRAFFIC_NOT_PHYSICAL_DRAM_TRAFFIC"
    )

    batch_two_spec = spec.model_copy(update={
        "decode": spec.decode.model_copy(update={"batch_size": 2})
    })
    batch_two = resolve_llm_decode_demand(
        batch_two_spec, evaluate_llm_decode(batch_two_spec.decode)
    )
    assert batch_two.required_capacity_bytes != batch_two.read_bytes_per_output


def test_workload_demand_rejects_metrics_from_another_workload() -> None:
    spec = load_workload_spec(WORKLOAD_PATH, project_root=ROOT)
    mismatched = evaluate_llm_decode(
        spec.decode.model_copy(update={"context_length": 1})
    )

    with pytest.raises(ValueError, match="do not match"):
        resolve_llm_decode_demand(spec, mismatched)


def test_generic_capacity_evaluator_consumes_workload_demand() -> None:
    spec = load_workload_spec(WORKLOAD_PATH, project_root=ROOT)
    demand = resolve_llm_decode_demand(spec, evaluate_llm_decode(spec.decode))

    result = evaluate_capacity_feasibility(
        demand,
        physical_capacity_bytes=demand.required_capacity_bytes,
        reserved_capacity_bytes=0,
    )

    assert result.capacity_feasible is True
    assert result.capacity_margin_bytes == 0
