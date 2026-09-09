"""Mixed-scope, state-conservation, growing-KV, and placement audit gates."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from om3dthermal.serving import (
    EvaluationSemantics,
    MixedPhaseServingCase,
    RequestKVState,
    decode_step,
    evaluate_conventional_hbm_resident_wave_growing_kv,
    evaluate_conventional_prefill_first_state_window,
    evaluate_m3d_growing_kv_capacity,
    evaluate_nmp_decode_batch,
    free_local,
    initial_decode_state,
    initial_prefill_state,
    require_comparable_semantics,
    resolve_conventional_hbm_backend,
    transfer_host_to_local,
    transfer_local_to_host,
)
from om3dthermal.workload import load_dense_model_registry


ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registry():
    return load_dense_model_registry(ROOT/"configs/workload/models")


def test_qwen_14_14_decode_only_and_mixed_window_are_distinct(registry):
    case=MixedPhaseServingCase(model_id="qwen25_7b",context_length=131072,
        batch_size=28,prefill_requests=14,decode_requests=14)
    result=evaluate_conventional_prefill_first_state_window(
        project_root=ROOT,model=registry["qwen25_7b"],case=case)
    assert result.evaluation_scope=="MIXED_SERVICE_WINDOW"
    assert result.schedule_policy=="NO_OVERLAP__PREFILL_FIRST"
    assert result.resident_decode_requests==14
    assert result.resident_prefill_requests==3
    assert result.host_prefill_requests==11
    assert result.host_prefill_kv_write_GB==pytest.approx(82.678120448)
    host = resolve_conventional_hbm_backend(ROOT).host_offload
    assert result.host_prefill_transfer_time_ms == pytest.approx(
        result.host_prefill_kv_write_GB*1e9
        / host.effective_bandwidth_bytes_per_second*1e3)
    assert sum(event.bytes for event in result.events)==pytest.approx(82.678120448e9)


def test_unstarted_prefill_has_no_fabricated_kv_and_live_kv_cannot_disappear():
    pending=initial_prefill_state("P0")
    assert pending.current_kv_length==0
    assert pending.local_valid_kv_length==pending.host_valid_kv_length==0
    with pytest.raises(ValidationError,match="no complete valid copy"):
        RequestKVState(request_id="bad",phase="ACTIVE_DECODE",current_kv_length=4,
            local_valid_kv_length=0,host_valid_kv_length=0)


def test_initial_host_admission_dirty_suffix_writeback_and_legal_free(registry):
    backend=resolve_conventional_hbm_backend(ROOT); host=backend.host_offload
    kv_per_token=131072.0
    state=initial_decode_state("D0",10,local=False)
    admission=transfer_host_to_local(state,kv_bytes_per_token=kv_per_token,
        bandwidth_bytes_per_s=host.effective_bandwidth_bytes_per_second,
        e_pcie_J_per_bit=host.host_link_dynamic_J_per_bit,
        e_ddr_J_per_bit=host.host_memory_dynamic_J_per_bit)
    step=decode_step(admission.after,kv_append_bytes=kv_per_token)
    writeback=transfer_local_to_host(step.after,kv_bytes_per_token=kv_per_token,
        bandwidth_bytes_per_s=host.effective_bandwidth_bytes_per_second,
        e_pcie_J_per_bit=host.host_link_dynamic_J_per_bit,
        e_ddr_J_per_bit=host.host_memory_dynamic_J_per_bit)
    assert writeback.bytes==kv_per_token
    repeated=transfer_local_to_host(writeback.after,kv_bytes_per_token=kv_per_token,
        bandwidth_bytes_per_s=host.effective_bandwidth_bytes_per_second,
        e_pcie_J_per_bit=host.host_link_dynamic_J_per_bit,
        e_ddr_J_per_bit=host.host_memory_dynamic_J_per_bit)
    assert repeated.bytes==0.0
    finished=repeated.after.model_copy(update={"phase":"FINISHED","completed":True})
    released=free_local(finished)
    assert released.after.local_valid_kv_length==0
    assert released.after.has_host_copy


def test_first_wave_zero_requires_local_initial_state(registry):
    local=initial_decode_state("D0",131072,local=True)
    assert local.local_valid_kv_length==local.current_kv_length
    host=initial_decode_state("D1",131072,local=False)
    with pytest.raises(ValueError,match="complete local KV"):
        decode_step(host,kv_append_bytes=131072)


def test_short_g_growing_reference_closes(registry):
    case=MixedPhaseServingCase(model_id="qwen25_7b",context_length=131072,
        batch_size=28,prefill_requests=14,decode_requests=14)
    result=evaluate_conventional_hbm_resident_wave_growing_kv(
        project_root=ROOT,model=registry["qwen25_7b"],case=case,
        generated_decode_steps=2)
    assert result.step_context_lengths==(131072,131073)
    assert result.high_water_context_length==131074
    assert result.total_generated_tokens==28
    assert result.total_compute_time_ms==pytest.approx(
        sum(sum(row) for row in result.step_times_ms_by_wave))
    assert result.total_completion_time_ms==pytest.approx(
        result.total_compute_time_ms+result.total_admission_time_ms)


@pytest.mark.parametrize("prefill,decode,growth",[
    (1,27,1811939328.0),(14,14,939524096.0)])
def test_llama_b28_g512_crosses_m3d_capacity(registry,prefill,decode,growth):
    case=MixedPhaseServingCase(model_id="llama31_8b",context_length=131072,
        batch_size=28,prefill_requests=prefill,decode_requests=decode)
    result=evaluate_m3d_growing_kv_capacity(
        project_root=ROOT,model=registry["llama31_8b"],case=case,
        generated_decode_steps=512)
    assert result.growing_kv_bytes==growth
    assert result.capacity_status=="CAPACITY_INFEASIBLE"
    assert result.capacity_margin_GB<0.0


def test_nmp_execution_projects_one_persistent_resident_layout(registry):
    workload=registry["llama31_8b"].decode_input(
        batch_size=14,context_length=131072)
    result=evaluate_nmp_decode_batch(workload,project_root=ROOT,
        active_capacity_requests=28)
    trace=result.execution_trace
    assert result.resident_execution_placement_status==(
        "SINGLE_PERSISTENT_RESIDENT_LAYOUT_ACTIVE_LOAD_PROJECTION")
    resident={load.unit.unit_id:owners for load,owners in zip(
        trace.resident_placement.unit_loads,trace.resident_placement.ownership)}
    assert all(resident[load.unit.unit_id]==owners for load,owners in zip(
        trace.placement.unit_loads,trace.placement.ownership))
    assert trace.placement.resident_used_bytes_per_die==(
        trace.resident_placement.resident_used_bytes_per_die)
    assert result.post_step_capacity_violations==0
    assert result.kv_append_allocation_status.startswith(
        "WHOLE_VECTOR_APPEND_DIE_OWNER_AND_CAPACITY_VALIDATED")


def test_kv_append_is_256_byte_atomic_and_kv_pair_owner_consistent(registry):
    result=evaluate_nmp_decode_batch(registry["llama31_8b"].decode_input(
        batch_size=1,context_length=131072),project_root=ROOT)
    activity=result.execution_trace.activity
    assert all(item.kv_write_bytes/256==round(item.kv_write_bytes/256)
               for item in activity.activities)
    owners=activity.kv_append_owner_by_layer_request
    for layer in range(32):
        assert owners[layer,0,"ATTENTION_QK"]==owners[layer,0,"ATTENTION_AV"]


def test_comparison_rejects_scope_or_context_mismatch():
    common=dict(initial_state_requirement="S_CACHE_VALID",
        final_state_requirement="G_TOKENS_GENERATED",generated_decode_steps=2,
        schedule_policy="NO_OVERLAP")
    decode=EvaluationSemantics(evaluation_scope="DECODE_SERVICE_ONLY",
        context_evolution="GROWING_KV",**common)
    mixed=EvaluationSemantics(evaluation_scope="MIXED_SERVICE_WINDOW",
        context_evolution="GROWING_KV",**common)
    with pytest.raises(ValueError,match="INCOMPARABLE"):
        require_comparable_semantics(decode,mixed)
    fixed=decode.model_copy(update={"context_evolution":"FIXED_CONTEXT_SNAPSHOT"})
    with pytest.raises(ValueError,match="INCOMPARABLE"):
        require_comparable_semantics(decode,fixed)
