"""Unified horizon, live workspace, physical allocation, and energy gates."""

from pathlib import Path

import pytest

from om3dthermal.serving import (
    KVPhysicalAllocator,
    PersistentMixedServiceCase,
    assert_no_live_overlap,
    compare_persistent_horizons,
    evaluate_conventional_overflow_policy,
    evaluate_decode_workspace,
    evaluate_persistent_mixed_service_horizon,
    evaluate_prefill_workspace,
    load_serving_e2e_closure_spec,
)
from om3dthermal.workload import load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registry():
    return load_dense_model_registry(ROOT/"configs/workload/models")


@pytest.fixture(scope="module")
def spec():
    return load_serving_e2e_closure_spec(
        ROOT/"configs/experiment/serving_e2e_closure.yaml")


def test_workspace_is_peak_live_set_not_sum(registry, spec):
    result = evaluate_prefill_workspace(
        registry["llama31_8b"], batch_size=14, context_length=131072,
        config=spec.workspace)
    assert result.peak_bytes > 0
    assert result.peak_bytes == max(stage.live_bytes for stage in result.stages)
    assert result.peak_bytes < result.sum_of_stage_bytes
    assert result.peak_stage == "FFN_ACT"
    assert result.chunk_tokens == 2048


def test_decode_workspace_grows_with_context_and_nmp_boundary(registry, spec):
    gpu = evaluate_decode_workspace(
        registry["llama31_8b"], batch_size=14, context_length=131072,
        config=spec.workspace)
    grown = evaluate_decode_workspace(
        registry["llama31_8b"], batch_size=14, context_length=131073,
        config=spec.workspace)
    nmp = evaluate_decode_workspace(
        registry["llama31_8b"], batch_size=14, context_length=131072,
        config=spec.workspace, proposed_nmp=True, nmp_die_count=106)
    assert grown.peak_bytes > gpu.peak_bytes
    assert nmp.peak_bytes >= gpu.peak_bytes




def test_physical_whole_vectors_are_integer_nonoverlap_and_kv_paired():
    allocator = KVPhysicalAllocator(
        die_count=4, pages_per_die=2, page_size_bytes=1024)
    owners = (0, 1, 2, 3)
    k = allocator.allocate_vector(
        request_id="D0", layer_id=0, token_id=7, kind="K", kv_head_id=3,
        vector_bytes=256, owners=owners)
    v = allocator.allocate_vector(
        request_id="D0", layer_id=0, token_id=7, kind="V", kv_head_id=3,
        vector_bytes=256, owners=owners)
    assert sum(item.size_bytes for item in k) == 256
    assert sum(item.size_bytes for item in v) == 256
    assert len({item.die_id for item in k}) == 1
    assert k[0].die_id == v[0].die_id
    assert_no_live_overlap(allocator.live_allocations, 1024)


def test_free_reuse_and_migration_conserve_bytes():
    allocator = KVPhysicalAllocator(
        die_count=1, pages_per_die=1, page_size_bytes=1024)
    first = allocator.allocate_bytes("kv:D0", die_id=0, size_bytes=700)
    first_location = (first[0].page_id, first[0].offset_bytes)
    assert allocator.migrate_local_to_host("kv:D0", retain_local=False) == 700
    second = allocator.migrate_host_to_local(
        "kv:D1", die_id=0, size_bytes=700)
    assert (second[0].page_id, second[0].offset_bytes) == first_location
    assert allocator.allocated_bytes_per_die == (700,)


def test_conventional_energy_unknown_is_partial_not_zero(registry, spec):
    case = PersistentMixedServiceCase(
        model_id="llama31_8b", context_length=32768,
        prefill_requests=1, decode_requests=3, generated_decode_steps=2)
    result = evaluate_conventional_overflow_policy(
        project_root=ROOT, model=registry["llama31_8b"], case=case,
        policy="HOST_KV_OFFLOAD", workspace_config=spec.workspace)
    assert result.status == "EVALUATED"
    assert result.energy_status == (
        "INCOMPLETE_CONVENTIONAL_HBM_WRITE_ENERGY_UNRESOLVED")
    assert result.known_energy_J > 0
    assert result.unresolved_HBM_write_bytes > 0


def test_comparison_rejects_workload_mismatch(registry, spec):
    case = PersistentMixedServiceCase(
        model_id="llama31_8b", context_length=32768,
        prefill_requests=1, decode_requests=3, generated_decode_steps=2)
    baseline = evaluate_persistent_mixed_service_horizon(
        project_root=ROOT, model=registry["llama31_8b"], case=case,
        system="ORTHOGONAL_M3D_IGZO_MEMORY_ONLY", workspace_config=spec.workspace)
    changed = baseline.model_copy(update={"G": 3})
    with pytest.raises(ValueError, match="INCOMPARABLE"):
        compare_persistent_horizons(baseline, changed)
