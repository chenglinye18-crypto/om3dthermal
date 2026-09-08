"""Final non-thermal mixed-phase framework and conventional HBM closure."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from om3dthermal.serving import (
    SYSTEM_CONFIGURATIONS,
    AnalyticalRooflineGPUModel,
    MixedPhaseE2EResult,
    MixedPhaseServingCase,
    evaluate_conventional_hbm_mixed_phase,
    evaluate_mixed_phase_e2e,
    load_final_dense_e2e_matrix,
    resolve_conventional_hbm_backend,
)
from om3dthermal.workload import (
    evaluate_llm_decode,
    evaluate_llm_prefill,
    load_dense_model_registry,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registry():
    return load_dense_model_registry(ROOT / "configs/workload/models")


@pytest.fixture(scope="module")
def backend():
    return resolve_conventional_hbm_backend(ROOT)


@pytest.fixture(scope="module")
def local_result(registry):
    case = MixedPhaseServingCase(
        model_id="llama31_8b", context_length=131072,
        batch_size=4, prefill_requests=1, decode_requests=3)
    return evaluate_conventional_hbm_mixed_phase(
        project_root=ROOT, model=registry["llama31_8b"], case=case)


@pytest.fixture(scope="module")
def spill_result(registry):
    case = MixedPhaseServingCase(
        model_id="llama31_8b", context_length=131072,
        batch_size=32, prefill_requests=1, decode_requests=31)
    return evaluate_conventional_hbm_mixed_phase(
        project_root=ROOT, model=registry["llama31_8b"], case=case)


def test_model_registry_is_single_source_and_all_matrix_models_resolve(registry):
    resolved = registry["llama31_8b"]
    assert resolved.decode_input(batch_size=1, context_length=131072).n_param == 8_000_000_000
    assert resolved.prefill_input(batch_size=1, prompt_length=131072).d_ff == 14336
    assert registry["qwen25_7b"].model_spec_status == "RESOLVED"
    assert registry["qwen25_7b"].decode_input(
        batch_size=1, context_length=131072).n_heads_kv == 4
    assert registry["llama2_7b"].context_status == (
        "ARCHITECTURE_SCALING_CASE_BEYOND_NATIVE_CONTEXT")
    assert registry["llama2_7b"].decode_input(
        batch_size=1, context_length=131072).n_heads_kv == 32


def test_manifest_validates_only_the_frozen_3_by_4_by_3_shape():
    manifest = load_final_dense_e2e_matrix(
        ROOT / "configs/experiment/final_dense_e2e_matrix.yaml")
    assert (len(manifest.models), len(manifest.mixed_points), len(manifest.systems)) == (3, 4, 3)
    assert manifest.context_length == 131072
    assert all(
        point["prefill_requests"] + point["decode_requests"] == point["batch_size"]
        for point in manifest.mixed_points)
    assert manifest.mixed_points == (
        {"batch_size": 4, "prefill_requests": 1, "decode_requests": 3},
        {"batch_size": 4, "prefill_requests": 2, "decode_requests": 2},
        {"batch_size": 28, "prefill_requests": 1, "decode_requests": 27},
        {"batch_size": 28, "prefill_requests": 14, "decode_requests": 14},
    )


def test_mixed_case_rejects_nonclosing_batch():
    with pytest.raises(ValidationError, match="must equal batch_size"):
        MixedPhaseServingCase(
            model_id="llama31_8b", context_length=131072,
            batch_size=4, prefill_requests=2, decode_requests=3)


def test_prefill_read_write_direction_closes_frozen_total(registry):
    metrics = evaluate_llm_prefill(
        registry["llama31_8b"].prefill_input(
            batch_size=1, prompt_length=131072))
    assert metrics.prefill_read_bytes + metrics.prefill_write_bytes == pytest.approx(
        metrics.total_memory_bytes)
    assert metrics.kv_write_bytes == metrics.final_kv_cache_bytes
    assert metrics.total_memory_bytes == pytest.approx(1_081_235_212_800.0)
    assert metrics.total_flops == 6_333_222_386_401_280


def test_conventional_hbm_backend_resolves_only_evidenced_terms(backend):
    assert backend.capacity_bytes == 144_955_146_240.0
    assert backend.read_energy_pJ_per_bit == pytest.approx(1.3970979848163718)
    assert backend.refresh_power_W == pytest.approx(0.9614665609424703)
    assert backend.sustained_bandwidth_bytes_per_s == pytest.approx(2.4e12)
    assert backend.write_energy_pJ_per_bit is None
    assert backend.hbm_write_energy_status == "UNRESOLVED"
    assert "no WR command" in backend.hbm_write_energy_reason


def test_local_baseline_has_no_host_time_or_energy(local_result):
    assert local_result.capacity_status == "FULLY_LOCAL"
    assert local_result.resident_requests == 4
    assert local_result.spilled_requests == 0
    assert local_result.host_read_GB == 0.0
    assert local_result.host_write_GB == 0.0
    assert local_result.prefill_host_ddr_J == 0.0
    assert local_result.prefill_host_pcie_J == 0.0
    assert local_result.decode_host_ddr_J == 0.0
    assert local_result.decode_host_pcie_J == 0.0
    assert local_result.thermal is None


def test_prefill_hbm_read_energy_consumes_directional_physical_traffic(
    registry, backend, local_result,
):
    metrics = evaluate_llm_prefill(
        registry["llama31_8b"].prefill_input(
            batch_size=1, prompt_length=131072))
    assert local_result.prefill_memory_read_dynamic_J == pytest.approx(
        8 * metrics.prefill_read_bytes
        * backend.read_energy_pJ_per_bit * 1e-12)


def test_capacity_pressure_uses_shared_weight_and_decode_first(registry, spill_result):
    model = registry["llama31_8b"]
    metrics = evaluate_llm_decode(
        model.decode_input(batch_size=32, context_length=131072))
    expected = metrics.weight_footprint_bytes + 32 * metrics.kv_bytes_per_request
    assert spill_result.required_capacity_GB * 1e9 == expected
    assert spill_result.capacity_status == "CAPACITY_PRESSURED"
    assert spill_result.resident_requests == 7
    assert spill_result.spilled_requests == 25
    assert spill_result.resident_decode_requests == 7
    assert spill_result.spilled_decode_requests == 24
    assert spill_result.spilled_prefill_requests == 1
    assert spill_result.residency_policy == "DECODE_FIRST_LOCAL_RESIDENCY"


def test_host_bytes_energy_and_transfer_time_close(
    registry, backend, local_result, spill_result,
):
    metrics = evaluate_llm_decode(
        registry["llama31_8b"].decode_input(
            batch_size=31, context_length=131072))
    decode_read = 24 * metrics.kv_bytes_per_request
    decode_write = 24 * metrics.kv_write_bytes_per_token
    prefill_write = metrics.kv_bytes_per_request
    total_host_decode = decode_read + decode_write
    assert spill_result.host_read_GB * 1e9 == decode_read
    assert spill_result.host_write_GB * 1e9 == prefill_write + decode_write
    assert spill_result.decode_host_ddr_J == pytest.approx(
        8 * total_host_decode * backend.host_offload.e_ddr_dynamic_J_per_bit)
    assert spill_result.decode_host_pcie_J == pytest.approx(
        8 * total_host_decode * backend.host_offload.e_pcie_dynamic_J_per_bit)
    host_ms = total_host_decode / backend.host_offload.effective_bandwidth_bytes_per_second * 1e3
    gpu = AnalyticalRooflineGPUModel(
        matched_payload_bandwidth_bits_per_second=(
            8 * backend.sustained_bandwidth_bytes_per_s),
        effective_compute_flops_per_second=989.5e12,
    ).evaluate(
        registry["llama31_8b"].decode_input(
            batch_size=31, context_length=131072),
        batch_size=31,
    )
    assert spill_result.decode_service_time_ms == pytest.approx(
        gpu.decode_step_time_ms + host_ms)
    prefill_host_ms = (
        prefill_write / backend.host_offload.effective_bandwidth_bytes_per_second
        * 1e3)
    assert spill_result.prefill_service_time_ms == pytest.approx(
        local_result.prefill_service_time_ms + prefill_host_ms)


def test_no_overlap_time_and_incomplete_energy_are_explicit(local_result, spill_result):
    for result in (local_result, spill_result):
        assert result.mixed_epoch_time_ms == pytest.approx(
            result.prefill_service_time_ms + result.decode_service_time_ms)
        assert result.energy_status == (
            "INCOMPLETE_CONVENTIONAL_HBM_WRITE_ENERGY_UNRESOLVED")
        assert result.hbm_write_energy_pJ_per_bit is None
        assert result.prefill_total_J is None
        assert result.decode_total_J is None
        assert result.mixed_total_energy_J is None
        assert result.decode_tokens_per_J is None


def test_complete_result_schema_enforces_phase_and_mixed_energy_closure():
    common = dict(
        system_id="CONVENTIONAL_HBM_GPU", comparison_role="BASELINE",
        model_id="toy", context_length=1, batch_size=2,
        prefill_requests=1, decode_requests=1,
        evaluation_status="EVALUATED", energy_status="EVALUATED",
        nmp_batch_generalization_status="NOT_APPLICABLE",
        prefill_service_time_ms=2.0, decode_service_time_ms=3.0,
        mixed_epoch_time_ms=5.0,
        prefill_gpu_dynamic_J=1.0, prefill_gpu_static_J=2.0,
        prefill_memory_dynamic_J=3.0, prefill_refresh_J=4.0,
        prefill_host_ddr_J=5.0, prefill_host_pcie_J=6.0,
        prefill_total_J=21.0,
        decode_gpu_dynamic_J=1.0, decode_gpu_static_J=1.0,
        decode_memory_dynamic_J=1.0, decode_refresh_J=1.0,
        decode_host_ddr_J=1.0, decode_host_pcie_J=1.0,
        decode_total_J=6.0, mixed_total_energy_J=27.0,
    )
    assert MixedPhaseE2EResult(**common).mixed_total_energy_J == 27.0
    with pytest.raises(ValidationError, match="mixed energy does not close"):
        MixedPhaseE2EResult(**{**common, "mixed_total_energy_J": 28.0})


@pytest.mark.parametrize("batch,prefill,decode", [
    (4, 1, 3), (4, 2, 2), (28, 1, 27), (28, 14, 14),
])
def test_system_roles_and_resolved_nmp_batch_path(
    registry, batch, prefill, decode,
):
    assert SYSTEM_CONFIGURATIONS["CONVENTIONAL_HBM_GPU"].comparison_role == "BASELINE"
    memory_only = SYSTEM_CONFIGURATIONS["ORTHOGONAL_M3D_IGZO_MEMORY_ONLY"]
    assert memory_only.comparison_role == "ABLATION"
    assert memory_only.decode_executor == "GPU"
    proposed = SYSTEM_CONFIGURATIONS["IOM3D_FEOL_NMP"]
    assert proposed.decode_executor == "FEOL_NMP_GPU_HYBRID"
    case = MixedPhaseServingCase(
        model_id="llama31_8b", context_length=131072,
        batch_size=batch, prefill_requests=prefill, decode_requests=decode)
    result = evaluate_mixed_phase_e2e(
        project_root=ROOT, model=registry["llama31_8b"], case=case,
        system_id="IOM3D_FEOL_NMP")
    assert result.nmp_batch_generalization_status == (
        "RESOLVED_ANALYTICAL_BATCH_MODEL")
    assert result.evaluation_status == "EVALUATED"
    assert result.capacity_status == "FULLY_LOCAL"
    assert result.capacity_violations == 0
    assert result.resident_requests == batch
    assert result.decode_tokens_per_s > 0.0
    assert result.decode_total_J > 0.0
    assert result.prefill_memory_write_dynamic_J > 0.0
    assert result.prefill_memory_dynamic_J == pytest.approx(
        result.prefill_memory_read_dynamic_J
        + result.prefill_memory_write_dynamic_J)
    assert result.mixed_epoch_time_ms == pytest.approx(
        result.prefill_service_time_ms + result.decode_service_time_ms)
    assert result.thermal is None


def test_memory_only_backend_is_evaluated_and_gpu_only(registry):
    case = MixedPhaseServingCase(
        model_id="llama31_8b", context_length=131072,
        batch_size=4, prefill_requests=1, decode_requests=3)
    result = evaluate_mixed_phase_e2e(
        project_root=ROOT, model=registry["llama31_8b"], case=case,
        system_id="ORTHOGONAL_M3D_IGZO_MEMORY_ONLY")
    assert result.evaluation_status == "EVALUATED"
    assert result.capacity_status == "FULLY_LOCAL"
    assert result.capacity_violations == 0
    assert result.decode_nmp_mac_dynamic_J == 0.0
    assert result.decode_residual_interface_J == 0.0
    assert result.decode_total_J > 0.0
    assert result.prefill_memory_write_dynamic_J > 0.0
    assert result.nmp_batch_generalization_status == "NOT_APPLICABLE"
    assert result.thermal is None


def test_memory_only_capacity_failure_is_structured(registry):
    case = MixedPhaseServingCase(
        model_id="llama2_7b", context_length=131072,
        batch_size=28, prefill_requests=1, decode_requests=27)
    result = evaluate_mixed_phase_e2e(
        project_root=ROOT, model=registry["llama2_7b"], case=case,
        system_id="ORTHOGONAL_M3D_IGZO_MEMORY_ONLY")
    assert result.evaluation_status == "CAPACITY_INFEASIBLE"
    assert result.capacity_status == "CAPACITY_INFEASIBLE"
    assert result.mixed_epoch_time_ms is None


def test_capacity_infeasible_design_has_no_comparative_claim(registry):
    from om3dthermal.serving import compare_mixed_phase_results

    case = MixedPhaseServingCase(
        model_id="llama2_7b", context_length=131072,
        batch_size=28, prefill_requests=1, decode_requests=27)
    baseline = evaluate_mixed_phase_e2e(
        project_root=ROOT, model=registry["llama2_7b"], case=case,
        system_id="CONVENTIONAL_HBM_GPU")
    design = evaluate_mixed_phase_e2e(
        project_root=ROOT, model=registry["llama2_7b"], case=case,
        system_id="ORTHOGONAL_M3D_IGZO_MEMORY_ONLY")
    comparison = compare_mixed_phase_results(baseline, (design,))[0]
    assert comparison.e2e_speedup_vs_baseline is None
    assert comparison.spill_reduction is None
