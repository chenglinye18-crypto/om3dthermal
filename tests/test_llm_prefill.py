"""Dense prefill primitive tests; intentionally no thermal execution."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from om3dthermal.experiment import load_platform_spec, load_prefill_workload_spec
from om3dthermal.platform import resolve_gpu_bandwidth_service
from om3dthermal.workload import (
    LLMDecodeInput,
    LLMPrefillInput,
    evaluate_gpu_prefill_roofline,
    evaluate_llm_decode,
    evaluate_llm_prefill,
)

ROOT = Path(__file__).parents[1]
WORKLOAD_PATH = ROOT / "configs/workload/llama31_8b_prefill_b1_s131072.yaml"
PLATFORM_PATH = ROOT / "configs/platform/gpu_package_h200_reference.yaml"


def _input(*, prompt_length: int = 8) -> LLMPrefillInput:
    return LLMPrefillInput(
        n_param=8_000_000_000, n_layers=32,
        n_heads_q=32, n_heads_kv=8, d_model=4096, d_ff=14336,
        vocab_size=128_256, batch_size=1, prompt_length=prompt_length,
        weight_bits=16, kv_bits=16)


def _canonical_roofline(metrics):
    platform = load_platform_spec(PLATFORM_PATH, project_root=ROOT)
    assert platform.gpu_decode_power is not None
    assert platform.gpu_compute_power is not None
    bandwidth = resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=(
            platform.gpu_decode_power.peak_memory_bandwidth_bytes_per_s),
        gpu_bandwidth_utilization=platform.gpu_bandwidth_service.nominal_utilization,
        utilization_status=platform.gpu_bandwidth_service.utilization_status,
        utilization_provenance=platform.gpu_bandwidth_service.provenance)
    compute = platform.gpu_compute_power
    return evaluate_gpu_prefill_roofline(
        metrics,
        peak_compute_flops_per_s=compute.peak_compute_BF16_dense_flops_per_s,
        sustained_memory_bandwidth_bytes_per_s=(
            bandwidth.sustained_bandwidth_bytes_per_s),
        static_power_W=compute.static_power_W,
        e_compute_dynamic_J_per_FLOP_min=(
            compute.e_compute_dynamic_J_per_FLOP_min),
        e_compute_dynamic_J_per_FLOP_max=(
            compute.e_compute_dynamic_J_per_FLOP_max))


def test_exact_head_dimension_and_uniform_gqa_validation() -> None:
    base = _input()
    raw = base.model_dump(exclude={"d_head"})
    assert base.d_head == 128
    with pytest.raises(ValidationError, match="must not exceed"):
        LLMPrefillInput.model_validate(raw | {"n_heads_kv": 64})
    with pytest.raises(ValidationError, match="evenly divisible"):
        LLMPrefillInput.model_validate(raw | {
            "n_heads_q": 30, "n_heads_kv": 8, "d_model": 4096})
    with pytest.raises(ValidationError, match="d_model"):
        LLMPrefillInput.model_validate(raw | {
            "n_heads_q": 30, "n_heads_kv": 5, "d_model": 4096})


def test_causal_attention_and_total_flop_closures() -> None:
    inp = _input(prompt_length=7)
    metrics = evaluate_llm_prefill(inp)
    expected_pairs = 7 * 8 // 2
    assert metrics.causal_token_pairs == expected_pairs
    assert metrics.qk_flops == metrics.av_flops
    assert metrics.qk_flops == (
        inp.batch_size * inp.n_layers * inp.n_heads_q * inp.d_head
        * inp.prompt_length * (inp.prompt_length + 1))
    assert metrics.attention_flops == metrics.qk_flops + metrics.av_flops
    assert metrics.lm_head_flops == 2 * inp.batch_size * inp.d_model * inp.vocab_size
    assert metrics.total_flops == (
        metrics.linear_flops + metrics.attention_flops + metrics.lm_head_flops)
    assert metrics.linear_flop_fraction + metrics.attention_flop_fraction + metrics.lm_head_flop_fraction == pytest.approx(1.0)


def test_nominal_gqa_tensor_widths_and_projection_traffic() -> None:
    metrics = evaluate_llm_prefill(_input(prompt_length=131_072))
    assert metrics.q_width == 4096
    assert metrics.kv_width == 1024
    assert metrics.q_projection_output_bytes == 34_359_738_368.0
    assert metrics.k_projection_cache_write_bytes == 8_589_934_592.0
    assert metrics.v_projection_cache_write_bytes == 8_589_934_592.0
    assert metrics.k_projection_cache_write_bytes / metrics.q_projection_output_bytes == 0.25
    assert metrics.v_projection_cache_write_bytes / metrics.q_projection_output_bytes == 0.25


def test_kv_projection_write_alias_and_tiled_read_closures() -> None:
    metrics = evaluate_llm_prefill(_input(prompt_length=131_072))
    logical_kv = (
        metrics.k_projection_cache_write_bytes
        + metrics.v_projection_cache_write_bytes)
    assert logical_kv == metrics.logical_kv_projection_tensor_bytes
    assert logical_kv == metrics.final_kv_cache_bytes
    assert metrics.physical_kv_projection_write_bytes == metrics.kv_write_bytes
    assert metrics.kv_write_bytes == metrics.final_kv_cache_bytes
    assert metrics.kv_projection_output_aliases_final_cache_write is True
    assert metrics.total_kv_cache_write_double_count_status == "PASS"
    assert metrics.total_memory_bytes == (
        metrics.active_weight_read_bytes + metrics.activation_memory_bytes
        + metrics.kv_write_bytes)
    assert metrics.attention_k_read_bytes + metrics.attention_v_read_bytes == (
        metrics.attention_kv_tiled_read_bytes)
    assert metrics.attention_kv_tiled_read_bytes == metrics.final_kv_cache_bytes
    assert metrics.kv_projection_write_provenance == (
        "K_V_PROJECTION_OUTPUTS_DIRECTLY_MATERIALIZE_FINAL_PREFILL_KV_CACHE")


def test_prefill_footprints_close_exactly_to_decode() -> None:
    inp = _input(prompt_length=131_072)
    prefill = evaluate_llm_prefill(inp)
    decode = evaluate_llm_decode(LLMDecodeInput(
        n_param=inp.n_param, n_layers=inp.n_layers,
        n_heads_q=inp.n_heads_q, n_heads_kv=inp.n_heads_kv,
        d_model=inp.d_model, d_ff=inp.d_ff,
        vocab_size=inp.vocab_size, batch_size=inp.batch_size,
        context_length=inp.prompt_length, weight_bits=inp.weight_bits,
        kv_bits=inp.kv_bits))
    assert prefill.weight_footprint_bytes == decode.weight_footprint_bytes
    assert prefill.final_kv_cache_bytes == decode.kv_footprint_bytes
    assert prefill.kv_write_bytes == prefill.final_kv_cache_bytes


def test_active_weights_are_read_once_and_score_matrix_is_not_materialized() -> None:
    short = evaluate_llm_prefill(_input(prompt_length=8))
    long = evaluate_llm_prefill(_input(prompt_length=16))
    assert short.active_weight_read_bytes == long.active_weight_read_bytes
    assert short.active_weight_read_bytes == pytest.approx(15_009_316_864.0)
    assert long.active_weight_read_bytes != long.weight_footprint_bytes * 16
    assert short.attention_score_matrix_materialized_bytes == 0
    assert long.attention_score_matrix_materialized_bytes == 0
    assert short.attention_probability_matrix_materialized_bytes == 0
    assert long.attention_probability_matrix_materialized_bytes == 0
    # Every modeled traffic term is constant or O(S), even though FLOPs are O(S^2).
    constant = short.active_weight_read_bytes + short.final_logits_bytes
    assert long.total_memory_bytes - constant == pytest.approx(
        2 * (short.total_memory_bytes - constant))
    assert long.traffic_provenance == (
        "PREFILL_ATTENTION_FUSED_TILED_NO_DECODE_STYLE_FULL_KV_REREAD")
    assert long.attention_tensor_read_provenance == (
        "PREFILL_ATTENTION_FUSED_TILED_LINEAR_QKV_READ")
    assert long.kv_read_semantics_status == (
        "NOT_DECODE_STYLE_FULL_HISTORY_REREAD_PER_TOKEN")
    assert long.kv_dram_scaling_status == "NO_S2_KV_DRAM_TRAFFIC"
    assert long.score_dram_scaling_status == "NO_S2_SCORE_DRAM_TRAFFIC"


def test_nominal_roofline_and_compute_energy_use_canonical_platform() -> None:
    spec = load_prefill_workload_spec(WORKLOAD_PATH, project_root=ROOT)
    metrics = evaluate_llm_prefill(spec.prefill)
    roofline = _canonical_roofline(metrics)
    platform = load_platform_spec(PLATFORM_PATH, project_root=ROOT)
    assert platform.gpu_compute_power is not None
    compute = platform.gpu_compute_power
    assert roofline.peak_compute_flops_per_s == 989.5e12
    assert roofline.sustained_memory_bandwidth_bytes_per_s == 2.4e12
    assert roofline.roofline_lower_bound_ms == max(
        roofline.compute_lower_bound_ms, roofline.memory_lower_bound_ms)
    assert roofline.roofline_bottleneck == "COMPUTE"
    assert roofline.compute_time_status == (
        "THEORETICAL_PEAK_COMPUTE_LOWER_BOUND")
    assert roofline.memory_time_status == "MEMORY_LOWER_BOUND"
    assert roofline.roofline_time_status == "ROOFLINE_LOWER_BOUND"
    assert roofline.achieved_prefill_throughput_status == "UNRESOLVED"
    assert roofline.achieved_prefill_throughput_resolved == "NO"
    assert roofline.dynamic_compute_energy_J_min == pytest.approx(
        metrics.total_flops * compute.e_compute_dynamic_J_per_FLOP_min)
    assert roofline.dynamic_compute_energy_J_max == pytest.approx(
        metrics.total_flops * compute.e_compute_dynamic_J_per_FLOP_max)
    assert roofline.static_energy_J_at_roofline_bound == pytest.approx(
        compute.static_power_W * roofline.roofline_lower_bound_ms / 1e3)
    assert roofline.static_energy_status == (
        "STATIC_ENERGY_AT_PEAK_ROOFLINE_LOWER_BOUND")


def test_nominal_prefill_sanity_ranges_and_transition() -> None:
    spec = load_prefill_workload_spec(WORKLOAD_PATH, project_root=ROOT)
    metrics = evaluate_llm_prefill(spec.prefill)
    assert metrics.linear_flops == 1_829_587_348_619_264
    assert metrics.qk_flops == 2_251_816_993_554_432
    assert metrics.av_flops == 2_251_816_993_554_432
    assert metrics.attention_flops == 4_503_633_987_108_864
    assert metrics.lm_head_flops == 1_050_673_152
    assert metrics.total_flops == 6_333_222_386_401_280
    assert metrics.attention_flop_fraction == pytest.approx(0.711, rel=0.01)
    assert metrics.activation_memory_bytes == 1_049_046_026_752.0
    assert metrics.total_memory_bytes == 1_081_235_212_800.0
    assert metrics.transition_status == "PREFILL_KV_RESIDENT_READY_FOR_DECODE"
