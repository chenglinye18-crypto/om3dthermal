"""Dense causal LLM prefill accounting and a GPU roofline primitive.

This is independent from autoregressive decode. It models one complete prompt
pass, leaves the final KV cache resident for decode, and never materializes an
S-by-S attention-score matrix in DRAM.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

ACTIVATION_BYTES = 2
PREFILL_TRAFFIC_PROVENANCE = "PREFILL_ATTENTION_FUSED_TILED_NO_DECODE_STYLE_FULL_KV_REREAD"
PREFILL_TRANSITION_STATUS = "PREFILL_KV_RESIDENT_READY_FOR_DECODE"
_ACTIVE_MATRIX_OPERATORS = frozenset({
    "Q", "K", "V", "O", "FFN_GATE", "FFN_UP", "FFN_DOWN", "LM_HEAD",
})


class LLMPrefillInput(BaseModel):
    """Dimensions and precisions for one dense causal prompt pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    n_param: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    n_heads_q: int = Field(gt=0)
    n_heads_kv: int = Field(gt=0)
    d_model: int = Field(gt=0)
    d_ff: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    prompt_length: int = Field(gt=0)
    weight_bits: int = Field(gt=0)
    kv_bits: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_heads(self) -> "LLMPrefillInput":
        if self.n_heads_kv > self.n_heads_q:
            raise ValueError("n_heads_kv must not exceed n_heads_q")
        if self.n_heads_q % self.n_heads_kv != 0:
            raise ValueError(
                f"n_heads_q ({self.n_heads_q}) must be evenly divisible by "
                f"n_heads_kv ({self.n_heads_kv}) for uniform GQA/MQA/MHA")
        if self.d_model % self.n_heads_q != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be evenly divisible by "
                f"n_heads_q ({self.n_heads_q})")
        return self

    @computed_field(return_type=int)
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads_q


class LLMPrefillMetrics(BaseModel):
    """Exact FLOP/capacity accounting plus first-order DRAM traffic."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    causal_token_pairs: int
    prefill_input_tokens: int
    linear_flops: int
    qk_flops: int
    av_flops: int
    attention_flops: int
    lm_head_flops: int
    total_flops: int
    linear_flop_fraction: float
    attention_flop_fraction: float
    lm_head_flop_fraction: float
    weight_footprint_bytes: float
    active_weight_read_bytes: float
    final_kv_cache_bytes: float
    kv_write_bytes: float
    required_capacity_bytes: float
    embedding_input_bytes: float
    q_width: int
    kv_width: int
    qkv_input_read_bytes: float
    q_projection_output_bytes: float
    k_projection_cache_write_bytes: float
    v_projection_cache_write_bytes: float
    qkv_activation_bytes: float
    logical_kv_projection_tensor_bytes: float
    physical_kv_projection_write_bytes: float
    kv_projection_output_aliases_final_cache_write: Literal[True] = True
    kv_projection_write_provenance: Literal[
        "K_V_PROJECTION_OUTPUTS_DIRECTLY_MATERIALIZE_FINAL_PREFILL_KV_CACHE"
    ] = "K_V_PROJECTION_OUTPUTS_DIRECTLY_MATERIALIZE_FINAL_PREFILL_KV_CACHE"
    total_kv_cache_write_double_count_status: Literal["PASS"] = "PASS"
    attention_q_read_bytes: float
    attention_k_read_bytes: float
    attention_v_read_bytes: float
    attention_kv_tiled_read_bytes: float
    attention_output_write_bytes: float
    attention_memory_bytes: float
    o_projection_input_read_bytes: float
    o_projection_output_write_bytes: float
    output_projection_memory_bytes: float
    ffn_activation_bytes: float
    final_logits_bytes: float
    ffn_read_bytes: float
    ffn_write_bytes: float
    final_logits_read_bytes: float
    final_logits_write_bytes: float
    prefill_read_bytes: float
    prefill_write_bytes: float
    activation_memory_bytes: float
    linear_memory_bytes: float
    total_memory_bytes: float
    arithmetic_intensity_flop_per_byte: float
    linear_arithmetic_intensity_flop_per_byte: float
    attention_score_matrix_materialized_bytes: Literal[0] = 0
    attention_probability_matrix_materialized_bytes: Literal[0] = 0
    traffic_provenance: Literal[
        "PREFILL_ATTENTION_FUSED_TILED_NO_DECODE_STYLE_FULL_KV_REREAD"
    ] = PREFILL_TRAFFIC_PROVENANCE
    attention_tensor_read_provenance: Literal[
        "PREFILL_ATTENTION_FUSED_TILED_LINEAR_QKV_READ"
    ] = "PREFILL_ATTENTION_FUSED_TILED_LINEAR_QKV_READ"
    kv_read_semantics_status: Literal[
        "NOT_DECODE_STYLE_FULL_HISTORY_REREAD_PER_TOKEN"
    ] = "NOT_DECODE_STYLE_FULL_HISTORY_REREAD_PER_TOKEN"
    kv_dram_scaling_status: Literal[
        "NO_S2_KV_DRAM_TRAFFIC"
    ] = "NO_S2_KV_DRAM_TRAFFIC"
    score_dram_scaling_status: Literal[
        "NO_S2_SCORE_DRAM_TRAFFIC"
    ] = "NO_S2_SCORE_DRAM_TRAFFIC"
    transition_status: Literal[
        "PREFILL_KV_RESIDENT_READY_FOR_DECODE"
    ] = PREFILL_TRANSITION_STATUS

    @model_validator(mode="after")
    def _traffic_closure(self) -> "LLMPrefillMetrics":
        closures = (
            (self.logical_kv_projection_tensor_bytes,
             self.k_projection_cache_write_bytes
             + self.v_projection_cache_write_bytes,
             "logical K/V projection tensor"),
            (self.logical_kv_projection_tensor_bytes,
             self.final_kv_cache_bytes,
             "logical K/V projection to final cache"),
            (self.physical_kv_projection_write_bytes, self.kv_write_bytes,
             "physical K/V projection write alias"),
            (self.kv_write_bytes, self.final_kv_cache_bytes,
             "final KV cache write"),
            (self.qkv_activation_bytes,
             self.qkv_input_read_bytes + self.q_projection_output_bytes,
             "physical QKV activation excluding aliased K/V cache write"),
            (self.attention_kv_tiled_read_bytes,
             self.attention_k_read_bytes + self.attention_v_read_bytes,
             "attention K/V tiled read"),
            (self.attention_memory_bytes,
             self.attention_q_read_bytes + self.attention_k_read_bytes
             + self.attention_v_read_bytes
             + self.attention_output_write_bytes,
             "attention tensor traffic"),
            (self.output_projection_memory_bytes,
             self.o_projection_input_read_bytes
             + self.o_projection_output_write_bytes,
             "output projection tensor traffic"),
            (self.activation_memory_bytes,
             self.embedding_input_bytes + self.qkv_activation_bytes
             + self.attention_memory_bytes
             + self.output_projection_memory_bytes
             + self.ffn_activation_bytes + self.final_logits_bytes,
             "physical activation traffic"),
            (self.linear_memory_bytes,
             self.active_weight_read_bytes + self.embedding_input_bytes
             + self.qkv_activation_bytes + self.kv_write_bytes
             + self.output_projection_memory_bytes
             + self.ffn_activation_bytes + self.final_logits_bytes,
             "linear tensor traffic"),
            (self.total_memory_bytes,
             self.active_weight_read_bytes + self.activation_memory_bytes
             + self.kv_write_bytes,
             "total physical memory traffic"),
            (self.ffn_activation_bytes,
             self.ffn_read_bytes + self.ffn_write_bytes,
             "FFN read/write traffic"),
            (self.final_logits_bytes,
             self.final_logits_read_bytes + self.final_logits_write_bytes,
             "final logits read/write traffic"),
            (self.total_memory_bytes,
             self.prefill_read_bytes + self.prefill_write_bytes,
             "Prefill physical read/write traffic"),
        )
        for actual, expected, name in closures:
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6):
                raise ValueError(f"{name} closure failed")
        return self


class GPUPrefillRooflineMetrics(BaseModel):
    """Peak bounds and reference-calibrated nominal Prefill accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    peak_compute_flops_per_s: float
    peak_compute_capability_status: Literal[
        "VENDOR_REPORTED_BF16_DENSE_PEAK"
    ] = "VENDOR_REPORTED_BF16_DENSE_PEAK"
    large_gemm_effective_flops_per_s: float
    causal_attention_effective_flops_per_s: float
    effective_compute_capability_status: Literal[
        "REFERENCE_CALIBRATED_EFFECTIVE_THROUGHPUT"
    ] = "REFERENCE_CALIBRATED_EFFECTIVE_THROUGHPUT"
    large_gemm_reference_efficiency: float
    causal_attention_reference_efficiency: float
    sustained_memory_bandwidth_bytes_per_s: float
    peak_compute_lower_bound_s: float
    memory_lower_bound_s: float
    peak_roofline_lower_bound_s: float
    linear_nominal_compute_s: float
    attention_nominal_compute_s: float
    nominal_compute_s: float
    nominal_prefill_latency_s: float
    prefill_input_tokens_per_s: float
    nominal_bottleneck: Literal["COMPUTE", "MEMORY"]
    compute_time_status: Literal[
        "THEORETICAL_PEAK_COMPUTE_LOWER_BOUND"
    ] = "THEORETICAL_PEAK_COMPUTE_LOWER_BOUND"
    memory_time_status: Literal[
        "MEMORY_LOWER_BOUND"
    ] = "MEMORY_LOWER_BOUND"
    roofline_time_status: Literal[
        "ROOFLINE_LOWER_BOUND"
    ] = "ROOFLINE_LOWER_BOUND"
    nominal_prefill_latency_status: Literal[
        "REFERENCE_CALIBRATED_EFFECTIVE_GPU_MODEL"
    ] = "REFERENCE_CALIBRATED_EFFECTIVE_GPU_MODEL"
    prefill_effective_model: Literal[
        "REFERENCE_CALIBRATED"
    ] = "REFERENCE_CALIBRATED"
    static_power_W: float
    peak_reference_dynamic_pJ_per_FLOP_min: float
    peak_reference_dynamic_pJ_per_FLOP_max: float
    nominal_gemm_dynamic_pJ_per_FLOP_min: float
    nominal_gemm_dynamic_pJ_per_FLOP_max: float
    nominal_attention_dynamic_pJ_per_FLOP_min: float
    nominal_attention_dynamic_pJ_per_FLOP_max: float
    peak_reference_dynamic_energy_J_min: float
    peak_reference_dynamic_energy_J_max: float
    linear_dynamic_energy_J_min: float
    linear_dynamic_energy_J_max: float
    attention_dynamic_energy_J_min: float
    attention_dynamic_energy_J_max: float
    total_dynamic_energy_J_min: float
    total_dynamic_energy_J_max: float
    static_energy_J_at_peak_roofline_lower_bound: float
    static_energy_status: Literal[
        "STATIC_ENERGY_AT_PEAK_ROOFLINE_LOWER_BOUND"
    ] = "STATIC_ENERGY_AT_PEAK_ROOFLINE_LOWER_BOUND"
    nominal_static_energy_J: float
    nominal_static_energy_status: Literal[
        "STATIC_ENERGY_AT_REFERENCE_CALIBRATED_NOMINAL_PREFILL_LATENCY"
    ] = "STATIC_ENERGY_AT_REFERENCE_CALIBRATED_NOMINAL_PREFILL_LATENCY"
    peak_bound_compute_plus_static_energy_J_min: float
    peak_bound_compute_plus_static_energy_J_max: float
    nominal_compute_plus_static_energy_J_min: float
    nominal_compute_plus_static_energy_J_max: float
    implied_total_power_W_min: float
    implied_total_power_W_max: float
    power_closure_status: Literal[
        "PASS", "FAIL", "NOT_APPLICABLE_MEMORY_BOUND"
    ]
    nominal_energy_status: Literal[
        "REFERENCE_CALIBRATED_COMPUTE_BOUND_POWER_CLOSED",
        "REFERENCE_CALIBRATED_COMPUTE_BOUND_POWER_NOT_CLOSED",
    ]
    prefill_memory_dynamic_energy_completeness: Literal[
        "UNRESOLVED_NOT_INCLUDED"
    ] = "UNRESOLVED_NOT_INCLUDED"
    energy_model_status: Literal[
        "REFERENCE_CALIBRATED_POWER_CLOSED__MEMORY_INTERFACE_DYNAMIC_UNRESOLVED"
    ] = "REFERENCE_CALIBRATED_POWER_CLOSED__MEMORY_INTERFACE_DYNAMIC_UNRESOLVED"


def _active_matrix_weight_bytes(inp: LLMPrefillInput) -> float:
    """Reuse the canonical dense ledger without any decode KV traffic."""
    from .dense_decode_ledger import build_dense_decode_placement_units
    from .llm_decode import LLMDecodeInput

    dimensions = LLMDecodeInput(
        n_param=inp.n_param, n_layers=inp.n_layers,
        n_heads_q=inp.n_heads_q, n_heads_kv=inp.n_heads_kv,
        d_model=inp.d_model, d_ff=inp.d_ff, vocab_size=inp.vocab_size,
        batch_size=1, context_length=0, weight_bits=inp.weight_bits,
        kv_bits=inp.kv_bits,
        weight_activity_model="dimension_derived_active_operators")
    return sum(
        unit.active_weight_read_bytes
        for unit in build_dense_decode_placement_units(dimensions)
        if unit.operator_type in _ACTIVE_MATRIX_OPERATORS)


def evaluate_llm_prefill(inp: LLMPrefillInput) -> LLMPrefillMetrics:
    """Evaluate one dense causal prefill pass and its tensor traffic."""
    B, S, L = inp.batch_size, inp.prompt_length, inp.n_layers
    D, Dff, V = inp.d_model, inp.d_ff, inp.vocab_size
    Hq, Hkv, dh = inp.n_heads_q, inp.n_heads_kv, inp.d_head
    kv_width = Hkv * dh

    causal_pairs = S * (S + 1) // 2
    linear_flops = B * S * L * (
        2 * D * (D + 2 * kv_width) + 2 * D * D + 6 * D * Dff)
    attention_qk_flops = B * L * Hq * dh * S * (S + 1)
    attention_av_flops = attention_qk_flops
    attention_flops = attention_qk_flops + attention_av_flops
    lm_head_flops = 2 * B * D * V
    total_flops = linear_flops + attention_flops + lm_head_flops

    weight_footprint = inp.n_param * inp.weight_bits / 8
    active_weights = _active_matrix_weight_bytes(inp)
    final_kv = 2 * B * L * S * Hkv * dh * inp.kv_bits / 8

    # First-order BF16 tensor reads/writes. QKV input is a single tensor-level
    # read. K/V projection outputs directly materialize the final cache, so
    # their logical output tensors alias (rather than add to) final_kv.
    embedding_input = B * S * D * ACTIVATION_BYTES
    qkv_input_read = L * B * S * D * ACTIVATION_BYTES
    q_projection_output = L * B * S * D * ACTIVATION_BYTES
    k_projection_cache_write = L * B * S * kv_width * inp.kv_bits / 8
    v_projection_cache_write = L * B * S * kv_width * inp.kv_bits / 8
    logical_kv_projection = k_projection_cache_write + v_projection_cache_write
    qkv_activation = qkv_input_read + q_projection_output

    # Fused/tiled attention services each Q/K/V tensor once at O(S) DRAM
    # traffic. Scores and probabilities remain tile-local and contribute zero
    # external S-by-S materialization traffic.
    attention_q_read = L * B * S * D * ACTIVATION_BYTES
    attention_k_read = L * B * S * kv_width * inp.kv_bits / 8
    attention_v_read = L * B * S * kv_width * inp.kv_bits / 8
    attention_kv_tiled_read = attention_k_read + attention_v_read
    attention_output_write = L * B * S * D * ACTIVATION_BYTES
    attention_memory = (
        attention_q_read + attention_k_read + attention_v_read
        + attention_output_write)

    o_projection_input_read = L * B * S * D * ACTIVATION_BYTES
    o_projection_output_write = L * B * S * D * ACTIVATION_BYTES
    output_projection = o_projection_input_read + o_projection_output_write
    ffn_activation = L * B * S * (3 * D + 6 * Dff) * ACTIVATION_BYTES
    final_logits = B * (D + V) * ACTIVATION_BYTES
    # Preserve the existing first-order tensor ledger while making direction
    # explicit.  The FFN term is the same fused SwiGLU traffic already counted
    # above: input/intermediate reads plus intermediate/output writes.
    ffn_read = L * B * S * (2 * D + 4 * Dff) * ACTIVATION_BYTES
    ffn_write = L * B * S * (D + 2 * Dff) * ACTIVATION_BYTES
    final_logits_read = B * D * ACTIVATION_BYTES
    final_logits_write = B * V * ACTIVATION_BYTES
    activation_traffic = (
        embedding_input + qkv_activation + attention_memory
        + output_projection
        + ffn_activation + final_logits)
    linear_memory = (
        active_weights + embedding_input + qkv_activation
        + final_kv + output_projection + ffn_activation + final_logits)
    total_memory = active_weights + final_kv + activation_traffic
    prefill_read = (
        active_weights + embedding_input + qkv_input_read
        + attention_q_read + attention_k_read + attention_v_read
        + o_projection_input_read + ffn_read + final_logits_read)
    prefill_write = (
        q_projection_output + final_kv + attention_output_write
        + o_projection_output_write + ffn_write + final_logits_write)

    return LLMPrefillMetrics(
        causal_token_pairs=causal_pairs, prefill_input_tokens=B * S,
        linear_flops=linear_flops,
        qk_flops=attention_qk_flops,
        av_flops=attention_av_flops,
        attention_flops=attention_flops, lm_head_flops=lm_head_flops,
        total_flops=total_flops,
        linear_flop_fraction=linear_flops / total_flops,
        attention_flop_fraction=attention_flops / total_flops,
        lm_head_flop_fraction=lm_head_flops / total_flops,
        weight_footprint_bytes=weight_footprint,
        active_weight_read_bytes=active_weights,
        final_kv_cache_bytes=final_kv, kv_write_bytes=final_kv,
        required_capacity_bytes=weight_footprint + final_kv,
        embedding_input_bytes=embedding_input,
        q_width=D, kv_width=kv_width,
        qkv_input_read_bytes=qkv_input_read,
        q_projection_output_bytes=q_projection_output,
        k_projection_cache_write_bytes=k_projection_cache_write,
        v_projection_cache_write_bytes=v_projection_cache_write,
        qkv_activation_bytes=qkv_activation,
        logical_kv_projection_tensor_bytes=logical_kv_projection,
        physical_kv_projection_write_bytes=final_kv,
        attention_q_read_bytes=attention_q_read,
        attention_k_read_bytes=attention_k_read,
        attention_v_read_bytes=attention_v_read,
        attention_kv_tiled_read_bytes=attention_kv_tiled_read,
        attention_output_write_bytes=attention_output_write,
        attention_memory_bytes=attention_memory,
        o_projection_input_read_bytes=o_projection_input_read,
        o_projection_output_write_bytes=o_projection_output_write,
        output_projection_memory_bytes=output_projection,
        ffn_activation_bytes=ffn_activation,
        final_logits_bytes=final_logits,
        ffn_read_bytes=ffn_read,
        ffn_write_bytes=ffn_write,
        final_logits_read_bytes=final_logits_read,
        final_logits_write_bytes=final_logits_write,
        prefill_read_bytes=prefill_read,
        prefill_write_bytes=prefill_write,
        activation_memory_bytes=activation_traffic,
        linear_memory_bytes=linear_memory,
        total_memory_bytes=total_memory,
        arithmetic_intensity_flop_per_byte=total_flops / total_memory,
        linear_arithmetic_intensity_flop_per_byte=linear_flops / linear_memory)


def evaluate_gpu_prefill_roofline(
    metrics: LLMPrefillMetrics, *, peak_compute_flops_per_s: float,
    large_gemm_effective_flops_per_s: float,
    causal_attention_effective_flops_per_s: float,
    sustained_memory_bandwidth_bytes_per_s: float, static_power_W: float,
    peak_reference_dynamic_J_per_FLOP_min: float,
    peak_reference_dynamic_J_per_FLOP_max: float,
    nominal_gemm_dynamic_J_per_FLOP_min: float,
    nominal_gemm_dynamic_J_per_FLOP_max: float,
    nominal_attention_dynamic_J_per_FLOP_min: float,
    nominal_attention_dynamic_J_per_FLOP_max: float,
    compute_bound_total_power_W_min: float,
    compute_bound_total_power_W_max: float,
) -> GPUPrefillRooflineMetrics:
    """Apply a dense-BF16 GPU roofline and compute-energy range."""
    values = (
        peak_compute_flops_per_s, large_gemm_effective_flops_per_s,
        causal_attention_effective_flops_per_s,
        sustained_memory_bandwidth_bytes_per_s, static_power_W,
        peak_reference_dynamic_J_per_FLOP_min,
        peak_reference_dynamic_J_per_FLOP_max,
        nominal_gemm_dynamic_J_per_FLOP_min,
        nominal_gemm_dynamic_J_per_FLOP_max,
        nominal_attention_dynamic_J_per_FLOP_min,
        nominal_attention_dynamic_J_per_FLOP_max,
        compute_bound_total_power_W_min,
        compute_bound_total_power_W_max)
    if any(value <= 0 for value in values):
        raise ValueError("GPU roofline rates, power, and energy coefficients must be positive")
    ranges = (
        (peak_reference_dynamic_J_per_FLOP_min,
         peak_reference_dynamic_J_per_FLOP_max),
        (nominal_gemm_dynamic_J_per_FLOP_min,
         nominal_gemm_dynamic_J_per_FLOP_max),
        (nominal_attention_dynamic_J_per_FLOP_min,
         nominal_attention_dynamic_J_per_FLOP_max),
        (compute_bound_total_power_W_min, compute_bound_total_power_W_max),
    )
    if any(maximum < minimum for minimum, maximum in ranges):
        raise ValueError("Prefill energy and power maxima must not be below minima")
    if max(
        large_gemm_effective_flops_per_s,
        causal_attention_effective_flops_per_s,
    ) > peak_compute_flops_per_s:
        raise ValueError("effective Prefill throughput cannot exceed vendor peak")
    peak_compute_s = metrics.total_flops / peak_compute_flops_per_s
    memory_s = metrics.total_memory_bytes / sustained_memory_bandwidth_bytes_per_s
    peak_roofline_s = max(peak_compute_s, memory_s)
    linear_nominal_s = (
        metrics.linear_flops + metrics.lm_head_flops
    ) / large_gemm_effective_flops_per_s
    attention_nominal_s = (
        metrics.attention_flops / causal_attention_effective_flops_per_s)
    nominal_compute_s = linear_nominal_s + attention_nominal_s
    nominal_prefill_s = max(nominal_compute_s, memory_s)
    peak_static_energy = static_power_W * peak_roofline_s
    nominal_static_energy = static_power_W * nominal_prefill_s
    linear_flops = metrics.linear_flops + metrics.lm_head_flops
    peak_dynamic_min = (
        metrics.total_flops * peak_reference_dynamic_J_per_FLOP_min)
    peak_dynamic_max = (
        metrics.total_flops * peak_reference_dynamic_J_per_FLOP_max)
    linear_dynamic_min = (
        linear_flops * nominal_gemm_dynamic_J_per_FLOP_min)
    linear_dynamic_max = (
        linear_flops * nominal_gemm_dynamic_J_per_FLOP_max)
    attention_dynamic_min = (
        metrics.attention_flops
        * nominal_attention_dynamic_J_per_FLOP_min)
    attention_dynamic_max = (
        metrics.attention_flops
        * nominal_attention_dynamic_J_per_FLOP_max)
    total_dynamic_min = linear_dynamic_min + attention_dynamic_min
    total_dynamic_max = linear_dynamic_max + attention_dynamic_max
    nominal_energy_min = total_dynamic_min + nominal_static_energy
    nominal_energy_max = total_dynamic_max + nominal_static_energy
    implied_power_min = nominal_energy_min / nominal_prefill_s
    implied_power_max = nominal_energy_max / nominal_prefill_s
    closure_pass = (
        math.isclose(implied_power_min, compute_bound_total_power_W_min,
                     rel_tol=1e-12, abs_tol=1e-9)
        and math.isclose(implied_power_max, compute_bound_total_power_W_max,
                         rel_tol=1e-12, abs_tol=1e-9))
    return GPUPrefillRooflineMetrics(
        peak_compute_flops_per_s=peak_compute_flops_per_s,
        large_gemm_effective_flops_per_s=large_gemm_effective_flops_per_s,
        causal_attention_effective_flops_per_s=(
            causal_attention_effective_flops_per_s),
        large_gemm_reference_efficiency=(
            large_gemm_effective_flops_per_s / peak_compute_flops_per_s),
        causal_attention_reference_efficiency=(
            causal_attention_effective_flops_per_s
            / peak_compute_flops_per_s),
        sustained_memory_bandwidth_bytes_per_s=sustained_memory_bandwidth_bytes_per_s,
        peak_compute_lower_bound_s=peak_compute_s,
        memory_lower_bound_s=memory_s,
        peak_roofline_lower_bound_s=peak_roofline_s,
        linear_nominal_compute_s=linear_nominal_s,
        attention_nominal_compute_s=attention_nominal_s,
        nominal_compute_s=nominal_compute_s,
        nominal_prefill_latency_s=nominal_prefill_s,
        prefill_input_tokens_per_s=(
            metrics.prefill_input_tokens / nominal_prefill_s),
        nominal_bottleneck=(
            "COMPUTE" if nominal_compute_s >= memory_s else "MEMORY"),
        static_power_W=static_power_W,
        peak_reference_dynamic_pJ_per_FLOP_min=(
            peak_reference_dynamic_J_per_FLOP_min * 1e12),
        peak_reference_dynamic_pJ_per_FLOP_max=(
            peak_reference_dynamic_J_per_FLOP_max * 1e12),
        nominal_gemm_dynamic_pJ_per_FLOP_min=(
            nominal_gemm_dynamic_J_per_FLOP_min * 1e12),
        nominal_gemm_dynamic_pJ_per_FLOP_max=(
            nominal_gemm_dynamic_J_per_FLOP_max * 1e12),
        nominal_attention_dynamic_pJ_per_FLOP_min=(
            nominal_attention_dynamic_J_per_FLOP_min * 1e12),
        nominal_attention_dynamic_pJ_per_FLOP_max=(
            nominal_attention_dynamic_J_per_FLOP_max * 1e12),
        peak_reference_dynamic_energy_J_min=peak_dynamic_min,
        peak_reference_dynamic_energy_J_max=peak_dynamic_max,
        linear_dynamic_energy_J_min=linear_dynamic_min,
        linear_dynamic_energy_J_max=linear_dynamic_max,
        attention_dynamic_energy_J_min=attention_dynamic_min,
        attention_dynamic_energy_J_max=attention_dynamic_max,
        total_dynamic_energy_J_min=total_dynamic_min,
        total_dynamic_energy_J_max=total_dynamic_max,
        static_energy_J_at_peak_roofline_lower_bound=peak_static_energy,
        nominal_static_energy_J=nominal_static_energy,
        peak_bound_compute_plus_static_energy_J_min=(
            peak_dynamic_min + peak_static_energy),
        peak_bound_compute_plus_static_energy_J_max=(
            peak_dynamic_max + peak_static_energy),
        nominal_compute_plus_static_energy_J_min=nominal_energy_min,
        nominal_compute_plus_static_energy_J_max=nominal_energy_max,
        implied_total_power_W_min=implied_power_min,
        implied_total_power_W_max=implied_power_max,
        power_closure_status=(
            "PASS" if closure_pass
            else "NOT_APPLICABLE_MEMORY_BOUND"
            if nominal_compute_s < memory_s else "FAIL"),
        nominal_energy_status=(
            "REFERENCE_CALIBRATED_COMPUTE_BOUND_POWER_CLOSED"
            if closure_pass else
            "REFERENCE_CALIBRATED_COMPUTE_BOUND_POWER_NOT_CLOSED"))
