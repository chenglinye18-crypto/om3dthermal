"""Dense causal LLM prefill accounting and a GPU roofline primitive.

This is independent from autoregressive decode. It models one complete prompt
pass, leaves the final KV cache resident for decode, and never materializes an
S-by-S attention-score matrix in DRAM.
"""

from __future__ import annotations

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
    qkv_activation_bytes: float
    attention_kernel_activation_bytes: float
    output_projection_activation_bytes: float
    attention_output_bytes: float
    ffn_activation_bytes: float
    final_logits_bytes: float
    activation_memory_bytes: float
    linear_memory_bytes: float
    total_memory_bytes: float
    arithmetic_intensity_flop_per_byte: float
    linear_arithmetic_intensity_flop_per_byte: float
    attention_score_matrix_materialized_bytes: Literal[0] = 0
    traffic_provenance: Literal[
        "PREFILL_ATTENTION_FUSED_TILED_NO_DECODE_STYLE_FULL_KV_REREAD"
    ] = PREFILL_TRAFFIC_PROVENANCE
    transition_status: Literal[
        "PREFILL_KV_RESIDENT_READY_FOR_DECODE"
    ] = PREFILL_TRANSITION_STATUS


class GPUPrefillRooflineMetrics(BaseModel):
    """Compute/memory roofline time and compute-dynamic GPU energy range."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    peak_compute_flops_per_s: float
    sustained_memory_bandwidth_bytes_per_s: float
    compute_lower_bound_ms: float
    memory_lower_bound_ms: float
    roofline_lower_bound_ms: float
    roofline_bottleneck: Literal["COMPUTE", "MEMORY"]
    static_power_W: float
    dynamic_compute_energy_J_min: float
    dynamic_compute_energy_J_max: float
    static_energy_J_at_roofline_bound: float
    gpu_energy_J_min: float
    gpu_energy_J_max: float
    energy_model_status: Literal[
        "COMPUTE_DYNAMIC_PER_FLOP_RANGE_PLUS_STATIC_ROOFLINE_TIME"
    ] = "COMPUTE_DYNAMIC_PER_FLOP_RANGE_PLUS_STATIC_ROOFLINE_TIME"


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

    # First-order BF16 tensor reads/writes. K/V projection results are consumed
    # by tiled attention on-chip and written exactly once as the final cache.
    embedding_input = B * S * D * ACTIVATION_BYTES
    qkv_activation = L * B * S * (3 * D + D) * ACTIVATION_BYTES
    attention_kernel = L * B * S * (D + D) * ACTIVATION_BYTES
    output_projection = L * B * S * (D + D) * ACTIVATION_BYTES
    attention_output = attention_kernel + output_projection
    ffn_activation = L * B * S * (3 * D + 6 * Dff) * ACTIVATION_BYTES
    final_logits = B * (D + V) * ACTIVATION_BYTES
    activation_traffic = (
        embedding_input + qkv_activation + attention_output
        + ffn_activation + final_logits)
    linear_memory = (
        active_weights + embedding_input + qkv_activation
        + output_projection + ffn_activation + final_logits)
    total_memory = active_weights + final_kv + activation_traffic

    return LLMPrefillMetrics(
        causal_token_pairs=causal_pairs, linear_flops=linear_flops,
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
        qkv_activation_bytes=qkv_activation,
        attention_kernel_activation_bytes=attention_kernel,
        output_projection_activation_bytes=output_projection,
        attention_output_bytes=attention_output,
        ffn_activation_bytes=ffn_activation,
        final_logits_bytes=final_logits,
        activation_memory_bytes=activation_traffic,
        linear_memory_bytes=linear_memory,
        total_memory_bytes=total_memory,
        arithmetic_intensity_flop_per_byte=total_flops / total_memory,
        linear_arithmetic_intensity_flop_per_byte=linear_flops / linear_memory)


def evaluate_gpu_prefill_roofline(
    metrics: LLMPrefillMetrics, *, peak_compute_flops_per_s: float,
    sustained_memory_bandwidth_bytes_per_s: float, static_power_W: float,
    e_compute_dynamic_J_per_FLOP_min: float,
    e_compute_dynamic_J_per_FLOP_max: float,
) -> GPUPrefillRooflineMetrics:
    """Apply a dense-BF16 GPU roofline and compute-energy range."""
    values = (peak_compute_flops_per_s, sustained_memory_bandwidth_bytes_per_s,
              static_power_W, e_compute_dynamic_J_per_FLOP_min,
              e_compute_dynamic_J_per_FLOP_max)
    if any(value <= 0 for value in values):
        raise ValueError("GPU roofline rates, power, and energy coefficients must be positive")
    if e_compute_dynamic_J_per_FLOP_max < e_compute_dynamic_J_per_FLOP_min:
        raise ValueError("compute dynamic energy maximum must not be below minimum")
    compute_s = metrics.total_flops / peak_compute_flops_per_s
    memory_s = metrics.total_memory_bytes / sustained_memory_bandwidth_bytes_per_s
    roofline_s = max(compute_s, memory_s)
    static_energy = static_power_W * roofline_s
    dynamic_min = metrics.total_flops * e_compute_dynamic_J_per_FLOP_min
    dynamic_max = metrics.total_flops * e_compute_dynamic_J_per_FLOP_max
    return GPUPrefillRooflineMetrics(
        peak_compute_flops_per_s=peak_compute_flops_per_s,
        sustained_memory_bandwidth_bytes_per_s=sustained_memory_bandwidth_bytes_per_s,
        compute_lower_bound_ms=compute_s * 1e3,
        memory_lower_bound_ms=memory_s * 1e3,
        roofline_lower_bound_ms=roofline_s * 1e3,
        roofline_bottleneck="COMPUTE" if compute_s >= memory_s else "MEMORY",
        static_power_W=static_power_W,
        dynamic_compute_energy_J_min=dynamic_min,
        dynamic_compute_energy_J_max=dynamic_max,
        static_energy_J_at_roofline_bound=static_energy,
        gpu_energy_J_min=dynamic_min + static_energy,
        gpu_energy_J_max=dynamic_max + static_energy)
