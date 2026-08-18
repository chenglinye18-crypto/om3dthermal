"""Analytical LLM autoregressive-decode workload primitive.

This module implements the frozen B1-R2 specification as a small,
architecture-independent workload model.  It computes per-token
footprint, traffic, and FLOPs for dense Transformer decoder inference.

All assumptions are documented as either formula/algorithmic definitions
or explicit MODELING_CHOICE annotations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class LLMDecodeInput(BaseModel):
    """Input specification for a single LLM autoregressive decode workload.

    All dimensional fields are positive scalars.  No silent inference of
    missing dimensions is performed; the caller must supply every field.
    """

    # Model architecture
    n_param: int = Field(gt=0, description="Total trainable parameters")
    n_layers: int = Field(gt=0, description="Transformer decoder layers (L)")
    n_heads_q: int = Field(gt=0, description="Query heads (Hq)")
    n_heads_kv: int = Field(gt=0, description="Key/Value heads (Hkv)")
    d_head: int = Field(gt=0, description="Attention head dimension")
    d_model: int = Field(gt=0, description="Hidden dimension (Hq * Dhead)")
    d_ff: int = Field(gt=0, description="MLP intermediate dimension")
    vocab_size: int = Field(gt=0, description="Vocabulary size (V)")

    # Workload state
    batch_size: int = Field(gt=0, description="Concurrent sequences (B)")
    context_length: int = Field(
        ge=0, description="Current context length per sequence (S)"
    )

    # Precision
    weight_bits: int = Field(gt=0, description="Bits per weight parameter")
    kv_bits: int = Field(gt=0, description="Bits per KV-cache element")

    # Overhead
    runtime_bytes: int = Field(
        ge=0, default=0, description="Activation workspace / scheduler metadata bytes"
    )

    # Modeling choices (frozen for v0)
    weight_activity_model: Literal["full_footprint"] = Field(
        default="full_footprint",
        description="MODELING_CHOICE: v0 assumes all resident weights are read each step",
    )
    weight_reuse_model: Literal["tile_reuse"] = Field(
        default="tile_reuse",
        description=(
            "MODELING_CHOICE: v0 assumes a single weight tile is reused across "
            "batch inputs within an aggregate decode step"
        ),
    )
    kv_read_model: Literal["full_reread"] = Field(
        default="full_reread",
        description=(
            "MODELING_CHOICE: v0 assumes all historical K/V are re-read each step"
        ),
    )

    @field_validator("n_layers", "n_heads_q", "n_heads_kv", "d_head", "d_model",
                     "d_ff", "vocab_size", "batch_size", "weight_bits", "kv_bits")
    @classmethod
    def _positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @model_validator(mode="after")
    def _gqa_semantics(self) -> "LLMDecodeInput":
        """Validate GQA/MQA/MHA head relationships."""
        if self.n_heads_kv > self.n_heads_q:
            raise ValueError("n_heads_kv must not exceed n_heads_q")
        return self

    @model_validator(mode="after")
    def _dmodel_consistency(self) -> "LLMDecodeInput":
        """d_model must equal Hq * Dhead for the standard decomposition."""
        expected = self.n_heads_q * self.d_head
        if expected != self.d_model:
            raise ValueError(
                f"d_model ({self.d_model}) must equal n_heads_q * d_head "
                f"({expected})"
            )
        return self


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class LLMDecodeMetrics(BaseModel):
    """Output metrics for a single LLM decode evaluation.

    All byte fields are in bytes; all FLOP fields are scalar counts.
    """

    # Footprint
    weight_footprint_bytes: int
    weight_active_per_step_bytes: int
    kv_footprint_bytes: int
    runtime_bytes: int
    required_capacity_bytes: int

    # Traffic per generated token
    weight_read_bytes_per_token: int
    kv_read_bytes_per_token: int
    kv_write_bytes_per_token: int
    read_bytes_per_token: int
    write_bytes_per_token: int

    # Compute per generated token
    flops_per_token: int
    flops_sanity_per_token: int

    # Provenance / modeling choices ( echoed for audit )
    weight_activity_model: Literal["full_footprint"]
    weight_reuse_model: Literal["tile_reuse"]
    kv_read_model: Literal["full_reread"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_llm_decode(inp: LLMDecodeInput) -> LLMDecodeMetrics:
    """Evaluate the analytical LLM decode workload primitive.

    Computes footprint, per-token traffic, and per-token FLOPs according
    to the frozen B1-R2 specification.
    """
    L = inp.n_layers
    B = inp.batch_size
    S = inp.context_length
    Hq = inp.n_heads_q
    Hkv = inp.n_heads_kv
    Dhead = inp.d_head
    Dmodel = inp.d_model
    Dff = inp.d_ff
    V = inp.vocab_size
    Nparam = inp.n_param
    bw = inp.weight_bits
    bkv = inp.kv_bits
    runtime = inp.runtime_bytes

    # -------------------------------------------------------------
    # 4. Weight footprint (resident)
    # -------------------------------------------------------------
    # Formula: Nparam * bw / 8
    bytes_weight_footprint = Nparam * bw // 8

    # MODELING_CHOICE (v0): conservative – assume all resident weights read.
    bytes_weight_active_per_step = bytes_weight_footprint

    # -------------------------------------------------------------
    # 5. KV footprint
    # -------------------------------------------------------------
    # Formula: 2 * L * B * S * Hkv * Dhead * bkv / 8
    bytes_kv_footprint = (2 * L * B * S * Hkv * Dhead * bkv) // 8

    # -------------------------------------------------------------
    # 6. Required capacity on the workload side
    # -------------------------------------------------------------
    bytes_required = bytes_weight_footprint + bytes_kv_footprint + runtime

    # -------------------------------------------------------------
    # 7. Decode traffic – weight read per token
    # -------------------------------------------------------------
    # MODELING_CHOICE (v0): tile_reuse – weight tile services B inputs.
    # Therefore per-token weight read = active_weight / B.
    weight_read_per_token = bytes_weight_active_per_step // B

    # -------------------------------------------------------------
    # 8. KV traffic
    # -------------------------------------------------------------
    # MODELING_CHOICE (v0): full_reread – all historical K/V re-read.
    # KV read per token = 2 * L * S * Hkv * Dhead * bkv / 8  (independent of B)
    kv_read_per_token = (2 * L * S * Hkv * Dhead * bkv) // 8

    # KV write per token = 2 * L * Hkv * Dhead * bkv / 8  (independent of S)
    kv_write_per_token = (2 * L * Hkv * Dhead * bkv) // 8

    read_per_token = weight_read_per_token + kv_read_per_token
    write_per_token = kv_write_per_token

    # -------------------------------------------------------------
    # 9. FLOPs per token (detailed)
    # -------------------------------------------------------------
    # Per-layer decomposition:
    #   QKV proj:  2 * B * Dmodel * (Dmodel + 2*Hkv*Dhead)
    #   Out proj:  2 * B * Dmodel^2
    #   MLP:       6 * B * Dmodel * Dff
    #   Attention: 4 * B * Hq * S * Dhead
    # All layers + vocab projection: 2 * B * Dmodel * V
    # Then divide by B for per-token.
    flops_qkv_per_layer = 2 * Dmodel * (Dmodel + 2 * Hkv * Dhead)
    flops_out_proj_per_layer = 2 * Dmodel * Dmodel
    flops_mlp_per_layer = 6 * Dmodel * Dff
    flops_attention_per_layer = 4 * Hq * S * Dhead
    flops_vocab = 2 * Dmodel * V

    flops_per_token = (
        L * (
            flops_qkv_per_layer
            + flops_out_proj_per_layer
            + flops_mlp_per_layer
            + flops_attention_per_layer
        )
        + flops_vocab
    )

    # Sanity cross-check
    # Formula: 2 * Nparam + 4 * L * Hq * S * Dhead
    flops_sanity = 2 * Nparam + 4 * L * Hq * S * Dhead

    return LLMDecodeMetrics(
        weight_footprint_bytes=bytes_weight_footprint,
        weight_active_per_step_bytes=bytes_weight_active_per_step,
        kv_footprint_bytes=bytes_kv_footprint,
        runtime_bytes=runtime,
        required_capacity_bytes=bytes_required,
        weight_read_bytes_per_token=weight_read_per_token,
        kv_read_bytes_per_token=kv_read_per_token,
        kv_write_bytes_per_token=kv_write_per_token,
        read_bytes_per_token=read_per_token,
        write_bytes_per_token=write_per_token,
        flops_per_token=flops_per_token,
        flops_sanity_per_token=flops_sanity,
        weight_activity_model=inp.weight_activity_model,
        weight_reuse_model=inp.weight_reuse_model,
        kv_read_model=inp.kv_read_model,
    )
