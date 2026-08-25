"""Analytical LLM autoregressive-decode workload primitive.

This module implements the frozen B1-R2 specification as a small,
architecture-independent workload model.  It computes per-token
footprint, traffic, and FLOPs for dense Transformer decoder inference.

All assumptions are documented as either formula/algorithmic definitions
or explicit MODELING_CHOICE annotations.

Accounting policy (frozen for B1-R2):
- Internal canonical accounting computes integer bit counts first.
- Byte-valued analytical outputs use ordinary (true) division, e.g.
  ``bytes = bits / 8``.  Per-token averages use ordinary division,
  e.g. ``traffic_per_token = traffic_per_aggregate_step / batch_size``.
- These values are *exact analytical byte-equivalents*; they are not
  a physical byte-array allocation.  No implicit ``ceil`` packing is
  applied at this stage.  If a future task needs a physical packed
  allocation, that must be exposed as a separate metric; it must not
  silently ceil/floor the analytical metric.
- ``// 8`` and ``// batch_size`` are therefore forbidden for
  footprint or per-token traffic accounting in this module.
- FLOP counts remain integers; they are not subject to the same
  fractional-byte rule.
- ``runtime_bytes`` is supplied as an integer (bytes), and is added
  to other byte-equivalent totals consistently in the metrics
  output (no int/float semantic confusion in the sum).
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class LLMDecodeInput(BaseModel):
    """Input specification for a single LLM autoregressive decode workload.

    All user-supplied dimensional fields are positive scalars.  The attention
    head dimension is derived exactly as ``d_model / n_heads_q`` and remains
    present in serialized resolved inputs.
    """

    model_config = ConfigDict(extra="forbid")

    # Model architecture
    n_param: int = Field(gt=0, description="Total trainable parameters")
    n_layers: int = Field(gt=0, description="Transformer decoder layers (L)")
    n_heads_q: int = Field(gt=0, description="Query heads (Hq)")
    n_heads_kv: int = Field(gt=0, description="Key/Value heads (Hkv)")
    d_model: int = Field(gt=0, description="Transformer hidden dimension")
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

    @field_validator("n_layers", "n_heads_q", "n_heads_kv", "d_model",
                     "d_ff", "vocab_size", "batch_size", "weight_bits", "kv_bits")
    @classmethod
    def _positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @model_validator(mode="after")
    def _gqa_semantics(self) -> "LLMDecodeInput":
        """Validate GQA/MQA/MHA head relationships.

        Standard v0 supports only *uniform* grouped-query attention:
        every query head is bound to exactly one KV head, and the KV
        heads are evenly distributed across the query heads.  This
        requires two conditions:

        1. ``n_heads_kv <= n_heads_q`` (KV heads do not outnumber query
           heads; MHA = equal, GQA = fewer, MQA = one).
        2. ``n_heads_q % n_heads_kv == 0`` (the grouping is uniform,
           i.e. each KV head is shared by an integer number of query
           heads).

        Non-uniform mappings (e.g. ``Hq=6, Hkv=4``) are rejected.
        """
        if self.n_heads_kv > self.n_heads_q:
            raise ValueError("n_heads_kv must not exceed n_heads_q")
        if self.n_heads_q % self.n_heads_kv != 0:
            raise ValueError(
                f"n_heads_q ({self.n_heads_q}) must be evenly divisible by "
                f"n_heads_kv ({self.n_heads_kv}) for uniform GQA/MQA/MHA; "
                f"non-uniform head mappings are not supported in v0"
            )
        return self

    @model_validator(mode="after")
    def _head_dimension_is_integral(self) -> "LLMDecodeInput":
        """The standard v0 attention head dimension must be integral."""
        if self.d_model % self.n_heads_q != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be evenly divisible by "
                f"n_heads_q ({self.n_heads_q})"
            )
        return self

    @computed_field(return_type=int)
    @property
    def d_head(self) -> int:
        """SOFTWARE_DERIVED attention head dimension."""
        return self.d_model // self.n_heads_q


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class LLMDecodeMetrics(BaseModel):
    """Output metrics for a single LLM decode evaluation.

    All byte-valued fields are exact analytical *byte-equivalents*
    computed with true division (no floor).  They may be non-integer
    (e.g. ``0.125`` for a 1-bit / 1-parameter weight footprint).
    These values are NOT a physical byte-array allocation; see the
    module-level accounting policy for details.

    All FLOP-valued fields are integer scalar counts.
    """

    # Footprint (exact analytical byte-equivalents; may be fractional)
    weight_footprint_bytes: float
    weight_active_per_step_bytes: float
    kv_footprint_bytes: float
    runtime_bytes: float
    required_capacity_bytes: float

    # Traffic per generated token (exact analytical byte-equivalents; may be fractional)
    weight_read_bytes_per_token: float
    kv_read_bytes_per_token: float
    kv_write_bytes_per_token: float
    read_bytes_per_token: float
    write_bytes_per_token: float

    # Compute per generated token (integer counts)
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

    Per the accounting policy, byte-valued outputs use true division
    (``/ 8``, ``/ B``); they are *not* floored.  Sub-byte weight or KV
    configurations therefore preserve their analytical bit content
    exactly (e.g. ``n_param=1, weight_bits=1`` -> ``0.125`` bytes).
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
    # Formula: Nparam * bw / 8  (true division; exact analytical byte-equivalent)
    bytes_weight_footprint = Nparam * bw / 8

    # MODELING_CHOICE (v0): conservative – assume all resident weights read.
    bytes_weight_active_per_step = bytes_weight_footprint

    # -------------------------------------------------------------
    # 5. KV footprint
    # -------------------------------------------------------------
    # Formula: 2 * L * B * S * Hkv * Dhead * bkv / 8  (true division)
    bytes_kv_footprint = 2 * L * B * S * Hkv * Dhead * bkv / 8

    # -------------------------------------------------------------
    # 6. Required capacity on the workload side
    # -------------------------------------------------------------
    # Sum is consistent with the byte-equivalent type of all addends.
    bytes_required = bytes_weight_footprint + bytes_kv_footprint + runtime

    # -------------------------------------------------------------
    # 7. Decode traffic – weight read per token
    # -------------------------------------------------------------
    # MODELING_CHOICE (v0): tile_reuse – weight tile services B inputs.
    # Therefore per-token weight read = active_weight / B  (true division).
    weight_read_per_token = bytes_weight_active_per_step / B

    # -------------------------------------------------------------
    # 8. KV traffic
    # -------------------------------------------------------------
    # MODELING_CHOICE (v0): full_reread – all historical K/V re-read.
    # KV read per token = 2 * L * S * Hkv * Dhead * bkv / 8  (true division;
    # independent of B).
    kv_read_per_token = 2 * L * S * Hkv * Dhead * bkv / 8

    # KV write per token = 2 * L * Hkv * Dhead * bkv / 8  (true division;
    # independent of S).
    kv_write_per_token = 2 * L * Hkv * Dhead * bkv / 8

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
        runtime_bytes=float(runtime),
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
