"""Analytical autoregressive decode accounting for structural MoE models."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .llm_decode import calculate_kv_decode_accounting


class MoEDecodeInput(BaseModel):
    """Structural MoE input; capacity and active traffic remain distinct."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    num_hidden_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    num_local_experts: int = Field(gt=0)
    num_experts_per_tok: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    tie_word_embeddings: bool
    dtype: Literal["BF16"]
    weight_bits: int = Field(gt=0)
    kv_bits: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    context_length: int = Field(ge=0)
    max_position_embeddings: int = Field(gt=0)
    runtime_fixed_bytes: int = Field(default=0, ge=0)
    runtime_per_request_bytes: int = Field(default=0, ge=0)
    routing_baseline: Literal["UNIFORM_ROUTING_BASELINE"]
    real_expert_popularity_available: Literal["NO"] = "NO"
    weight_activity_model: Literal["full_active_parameter_footprint"] = (
        "full_active_parameter_footprint")
    weight_reuse_model: Literal["tile_reuse"] = "tile_reuse"
    kv_read_model: Literal["full_reread"] = "full_reread"

    @model_validator(mode="after")
    def _structural_closure(self) -> "MoEDecodeInput":
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden size must close to attention heads * head_dim")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("KV heads cannot exceed attention heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("attention heads must divide evenly over KV heads")
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError("top-k experts cannot exceed local experts")
        if self.context_length > self.max_position_embeddings:
            raise ValueError("context length exceeds model max_position_embeddings")
        if self.dtype == "BF16" and (self.weight_bits != 16 or self.kv_bits != 16):
            raise ValueError("BF16 canonical accounting requires 16-bit weight and KV")
        return self


class MoEDecodeMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    expert_count: int
    parameters_per_expert: int
    expert_parameters_total: int
    attention_parameters_total: int
    router_parameters_total: int
    rmsnorm_parameters_total: int
    token_embedding_parameters: int
    lm_head_parameters: int
    nonexpert_parameters_total: int
    total_parameters: int
    active_expert_parameters_per_token: int
    active_parameters_per_token: int
    active_to_total_parameter_ratio: float
    expert_footprint_bytes: float
    all_expert_footprint_bytes: float
    nonexpert_footprint_bytes: float
    total_weight_footprint_bytes: float
    active_expert_weight_bytes_per_decode_step: float
    active_nonexpert_weight_bytes_per_decode_step: float
    active_weight_bytes_per_decode_step: float
    nonexpert_weight_read_bytes_per_token: float
    expert_weight_read_bytes_per_token: float
    total_weight_read_bytes_per_token: float
    kv_bytes_per_token_per_request: float
    kv_bytes_per_request: float
    kv_footprint_bytes: float
    kv_read_bytes_per_token_per_request: float
    kv_write_bytes_per_token_per_request: float
    runtime_footprint_bytes: float
    required_capacity_bytes: float
    active_expert_flops_per_token: int
    flops_per_token: int
    expert_selection_probability: float
    expert_selection_probabilities_per_layer: tuple[float, ...]
    expert_selection_probability_sum_per_layer: float
    uniform_expected_read_bytes_per_expert_per_decode_step: float
    routing_semantics_status: Literal["STRUCTURAL_NEUTRAL_BASELINE"]
    routing_trace_status: Literal["NOT_REAL_ROUTING_TRACE"]
    expert_popularity_status: Literal["NOT_WORKLOAD_POPULARITY"]
    expert_demand_skew_status: Literal[
        "EXPERT_DEMAND_SKEW_UNAVAILABLE_WITHOUT_TRACE"]
    weight_reuse_status: Literal[
        "TOP_K_ACTIVE_SET_TILE_REUSED_ACROSS_BATCH_MODELING_CHOICE"]
    flop_status: Literal["ACTIVE_TOP_K_ANALYTICAL_FLOPS"]


def evaluate_moe_decode(inp: MoEDecodeInput) -> MoEDecodeMetrics:
    """Derive resident parameters and top-k active traffic from dimensions."""
    layers = inp.num_hidden_layers
    hidden = inp.hidden_size
    intermediate = inp.intermediate_size
    kv_output = inp.num_key_value_heads * inp.head_dim
    bytes_per_weight = inp.weight_bits / 8

    parameters_per_expert = 3 * hidden * intermediate
    expert_count = layers * inp.num_local_experts
    expert_total = expert_count * parameters_per_expert
    attention_per_layer = (
        hidden * hidden
        + hidden * kv_output
        + hidden * kv_output
        + hidden * hidden)
    attention_total = layers * attention_per_layer
    router_total = layers * hidden * inp.num_local_experts
    rmsnorm_total = (2 * layers + 1) * hidden
    embedding = inp.vocab_size * hidden
    lm_head = 0 if inp.tie_word_embeddings else inp.vocab_size * hidden
    nonexpert_total = (
        attention_total + router_total + rmsnorm_total + embedding + lm_head)
    total_parameters = expert_total + nonexpert_total
    active_expert = layers * inp.num_experts_per_tok * parameters_per_expert
    active_parameters = nonexpert_total + active_expert

    expert_footprint = parameters_per_expert * bytes_per_weight
    all_expert_footprint = expert_total * bytes_per_weight
    nonexpert_footprint = nonexpert_total * bytes_per_weight
    total_weight_footprint = total_parameters * bytes_per_weight
    active_expert_bytes = active_expert * bytes_per_weight
    active_nonexpert_bytes = nonexpert_footprint
    active_weight_bytes = active_nonexpert_bytes + active_expert_bytes

    kv = calculate_kv_decode_accounting(
        n_layers=layers,
        batch_size=inp.batch_size,
        context_length=inp.context_length,
        n_heads_kv=inp.num_key_value_heads,
        d_head=inp.head_dim,
        kv_bits=inp.kv_bits,
    )
    runtime = (
        inp.runtime_fixed_bytes
        + inp.batch_size * inp.runtime_per_request_bytes)
    probability = inp.num_experts_per_tok / inp.num_local_experts

    qkv_flops = 2 * hidden * (hidden + 2 * kv_output)
    output_projection_flops = 2 * hidden * hidden
    active_expert_flops_per_layer = (
        2 * inp.num_experts_per_tok * parameters_per_expert)
    router_flops = 2 * hidden * inp.num_local_experts
    attention_flops = (
        4 * inp.num_attention_heads * inp.context_length * inp.head_dim)
    vocab_flops = 2 * hidden * inp.vocab_size
    flops = layers * (
        qkv_flops
        + output_projection_flops
        + active_expert_flops_per_layer
        + router_flops
        + attention_flops
    ) + vocab_flops

    values = (
        total_parameters,
        active_parameters,
        total_weight_footprint,
        active_weight_bytes,
        kv.bytes_per_request,
        kv.footprint_bytes,
    )
    if any(not math.isfinite(float(value)) or value < 0 for value in values):
        raise ValueError("MoE analytical accounting must be finite and non-negative")
    return MoEDecodeMetrics(
        model_id=inp.model_id,
        expert_count=expert_count,
        parameters_per_expert=parameters_per_expert,
        expert_parameters_total=expert_total,
        attention_parameters_total=attention_total,
        router_parameters_total=router_total,
        rmsnorm_parameters_total=rmsnorm_total,
        token_embedding_parameters=embedding,
        lm_head_parameters=lm_head,
        nonexpert_parameters_total=nonexpert_total,
        total_parameters=total_parameters,
        active_expert_parameters_per_token=active_expert,
        active_parameters_per_token=active_parameters,
        active_to_total_parameter_ratio=active_parameters / total_parameters,
        expert_footprint_bytes=expert_footprint,
        all_expert_footprint_bytes=all_expert_footprint,
        nonexpert_footprint_bytes=nonexpert_footprint,
        total_weight_footprint_bytes=total_weight_footprint,
        active_expert_weight_bytes_per_decode_step=active_expert_bytes,
        active_nonexpert_weight_bytes_per_decode_step=active_nonexpert_bytes,
        active_weight_bytes_per_decode_step=active_weight_bytes,
        nonexpert_weight_read_bytes_per_token=(
            active_nonexpert_bytes / inp.batch_size),
        expert_weight_read_bytes_per_token=(
            active_expert_bytes / inp.batch_size),
        total_weight_read_bytes_per_token=(
            active_weight_bytes / inp.batch_size),
        kv_bytes_per_token_per_request=kv.write_bytes_per_token,
        kv_bytes_per_request=kv.bytes_per_request,
        kv_footprint_bytes=kv.footprint_bytes,
        kv_read_bytes_per_token_per_request=kv.read_bytes_per_token,
        kv_write_bytes_per_token_per_request=kv.write_bytes_per_token,
        runtime_footprint_bytes=float(runtime),
        required_capacity_bytes=(
            total_weight_footprint + kv.footprint_bytes + runtime),
        active_expert_flops_per_token=(
            layers * active_expert_flops_per_layer),
        flops_per_token=flops,
        expert_selection_probability=probability,
        expert_selection_probabilities_per_layer=tuple(
            probability for _ in range(inp.num_local_experts)),
        expert_selection_probability_sum_per_layer=(
            inp.num_local_experts * probability),
        uniform_expected_read_bytes_per_expert_per_decode_step=(
            expert_footprint * probability),
        routing_semantics_status="STRUCTURAL_NEUTRAL_BASELINE",
        routing_trace_status="NOT_REAL_ROUTING_TRACE",
        expert_popularity_status="NOT_WORKLOAD_POPULARITY",
        expert_demand_skew_status=(
            "EXPERT_DEMAND_SKEW_UNAVAILABLE_WITHOUT_TRACE"),
        weight_reuse_status=(
            "TOP_K_ACTIVE_SET_TILE_REUSED_ACROSS_BATCH_MODELING_CHOICE"),
        flop_status="ACTIVE_TOP_K_ANALYTICAL_FLOPS",
    )
