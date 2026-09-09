"""Cached-prefix incremental Prefill accounting for multi-turn serving."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .llm_prefill import LLMPrefillInput, evaluate_llm_prefill


class CachedPrefixIncrementalPrefillMetrics(BaseModel):
    """One incremental prompt pass whose historical KV already exists."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int
    cached_history_tokens: int
    new_prompt_tokens: int
    decode_start_context_tokens: int
    linear_ffn_tokens_computed_per_request: int
    prefill_input_tokens: int
    attention_query_tokens_per_request: int
    historical_kv_tokens_attended_per_query: int
    attention_pairs_per_request: int
    attention_pairs_total: int
    linear_flops: int
    qk_flops: int
    av_flops: int
    attention_flops: int
    lm_head_flops: int
    total_flops: int
    active_weight_read_bytes: float
    new_token_activation_read_bytes: float
    new_token_activation_write_bytes: float
    historical_cached_kv_read_bytes: float
    cached_kv_bytes_before: float
    new_kv_write_bytes: float
    final_kv_cache_bytes: float
    cached_kv_rewritten: bool = False
    total_read_bytes: float
    total_write_bytes: float
    total_memory_bytes: float
    provenance_status: str = (
        "CACHED_PREFIX_INCREMENTAL_PREFILL__HISTORY_KV_READ_ONLY__"
        "NEW_QUERY_LINEAR_AND_ATTENTION_COMPUTE")

    @model_validator(mode="after")
    def _closure(self) -> "CachedPrefixIncrementalPrefillMetrics":
        if self.linear_ffn_tokens_computed_per_request != self.new_prompt_tokens:
            raise ValueError("cached history must not execute linear/FFN operators")
        expected_pairs = (
            self.new_prompt_tokens*self.cached_history_tokens
            + self.new_prompt_tokens*(self.new_prompt_tokens+1)//2)
        if self.attention_pairs_per_request != expected_pairs:
            raise ValueError("incremental attention pair count does not close")
        if not math.isclose(
            self.final_kv_cache_bytes,
            self.cached_kv_bytes_before+self.new_kv_write_bytes,
            rel_tol=1e-12, abs_tol=1e-6,
        ):
            raise ValueError("cached plus appended KV does not close")
        if self.cached_kv_rewritten:
            raise ValueError("cached prefix KV must remain read-only")
        if not math.isclose(
            self.total_memory_bytes, self.total_read_bytes+self.total_write_bytes,
            rel_tol=1e-12, abs_tol=1e-6,
        ):
            raise ValueError("incremental Prefill traffic does not close")
        return self


def evaluate_cached_prefix_incremental_prefill(
    inp: LLMPrefillInput, *, cached_history_tokens: int,
) -> CachedPrefixIncrementalPrefillMetrics:
    """Evaluate only new-token compute while attending to cached historical KV.

    The existing fused/tiled Prefill traffic convention is retained: the
    historical K/V tensors are streamed once for the incremental attention
    pass and score/probability matrices are not materialized in memory.
    """
    if cached_history_tokens <= 0:
        raise ValueError("cached_history_tokens must be positive")
    base = evaluate_llm_prefill(inp)
    B, Q, L = inp.batch_size, inp.prompt_length, inp.n_layers
    kv_width = inp.n_heads_kv*inp.d_head
    pairs = Q*cached_history_tokens+Q*(Q+1)//2
    qk = 2*B*L*inp.n_heads_q*inp.d_head*pairs
    av = qk
    attention = qk+av
    cached_kv = (
        2*B*L*cached_history_tokens*kv_width*inp.kv_bits/8)
    historical_read = cached_kv
    activation_read = base.prefill_read_bytes-base.active_weight_read_bytes
    activation_write = base.prefill_write_bytes
    total_read = base.prefill_read_bytes+historical_read
    total_write = base.prefill_write_bytes
    total_flops = base.linear_flops+attention+base.lm_head_flops
    return CachedPrefixIncrementalPrefillMetrics(
        batch_size=B,
        cached_history_tokens=cached_history_tokens,
        new_prompt_tokens=Q,
        decode_start_context_tokens=cached_history_tokens+Q,
        linear_ffn_tokens_computed_per_request=Q,
        prefill_input_tokens=B*Q,
        attention_query_tokens_per_request=Q,
        historical_kv_tokens_attended_per_query=cached_history_tokens,
        attention_pairs_per_request=pairs,
        attention_pairs_total=B*pairs,
        linear_flops=base.linear_flops,
        qk_flops=qk,
        av_flops=av,
        attention_flops=attention,
        lm_head_flops=base.lm_head_flops,
        total_flops=total_flops,
        active_weight_read_bytes=base.active_weight_read_bytes,
        new_token_activation_read_bytes=activation_read,
        new_token_activation_write_bytes=activation_write,
        historical_cached_kv_read_bytes=historical_read,
        cached_kv_bytes_before=cached_kv,
        new_kv_write_bytes=base.final_kv_cache_bytes,
        final_kv_cache_bytes=cached_kv+base.final_kv_cache_bytes,
        total_read_bytes=total_read,
        total_write_bytes=total_write,
        total_memory_bytes=total_read+total_write,
    )
