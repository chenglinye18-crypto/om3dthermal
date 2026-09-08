"""Analytical live-tensor workspace for the unified serving timeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.workload import DenseLLMModelSpec


class WorkspaceExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prefill_chunk_tokens: int = Field(gt=0)
    attention_tile_tokens: int = Field(gt=0)
    activation_bytes: int = Field(gt=0)
    attention_accumulator_bytes: int = Field(gt=0)
    provenance_status: Literal[
        "MODELING_CHOICE_UNIFORM_CHUNKED_FUSED_ATTENTION"]
    provenance: str


class WorkspaceStage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    live_tensors: dict[str, int]
    live_bytes: int

    @model_validator(mode="after")
    def _closure(self) -> "WorkspaceStage":
        if self.live_bytes != sum(self.live_tensors.values()):
            raise ValueError("workspace stage tensor bytes do not close")
        return self


class WorkspacePeak(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: Literal["PREFILL", "DECODE"]
    batch_size: int
    context_length: int
    chunk_tokens: int
    stages: tuple[WorkspaceStage, ...]
    peak_stage: str
    peak_bytes: int
    sum_of_stage_bytes: int
    lifetime_status: Literal["MAX_SIMULTANEOUS_LIVE_TENSORS_WITH_BUFFER_REUSE"] = (
        "MAX_SIMULTANEOUS_LIVE_TENSORS_WITH_BUFFER_REUSE")

    @model_validator(mode="after")
    def _peak_closure(self) -> "WorkspacePeak":
        if self.peak_bytes != max(stage.live_bytes for stage in self.stages):
            raise ValueError("workspace peak must be the maximum stage live set")
        if self.sum_of_stage_bytes != sum(stage.live_bytes for stage in self.stages):
            raise ValueError("workspace diagnostic stage sum does not close")
        return self


def _stage(name: str, **tensors: int) -> WorkspaceStage:
    clean = {key: int(value) for key, value in tensors.items() if value}
    return WorkspaceStage(
        stage=name, live_tensors=clean, live_bytes=sum(clean.values()))


def _finish(
    phase: Literal["PREFILL", "DECODE"], batch_size: int,
    context_length: int, chunk_tokens: int,
    stages: tuple[WorkspaceStage, ...],
) -> WorkspacePeak:
    peak = max(stages, key=lambda item: item.live_bytes)
    return WorkspacePeak(
        phase=phase, batch_size=batch_size, context_length=context_length,
        chunk_tokens=chunk_tokens, stages=stages, peak_stage=peak.stage,
        peak_bytes=peak.live_bytes,
        sum_of_stage_bytes=sum(item.live_bytes for item in stages))


def evaluate_prefill_workspace(
    model: DenseLLMModelSpec, *, batch_size: int, context_length: int,
    config: WorkspaceExecutionConfig,
) -> WorkspacePeak:
    """Peak storage for one chunk of the existing fused/tiled Prefill model."""
    dims = model._resolved_dimensions()
    b = batch_size
    c = min(context_length, config.prefill_chunk_tokens)
    t = min(c, config.attention_tile_tokens)
    a = config.activation_bytes
    acc = config.attention_accumulator_bytes
    d = dims["d_model"]
    dff = dims["d_ff"]
    hq = dims["n_heads_q"]
    hkv = dims["n_heads_kv"]
    dh = d // hq
    hidden = b * c * d * a
    q = hidden
    kv = b * c * hkv * dh * a
    score_tile = b * hq * t * t * acc
    row_stats = b * hq * t * 2 * acc
    ffn_branch = b * c * dff * a
    logits = b * dims["vocab_size"] * a
    stages = (
        _stage("EMBED", input_hidden=hidden, output_hidden=hidden),
        _stage("ATTN_PROJ", input_hidden=hidden, q=q, k=kv, v=kv),
        _stage("ATTN_QK", q_tile=b*hq*t*dh*a, k_tile=b*hkv*t*dh*a,
               score_tile=score_tile, row_stats=row_stats),
        _stage("SOFTMAX", score_tile=score_tile, row_stats=row_stats),
        _stage("ATTN_AV", probability_tile=score_tile,
               v_tile=b*hkv*t*dh*a, attention_output=hidden),
        _stage("ATTN_OUT", attention_output=hidden, projected_hidden=hidden),
        _stage("FFN_GATE_UP", hidden=hidden, gate=ffn_branch, up=ffn_branch),
        _stage("FFN_ACT", gate=ffn_branch, up=ffn_branch,
               activated=ffn_branch),
        _stage("FFN_DOWN", activated=ffn_branch, output_hidden=hidden),
        _stage("LM_HEAD", final_hidden=b*d*a, logits=logits),
    )
    return _finish("PREFILL", b, context_length, c, stages)


def evaluate_decode_workspace(
    model: DenseLLMModelSpec, *, batch_size: int, context_length: int,
    config: WorkspaceExecutionConfig, proposed_nmp: bool = False,
    nmp_die_count: int = 0,
) -> WorkspacePeak:
    """Peak token-step storage; attention vectors grow with current context."""
    dims = model._resolved_dimensions()
    b = batch_size
    a = config.activation_bytes
    acc = config.attention_accumulator_bytes
    d = dims["d_model"]
    dff = dims["d_ff"]
    hq = dims["n_heads_q"]
    hkv = dims["n_heads_kv"]
    dh = d // hq
    hidden = b * d * a
    q = hidden
    kv_token = b * hkv * dh * a
    scores = b * hq * context_length * a
    row_stats = b * hq * 2 * acc
    ffn_branch = b * dff * a
    logits = b * dims["vocab_size"] * a
    boundary = b * nmp_die_count * d * acc if proposed_nmp else 0
    stages = (
        _stage("EMBED", input_hidden=hidden, output_hidden=hidden),
        _stage("ATTN_PROJ", input_hidden=hidden, q=q, k_token=kv_token,
               v_token=kv_token),
        _stage("ATTN_QK", q=q, attention_scores=scores,
               row_stats=row_stats),
        _stage("SOFTMAX", attention_scores=scores, row_stats=row_stats),
        _stage("ATTN_AV", probabilities=scores, av_result=hidden,
               nmp_boundary_partial=boundary),
        _stage("ATTN_OUT", av_result=hidden, projected_hidden=hidden),
        _stage("FFN_GATE_UP", hidden=hidden, gate=ffn_branch, up=ffn_branch),
        _stage("FFN_ACT", gate=ffn_branch, up=ffn_branch,
               activated=ffn_branch),
        _stage("FFN_DOWN", activated=ffn_branch, output_hidden=hidden),
        _stage("LM_HEAD", final_hidden=hidden, logits=logits),
    )
    return _finish("DECODE", b, context_length, 1, stages)
