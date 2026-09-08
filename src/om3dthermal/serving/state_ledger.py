"""Lightweight request-KV state and transfer-event conservation ledger."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RequestPhase = Literal[
    "PREFILL_PENDING", "READY_FOR_DECODE", "ACTIVE_DECODE", "FINISHED"]
EventKind = Literal[
    "PREFILL", "DECODE_STEP", "HOST_TO_LOCAL", "LOCAL_TO_HOST", "FREE"]


class EvaluationSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_scope: Literal["DECODE_SERVICE_ONLY", "MIXED_SERVICE_WINDOW"]
    context_evolution: Literal["FIXED_CONTEXT_SNAPSHOT", "GROWING_KV"]
    initial_state_requirement: str
    final_state_requirement: str
    generated_decode_steps: int | None = Field(default=None, gt=0)
    schedule_policy: str


def require_comparable_semantics(
        left: EvaluationSemantics, right: EvaluationSemantics) -> None:
    """Reject ratios whose work, state boundary, G, or schedule differs."""
    if left != right:
        raise ValueError(
            "INCOMPARABLE_EVALUATION_SEMANTICS: scope/state/G/context/schedule "
            "must match before computing a ratio")


class RequestKVState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    phase: RequestPhase
    current_kv_length: int = Field(ge=0)
    local_valid_kv_length: int = Field(ge=0)
    host_valid_kv_length: int = Field(ge=0)
    local_dirty_from_token: int | None = Field(default=None, ge=0)
    completed: bool = False

    @model_validator(mode="after")
    def _valid_ranges(self) -> "RequestKVState":
        if self.local_valid_kv_length > self.current_kv_length:
            raise ValueError("local KV range exceeds live KV")
        if self.host_valid_kv_length > self.current_kv_length:
            raise ValueError("host KV range exceeds live KV")
        if self.current_kv_length and not (
                self.local_valid_kv_length == self.current_kv_length
                or self.host_valid_kv_length == self.current_kv_length):
            raise ValueError("live KV has no complete valid copy")
        if self.completed and self.phase != "FINISHED":
            raise ValueError("completed request must be FINISHED")
        return self

    @property
    def has_host_copy(self) -> bool:
        return self.host_valid_kv_length == self.current_kv_length


class ServingStateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_kind: EventKind
    request_id: str
    bytes: float = Field(ge=0.0)
    direction: str
    time_ms: float = Field(ge=0.0)
    pcie_dynamic_J: float | None = Field(default=None, ge=0.0)
    ddr_dynamic_J: float | None = Field(default=None, ge=0.0)
    local_memory_dynamic_J: float | None = Field(default=None, ge=0.0)
    before: RequestKVState
    after: RequestKVState
    status: str


def initial_decode_state(
        request_id: str, context_length: int, *, local: bool) -> RequestKVState:
    return RequestKVState(
        request_id=request_id, phase="ACTIVE_DECODE",
        current_kv_length=context_length,
        local_valid_kv_length=context_length if local else 0,
        host_valid_kv_length=0 if local else context_length)


def initial_prefill_state(request_id: str) -> RequestKVState:
    return RequestKVState(
        request_id=request_id, phase="PREFILL_PENDING", current_kv_length=0,
        local_valid_kv_length=0, host_valid_kv_length=0)


def complete_prefill(
        state: RequestKVState, *, prompt_length: int, kv_bytes_per_token: float,
        destination: Literal["LOCAL", "HOST"], transfer_bandwidth_bytes_per_s: float,
        e_pcie_J_per_bit: float | None, e_ddr_J_per_bit: float | None,
) -> ServingStateEvent:
    if state.phase != "PREFILL_PENDING" or state.current_kv_length != 0:
        raise ValueError("Prefill requires an unstarted request")
    bytes_ = prompt_length * kv_bytes_per_token
    host = destination == "HOST"
    after = state.model_copy(update={
        "phase": "READY_FOR_DECODE", "current_kv_length": prompt_length,
        "local_valid_kv_length": 0 if host else prompt_length,
        "host_valid_kv_length": prompt_length if host else 0,
        "local_dirty_from_token": None if host else 0,
    })
    transfer = bytes_ if host else 0.0
    return ServingStateEvent(
        event_kind="PREFILL", request_id=state.request_id, bytes=transfer,
        direction=("GPU_TO_HOST_DIRECT_FINAL_KV_MATERIALIZATION"
                   if host else "GPU_TO_LOCAL_FINAL_KV_MATERIALIZATION"),
        time_ms=(transfer / transfer_bandwidth_bytes_per_s * 1e3
                 if transfer else 0.0),
        pcie_dynamic_J=(None if e_pcie_J_per_bit is None else
                        8.0 * transfer * e_pcie_J_per_bit),
        ddr_dynamic_J=(None if e_ddr_J_per_bit is None else
                       8.0 * transfer * e_ddr_J_per_bit),
        local_memory_dynamic_J=None,
        before=state, after=after,
        status=("HOST_FINAL_KV_MATERIALIZED_ONCE"
                if host else "LOCAL_FINAL_KV_RETAINED"))


def transfer_host_to_local(
        state: RequestKVState, *, kv_bytes_per_token: float,
        bandwidth_bytes_per_s: float, e_pcie_J_per_bit: float | None,
        e_ddr_J_per_bit: float | None) -> ServingStateEvent:
    if not state.has_host_copy:
        raise ValueError("HOST_TO_LOCAL requires a complete valid host copy")
    bytes_ = state.current_kv_length * kv_bytes_per_token
    after = state.model_copy(update={
        "local_valid_kv_length": state.current_kv_length,
        "local_dirty_from_token": None})
    return ServingStateEvent(
        event_kind="HOST_TO_LOCAL", request_id=state.request_id, bytes=bytes_,
        direction="HOST_TO_LOCAL", time_ms=bytes_ / bandwidth_bytes_per_s * 1e3,
        pcie_dynamic_J=(None if e_pcie_J_per_bit is None else
                        8.0 * bytes_ * e_pcie_J_per_bit),
        ddr_dynamic_J=(None if e_ddr_J_per_bit is None else
                       8.0 * bytes_ * e_ddr_J_per_bit),
        before=state, after=after, status="COMPLETE_HOST_COPY_ADMITTED")


def decode_step(
        state: RequestKVState, *, kv_append_bytes: float,
) -> ServingStateEvent:
    if state.phase != "ACTIVE_DECODE" or (
            state.local_valid_kv_length != state.current_kv_length):
        raise ValueError("Decode requires complete local KV")
    old_length = state.current_kv_length
    after = state.model_copy(update={
        "current_kv_length": old_length + 1,
        "local_valid_kv_length": old_length + 1,
        "local_dirty_from_token": (old_length if state.local_dirty_from_token is None
                                   else state.local_dirty_from_token)})
    return ServingStateEvent(
        event_kind="DECODE_STEP", request_id=state.request_id,
        bytes=kv_append_bytes, direction="LOCAL_APPEND", time_ms=0.0,
        before=state, after=after, status="LIVE_KV_GREW_BY_ONE_TOKEN")


def transfer_local_to_host(
        state: RequestKVState, *, kv_bytes_per_token: float,
        bandwidth_bytes_per_s: float, e_pcie_J_per_bit: float | None,
        e_ddr_J_per_bit: float | None) -> ServingStateEvent:
    if state.local_valid_kv_length != state.current_kv_length:
        raise ValueError("LOCAL_TO_HOST requires a complete local copy")
    missing_tokens = state.current_kv_length - state.host_valid_kv_length
    bytes_ = missing_tokens * kv_bytes_per_token
    after = state.model_copy(update={
        "host_valid_kv_length": state.current_kv_length,
        "local_dirty_from_token": None})
    return ServingStateEvent(
        event_kind="LOCAL_TO_HOST", request_id=state.request_id, bytes=bytes_,
        direction="LOCAL_TO_HOST", time_ms=bytes_ / bandwidth_bytes_per_s * 1e3,
        pcie_dynamic_J=(None if e_pcie_J_per_bit is None else
                        8.0 * bytes_ * e_pcie_J_per_bit),
        ddr_dynamic_J=(None if e_ddr_J_per_bit is None else
                       8.0 * bytes_ * e_ddr_J_per_bit),
        before=state, after=after,
        status=("NOOP_HOST_COPY_ALREADY_CURRENT" if bytes_ == 0.0
                else "DIRTY_SUFFIX_WRITTEN_ONCE"))


def free_local(state: RequestKVState) -> ServingStateEvent:
    if state.phase != "FINISHED" and not state.completed:
        raise ValueError("FREE is legal only for a finished request")
    after = state.model_copy(update={
        "local_valid_kv_length": 0, "local_dirty_from_token": None,
        "phase": "FINISHED", "completed": True})
    return ServingStateEvent(
        event_kind="FREE", request_id=state.request_id, bytes=0.0,
        direction="LOCAL_RELEASE", time_ms=0.0, before=state, after=after,
        status="FINISHED_LOCAL_KV_RELEASED")
