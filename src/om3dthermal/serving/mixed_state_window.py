"""PREFILL_FIRST mixed-service request-state conservation audit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from om3dthermal.workload import DenseLLMModelSpec, evaluate_llm_decode

from .mixed_phase_e2e import (
    MixedPhaseServingCase,
    resolve_conventional_hbm_backend,
)
from .residency import ServingCapacitySource, evaluate_capacity_residency
from .state_ledger import (
    RequestKVState,
    ServingStateEvent,
    complete_prefill,
    initial_decode_state,
    initial_prefill_state,
)


class ConventionalMixedStateWindowResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    context_length: int
    batch_size: int
    prefill_requests: int
    decode_requests: int
    evaluation_scope: Literal["MIXED_SERVICE_WINDOW"] = "MIXED_SERVICE_WINDOW"
    schedule_policy: Literal["NO_OVERLAP__PREFILL_FIRST"] = (
        "NO_OVERLAP__PREFILL_FIRST")
    resident_decode_requests: int
    resident_prefill_requests: int
    host_decode_requests: int
    host_prefill_requests: int
    host_prefill_kv_write_GB: float
    host_prefill_transfer_time_ms: float
    local_live_kv_GB_after_prefill: float
    capacity_limit_requests: int
    capacity_status: str
    capacity_closure_status: Literal[
        "MODEL_DECLARED_WEIGHTS_KV_RUNTIME_ONLY__WORKSPACE_UNRESOLVED"]
    prefill_host_path_status: Literal[
        "GPU_TO_HOST_DIRECT_FINAL_KV_MATERIALIZATION"]
    energy_status: Literal[
        "PARTIAL_HOST_TRANSFER_RESOLVED__HBM_WRITE_UNRESOLVED"]
    states_after_prefill: tuple[RequestKVState, ...]
    events: tuple[ServingStateEvent, ...]
    thermal: None = None


def evaluate_conventional_prefill_first_state_window(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    case: MixedPhaseServingCase,
) -> ConventionalMixedStateWindowResult:
    """Materialize P requests without conflating mixed and Decode-only work."""
    if model.model_id != case.model_id:
        raise ValueError("model and mixed case model_id must match")
    root = Path(project_root).resolve()
    backend = resolve_conventional_hbm_backend(root)
    host = backend.host_offload
    host_bw = host.effective_bandwidth_bytes_per_second
    if host_bw is None:
        raise ValueError("canonical host bandwidth is unresolved")
    all_metrics = evaluate_llm_decode(model.decode_input(
        batch_size=case.batch_size, context_length=case.context_length))
    capacity = ServingCapacitySource(
        architecture=backend.architecture,
        usable_capacity_bytes=backend.capacity_bytes,
        capacity_source_status=backend.capacity_source_status,
        provenance_status="CANONICAL_CONVENTIONAL_HBM_CASE")
    residency = evaluate_capacity_residency(
        all_metrics, capacity, requested_requests=case.batch_size)
    limit = residency.max_resident_requests
    if limit is None:
        limit = case.batch_size
    resident_decode = min(case.decode_requests, limit)
    resident_prefill = min(case.prefill_requests, max(0, limit-resident_decode))
    kv_bytes_per_token = all_metrics.kv_bytes_per_request/case.context_length
    states: list[RequestKVState] = [
        initial_decode_state(f"D{index}", case.context_length,
                             local=index < resident_decode)
        for index in range(case.decode_requests)]
    events: list[ServingStateEvent] = []
    for index in range(case.prefill_requests):
        state = initial_prefill_state(f"P{index}")
        event = complete_prefill(
            state, prompt_length=case.context_length,
            kv_bytes_per_token=kv_bytes_per_token,
            destination=("LOCAL" if index < resident_prefill else "HOST"),
            transfer_bandwidth_bytes_per_s=host_bw,
            e_pcie_J_per_bit=host.host_link_dynamic_J_per_bit,
            e_ddr_J_per_bit=host.host_memory_dynamic_J_per_bit)
        events.append(event); states.append(event.after)
    host_bytes = sum(event.bytes for event in events
                     if event.direction.startswith("GPU_TO_HOST"))
    local_kv = sum(state.current_kv_length for state in states
                   if state.local_valid_kv_length == state.current_kv_length)
    local_kv_bytes = local_kv*kv_bytes_per_token
    declared_used = (all_metrics.weight_footprint_bytes
                     + all_metrics.runtime_fixed_bytes
                     + local_kv_bytes)
    if declared_used > backend.capacity_bytes and not math.isclose(
            declared_used, backend.capacity_bytes, rel_tol=1e-12):
        raise RuntimeError("declared live local state exceeds HBM capacity")
    return ConventionalMixedStateWindowResult(
        model_id=model.model_id, context_length=case.context_length,
        batch_size=case.batch_size, prefill_requests=case.prefill_requests,
        decode_requests=case.decode_requests,
        resident_decode_requests=resident_decode,
        resident_prefill_requests=resident_prefill,
        host_decode_requests=case.decode_requests-resident_decode,
        host_prefill_requests=case.prefill_requests-resident_prefill,
        host_prefill_kv_write_GB=host_bytes/1e9,
        host_prefill_transfer_time_ms=host_bytes/host_bw*1e3,
        local_live_kv_GB_after_prefill=local_kv_bytes/1e9,
        capacity_limit_requests=limit, capacity_status=residency.capacity_status,
        capacity_closure_status=(
            "MODEL_DECLARED_WEIGHTS_KV_RUNTIME_ONLY__WORKSPACE_UNRESOLVED"),
        prefill_host_path_status="GPU_TO_HOST_DIRECT_FINAL_KV_MATERIALIZATION",
        energy_status="PARTIAL_HOST_TRANSFER_RESOLVED__HBM_WRITE_UNRESOLVED",
        states_after_prefill=tuple(states), events=tuple(events))
