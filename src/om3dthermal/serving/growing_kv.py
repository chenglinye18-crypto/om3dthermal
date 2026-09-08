"""Cross-system growing-KV high-water capacity semantics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from om3dthermal.resident_pages import ResidentDataObject
from om3dthermal.workload import DenseLLMModelSpec, evaluate_llm_decode

from .mixed_phase_e2e import MixedPhaseServingCase
from .nmp_decode import resolve_m3d_architecture_backend, rounded_capacity_bytes


class M3DGrowingKVCapacityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    initial_context_length: int
    generated_decode_steps: int
    retained_prefill_requests: int
    growing_decode_requests: int
    initial_logical_capacity_GB: float
    growing_kv_bytes: float
    high_water_logical_capacity_GB: float
    high_water_page_rounded_capacity_GB: float
    available_capacity_GB: float
    capacity_margin_GB: float
    capacity_utilization: float
    capacity_status: Literal["FEASIBLE", "CAPACITY_INFEASIBLE"]
    context_evolution_status: Literal[
        "PREFILL_KV_RETAINED__DECODE_KV_GROWS_S_TO_S_PLUS_G"] = (
            "PREFILL_KV_RETAINED__DECODE_KV_GROWS_S_TO_S_PLUS_G")
    workspace_capacity_status: Literal[
        "WORKSPACE_UNRESOLVED__NOT_ASSUMED_ZERO"] = (
            "WORKSPACE_UNRESOLVED__NOT_ASSUMED_ZERO")
    thermal: None = None


def evaluate_m3d_growing_kv_capacity(
    *, project_root: str | Path, model: DenseLLMModelSpec,
    case: MixedPhaseServingCase, generated_decode_steps: int,
) -> M3DGrowingKVCapacityResult:
    """Reserve P at S and D at S+G using existing page/capacity primitives."""
    if generated_decode_steps <= 0:
        raise ValueError("generated_decode_steps must be positive")
    architecture = resolve_m3d_architecture_backend(project_root)
    base_input = model.decode_input(
        batch_size=case.batch_size, context_length=case.context_length)
    metrics = evaluate_llm_decode(base_input)
    kv_per_token = metrics.kv_bytes_per_request / case.context_length
    objects = [ResidentDataObject(
        "weights", "WEIGHT", int(metrics.weight_footprint_bytes))]
    for request in range(case.prefill_requests):
        objects.append(ResidentDataObject(
            f"kv.prefill.{request}", "KV", int(metrics.kv_bytes_per_request)))
    growing_size = int(metrics.kv_bytes_per_request
                       + generated_decode_steps * kv_per_token)
    for request in range(case.decode_requests):
        objects.append(ResidentDataObject(
            f"kv.decode.{request}", "KV", growing_size))
    runtime = int(metrics.runtime_bytes)
    if runtime:
        objects.append(ResidentDataObject("runtime", "OTHER", runtime))
    logical = sum(item.size_bytes for item in objects)
    rounded = rounded_capacity_bytes(
        tuple(objects), architecture.layout.slot_capacity_bytes)
    available = architecture.layout.total_capacity_bytes
    growth = case.decode_requests * generated_decode_steps * kv_per_token
    expected = metrics.required_capacity_bytes + growth
    if not math.isclose(logical, expected, rel_tol=0.0, abs_tol=1.0):
        raise RuntimeError("growing-KV logical capacity does not close")
    return M3DGrowingKVCapacityResult(
        model_id=model.model_id, initial_context_length=case.context_length,
        generated_decode_steps=generated_decode_steps,
        retained_prefill_requests=case.prefill_requests,
        growing_decode_requests=case.decode_requests,
        initial_logical_capacity_GB=metrics.required_capacity_bytes/1e9,
        growing_kv_bytes=growth,
        high_water_logical_capacity_GB=logical/1e9,
        high_water_page_rounded_capacity_GB=rounded/1e9,
        available_capacity_GB=available/1e9,
        capacity_margin_GB=(available-rounded)/1e9,
        capacity_utilization=rounded/available,
        capacity_status=("FEASIBLE" if rounded<=available
                         else "CAPACITY_INFEASIBLE"))
