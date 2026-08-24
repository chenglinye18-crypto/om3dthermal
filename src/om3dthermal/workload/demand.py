"""Stable, architecture-independent workload demand contract.

The existing LLM decode implementation remains the numerical source of truth.
This module only adapts its validated metrics to a workload-neutral boundary
for downstream capacity, performance, and energy evaluators.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .llm_decode import LLMDecodeMetrics, evaluate_llm_decode
from .spec import WorkloadSpec


class WorkloadDemand(BaseModel):
    """Resolved resource demand per generated workload output.

    Footprint and traffic are deliberately separate.  Traffic fields describe
    algorithmic workload demand and must not be interpreted as physical DRAM
    traffic without an architecture mapping stage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    workload_id: str = Field(min_length=1)
    workload_type: str = Field(min_length=1)
    output_unit: Literal["GENERATED_TOKEN"] = "GENERATED_TOKEN"
    batch_size: int = Field(gt=0)
    context_length: int = Field(ge=0)

    weight_footprint_bytes: float = Field(ge=0.0)
    persistent_state_footprint_bytes: float = Field(ge=0.0)
    runtime_footprint_bytes: float = Field(ge=0.0)
    required_capacity_bytes: float = Field(ge=0.0)

    weight_read_bytes_per_output: float = Field(ge=0.0)
    persistent_state_read_bytes_per_output: float = Field(ge=0.0)
    persistent_state_write_bytes_per_output: float = Field(ge=0.0)
    read_bytes_per_output: float = Field(ge=0.0)
    write_bytes_per_output: float = Field(ge=0.0)
    flops_per_output: int = Field(ge=0)

    weight_activity_model: str = Field(min_length=1)
    weight_reuse_model: str = Field(min_length=1)
    persistent_state_read_model: str = Field(min_length=1)

    footprint_scope_status: Literal[
        "ANALYTICAL_BYTE_EQUIVALENT_NOT_PHYSICAL_ALLOCATION"
    ] = "ANALYTICAL_BYTE_EQUIVALENT_NOT_PHYSICAL_ALLOCATION"
    traffic_scope_status: Literal[
        "ALGORITHMIC_WORKLOAD_TRAFFIC_NOT_PHYSICAL_DRAM_TRAFFIC"
    ] = "ALGORITHMIC_WORKLOAD_TRAFFIC_NOT_PHYSICAL_DRAM_TRAFFIC"
    compute_scope_status: Literal[
        "ALGORITHMIC_FLOPS_PER_GENERATED_TOKEN"
    ] = "ALGORITHMIC_FLOPS_PER_GENERATED_TOKEN"

    @model_validator(mode="after")
    def _accounting_closure(self) -> "WorkloadDemand":
        footprint_total = (
            self.weight_footprint_bytes
            + self.persistent_state_footprint_bytes
            + self.runtime_footprint_bytes
        )
        if self.required_capacity_bytes != footprint_total:
            raise ValueError("workload footprint does not close")
        read_total = (
            self.weight_read_bytes_per_output
            + self.persistent_state_read_bytes_per_output
        )
        if self.read_bytes_per_output != read_total:
            raise ValueError("workload read traffic does not close")
        if (
            self.write_bytes_per_output
            != self.persistent_state_write_bytes_per_output
        ):
            raise ValueError("workload write traffic does not close")
        return self


def resolve_llm_decode_demand(
    spec: WorkloadSpec,
    metrics: LLMDecodeMetrics,
) -> WorkloadDemand:
    """Adapt validated metrics; re-evaluation is only a consistency gate."""

    expected = evaluate_llm_decode(spec.decode)
    if metrics != expected:
        raise ValueError("LLM decode metrics do not match the workload spec")
    return WorkloadDemand(
        workload_id=spec.workload_id,
        workload_type=spec.workload_type,
        batch_size=spec.decode.batch_size,
        context_length=spec.decode.context_length,
        weight_footprint_bytes=metrics.weight_footprint_bytes,
        persistent_state_footprint_bytes=metrics.kv_footprint_bytes,
        runtime_footprint_bytes=metrics.runtime_bytes,
        required_capacity_bytes=metrics.required_capacity_bytes,
        weight_read_bytes_per_output=metrics.weight_read_bytes_per_token,
        persistent_state_read_bytes_per_output=metrics.kv_read_bytes_per_token,
        persistent_state_write_bytes_per_output=metrics.kv_write_bytes_per_token,
        read_bytes_per_output=metrics.read_bytes_per_token,
        write_bytes_per_output=metrics.write_bytes_per_token,
        flops_per_output=metrics.flops_per_token,
        weight_activity_model=metrics.weight_activity_model,
        weight_reuse_model=metrics.weight_reuse_model,
        persistent_state_read_model=metrics.kv_read_model,
    )
