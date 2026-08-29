"""Named workload configuration around the frozen LLM decode primitive."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from om3dthermal.provenance import ProvenanceRecord

from .llm_decode import LLMDecodeInput
from .moe_decode import MoEDecodeInput


class WorkloadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    workload_id: str = Field(min_length=1)
    workload_type: Literal["llm_autoregressive_decode"]
    decode: LLMDecodeInput
    provenance: tuple[ProvenanceRecord, ...] = ()


class MoEWorkloadSpec(BaseModel):
    """Named structural MoE workload without changing dense semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    workload_id: str = Field(min_length=1)
    workload_type: Literal["moe_autoregressive_decode"]
    decode: MoEDecodeInput
    provenance: tuple[ProvenanceRecord, ...] = ()
