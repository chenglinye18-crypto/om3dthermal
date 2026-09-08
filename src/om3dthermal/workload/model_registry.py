"""Single-source dense-model dimensions shared by Prefill and Decode."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .llm_decode import LLMDecodeInput
from .llm_prefill import LLMPrefillInput


class DenseLLMModelSpec(BaseModel):
    """Small model registry entry; unresolved entries carry no guessed dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    model_id: str = Field(min_length=1)
    model_spec_status: Literal["RESOLVED", "UNRESOLVED_PENDING_SOURCE"]
    context_status: str
    n_param: int | None = Field(default=None, gt=0)
    n_layers: int | None = Field(default=None, gt=0)
    n_heads_q: int | None = Field(default=None, gt=0)
    n_heads_kv: int | None = Field(default=None, gt=0)
    d_model: int | None = Field(default=None, gt=0)
    d_ff: int | None = Field(default=None, gt=0)
    vocab_size: int | None = Field(default=None, gt=0)
    weight_bits: int | None = Field(default=None, gt=0)
    kv_bits: int | None = Field(default=None, gt=0)
    provenance: tuple[dict[str, object], ...] = ()

    @model_validator(mode="after")
    def _resolution_closure(self) -> "DenseLLMModelSpec":
        dimensions = (
            self.n_param, self.n_layers, self.n_heads_q, self.n_heads_kv,
            self.d_model, self.d_ff, self.vocab_size, self.weight_bits,
            self.kv_bits,
        )
        if self.model_spec_status == "RESOLVED" and any(
            value is None for value in dimensions
        ):
            raise ValueError("RESOLVED dense model requires every dimension")
        if self.model_spec_status == "UNRESOLVED_PENDING_SOURCE" and any(
            value is not None for value in dimensions
        ):
            raise ValueError(
                "unresolved dense model must not carry partial or guessed dimensions")
        return self

    def _resolved_dimensions(self) -> dict[str, int]:
        if self.model_spec_status != "RESOLVED":
            raise ValueError(f"model {self.model_id!r} is unresolved pending source")
        names = (
            "n_param", "n_layers", "n_heads_q", "n_heads_kv", "d_model",
            "d_ff", "vocab_size", "weight_bits", "kv_bits",
        )
        return {name: int(getattr(self, name)) for name in names}

    def decode_input(self, *, batch_size: int, context_length: int) -> LLMDecodeInput:
        return LLMDecodeInput(
            **self._resolved_dimensions(), batch_size=batch_size,
            context_length=context_length,
            weight_activity_model="dimension_derived_active_operators",
        )

    def prefill_input(self, *, batch_size: int, prompt_length: int) -> LLMPrefillInput:
        return LLMPrefillInput(
            **self._resolved_dimensions(), batch_size=batch_size,
            prompt_length=prompt_length,
        )


def load_dense_model_spec(path: str | Path) -> DenseLLMModelSpec:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("dense model spec root must be a mapping")
    return DenseLLMModelSpec.model_validate(raw)


def load_dense_model_registry(directory: str | Path) -> dict[str, DenseLLMModelSpec]:
    entries = tuple(load_dense_model_spec(path) for path in sorted(Path(directory).glob("*.yaml")))
    registry = {entry.model_id: entry for entry in entries}
    if len(registry) != len(entries):
        raise ValueError("dense model registry contains duplicate model_id values")
    return registry
