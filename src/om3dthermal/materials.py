"""Material definitions in their local coordinate systems."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Material(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    k_local: tuple[float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("k_local")
    @classmethod
    def conductivity_is_positive(cls, value: tuple[float, float, float] | None):
        if value is not None and any(component <= 0 for component in value):
            raise ValueError("all local thermal conductivity components must be positive")
        return value
