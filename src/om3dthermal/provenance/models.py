"""Small, serializable provenance records for reproducible experiments."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ProvenanceClassification = Literal[
    "PAPER_REPORTED",
    "DERIVED_FROM_PAPER",
    "MODELING_CHOICE",
    "NUMERICAL_CHOICE",
    "MATCHED_REFERENCE",
    "NOT_VALIDATED",
    "CONDITIONAL_LOWER_BOUND",
    "SOFTWARE_DERIVED",
]


class ProvenanceRecord(BaseModel):
    """One parameter/equation provenance record without free-form inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(min_length=1)
    classification: ProvenanceClassification
    source: str = Field(min_length=1)
    source_location: str | None = None
    status: str = Field(min_length=1)
    transformation: str | None = None
    notes: str | None = None
    parent_record_ids: tuple[str, ...] = ()


class RunProvenance(BaseModel):
    """Execution identity persisted with every formal experiment bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    main_repo_commit: str
    main_repo_branch: str | None
    main_repo_tracked_clean: bool
    dreamram_commit: str | None
    dreamram_branch: str | None
    python_version: str
    platform: str
    executable: str
    experiment_config_path: str
    input_sha256: dict[str, str]
    execution_started_utc: str
    execution_finished_utc: str | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
