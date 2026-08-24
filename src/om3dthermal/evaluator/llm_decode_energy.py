"""Architecture-independent LLM decode memory dynamic energy primitive.

Only actual per-generated-token memory traffic is consumed:
``LLMDecodeMetrics.read_bytes_per_token`` and
``LLMDecodeMetrics.write_bytes_per_token``.  Footprint and required-capacity
fields are deliberately excluded from the energy equation.

The accounting boundary is dynamic memory traffic energy only.  Compute,
refresh, background/static, power derivation, and thermal effects are excluded.
This module does not produce a complete system J/token metric.
"""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

from pydantic import BaseModel, field_validator, model_validator

from om3dthermal.evaluation import CapacityFeasibilityMetrics
from om3dthermal.workload.llm_decode import LLMDecodeMetrics


EnergyScalar: TypeAlias = int | float

STATUS_CAPACITY_INFEASIBLE = "CAPACITY_INFEASIBLE"
STATUS_EVALUATED = "EVALUATED_MEMORY_DYNAMIC_TRAFFIC_ENERGY"
ENERGY_SCOPE_STATUS = "MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY"
EXCLUDED_ACCOUNTING_COMPONENTS = (
    "COMPUTE_ENERGY",
    "REFRESH_ENERGY",
    "BACKGROUND_STATIC_ENERGY",
    "POWER_DERIVATION",
    "THERMAL_EFFECTS",
)


def _validate_nonnegative_real(
    name: str, value: EnergyScalar,
) -> EnergyScalar:
    """Return a finite non-negative real without inventing a default."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_traffic(name: str, value: float) -> float:
    """Validate an analytical fractional byte-equivalent traffic value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be finite")
    if value_float < 0:
        raise ValueError(f"{name} must be non-negative")
    return value_float


class LLMDecodeMemoryEnergyMetrics(BaseModel):
    """Per-token dynamic memory traffic energy, gated by capacity fit."""

    read_bytes_per_token: float
    write_bytes_per_token: float
    read_bits_per_token: float
    write_bits_per_token: float

    read_energy_pj_per_bit: EnergyScalar
    write_energy_pj_per_bit: EnergyScalar

    read_dynamic_energy_pj_per_token: float | None
    write_dynamic_energy_pj_per_token: float | None
    memory_dynamic_energy_pj_per_token: float | None
    memory_dynamic_energy_j_per_token: float | None

    capacity_feasible: bool
    evaluation_status: Literal[
        "CAPACITY_INFEASIBLE",
        "EVALUATED_MEMORY_DYNAMIC_TRAFFIC_ENERGY",
    ]
    energy_scope_status: Literal["MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY"]
    excluded_accounting_components: tuple[
        Literal[
            "COMPUTE_ENERGY",
            "REFRESH_ENERGY",
            "BACKGROUND_STATIC_ENERGY",
            "POWER_DERIVATION",
            "THERMAL_EFFECTS",
        ],
        ...,
    ]

    @field_validator(
        "read_energy_pj_per_bit", "write_energy_pj_per_bit", mode="before")
    @classmethod
    def _valid_energy_input(cls, value: EnergyScalar) -> EnergyScalar:
        return _validate_nonnegative_real("energy_pj_per_bit", value)

    @model_validator(mode="after")
    def _status_and_energy_closure(self) -> "LLMDecodeMemoryEnergyMetrics":
        energy_fields = (
            "read_dynamic_energy_pj_per_token",
            "write_dynamic_energy_pj_per_token",
            "memory_dynamic_energy_pj_per_token",
            "memory_dynamic_energy_j_per_token",
        )
        if not self.capacity_feasible:
            if self.evaluation_status != STATUS_CAPACITY_INFEASIBLE:
                raise ValueError(
                    "capacity-infeasible result must use CAPACITY_INFEASIBLE")
            if any(getattr(self, name) is not None for name in energy_fields):
                raise ValueError(
                    "energy fields must be None when capacity is infeasible")
            return self

        if self.evaluation_status != STATUS_EVALUATED:
            raise ValueError(
                "capacity-feasible result must use evaluated energy status")
        if any(getattr(self, name) is None for name in energy_fields):
            raise ValueError(
                "energy fields must be defined when capacity is feasible")

        assert self.read_dynamic_energy_pj_per_token is not None
        assert self.write_dynamic_energy_pj_per_token is not None
        assert self.memory_dynamic_energy_pj_per_token is not None
        assert self.memory_dynamic_energy_j_per_token is not None
        expected_pj = (
            self.read_dynamic_energy_pj_per_token
            + self.write_dynamic_energy_pj_per_token)
        if self.memory_dynamic_energy_pj_per_token != expected_pj:
            raise ValueError("dynamic energy contributions do not close in pJ")
        if self.memory_dynamic_energy_j_per_token != expected_pj * 1e-12:
            raise ValueError("pJ-to-J dynamic energy conversion does not close")
        return self


def evaluate_llm_decode_memory_energy(
    workload: LLMDecodeMetrics,
    capacity: CapacityFeasibilityMetrics,
    *,
    read_energy_pj_per_bit: EnergyScalar,
    write_energy_pj_per_bit: EnergyScalar,
) -> LLMDecodeMemoryEnergyMetrics:
    """Evaluate dynamic memory access energy per generated token.

    Read and write energy inputs are mandatory and independent.  The function
    never substitutes the read-access energy for the write-access energy.
    Workload traffic is already per generated token, so no batch factor is
    applied here.
    """
    read_energy = _validate_nonnegative_real(
        "read_energy_pj_per_bit", read_energy_pj_per_bit)
    write_energy = _validate_nonnegative_real(
        "write_energy_pj_per_bit", write_energy_pj_per_bit)
    read_bytes = _validate_traffic(
        "workload.read_bytes_per_token", workload.read_bytes_per_token)
    write_bytes = _validate_traffic(
        "workload.write_bytes_per_token", workload.write_bytes_per_token)
    read_bits = read_bytes * 8
    write_bits = write_bytes * 8

    common = {
        "read_bytes_per_token": read_bytes,
        "write_bytes_per_token": write_bytes,
        "read_bits_per_token": read_bits,
        "write_bits_per_token": write_bits,
        "read_energy_pj_per_bit": read_energy,
        "write_energy_pj_per_bit": write_energy,
        "capacity_feasible": capacity.capacity_feasible,
        "energy_scope_status": ENERGY_SCOPE_STATUS,
        "excluded_accounting_components": EXCLUDED_ACCOUNTING_COMPONENTS,
    }
    if not capacity.capacity_feasible:
        return LLMDecodeMemoryEnergyMetrics(
            **common,
            read_dynamic_energy_pj_per_token=None,
            write_dynamic_energy_pj_per_token=None,
            memory_dynamic_energy_pj_per_token=None,
            memory_dynamic_energy_j_per_token=None,
            evaluation_status=STATUS_CAPACITY_INFEASIBLE,
        )

    read_dynamic_pj = read_bits * read_energy
    write_dynamic_pj = write_bits * write_energy
    total_dynamic_pj = read_dynamic_pj + write_dynamic_pj
    return LLMDecodeMemoryEnergyMetrics(
        **common,
        read_dynamic_energy_pj_per_token=read_dynamic_pj,
        write_dynamic_energy_pj_per_token=write_dynamic_pj,
        memory_dynamic_energy_pj_per_token=total_dynamic_pj,
        memory_dynamic_energy_j_per_token=total_dynamic_pj * 1e-12,
        evaluation_status=STATUS_EVALUATED,
    )
