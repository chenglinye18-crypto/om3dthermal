"""Conditional architecture decode memory energy (E4).

Connects a frozen LLMDecodeMetrics workload, an
ArchitectureCapacityFeasibility gate, and a ResolvedSystemPower to a
conditional per-token memory dynamic energy estimate.

The architecture's nominal read energy is consumed verbatim from
``ResolvedSystemPower.memory_access_energy_pJ_per_bit``; no read-energy
model is reimplemented.  Write energy is derived as ``rho × Eread``,
where ``rho`` is an explicit dimensionless sensitivity parameter, not a
validated physical write-energy model.

Scope: memory dynamic traffic energy only.  Compute, refresh,
background, power derivation, thermal, and Tmax are excluded.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, field_validator

from om3dthermal.power.system import ResolvedSystemPower
from om3dthermal.workload.architecture_capacity import (
    ArchitectureCapacityFeasibility,
)
from om3dthermal.workload.capacity import (
    CapacityFeasibilityMetrics,
    evaluate_capacity_feasibility,
)
from om3dthermal.workload.llm_decode import LLMDecodeMetrics

from .llm_decode_energy import evaluate_llm_decode_memory_energy


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

READ_ENERGY_STATUS = "CURRENT_NOMINAL_ANALYTICAL_MODEL"
WRITE_ENERGY_STATUS = "RHO_SENSITIVITY_NOT_PHYSICAL_CLAIM"
ENERGY_SCOPE_STATUS = "MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY"
SCENARIO_STATUS = "CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY"
ZHU_TRANSFERABILITY_STATUS = "NOT_VALIDATED"

STATUS_NO_ARCHITECTURE_ENERGY = "NO_ARCHITECTURE_ENERGY_RESOLVED"
STATUS_CAPACITY_INFEASIBLE = "CAPACITY_INFEASIBLE"
STATUS_EVALUATED = "EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_rho(name: str, value: float) -> float:
    """Return a finite non-negative real, rejecting bool/NaN/inf/negative."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")
    v_float = float(value)
    if not math.isfinite(v_float):
        raise ValueError(f"{name} must be finite")
    if v_float < 0:
        raise ValueError(f"{name} must be non-negative")
    return v_float


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class ArchitectureDecodeMemoryEnergyMetrics(BaseModel):
    """Conditional per-token memory dynamic energy for one architecture."""

    # Identity
    architecture: str
    rho: float

    # Capacity gate
    capacity_feasible: bool

    # Traffic (echoed from workload for self-containment)
    read_bytes_per_token: float
    write_bytes_per_token: float

    # Energy inputs (None when architecture energy unavailable)
    read_energy_pj_per_bit: float | None
    write_energy_pj_per_bit: float | None

    # Energy results (None when capacity infeasible or energy unavailable)
    read_dynamic_energy_j_per_token: float | None
    write_dynamic_energy_j_per_token: float | None
    memory_dynamic_energy_j_per_token: float | None

    # Provenance / status
    read_energy_status: Literal[
        "CURRENT_NOMINAL_ANALYTICAL_MODEL",
        "NO_ARCHITECTURE_ENERGY_RESOLVED",
    ]
    write_energy_status: Literal[
        "RHO_SENSITIVITY_NOT_PHYSICAL_CLAIM",
        "NO_ARCHITECTURE_ENERGY_RESOLVED",
    ]
    energy_scope_status: Literal["MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY"]
    scenario_status: Literal[
        "CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY",
    ]
    zhu_transferability_status: Literal["NOT_VALIDATED"]
    evaluation_status: Literal[
        "EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY",
        "CAPACITY_INFEASIBLE",
        "NO_ARCHITECTURE_ENERGY_RESOLVED",
    ]

    @field_validator("rho", mode="before")
    @classmethod
    def _valid_rho(cls, value: float) -> float:
        return _validate_rho("rho", value)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_architecture_decode_memory_energy(
    workload: LLMDecodeMetrics,
    capacity: ArchitectureCapacityFeasibility,
    system: ResolvedSystemPower,
    *,
    rho: float,
) -> ArchitectureDecodeMemoryEnergyMetrics:
    """Evaluate conditional architecture decode memory energy.

    Parameters
    ----------
    workload:
        Frozen LLMDecodeMetrics from ``evaluate_llm_decode``.
    capacity:
        ArchitectureCapacityFeasibility gate.
    system:
        ResolvedSystemPower carrying the architecture's nominal read
        energy at ``memory_access_energy_pJ_per_bit``.
    rho:
        **Mandatory** dimensionless sensitivity parameter.
        ``Ewrite = rho × Eread``.  No default.  Must be a finite
        non-negative real.  ``bool``, ``NaN``, ``inf``, and negative
        values are rejected.

    Returns
    -------
    ArchitectureDecodeMemoryEnergyMetrics
        When ``capacity_feasible`` is ``False`` or the architecture's
        nominal read energy is ``None``, all energy-result fields are
        ``None``.  Provenance/status fields are always populated.
    """
    rho_valid = _validate_rho("rho", rho)

    # 1) Architecture identity gate: capacity and system must refer to the
    #    same canonical architecture.
    if capacity.architecture != system.case_name:
        raise ValueError(
            f"capacity.architecture ({capacity.architecture!r}) does not match "
            f"system.case_name ({system.case_name!r})"
        )

    # 2) Capacity / workload consistency gate: recompute feasibility from the
    #    workload and the architecture's physical capacity, then verify the
    #    supplied capacity gate was produced from the *same* workload.
    rebuilt = evaluate_capacity_feasibility(
        workload,
        physical_capacity_bytes=capacity.physical_capacity_bytes,
        reserved_capacity_bytes=capacity.reserved_capacity_bytes,
    )
    if (
        rebuilt.capacity_feasible != capacity.capacity_feasible
        or rebuilt.usable_capacity_bytes != capacity.usable_capacity_bytes
        or rebuilt.capacity_margin_bytes != capacity.capacity_margin_bytes
        or rebuilt.capacity_utilization != capacity.capacity_utilization
    ):
        raise ValueError(
            "capacity / workload mismatch: the supplied ArchitectureCapacityFeasibility "
            "was not produced from the given workload. "
            "Re-evaluate the workload against the architecture capacity."
        )

    e_read = system.memory_access_energy_pJ_per_bit
    has_energy = e_read is not None

    # Common fields always populated
    common = {
        "architecture": capacity.architecture,
        "rho": rho_valid,
        "capacity_feasible": capacity.capacity_feasible,
        "read_bytes_per_token": workload.read_bytes_per_token,
        "write_bytes_per_token": workload.write_bytes_per_token,
        "energy_scope_status": ENERGY_SCOPE_STATUS,
        "scenario_status": SCENARIO_STATUS,
        "zhu_transferability_status": ZHU_TRANSFERABILITY_STATUS,
    }

    # Case: no architecture energy resolved (unresolved / reference_fixed)
    if not has_energy:
        return ArchitectureDecodeMemoryEnergyMetrics(
            **common,
            read_energy_pj_per_bit=None,
            write_energy_pj_per_bit=None,
            read_dynamic_energy_j_per_token=None,
            write_dynamic_energy_j_per_token=None,
            memory_dynamic_energy_j_per_token=None,
            read_energy_status=STATUS_NO_ARCHITECTURE_ENERGY,
            write_energy_status=STATUS_NO_ARCHITECTURE_ENERGY,
            evaluation_status=STATUS_NO_ARCHITECTURE_ENERGY,
        )

    # Validate read energy
    if isinstance(e_read, bool) or not isinstance(e_read, (int, float)):
        raise TypeError("system.memory_access_energy_pJ_per_bit must be numeric")
    e_read_float = float(e_read)
    if not math.isfinite(e_read_float):
        raise ValueError("system.memory_access_energy_pJ_per_bit must be finite")
    if e_read_float < 0:
        raise ValueError("system.memory_access_energy_pJ_per_bit must be non-negative")

    e_write = rho_valid * e_read_float

    # Case: capacity infeasible
    if not capacity.capacity_feasible:
        return ArchitectureDecodeMemoryEnergyMetrics(
            **common,
            read_energy_pj_per_bit=e_read_float,
            write_energy_pj_per_bit=e_write,
            read_dynamic_energy_j_per_token=None,
            write_dynamic_energy_j_per_token=None,
            memory_dynamic_energy_j_per_token=None,
            read_energy_status=READ_ENERGY_STATUS,
            write_energy_status=WRITE_ENERGY_STATUS,
            evaluation_status=STATUS_CAPACITY_INFEASIBLE,
        )

    # Case: capacity feasible → delegate to existing energy primitive.
    # Build a minimal CapacityFeasibilityMetrics adapter, forwarding the
    # rebuilt utilization_status so it is not hard-coded here.
    capacity_adapter = CapacityFeasibilityMetrics(
        physical_capacity_bytes=capacity.physical_capacity_bytes,
        reserved_capacity_bytes=capacity.reserved_capacity_bytes,
        usable_capacity_bytes=capacity.usable_capacity_bytes,
        required_capacity_bytes=capacity.required_capacity_bytes,
        capacity_margin_bytes=capacity.capacity_margin_bytes,
        capacity_utilization=capacity.capacity_utilization,
        utilization_status=rebuilt.utilization_status,
        capacity_feasible=capacity.capacity_feasible,
        scope_status="AGGREGATE_CAPACITY_FEASIBILITY_ONLY",
    )
    energy = evaluate_llm_decode_memory_energy(
        workload,
        capacity_adapter,
        read_energy_pj_per_bit=e_read_float,
        write_energy_pj_per_bit=e_write,
    )

    # Convert per-component pJ to J for the output model
    read_j = energy.read_dynamic_energy_pj_per_token * 1e-12
    write_j = energy.write_dynamic_energy_pj_per_token * 1e-12
    total_j = energy.memory_dynamic_energy_j_per_token

    return ArchitectureDecodeMemoryEnergyMetrics(
        **common,
        read_energy_pj_per_bit=e_read_float,
        write_energy_pj_per_bit=e_write,
        read_dynamic_energy_j_per_token=read_j,
        write_dynamic_energy_j_per_token=write_j,
        memory_dynamic_energy_j_per_token=total_j,
        read_energy_status=READ_ENERGY_STATUS,
        write_energy_status=WRITE_ENERGY_STATUS,
        evaluation_status=STATUS_EVALUATED,
    )
