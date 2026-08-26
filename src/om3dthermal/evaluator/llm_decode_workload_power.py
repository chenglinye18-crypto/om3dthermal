"""Workload-dependent LLM decode memory/package power accounting (E5).

Dynamic memory power has exactly one source: conditional dynamic memory
energy per generated token multiplied by aggregate generated tokens per
second.  Existing configured-bandwidth access power is retained only as a
regression reference and is never added to the new total.

Refresh, memory-background, and logic-background power are consumed once from
``ResolvedSystemPower.memory_result``.  GPU power is the existing fixed
baseline; compute energy, system J/token, thermal mapping, and Tmax are outside
this module's accounting boundary.
"""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

from pydantic import BaseModel, model_validator

from om3dthermal.power.system import ResolvedSystemPower

from .llm_decode_architecture_energy import ArchitectureDecodeMemoryEnergyMetrics
from .llm_decode_performance import LLMDecodePerformanceMetrics


PowerScalar: TypeAlias = int | float

POLICY_REQUIRE_RESOLVED = "REQUIRE_RESOLVED"
POLICY_EXISTING_PLACEHOLDER_ZERO = "EXISTING_PLACEHOLDER_ZERO"
POLICY_PARAMETRIC_SENSITIVITY = "PARAMETRIC_SENSITIVITY"

STATUS_EVALUATED = "EVALUATED_WORKLOAD_DEPENDENT_MEMORY_POWER"
STATUS_BLOCKED = "BLOCKED_BY_CAPACITY_OR_UPSTREAM_EVALUATION"
STATUS_UNRESOLVED_STATIC = "UNRESOLVED_STATIC_POWER"

DYNAMIC_POWER_STATUS = (
    "WORKLOAD_J_PER_TOKEN_TIMES_AGGREGATE_TOKENS_PER_SECOND")
STATIC_POWER_STATUS = "EXISTING_POWER_MODEL_COMPONENTS_ADDED_ONCE"
GPU_POWER_STATUS = "FIXED_EXISTING_BASELINE_NOT_WORKLOAD_ENERGY_MODEL"
SYSTEM_ENERGY_STATUS = "NOT_AVAILABLE_COMPUTE_ENERGY_EXCLUDED"
SCENARIO_STATUS = "CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY"

LOGIC_STATUS_RESOLVED_ZERO = "RESOLVED_EXPLICIT_ZERO"
LOGIC_STATUS_RESOLVED_NUMERIC = "RESOLVED_EXPLICIT_NUMERIC"
LOGIC_STATUS_PLACEHOLDER_ZERO = (
    "EXISTING_PLACEHOLDER_ZERO_NOT_SEPARATELY_MODELED")
LOGIC_STATUS_UNRESOLVED = "UNRESOLVED_LOGIC_BACKGROUND"
LOGIC_STATUS_PARAMETRIC = "PARAMETRIC_SENSITIVITY_NOT_VALIDATED"

COMPLETENESS_RESOLVED = "RESOLVED_EXISTING_STATIC_COMPONENTS"
COMPLETENESS_CONDITIONAL = (
    "CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND")
COMPLETENESS_UNRESOLVED = "UNRESOLVED_STATIC_POWER"
COMPLETENESS_PARAMETRIC = "PARAMETRIC_SENSITIVITY"

_OLD_TOTAL_ABS_TOL_W = 1e-10


def _finite_nonnegative(name: str, value: PowerScalar) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


class LLMDecodeWorkloadPowerMetrics(BaseModel):
    """Workload-dependent dynamic plus existing static memory power."""

    architecture: str
    rho: float
    capacity_feasible: bool

    memory_dynamic_energy_j_per_token: float | None
    aggregate_tokens_per_second: float | None

    memory_dynamic_access_power_W: float | None
    refresh_power_W: float | None
    memory_background_power_W: float | None
    logic_background_raw_W: float | None
    logic_background_effective_W: float | None
    memory_workload_total_W: float | None
    fixed_gpu_power_W: float | None
    package_workload_total_W: float | None

    unresolved_logic_background_policy: Literal[
        "REQUIRE_RESOLVED", "EXISTING_PLACEHOLDER_ZERO",
        "PARAMETRIC_SENSITIVITY"]
    logic_background_status: Literal[
        "RESOLVED_EXPLICIT_ZERO",
        "RESOLVED_EXPLICIT_NUMERIC",
        "EXISTING_PLACEHOLDER_ZERO_NOT_SEPARATELY_MODELED",
        "UNRESOLVED_LOGIC_BACKGROUND",
        "PARAMETRIC_SENSITIVITY_NOT_VALIDATED",
    ]
    memory_total_completeness_status: Literal[
        "RESOLVED_EXISTING_STATIC_COMPONENTS",
        "CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND",
        "UNRESOLVED_STATIC_POWER",
        "PARAMETRIC_SENSITIVITY",
    ]
    evaluation_status: Literal[
        "EVALUATED_WORKLOAD_DEPENDENT_MEMORY_POWER",
        "BLOCKED_BY_CAPACITY_OR_UPSTREAM_EVALUATION",
        "UNRESOLVED_STATIC_POWER",
    ]
    dynamic_power_status: Literal[
        "WORKLOAD_J_PER_TOKEN_TIMES_AGGREGATE_TOKENS_PER_SECOND"]
    static_power_status: Literal[
        "EXISTING_POWER_MODEL_COMPONENTS_ADDED_ONCE"]
    gpu_power_status: Literal[
        "FIXED_EXISTING_BASELINE_NOT_WORKLOAD_ENERGY_MODEL"]
    system_energy_status: Literal[
        "NOT_AVAILABLE_COMPUTE_ENERGY_EXCLUDED"]
    scenario_status: Literal[
        "CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY"]

    @model_validator(mode="after")
    def _power_closure(self) -> "LLMDecodeWorkloadPowerMetrics":
        derived = (
            self.memory_dynamic_access_power_W,
            self.refresh_power_W,
            self.memory_background_power_W,
            self.logic_background_effective_W,
            self.memory_workload_total_W,
            self.fixed_gpu_power_W,
            self.package_workload_total_W,
        )
        if self.evaluation_status != STATUS_EVALUATED:
            if any(value is not None for value in derived):
                raise ValueError(
                    "derived power fields must be None when evaluation is blocked")
            return self

        if any(value is None for value in derived):
            raise ValueError("evaluated result must define every power field")
        assert self.memory_dynamic_access_power_W is not None
        assert self.refresh_power_W is not None
        assert self.memory_background_power_W is not None
        assert self.logic_background_effective_W is not None
        assert self.memory_workload_total_W is not None
        assert self.fixed_gpu_power_W is not None
        assert self.package_workload_total_W is not None
        expected_memory = (
            self.memory_dynamic_access_power_W + self.refresh_power_W
            + self.memory_background_power_W
            + self.logic_background_effective_W)
        if self.memory_workload_total_W != expected_memory:
            raise ValueError("memory workload power components do not close")
        if self.package_workload_total_W != (
                self.fixed_gpu_power_W + self.memory_workload_total_W):
            raise ValueError("package workload power components do not close")
        return self


def _common(
    energy: ArchitectureDecodeMemoryEnergyMetrics,
    policy: str,
) -> dict[str, object]:
    return {
        "architecture": energy.architecture,
        "rho": energy.rho,
        "capacity_feasible": energy.capacity_feasible,
        "unresolved_logic_background_policy": policy,
        "dynamic_power_status": DYNAMIC_POWER_STATUS,
        "static_power_status": STATIC_POWER_STATUS,
        "gpu_power_status": GPU_POWER_STATUS,
        "system_energy_status": SYSTEM_ENERGY_STATUS,
        "scenario_status": SCENARIO_STATUS,
    }


def evaluate_llm_decode_workload_power(
    energy: ArchitectureDecodeMemoryEnergyMetrics,
    performance: LLMDecodePerformanceMetrics,
    system: ResolvedSystemPower,
    *,
    unresolved_logic_background_policy: Literal[
        "REQUIRE_RESOLVED", "EXISTING_PLACEHOLDER_ZERO",
        "PARAMETRIC_SENSITIVITY"],
    logic_background_sensitivity_W: float | None = None,
) -> LLMDecodeWorkloadPowerMetrics:
    """Combine per-token memory energy, throughput, and static components.

    ``unresolved_logic_background_policy`` is mandatory.  The placeholder-zero
    policy is narrowly guarded by the existing system-total closure and cannot
    be applied when a numeric logic-background value already exists.
    """
    policy = unresolved_logic_background_policy
    if policy not in (POLICY_REQUIRE_RESOLVED,
                      POLICY_EXISTING_PLACEHOLDER_ZERO,
                      POLICY_PARAMETRIC_SENSITIVITY):
        raise ValueError("unsupported unresolved_logic_background_policy")

    if energy.architecture != system.case_name:
        raise ValueError("energy architecture does not match system.case_name")
    if performance.architecture != system.case_name:
        raise ValueError("performance architecture does not match system.case_name")
    if energy.capacity_feasible != performance.capacity_feasible:
        raise ValueError("energy/performance capacity feasibility mismatch")
    if energy.read_bytes_per_token != performance.read_bytes_per_token:
        raise ValueError("energy/performance read traffic mismatch")
    if energy.write_bytes_per_token != performance.write_bytes_per_token:
        raise ValueError("energy/performance write traffic mismatch")

    common = _common(energy, policy)
    blocked = (
        not energy.capacity_feasible
        or energy.memory_dynamic_energy_j_per_token is None
        or performance.aggregate_tokens_per_second is None
        or energy.evaluation_status != (
            "EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY")
        or performance.performance_status != (
            "EVALUATED_MATCHED_REFERENCE_SCENARIO")
    )
    if blocked:
        return LLMDecodeWorkloadPowerMetrics(
            **common,
            memory_dynamic_energy_j_per_token=None,
            aggregate_tokens_per_second=None,
            memory_dynamic_access_power_W=None,
            refresh_power_W=None,
            memory_background_power_W=None,
            logic_background_raw_W=None,
            logic_background_effective_W=None,
            memory_workload_total_W=None,
            fixed_gpu_power_W=None,
            package_workload_total_W=None,
            logic_background_status=LOGIC_STATUS_UNRESOLVED,
            memory_total_completeness_status=COMPLETENESS_UNRESOLVED,
            evaluation_status=STATUS_BLOCKED,
        )

    memory = system.memory_result
    if memory is None:
        return LLMDecodeWorkloadPowerMetrics(
            **common,
            memory_dynamic_energy_j_per_token=None,
            aggregate_tokens_per_second=None,
            memory_dynamic_access_power_W=None,
            refresh_power_W=None,
            memory_background_power_W=None,
            logic_background_raw_W=None,
            logic_background_effective_W=None,
            memory_workload_total_W=None,
            fixed_gpu_power_W=None,
            package_workload_total_W=None,
            logic_background_status=LOGIC_STATUS_UNRESOLVED,
            memory_total_completeness_status=COMPLETENESS_UNRESOLVED,
            evaluation_status=STATUS_UNRESOLVED_STATIC,
        )

    raw_logic = memory.P_logic_background_W
    if policy == POLICY_PARAMETRIC_SENSITIVITY:
        if raw_logic is not None:
            raise ValueError(
                "logic-background sensitivity requires unresolved nominal logic")
        if logic_background_sensitivity_W is None:
            raise ValueError(
                "PARAMETRIC_SENSITIVITY requires logic_background_sensitivity_W")
    elif logic_background_sensitivity_W is not None:
        raise ValueError(
            "logic_background_sensitivity_W requires PARAMETRIC_SENSITIVITY")
    if policy == POLICY_REQUIRE_RESOLVED and raw_logic is None:
        return LLMDecodeWorkloadPowerMetrics(
            **common,
            memory_dynamic_energy_j_per_token=None,
            aggregate_tokens_per_second=None,
            memory_dynamic_access_power_W=None,
            refresh_power_W=None,
            memory_background_power_W=None,
            logic_background_raw_W=None,
            logic_background_effective_W=None,
            memory_workload_total_W=None,
            fixed_gpu_power_W=None,
            package_workload_total_W=None,
            logic_background_status=LOGIC_STATUS_UNRESOLVED,
            memory_total_completeness_status=COMPLETENESS_UNRESOLVED,
            evaluation_status=STATUS_UNRESOLVED_STATIC,
        )
    if policy == POLICY_EXISTING_PLACEHOLDER_ZERO and raw_logic is not None:
        raise ValueError(
            "EXISTING_PLACEHOLDER_ZERO is forbidden when logic background "
            "is already numeric")

    energy_j = _finite_nonnegative(
        "energy.memory_dynamic_energy_j_per_token",
        energy.memory_dynamic_energy_j_per_token)
    throughput = _finite_nonnegative(
        "performance.aggregate_tokens_per_second",
        performance.aggregate_tokens_per_second)
    refresh = _finite_nonnegative("memory.P_refresh_W", memory.P_refresh_W)
    background = _finite_nonnegative(
        "memory.P_memory_background_W", memory.P_memory_background_W)
    gpu = _finite_nonnegative("system.gpu_power_W", system.gpu_power_W)

    if policy == POLICY_PARAMETRIC_SENSITIVITY:
        effective_logic = _finite_nonnegative(
            "logic_background_sensitivity_W",
            logic_background_sensitivity_W)
        logic_status = LOGIC_STATUS_PARAMETRIC
        completeness = COMPLETENESS_PARAMETRIC
    elif raw_logic is None:
        # The only permitted unresolved path: preserve raw None and use the
        # pre-existing zero placeholder iff the old resolved total closes.
        old_total = system.resolved_total_memory_power_W
        if old_total is None:
            raise ValueError(
                "placeholder-zero policy requires an existing resolved total")
        old_total_valid = _finite_nonnegative(
            "system.resolved_total_memory_power_W", old_total)
        old_access = _finite_nonnegative("memory.P_access_W", memory.P_access_W)
        expected_old = old_access + refresh + background
        if not math.isclose(
                old_total_valid, expected_old,
                rel_tol=0.0, abs_tol=_OLD_TOTAL_ABS_TOL_W):
            raise ValueError(
                "existing system total does not close with placeholder logic zero")
        effective_logic = 0.0
        logic_status = LOGIC_STATUS_PLACEHOLDER_ZERO
        completeness = COMPLETENESS_CONDITIONAL
    else:
        effective_logic = _finite_nonnegative(
            "memory.P_logic_background_W", raw_logic)
        logic_status = (
            LOGIC_STATUS_RESOLVED_ZERO if effective_logic == 0.0
            else LOGIC_STATUS_RESOLVED_NUMERIC)
        completeness = COMPLETENESS_RESOLVED

    dynamic = energy_j * throughput
    if not math.isfinite(dynamic):
        raise ValueError("derived dynamic access power must be finite")
    memory_total = dynamic + refresh + background + effective_logic
    package_total = gpu + memory_total
    if not math.isfinite(memory_total) or not math.isfinite(package_total):
        raise ValueError("derived workload power total must be finite")

    return LLMDecodeWorkloadPowerMetrics(
        **common,
        memory_dynamic_energy_j_per_token=energy_j,
        aggregate_tokens_per_second=throughput,
        memory_dynamic_access_power_W=dynamic,
        refresh_power_W=refresh,
        memory_background_power_W=background,
        logic_background_raw_W=raw_logic,
        logic_background_effective_W=effective_logic,
        memory_workload_total_W=memory_total,
        fixed_gpu_power_W=gpu,
        package_workload_total_W=package_total,
        logic_background_status=logic_status,
        memory_total_completeness_status=completeness,
        evaluation_status=STATUS_EVALUATED,
    )
