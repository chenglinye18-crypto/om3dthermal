"""GPU decode energy evaluator (E8) — affine utilization model, baseline path.

This stage consumes the committed E4 (conditional memory energy) and the
matched-reference performance result, and adds GPU-side decode energy and a
system-level J/token figure **without** touching the frozen E7 row or the
thermal path:

* the frozen fixed GPU power remains the thermal source;
* at ``u = 1`` (memory-bottleneck matched scenario) the affine power equals
  the configured peak decode power, which the canonical platform sets equal
  to the fixed 300 W baseline — the old fixed assumption is recovered as the
  special case of this model;
* system J/token is the sum of GPU energy and the conditional memory dynamic
  energy.  It excludes host CPU/DRAM, cooling, and networking, and it is
  analytical with measured-reference-range parameters, not a measurement.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from om3dthermal.platform import AffineGPUDecodePowerSpec

from .llm_decode_architecture_energy import (
    ArchitectureDecodeMemoryEnergyMetrics,
)
from .llm_decode_performance import LLMDecodePerformanceMetrics


STATUS_EVALUATED = "EVALUATED_ANALYTICAL_GPU_DECODE_ENERGY"
STATUS_BLOCKED = "BLOCKED_BY_CAPACITY"
GPU_POWER_MODEL_STATUS = "ANALYTICAL_AFFINE_UTILIZATION_MODEL"
PARAMETER_PROVENANCE_STATUS = (
    "PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE")
SYSTEM_ENERGY_SCOPE_STATUS = (
    "GPU_PLUS_MEMORY_DYNAMIC_ONLY__EXCLUDES_HOST_CPU_DRAM_COOLING_NETWORK")
UTILIZATION_SEMANTICS_STATUS = (
    "GPU_SIDE_PAYLOAD_BYTES_OVER_PEAK_BANDWIDTH_TIME__CLAMPED_AT_ONE")


class GPUDecodeEnergyMetrics(BaseModel):
    """GPU and system decode energy for one architecture/rho row."""

    architecture: str
    rho: float
    capacity_feasible: bool

    memory_bandwidth_utilization: float | None
    utilization_clamped: bool | None
    gpu_decode_power_W: float | None
    token_time_s: float | None
    gpu_energy_j_per_token: float | None
    memory_dynamic_energy_j_per_token: float | None
    system_energy_j_per_token: float | None

    evaluation_status: Literal[
        "EVALUATED_ANALYTICAL_GPU_DECODE_ENERGY",
        "BLOCKED_BY_CAPACITY",
    ]
    gpu_power_model_status: Literal["ANALYTICAL_AFFINE_UTILIZATION_MODEL"]
    parameter_provenance_status: Literal[
        "PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE"]
    bandwidth_status: Literal[
        "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED"]
    system_energy_scope_status: Literal[
        "GPU_PLUS_MEMORY_DYNAMIC_ONLY__EXCLUDES_HOST_CPU_DRAM_COOLING_NETWORK"]
    utilization_semantics_status: Literal[
        "GPU_SIDE_PAYLOAD_BYTES_OVER_PEAK_BANDWIDTH_TIME__CLAMPED_AT_ONE"]


def evaluate_gpu_decode_energy(
    performance: LLMDecodePerformanceMetrics,
    energy: ArchitectureDecodeMemoryEnergyMetrics,
    spec: AffineGPUDecodePowerSpec,
) -> GPUDecodeEnergyMetrics:
    """Evaluate affine GPU decode energy for one architecture/rho row."""
    if performance.architecture != energy.architecture:
        raise ValueError("performance/energy architecture identity mismatch")
    if performance.capacity_feasible != energy.capacity_feasible:
        raise ValueError("performance/energy capacity feasibility mismatch")

    common = {
        "architecture": energy.architecture,
        "rho": energy.rho,
        "capacity_feasible": energy.capacity_feasible,
        "gpu_power_model_status": GPU_POWER_MODEL_STATUS,
        "parameter_provenance_status": PARAMETER_PROVENANCE_STATUS,
        "bandwidth_status": spec.bandwidth_status,
        "system_energy_scope_status": SYSTEM_ENERGY_SCOPE_STATUS,
        "utilization_semantics_status": UTILIZATION_SEMANTICS_STATUS,
    }

    if not energy.capacity_feasible:
        if performance.performance_status != "BLOCKED_BY_CAPACITY":
            raise ValueError(
                "capacity-infeasible GPU energy requires blocked performance")
        return GPUDecodeEnergyMetrics(
            **common,
            memory_bandwidth_utilization=None,
            utilization_clamped=None,
            gpu_decode_power_W=None,
            token_time_s=None,
            gpu_energy_j_per_token=None,
            memory_dynamic_energy_j_per_token=None,
            system_energy_j_per_token=None,
            evaluation_status=STATUS_BLOCKED,
        )

    if performance.token_equivalent_time_s is None:
        raise ValueError("evaluated performance is missing token time")
    if energy.memory_dynamic_energy_j_per_token is None:
        raise ValueError("evaluated energy is missing memory dynamic energy")

    token_time_s = performance.token_equivalent_time_s
    gpu_side_bytes = (
        performance.read_bytes_per_token + performance.write_bytes_per_token)
    utilization = gpu_side_bytes / (
        spec.peak_memory_bandwidth_bytes_per_s * token_time_s)
    clamped = utilization > 1.0
    utilization = min(utilization, 1.0)
    gpu_power_W = (
        spec.static_power_W
        + (spec.peak_decode_power_W - spec.static_power_W) * utilization)
    gpu_energy = gpu_power_W * token_time_s
    memory_energy = energy.memory_dynamic_energy_j_per_token

    return GPUDecodeEnergyMetrics(
        **common,
        memory_bandwidth_utilization=utilization,
        utilization_clamped=clamped,
        gpu_decode_power_W=gpu_power_W,
        token_time_s=token_time_s,
        gpu_energy_j_per_token=gpu_energy,
        memory_dynamic_energy_j_per_token=memory_energy,
        system_energy_j_per_token=gpu_energy + memory_energy,
        evaluation_status=STATUS_EVALUATED,
    )
