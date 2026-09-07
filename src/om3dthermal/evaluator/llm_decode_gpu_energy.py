"""GPU decode energy evaluator (E8) — bandwidth-bounded baseline path.

This stage consumes the committed E4 (conditional memory energy) and the
matched-reference performance result, and adds GPU-side decode energy and a
scoped GPU-plus-memory-dynamic J/token figure. The formal runner evaluates
this stage before workload power and thermal mapping so they share one GPU
operating point:

* the GPU thermal source uses this stage's evaluated GPU power;
* memory-bound and compute-bound rows use their respective canonical
  rate-capped resolver; their dynamic terms are never added;
* system J/token is the sum of GPU energy and the conditional memory dynamic
  energy.  It excludes host CPU/DRAM, cooling, and networking, and it is
  analytical with measured-reference-range parameters, not a measurement.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

from om3dthermal.platform import (
    AffineGPUComputePowerSpec,
    AffineGPUDecodePowerSpec,
    GPUComputePowerOperatingPoint,
    GPUDecodePowerOperatingPoint,
    resolve_gpu_compute_power,
    resolve_gpu_decode_power,
)

from .llm_decode_architecture_energy import (
    ArchitectureDecodeMemoryEnergyMetrics,
)
from .llm_decode_performance import LLMDecodePerformanceMetrics


STATUS_EVALUATED = "EVALUATED_ANALYTICAL_GPU_DECODE_ENERGY"
STATUS_BLOCKED = "BLOCKED_BY_CAPACITY"
GPU_POWER_MODEL_STATUS = "ANALYTICAL_AFFINE_UTILIZATION_MODEL"
PARAMETER_PROVENANCE_STATUS = (
    "MEASURED_REFERENCE_RANGE_WITH_EXPLICIT_COMPUTE_SELECTION")
SYSTEM_ENERGY_SCOPE_STATUS = (
    "GPU_PLUS_MEMORY_DYNAMIC_ONLY__EXCLUDES_HOST_CPU_DRAM_COOLING_NETWORK")
UTILIZATION_SEMANTICS_STATUS = (
    "REGIME_ACTUAL_RATE_OVER_CEILING__STRICT_EXCEEDANCE_SATURATION")


class GPUDecodeEnergyMetrics(BaseModel):
    """GPU and system decode energy for one architecture/rho row."""

    architecture: str
    rho: float
    capacity_feasible: bool

    memory_bandwidth_utilization: float | None
    utilization_clamped: bool | None
    bandwidth_demand_bytes_per_s: float | None
    bandwidth_actual_bytes_per_s: float | None
    bandwidth_saturated: bool | None
    gpu_dynamic_power_W: float | None
    compute_demand_flops_per_s: float | None
    compute_actual_flops_per_s: float | None
    compute_utilization: float | None
    compute_saturated: bool | None
    gpu_dynamic_compute_power_W: float | None
    compute_energy_dynamic_J_per_FLOP: float | None
    gpu_power_regime: Literal["MEMORY", "COMPUTE"] | None
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
        "MEASURED_REFERENCE_RANGE_WITH_EXPLICIT_COMPUTE_SELECTION"]
    bandwidth_status: Literal[
        "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED",
        "VENDOR_SPEC_H200_PEAK_HBM3E_BANDWIDTH",
    ]
    system_energy_scope_status: Literal[
        "GPU_PLUS_MEMORY_DYNAMIC_ONLY__EXCLUDES_HOST_CPU_DRAM_COOLING_NETWORK"]
    utilization_semantics_status: Literal[
        "REGIME_ACTUAL_RATE_OVER_CEILING__STRICT_EXCEEDANCE_SATURATION"]

    @property
    def gpu_power_operating_point(
        self,
    ) -> GPUDecodePowerOperatingPoint | GPUComputePowerOperatingPoint:
        """Reify the already-resolved E8 point without recalculating power."""
        if self.evaluation_status != STATUS_EVALUATED:
            raise ValueError("blocked GPU energy has no operating point")
        assert self.gpu_decode_power_W is not None
        if self.gpu_power_regime == "MEMORY":
            assert self.bandwidth_demand_bytes_per_s is not None
            assert self.bandwidth_actual_bytes_per_s is not None
            assert self.memory_bandwidth_utilization is not None
            assert self.bandwidth_saturated is not None
            assert self.gpu_dynamic_power_W is not None
            return GPUDecodePowerOperatingPoint(
                bandwidth_demand_bytes_per_s=(
                    self.bandwidth_demand_bytes_per_s),
                bandwidth_actual_bytes_per_s=(
                    self.bandwidth_actual_bytes_per_s),
                bandwidth_utilization=self.memory_bandwidth_utilization,
                bandwidth_saturated=self.bandwidth_saturated,
                gpu_dynamic_power_W=self.gpu_dynamic_power_W,
                gpu_power_W=self.gpu_decode_power_W,
            )
        assert self.gpu_power_regime == "COMPUTE"
        assert self.compute_demand_flops_per_s is not None
        assert self.compute_actual_flops_per_s is not None
        assert self.compute_utilization is not None
        assert self.compute_saturated is not None
        assert self.gpu_dynamic_compute_power_W is not None
        return GPUComputePowerOperatingPoint(
            compute_demand_flops_per_s=self.compute_demand_flops_per_s,
            compute_actual_flops_per_s=self.compute_actual_flops_per_s,
            compute_utilization=self.compute_utilization,
            compute_saturated=self.compute_saturated,
            gpu_dynamic_compute_power_W=self.gpu_dynamic_compute_power_W,
            gpu_power_W=self.gpu_decode_power_W,
        )


def evaluate_gpu_decode_energy(
    performance: LLMDecodePerformanceMetrics,
    energy: ArchitectureDecodeMemoryEnergyMetrics,
    spec: AffineGPUDecodePowerSpec,
    compute_spec: AffineGPUComputePowerSpec | None = None,
    *,
    compute_energy_dynamic_J_per_FLOP: float | None = None,
) -> GPUDecodeEnergyMetrics:
    """Evaluate regime-selected GPU decode energy for one architecture/rho.

    Memory-bound rows use the bandwidth resolver. Compute-bound rows require
    an explicitly selected dynamic coefficient from the configured range;
    there is no implicit nominal. Balanced rows are intentionally unresolved
    because bandwidth and compute dynamic terms must not be added or blended.
    """
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
            bandwidth_demand_bytes_per_s=None,
            bandwidth_actual_bytes_per_s=None,
            bandwidth_saturated=None,
            gpu_dynamic_power_W=None,
            compute_demand_flops_per_s=None,
            compute_actual_flops_per_s=None,
            compute_utilization=None,
            compute_saturated=None,
            gpu_dynamic_compute_power_W=None,
            compute_energy_dynamic_J_per_FLOP=None,
            gpu_power_regime=None,
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
    if compute_spec is not None:
        if compute_spec.static_power_W != spec.static_power_W:
            raise ValueError(
                "GPU decode and compute power models must share static power")
        if (performance.effective_compute_flops_per_second
                > compute_spec.peak_compute_BF16_dense_flops_per_s):
            raise ValueError(
                "effective compute ceiling cannot exceed vendor peak")

    bandwidth_values = {
        "memory_bandwidth_utilization": None,
        "utilization_clamped": None,
        "bandwidth_demand_bytes_per_s": None,
        "bandwidth_actual_bytes_per_s": None,
        "bandwidth_saturated": None,
        "gpu_dynamic_power_W": None,
    }
    compute_values = {
        "compute_demand_flops_per_s": None,
        "compute_actual_flops_per_s": None,
        "compute_utilization": None,
        "compute_saturated": None,
        "gpu_dynamic_compute_power_W": None,
        "compute_energy_dynamic_J_per_FLOP": None,
    }

    if performance.bottleneck == "MEMORY":
        gpu_side_bytes = (
            performance.read_bytes_per_token
            + performance.write_bytes_per_token)
        operating_point = resolve_gpu_decode_power(
            static_power_W=spec.static_power_W,
            e_decode_J_per_bit=spec.e_decode_J_per_bit,
            bandwidth_demand_bytes_per_s=gpu_side_bytes / token_time_s,
            peak_bandwidth_bytes_per_s=(
                spec.peak_memory_bandwidth_bytes_per_s),
        )
        bandwidth_values = {
            "memory_bandwidth_utilization": (
                operating_point.bandwidth_utilization),
            "utilization_clamped": operating_point.bandwidth_saturated,
            "bandwidth_demand_bytes_per_s": (
                operating_point.bandwidth_demand_bytes_per_s),
            "bandwidth_actual_bytes_per_s": (
                operating_point.bandwidth_actual_bytes_per_s),
            "bandwidth_saturated": operating_point.bandwidth_saturated,
            "gpu_dynamic_power_W": operating_point.gpu_dynamic_power_W,
        }
        gpu_power_W = operating_point.gpu_power_W
        gpu_power_regime = "MEMORY"
    elif performance.bottleneck == "COMPUTE":
        if compute_spec is None:
            raise ValueError(
                "compute-bound GPU power requires a compute power spec")
        if compute_energy_dynamic_J_per_FLOP is None:
            raise ValueError(
                "compute-bound GPU power requires an explicit dynamic "
                "coefficient selection; no nominal is configured")
        coefficient = float(compute_energy_dynamic_J_per_FLOP)
        if not math.isfinite(coefficient) or not (
            compute_spec.e_compute_dynamic_J_per_FLOP_min
            <= coefficient
            <= compute_spec.e_compute_dynamic_J_per_FLOP_max
        ):
            raise ValueError(
                "selected compute dynamic coefficient must be finite and "
                "within the configured sensitivity range")
        compute_demand = (
            performance
            .compute_throughput_required_to_match_memory_flops_per_second)
        if compute_demand is None or not math.isfinite(compute_demand):
            raise ValueError(
                "compute-bound power requires a finite workload compute demand")
        operating_point = resolve_gpu_compute_power(
            static_power_W=compute_spec.static_power_W,
            compute_energy_dynamic_J_per_FLOP=coefficient,
            compute_demand_FLOP_per_s=compute_demand,
            effective_compute_ceiling_FLOP_per_s=(
                performance.effective_compute_flops_per_second),
        )
        compute_values = {
            "compute_demand_flops_per_s": (
                operating_point.compute_demand_flops_per_s),
            "compute_actual_flops_per_s": (
                operating_point.compute_actual_flops_per_s),
            "compute_utilization": operating_point.compute_utilization,
            "compute_saturated": operating_point.compute_saturated,
            "gpu_dynamic_compute_power_W": (
                operating_point.gpu_dynamic_compute_power_W),
            "compute_energy_dynamic_J_per_FLOP": coefficient,
        }
        gpu_power_W = operating_point.gpu_power_W
        gpu_power_regime = "COMPUTE"
    elif performance.bottleneck == "BALANCED":
        raise ValueError(
            "balanced GPU power is unresolved; bandwidth and compute "
            "dynamic terms must not be combined")
    else:
        raise ValueError("evaluated GPU energy has an invalid bottleneck")

    gpu_energy = gpu_power_W * token_time_s
    memory_energy = energy.memory_dynamic_energy_j_per_token

    return GPUDecodeEnergyMetrics(
        **common,
        **bandwidth_values,
        **compute_values,
        gpu_power_regime=gpu_power_regime,
        gpu_decode_power_W=gpu_power_W,
        token_time_s=token_time_s,
        gpu_energy_j_per_token=gpu_energy,
        memory_dynamic_energy_j_per_token=memory_energy,
        system_energy_j_per_token=gpu_energy + memory_energy,
        evaluation_status=STATUS_EVALUATED,
    )
