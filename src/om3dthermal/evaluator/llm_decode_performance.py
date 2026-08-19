"""LLM Decode Performance Primitive v0 — Matched-Bandwidth Scenario Only.

This module connects a frozen :class:`LLMDecodeMetrics` result and an
:class:`ArchitectureCapacityFeasibility` gate to an explicit hardware
bandwidth and compute throughput scenario.  It then computes the
per-token-equivalent memory and compute times under a roofline-max
overlap model, and derives aggregate and per-sequence decode
throughput.

Scope (frozen for v0):

* **Matched-reference scenario only.**  The 39.2 Tb/s bandwidth input
  is a ``MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED`` scenario input;
  it is **not** a measured hardware capability, an architecture
  maximum, or a sustained production bandwidth.  Likewise, the
  100 TFLOP/s compute throughput is a
  ``NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED`` scenario input.  No
  value here may be cited as an M3D, HBM, or any other architecture
  capability.
* **Capacity gate is mandatory.**  When the upstream
  ``ArchitectureCapacityFeasibility.capacity_feasible`` flag is
  ``False``, this evaluator returns a ``BLOCKED_BY_CAPACITY`` status
  and sets every numeric performance field to ``None``.  It must
  never emit a deceptive tokens/s number for a workload that does
  not fit in the available hardware capacity.
* **SHARED_READ_WRITE_PAYLOAD_BANDWIDTH.**  Read and write traffic
  share one aggregate payload bandwidth budget.  Full-duplex and
  independent read/write channels are explicitly out of scope for
  v0; they must be added as a separate task with a new
  ``memory_bandwidth_model`` literal if needed.
* **ROOFLINE_MAX overlap.**  The per-token-equivalent resource
  time is the maximum of memory and compute time, not their sum.
  An additive model must be added as a separate task with a new
  ``overlap_model`` literal if needed.
* **Aggregate vs per-sequence semantics are explicit.**  The
  ``token_equivalent_time_s`` is *not* a per-sequence step latency.
  One aggregate decode step generates ``B`` tokens, so the
  per-sequence step latency equals the aggregate step time, which
  spans ``B`` token-equivalent work units.  Aggregate throughput
  and per-sequence throughput are derived from the same
  ``token_equivalent_time_s`` and differ only by the ``batch_size``
  factor.

This module is performance-only.  Energy, J/token, power, thermal,
and Tmax are explicitly out of scope and must not be added here.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from om3dthermal.workload.architecture_capacity import (
    ArchitectureCapacityFeasibility,
)
from om3dthermal.workload.llm_decode import LLMDecodeMetrics


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Status emitted when the workload does not fit in the architecture's
#: usable capacity.  All numeric performance fields are set to ``None``;
#: ``bottleneck`` is set to ``NOT_EVALUATED_CAPACITY_INFEASIBLE``.
STATUS_BLOCKED_BY_CAPACITY = "BLOCKED_BY_CAPACITY"

#: Status emitted when the workload is capacity-feasible and the
#: performance evaluation ran on the supplied matched-reference
#: scenario inputs.  This is not an architecture capability
#: comparison.
STATUS_EVALUATED_MATCHED = "EVALUATED_MATCHED_REFERENCE_SCENARIO"

#: Sentinel bottleneck string used when the capacity gate is closed.
BOTTLENECK_NOT_EVALUATED = "NOT_EVALUATED_CAPACITY_INFEASIBLE"

#: Matched-reference bandwidth status echoed from caller input.
MATCHED_BW_STATUS = "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED"

#: Illustrative compute throughput status echoed from caller input.
COMPUTE_NUMERICAL_STATUS = "NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED"

#: The only supported memory bandwidth model in v0.
MEMORY_BANDWIDTH_MODEL = "SHARED_READ_WRITE_PAYLOAD_BANDWIDTH"

#: The only supported overlap model in v0.
OVERLAP_MODEL = "ROOFLINE_MAX"

#: Bottleneck labels emitted by the roofline-max classifier.
BOTTLENECK_MEMORY = "MEMORY"
BOTTLENECK_COMPUTE = "COMPUTE"
BOTTLENECK_BALANCED = "BALANCED"

#: Tolerance used to decide whether memory and compute time are
#: numerically equal at the per-token-equivalent scale.  Chosen
#: tight (rel_tol=1e-12, abs_tol=0.0) so equal-times calls do not
#: silently hide a small but real gap.
_BALANCE_REL_TOL = 1e-12
_BALANCE_ABS_TOL = 0.0


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class LLMDecodePerformanceMetrics(BaseModel):
    """Performance evaluation result for one architecture + scenario.

    All byte- and time-valued fields are floats.  When the upstream
    capacity gate is closed (``capacity_feasible=False``), every
    numeric time and throughput field is ``None``; only the gate
    signal, the workload scenario inputs, and the provenance labels
    are populated.

    The provenance labels (bandwidth, compute, memory-bandwidth
    model, overlap model) are echoed verbatim from caller input; the
    evaluator does not attempt to reclassify them.
    """

    # Architecture and gating
    architecture: str
    batch_size: int
    capacity_feasible: bool

    # Workload (echoed from LLMDecodeMetrics for self-containment)
    read_bytes_per_token: float
    write_bytes_per_token: float
    traffic_bits_per_token: float
    flops_per_token: int

    # Scenario inputs (echoed verbatim)
    matched_payload_bandwidth_bits_per_second: float
    effective_compute_flops_per_second: float

    # Per-token-equivalent resource times (None if capacity infeasible)
    memory_time_per_token_equivalent_s: float | None
    compute_time_per_token_equivalent_s: float | None
    token_equivalent_time_s: float | None

    # Aggregate and per-sequence throughput (None if capacity infeasible)
    aggregate_step_time_s: float | None
    aggregate_tokens_per_second: float | None
    per_sequence_tokens_per_second: float | None
    per_sequence_step_latency_s: float | None

    # Crossover diagnostic (None if capacity infeasible OR if
    # memory_time == 0; see compute_throughput_required_to_match_memory
    # for the divide-by-zero policy)
    compute_throughput_required_to_match_memory_flops_per_second: float | None

    # Classification
    bottleneck: Literal[
        "MEMORY",
        "COMPUTE",
        "BALANCED",
        "NOT_EVALUATED_CAPACITY_INFEASIBLE",
    ]

    # Status and provenance
    performance_status: Literal[
        "EVALUATED_MATCHED_REFERENCE_SCENARIO",
        "BLOCKED_BY_CAPACITY",
    ]
    bandwidth_status: Literal["MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED"]
    compute_throughput_status: Literal[
        "NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED",
    ]
    memory_bandwidth_model: Literal["SHARED_READ_WRITE_PAYLOAD_BANDWIDTH"]
    overlap_model: Literal["ROOFLINE_MAX"]

    @field_validator("matched_payload_bandwidth_bits_per_second",
                     "effective_compute_flops_per_second")
    @classmethod
    def _positive_finite(cls, v: float) -> float:
        # Defensive duplicate of input-time validation: Pydantic will
        # surface this if a caller bypasses the public function and
        # constructs the model directly with a bad value.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError("scenario input must be a real number")
        v_float = float(v)
        if not math.isfinite(v_float):
            raise ValueError("scenario input must be finite")
        if v_float <= 0:
            raise ValueError("scenario input must be strictly positive")
        return v_float

    @field_validator("batch_size")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError("batch_size must be an int")
        if v <= 0:
            raise ValueError("batch_size must be a positive integer")
        return v

    @model_validator(mode="after")
    def _blocked_consistency(self) -> "LLMDecodePerformanceMetrics":
        """When capacity is infeasible, every numeric performance
        field must be ``None`` and ``bottleneck`` must be the
        not-evaluated sentinel.  When capacity is feasible, none of
        those fields may be ``None``.
        """
        if not self.capacity_feasible:
            none_fields = (
                "memory_time_per_token_equivalent_s",
                "compute_time_per_token_equivalent_s",
                "token_equivalent_time_s",
                "aggregate_step_time_s",
                "aggregate_tokens_per_second",
                "per_sequence_tokens_per_second",
                "per_sequence_step_latency_s",
                "compute_throughput_required_to_match_memory_flops_per_second",
            )
            for name in none_fields:
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"{name} must be None when capacity_feasible=False")
            if self.bottleneck != BOTTLENECK_NOT_EVALUATED:
                raise ValueError(
                    "bottleneck must be NOT_EVALUATED_CAPACITY_INFEASIBLE "
                    "when capacity_feasible=False")
            if self.performance_status != STATUS_BLOCKED_BY_CAPACITY:
                raise ValueError(
                    "performance_status must be BLOCKED_BY_CAPACITY "
                    "when capacity_feasible=False")
        else:
            none_fields = (
                "memory_time_per_token_equivalent_s",
                "compute_time_per_token_equivalent_s",
                "token_equivalent_time_s",
                "aggregate_step_time_s",
                "aggregate_tokens_per_second",
                "per_sequence_tokens_per_second",
                "per_sequence_step_latency_s",
            )
            for name in none_fields:
                if getattr(self, name) is None:
                    raise ValueError(
                        f"{name} must not be None when capacity_feasible=True")
        return self


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_positive_real(name: str, value: float) -> float:
    """Reject NaN, +-inf, zero, and negative values without coercing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    v_float = float(value)
    if not math.isfinite(v_float):
        raise ValueError(f"{name} must be finite")
    if v_float <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return v_float


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_nonneg_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    v_float = float(value)
    if not math.isfinite(v_float):
        raise ValueError(f"{name} must be finite")
    if v_float < 0:
        raise ValueError(f"{name} must be non-negative")
    return v_float


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_llm_decode_performance(
    workload: LLMDecodeMetrics,
    capacity: ArchitectureCapacityFeasibility,
    *,
    batch_size: int,
    matched_payload_bandwidth_bits_per_second: float,
    effective_compute_flops_per_second: float,
    memory_bandwidth_model: Literal[
        "SHARED_READ_WRITE_PAYLOAD_BANDWIDTH",
    ] = MEMORY_BANDWIDTH_MODEL,
    overlap_model: Literal["ROOFLINE_MAX"] = OVERLAP_MODEL,
    bandwidth_status: Literal[
        "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED",
    ] = MATCHED_BW_STATUS,
    compute_throughput_status: Literal[
        "NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED",
    ] = COMPUTE_NUMERICAL_STATUS,
) -> LLMDecodePerformanceMetrics:
    """Evaluate aggregate / per-sequence decode performance under one
    explicit matched-bandwidth scenario.

    The scenario inputs are **mandatory**: ``batch_size``,
    ``matched_payload_bandwidth_bits_per_second``, and
    ``effective_compute_flops_per_second`` are keyword-only and have
    no implicit default.  The four label / model keywords are
    defaulted to their frozen v0 values to keep call sites compact
    while remaining explicit enough that audit / test code can
    override them.

    Capacity gate:
        If ``capacity.capacity_feasible`` is ``False``, the result
        carries ``performance_status = BLOCKED_BY_CAPACITY``, every
        numeric performance field is ``None``, and ``bottleneck`` is
        the ``NOT_EVALUATED_CAPACITY_INFEASIBLE`` sentinel.  This
        evaluator does not emit a deceptive tokens/s number for a
        workload that does not fit in the architecture.

    Bandwidth model:
        ``SHARED_READ_WRITE_PAYLOAD_BANDWIDTH`` (v0).  Read and write
        traffic are merged into a single ``traffic_bits_per_token``
        and divided by ``matched_payload_bandwidth_bits_per_second``.
        No full-duplex and no independent read/write channels.

    Overlap model:
        ``ROOFLINE_MAX`` (v0).  The per-token-equivalent resource
        time is the maximum of memory and compute time.

    Batch semantics:
        One aggregate decode step generates ``B`` tokens.  Therefore
        ``aggregate_step_time_s = B * token_equivalent_time_s`` and
        ``per_sequence_step_latency_s = aggregate_step_time_s``.
        ``aggregate_tokens_per_second = B / aggregate_step_time_s =
        1 / token_equivalent_time_s``;
        ``per_sequence_tokens_per_second = 1 /
        aggregate_step_time_s = aggregate_tokens_per_second / B``.

    Crossover diagnostic policy:
        ``compute_throughput_required_to_match_memory_flops_per_second
        = flops_per_token / memory_time_per_token_equivalent_s``.
        When ``memory_time == 0`` (workload has zero memory traffic
        for the supported dimensions), the divide-by-zero policy is
        to return ``math.inf`` — i.e. no finite compute throughput
        can match a zero memory time, so memory cannot be the
        bottleneck.  Callers can detect this with ``math.isinf``.
    """
    # ----- Validate scenario inputs (no silent coercion) ---------------
    bsz = _validate_positive_int("batch_size", batch_size)
    bw = _validate_positive_real(
        "matched_payload_bandwidth_bits_per_second",
        matched_payload_bandwidth_bits_per_second,
    )
    flops_per_s = _validate_positive_real(
        "effective_compute_flops_per_second",
        effective_compute_flops_per_second,
    )

    # ----- Validate workload (no silent coercion) -----------------------
    r_bpt = _validate_nonneg_real(
        "workload.read_bytes_per_token", workload.read_bytes_per_token)
    w_bpt = _validate_nonneg_real(
        "workload.write_bytes_per_token", workload.write_bytes_per_token)
    flops = workload.flops_per_token
    if not isinstance(flops, int) or isinstance(flops, bool):
        raise TypeError("workload.flops_per_token must be an int")
    if flops < 0:
        raise ValueError("workload.flops_per_token must be non-negative")

    # ----- Capacity gate ----------------------------------------------
    if not capacity.capacity_feasible:
        return LLMDecodePerformanceMetrics(
            architecture=capacity.architecture,
            batch_size=bsz,
            capacity_feasible=False,
            read_bytes_per_token=r_bpt,
            write_bytes_per_token=w_bpt,
            traffic_bits_per_token=(r_bpt + w_bpt) * 8.0,
            flops_per_token=flops,
            matched_payload_bandwidth_bits_per_second=float(bw),
            effective_compute_flops_per_second=float(flops_per_s),
            memory_time_per_token_equivalent_s=None,
            compute_time_per_token_equivalent_s=None,
            token_equivalent_time_s=None,
            aggregate_step_time_s=None,
            aggregate_tokens_per_second=None,
            per_sequence_tokens_per_second=None,
            per_sequence_step_latency_s=None,
            compute_throughput_required_to_match_memory_flops_per_second=None,
            bottleneck=BOTTLENECK_NOT_EVALUATED,
            performance_status=STATUS_BLOCKED_BY_CAPACITY,
            bandwidth_status=bandwidth_status,
            compute_throughput_status=compute_throughput_status,
            memory_bandwidth_model=memory_bandwidth_model,
            overlap_model=overlap_model,
        )

    # ----- Memory boundary (SHARED_READ_WRITE_PAYLOAD_BANDWIDTH) -----
    traffic_bytes_per_token = r_bpt + w_bpt
    traffic_bits_per_token = traffic_bytes_per_token * 8.0
    memory_time = traffic_bits_per_token / float(bw)

    # ----- Compute boundary -------------------------------------------
    compute_time = float(flops) / float(flops_per_s)

    # ----- Roofline-max overlap ---------------------------------------
    token_equivalent_time = max(memory_time, compute_time)
    if math.isclose(memory_time, compute_time,
                    rel_tol=_BALANCE_REL_TOL, abs_tol=_BALANCE_ABS_TOL):
        bottleneck = BOTTLENECK_BALANCED
    elif memory_time > compute_time:
        bottleneck = BOTTLENECK_MEMORY
    else:
        bottleneck = BOTTLENECK_COMPUTE

    # ----- Batch semantics --------------------------------------------
    aggregate_step_time = bsz * token_equivalent_time
    aggregate_tokens_per_s = 1.0 / token_equivalent_time
    per_seq_tokens_per_s = 1.0 / aggregate_step_time
    per_seq_step_latency = aggregate_step_time

    # ----- Crossover diagnostic ---------------------------------------
    if memory_time > 0.0:
        crossover_compute_flops_per_s = float(flops) / memory_time
    else:
        # Divide-by-zero policy: no finite compute throughput can
        # match a zero memory time, so memory cannot be the
        # bottleneck.  ``math.inf`` is the documented sentinel.
        crossover_compute_flops_per_s = math.inf

    return LLMDecodePerformanceMetrics(
        architecture=capacity.architecture,
        batch_size=bsz,
        capacity_feasible=True,
        read_bytes_per_token=r_bpt,
        write_bytes_per_token=w_bpt,
        traffic_bits_per_token=traffic_bits_per_token,
        flops_per_token=flops,
        matched_payload_bandwidth_bits_per_second=float(bw),
        effective_compute_flops_per_second=float(flops_per_s),
        memory_time_per_token_equivalent_s=memory_time,
        compute_time_per_token_equivalent_s=compute_time,
        token_equivalent_time_s=token_equivalent_time,
        aggregate_step_time_s=aggregate_step_time,
        aggregate_tokens_per_second=aggregate_tokens_per_s,
        per_sequence_tokens_per_second=per_seq_tokens_per_s,
        per_sequence_step_latency_s=per_seq_step_latency,
        compute_throughput_required_to_match_memory_flops_per_second=(
            crossover_compute_flops_per_s),
        bottleneck=bottleneck,
        performance_status=STATUS_EVALUATED_MATCHED,
        bandwidth_status=bandwidth_status,
        compute_throughput_status=compute_throughput_status,
        memory_bandwidth_model=memory_bandwidth_model,
        overlap_model=overlap_model,
    )
