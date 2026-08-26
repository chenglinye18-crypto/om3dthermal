"""GPU decode-step performance interfaces for capacity-aware serving."""

from __future__ import annotations

import math
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.evaluation import ArchitectureCapacityFeasibility
from om3dthermal.evaluator import evaluate_llm_decode_performance
from om3dthermal.workload import LLMDecodeInput, evaluate_llm_decode


class GPUDecodeStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int
    decode_step_time_ms: float = Field(gt=0.0)
    aggregate_tokens_per_s: float = Field(gt=0.0)
    backend: Literal["ANALYTICAL_CONDITIONAL", "MEASURED_BATCH_CURVE"]
    backend_status: str


class GPUDecodePerformanceModel(Protocol):
    def evaluate(
        self, workload: LLMDecodeInput, *, batch_size: int
    ) -> GPUDecodeStepResult: ...


class AnalyticalRooflineGPUModel:
    """Adapter over the existing matched-reference roofline evaluator."""

    def __init__(
        self,
        *,
        matched_payload_bandwidth_bits_per_second: float,
        effective_compute_flops_per_second: float,
    ) -> None:
        self.bandwidth = matched_payload_bandwidth_bits_per_second
        self.compute = effective_compute_flops_per_second

    def evaluate(
        self, workload: LLMDecodeInput, *, batch_size: int
    ) -> GPUDecodeStepResult:
        metrics = evaluate_llm_decode(
            workload.model_copy(update={"batch_size": batch_size})
        )
        # The serving capacity gate is evaluated separately. This adapter uses
        # an explicit always-feasible boundary solely to reuse the frozen
        # roofline arithmetic for the active GPU batch.
        required = metrics.required_capacity_bytes
        capacity = ArchitectureCapacityFeasibility(
            architecture="serving_gpu_resource_boundary",
            physical_capacity_bytes=required,
            physical_capacity_GiB=required / 2**30,
            reserved_capacity_bytes=0,
            usable_capacity_bytes=required,
            required_capacity_bytes=required,
            capacity_margin_bytes=0.0,
            capacity_utilization=(1.0 if required > 0.0 else 0.0),
            capacity_feasible=True,
            capacity_scope_status="GPU_RESOURCE_TIME_CAPACITY_GATED_SEPARATELY",
            capacity_source_status="SERVING_ADAPTER_NOT_ARCHITECTURE_CAPACITY",
        )
        result = evaluate_llm_decode_performance(
            metrics,
            capacity,
            batch_size=batch_size,
            matched_payload_bandwidth_bits_per_second=self.bandwidth,
            effective_compute_flops_per_second=self.compute,
        )
        assert result.aggregate_step_time_s is not None
        assert result.aggregate_tokens_per_second is not None
        return GPUDecodeStepResult(
            batch_size=batch_size,
            decode_step_time_ms=result.aggregate_step_time_s * 1e3,
            aggregate_tokens_per_s=result.aggregate_tokens_per_second,
            backend="ANALYTICAL_CONDITIONAL",
            backend_status="MATCHED_REFERENCE_NOT_VALIDATED_BATCH_CURVE",
        )


class MeasuredBatchCurvePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(gt=0)
    decode_step_ms: float = Field(gt=0.0)
    aggregate_tokens_per_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _token_step_closure(self) -> "MeasuredBatchCurvePoint":
        expected = self.batch_size / (self.decode_step_ms * 1e-3)
        if not math.isclose(
            self.aggregate_tokens_per_s, expected, rel_tol=1e-9, abs_tol=0.0
        ):
            raise ValueError(
                "measured curve point must close: tokens/s = batch/step_time"
            )
        return self


class MeasuredBatchCurveGPUModel:
    """Deterministic bounded linear interpolation of measured step time."""

    def __init__(self, points: tuple[MeasuredBatchCurvePoint, ...]) -> None:
        if not points:
            raise ValueError("measured batch curve must not be empty")
        ordered = tuple(sorted(points, key=lambda point: point.batch_size))
        if len({point.batch_size for point in ordered}) != len(ordered):
            raise ValueError("measured batch sizes must be unique")
        self.points = ordered

    def evaluate(
        self, workload: LLMDecodeInput, *, batch_size: int
    ) -> GPUDecodeStepResult:
        del workload
        if batch_size < self.points[0].batch_size or batch_size > self.points[-1].batch_size:
            raise ValueError("batch size lies outside measured curve bounds")
        for point in self.points:
            if point.batch_size == batch_size:
                step_ms = point.decode_step_ms
                break
        else:
            upper_index = next(
                index
                for index, point in enumerate(self.points)
                if point.batch_size > batch_size
            )
            lower = self.points[upper_index - 1]
            upper = self.points[upper_index]
            fraction = (
                (batch_size - lower.batch_size)
                / (upper.batch_size - lower.batch_size)
            )
            step_ms = lower.decode_step_ms + fraction * (
                upper.decode_step_ms - lower.decode_step_ms
            )
        return GPUDecodeStepResult(
            batch_size=batch_size,
            decode_step_time_ms=step_ms,
            aggregate_tokens_per_s=batch_size / (step_ms * 1e-3),
            backend="MEASURED_BATCH_CURVE",
            backend_status="USER_SUPPLIED_MEASURED_CURVE",
        )
