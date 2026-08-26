"""Formal orchestration for capacity-aware analytical serving only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from om3dthermal.adapters import resolve_architecture_spec
from om3dthermal.evaluation import evaluate_architecture_capacity_feasibility
from om3dthermal.serving import (
    AnalyticalRooflineGPUModel,
    HostOverlapSpec,
    MeasuredBatchCurveGPUModel,
    ServingCapacitySource,
    ServingOperatingPointResult,
    search_serving_operating_point,
)
from om3dthermal.workload import evaluate_llm_decode, resolve_llm_decode_demand

from .config import (
    ServingExperimentSpec,
    load_architecture_spec,
    load_platform_spec,
    load_serving_experiment_spec,
    load_workload_spec,
)


@dataclass(frozen=True)
class ServingExperimentRunResult:
    experiment: ServingExperimentSpec
    operating_points: tuple[ServingOperatingPointResult, ...]
    unresolved_capacity_references: tuple[str, ...]


def run_serving_experiment(
    config_path: str | Path,
    *,
    project_root: Path,
) -> ServingExperimentRunResult:
    """Evaluate capacity/host/GPU serving without thermal construction."""
    root = project_root.resolve()
    experiment = load_serving_experiment_spec(config_path, project_root=root)
    platform = load_platform_spec(experiment.platform_config, project_root=root)
    if platform.host_offload is None:
        raise ValueError("serving experiment requires platform.host_offload")
    workload_spec = load_workload_spec(
        experiment.workload_config, project_root=root)
    base_metrics = evaluate_llm_decode(workload_spec.decode)
    demand = resolve_llm_decode_demand(workload_spec, base_metrics)
    overlap = HostOverlapSpec(
        policy=experiment.serving.host_overlap_policy,
        overlap_fraction=experiment.serving.host_overlap_fraction,
    )
    gpu_spec = experiment.serving.gpu_performance
    if gpu_spec.backend == "ANALYTICAL_CONDITIONAL":
        assert gpu_spec.matched_payload_bandwidth_bits_per_second is not None
        assert gpu_spec.effective_compute_flops_per_second is not None
        gpu_model = AnalyticalRooflineGPUModel(
            matched_payload_bandwidth_bits_per_second=(
                gpu_spec.matched_payload_bandwidth_bits_per_second),
            effective_compute_flops_per_second=(
                gpu_spec.effective_compute_flops_per_second),
        )
    else:
        gpu_model = MeasuredBatchCurveGPUModel(gpu_spec.measured_points)

    results: list[ServingOperatingPointResult] = []
    for config in experiment.architecture_configs:
        spec = load_architecture_spec(config, project_root=root)
        resolved = resolve_architecture_spec(spec, project_root=root)
        capacity = evaluate_architecture_capacity_feasibility(
            demand,
            resolved.packing,
            reserved_capacity_bytes=experiment.serving.reserved_capacity_bytes,
        )
        results.append(search_serving_operating_point(
            architecture=spec.architecture_id,
            workload_id=workload_spec.workload_id,
            workload=workload_spec.decode,
            capacity=capacity,
            requested_request_points=experiment.serving.requested_requests,
            host_offload=platform.host_offload,
            overlap=overlap,
            gpu_model=gpu_model,
        ))

    unresolved: list[str] = []
    for reference in experiment.serving.capacity_references:
        if reference.status == "UNRESOLVED":
            unresolved.append(reference.reference_id)
            continue
        assert reference.usable_capacity_bytes is not None
        capacity = ServingCapacitySource(
            architecture=reference.reference_id,
            usable_capacity_bytes=reference.usable_capacity_bytes,
            capacity_source_status="CAPACITY_REFERENCE_ONLY_NOT_ARCHITECTURE",
            provenance_status="CONFIGURED_REFERENCE_WITH_EXPLICIT_PROVENANCE",
        )
        results.append(search_serving_operating_point(
            architecture=reference.reference_id,
            workload_id=workload_spec.workload_id,
            workload=workload_spec.decode,
            capacity=capacity,
            requested_request_points=experiment.serving.requested_requests,
            host_offload=platform.host_offload,
            overlap=overlap,
            gpu_model=gpu_model,
        ))
    return ServingExperimentRunResult(
        experiment=experiment,
        operating_points=tuple(results),
        unresolved_capacity_references=tuple(unresolved),
    )
