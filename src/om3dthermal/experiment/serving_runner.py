"""Formal orchestration for capacity-aware analytical serving only."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    sensitivity_operating_points: tuple["ServingSensitivityRunResult", ...]
    unresolved_capacity_references: tuple[str, ...]


@dataclass(frozen=True)
class ServingSensitivityRunResult:
    sensitivity_id: str
    status: str
    operating_points: tuple[ServingOperatingPointResult, ...]


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

    capacity_targets: list[tuple[str, object]] = []
    for config in experiment.architecture_configs:
        spec = load_architecture_spec(config, project_root=root)
        resolved = resolve_architecture_spec(spec, project_root=root)
        capacity = evaluate_architecture_capacity_feasibility(
            demand,
            resolved.packing,
            reserved_capacity_bytes=experiment.serving.reserved_capacity_bytes,
        )
        capacity_targets.append((spec.architecture_id, capacity))

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
        capacity_targets.append((reference.reference_id, capacity))

    def evaluate_targets(host_offload, host_overlap):
        return tuple(search_serving_operating_point(
            architecture=architecture,
            workload_id=workload_spec.workload_id,
            workload=workload_spec.decode,
            capacity=capacity,
            requested_request_points=experiment.serving.requested_requests,
            host_offload=host_offload,
            overlap=host_overlap,
            gpu_model=gpu_model,
        ) for architecture, capacity in capacity_targets)

    results = evaluate_targets(platform.host_offload, overlap)
    sensitivity_results = tuple(
        ServingSensitivityRunResult(
            sensitivity_id=point.sensitivity_id,
            status=point.status,
            operating_points=evaluate_targets(
                point.host_offload,
                HostOverlapSpec(
                    policy=point.host_overlap_policy,
                    overlap_fraction=point.host_overlap_fraction,
                ),
            ),
        )
        for point in experiment.serving.sensitivity_points
    )
    return ServingExperimentRunResult(
        experiment=experiment,
        operating_points=results,
        sensitivity_operating_points=sensitivity_results,
        unresolved_capacity_references=tuple(unresolved),
    )


def write_serving_experiment_csvs(
    result: ServingExperimentRunResult,
) -> tuple[Path, Path]:
    """Write plot-ready nominal and sensitivity rows; no figures are created."""
    output_dir = result.experiment.output_dir
    if output_dir.exists() and result.experiment.output_policy == "ERROR_IF_EXISTS":
        raise FileExistsError(f"serving output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    nominal_path = output_dir / "serving_nominal.csv"
    sensitivity_path = output_dir / "serving_sensitivity.csv"
    _write_rows(nominal_path, _flatten_rows(result.operating_points, "nominal", "NOMINAL"))
    sensitivity_rows = (
        row
        for sensitivity in result.sensitivity_operating_points
        for row in _flatten_rows(
            sensitivity.operating_points,
            sensitivity.sensitivity_id,
            sensitivity.status,
        )
    )
    _write_rows(sensitivity_path, sensitivity_rows)
    return nominal_path, sensitivity_path


def _flatten_rows(
    operating_points: Iterable[ServingOperatingPointResult],
    scenario_id: str,
    scenario_status: str,
) -> Iterable[dict[str, object]]:
    for operating_point in operating_points:
        for row in operating_point.rows:
            values = row.model_dump()
            values["scenario_id"] = scenario_id
            values["scenario_status"] = scenario_status
            values["usable_capacity_GB"] = row.usable_capacity_bytes / 1e9
            values["kv_bytes_per_request_GiB"] = row.kv_bytes_per_request / 2**30
            values["host_read_GB_per_step"] = row.host_read_bytes_per_step / 1e9
            values["host_write_GB_per_step"] = row.host_write_bytes_per_step / 1e9
            values["host_total_GB_per_step"] = row.host_transfer_bytes_per_step / 1e9
            effective = row.host_effective_bandwidth_bytes_per_second
            values["host_effective_bandwidth_GBps"] = (
                None if effective is None else effective / 1e9)
            yield values


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("serving CSV requires at least one row")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
