"""Single formal orchestration path for the conditional LLM decode E2E flow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import math
import platform as platform_module
from pathlib import Path
import subprocess
import sys
from typing import Any

from om3dthermal import _git_metadata
from om3dthermal.adapters import resolve_architecture_spec
from om3dthermal.evaluator import (
    ConditionalLLMDecodeE2ERow,
    GPUDecodeEnergyMetrics,
    assemble_conditional_llm_decode_e2e_row,
    evaluate_architecture_decode_memory_energy,
    evaluate_gpu_decode_energy,
    evaluate_llm_decode_performance,
    evaluate_llm_decode_workload_power,
    map_workload_power_to_thermal,
    run_llm_decode_workload_thermal,
    validate_conditional_llm_decode_e2e_rows,
)
from om3dthermal.evaluation import evaluate_architecture_capacity_feasibility
from om3dthermal.provenance import RunProvenance
from om3dthermal.platform import (
    LocalMemoryGPUTransferOperatingPoint,
    resolve_gpu_decode_power,
    resolve_local_memory_gpu_transfer,
)
from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    resolve_case_geometry,
    resolve_effective_bandwidth,
    resolve_system_power,
)
from om3dthermal.result import write_result_bundle
from om3dthermal.workload import (
    evaluate_llm_decode,
    resolve_llm_decode_demand,
)

from .config import (
    ExperimentSpec,
    derive_orthogonal_slab_io_bandwidth_bits_per_second,
    load_architecture_spec,
    load_experiment_spec,
    load_platform_spec,
    load_workload_spec,
)
from .m3d_sensitivity import (
    M3DParameterSensitivityResult,
    run_m3d_parameter_sensitivity,
)


@dataclass(frozen=True)
class ExperimentRunResult:
    experiment: ExperimentSpec
    rows: tuple[ConditionalLLMDecodeE2ERow, ...]
    output_dir: Path | None
    provenance: RunProvenance
    m3d_parameter_sensitivity: M3DParameterSensitivityResult | None = None
    gpu_decode_energy: tuple[GPUDecodeEnergyMetrics, ...] | None = None


@dataclass(frozen=True)
class _ArchitectureCaseInput:
    spec: Any
    case: Any
    geometry: Any


def _project_root(path: Path) -> Path:
    current = path.resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"cannot locate project root from {path}")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _tracked_clean(project_root: Path) -> bool:
    result = subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain",
         "--untracked-files=no"],
        text=True,
    )
    return not result.strip()


def _resolved_architecture_payload(resolved) -> dict[str, Any]:
    memory = resolved.system_power.memory_result
    assert memory is not None
    return {
        "spec": resolved.spec,
        "canonical_case": resolved.case,
        "resolved_geometry": resolved.geometry,
        "resolved_packing": resolved.packing,
        "resolved_energy_primitives": {
            "read_energy_pj_per_bit": (
                resolved.system_power.memory_access_energy_pJ_per_bit),
            "memory_internal_pj_per_bit": memory.E_memory_internal_pj_bit,
            "vertical_pj_per_bit": memory.E_vertical_pj_bit,
            "feol_route_pj_per_bit": memory.E_feol_route_pj_bit,
            "base_route_pj_per_bit": memory.E_base_route_pj_bit,
            "interface_pj_per_bit": memory.E_interface_pj_bit,
        },
        "resolved_static_power": {
            "refresh_W": memory.P_refresh_W,
            "memory_background_W": memory.P_memory_background_W,
            "logic_background_W": memory.P_logic_background_W,
        },
    }


def _resolve_matched_bandwidth_bits_per_second(
    scenario,
    architecture_inputs,
) -> tuple[float, float | None]:
    """Resolve (applied, capability) matched payload bandwidth [bit/s].

    Literal scenarios return (literal, None).  Derived scenarios compute
    slab_count x links_per_slab x data_rate_gbps_per_link from the referenced
    architecture's geometry and contactless-interface spec; an optional cap pins
    the applied value below the derived capability, with the difference
    recorded as design margin.  The platform GPU decode-power peak bandwidth
    is a separate hardware capability anchor (their ratio defines the
    memory-bandwidth utilization u) and is intentionally not forced to equal
    the scenario bandwidth.
    """

    derivation = scenario.matched_bandwidth_derivation
    if derivation is None:
        literal = scenario.matched_payload_bandwidth_bits_per_second
        assert literal is not None  # guaranteed by scenario validation
        return float(literal), None
    matches = tuple(
        item for item in architecture_inputs
        if item.spec.architecture_id == derivation.architecture_id)
    if len(matches) != 1:
        raise ValueError(
            "matched bandwidth derivation requires exactly one "
            f"architecture {derivation.architecture_id!r} in the experiment")
    capability = derive_orthogonal_slab_io_bandwidth_bits_per_second(
        matches[0].case.geometry.orthogonal,
        matches[0].case.architecture.memory_service.coil,
        architecture_id=derivation.architecture_id)
    applied = capability
    if derivation.cap_bits_per_second is not None:
        applied = min(capability, derivation.cap_bits_per_second)
    return applied, capability


def _resolve_m3d_transfer_operating_point(
    item: _ArchitectureCaseInput,
    *,
    project_root: Path,
    bandwidth_demand_bytes_per_s: float,
    gpu_peak_bandwidth_bytes_per_s: float,
) -> LocalMemoryGPUTransferOperatingPoint | None:
    """Resolve the M3D raw capability, then the shared transfer rate."""

    if item.case.geometry.type != "orthogonal_m3d":
        return None
    intrinsic = calculate_memory_power(
        item.case,
        project_root=project_root,
        geometry=item.geometry,
        read_bandwidth_gbps=0.0,
    )
    closure = intrinsic.architecture_bandwidth_closure
    if closure is None:
        raise ValueError("M3D transfer requires architecture bandwidth closure")
    raw = resolve_effective_bandwidth(
        closure,
        closure.average_service_cycle_ns / closure.service_cycle_scale,
    )
    return resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=bandwidth_demand_bytes_per_s,
        memory_capability_bytes_per_s=raw.effective_bandwidth_bytes_per_s,
        gpu_peak_bandwidth_bytes_per_s=gpu_peak_bandwidth_bytes_per_s,
    )


def run_experiment(
    config_path: str | Path,
    *,
    project_root: Path | None = None,
    output_dir_override: Path | None = None,
    write_bundle: bool = True,
) -> ExperimentRunResult:
    """Run the configured stages using existing validated calculators."""

    path = Path(config_path).resolve()
    root = project_root.resolve() if project_root else _project_root(path)
    started = datetime.now(timezone.utc).isoformat()
    experiment = load_experiment_spec(path, project_root=root)
    output_dir = output_dir_override or experiment.output_dir
    if write_bundle and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "experiment output directory is not empty under "
            f"{experiment.output_policy}: {output_dir}")
    platform = load_platform_spec(
        experiment.platform_config, project_root=root)
    if platform.gpu_decode_power is None:
        raise ValueError("formal decode evaluation requires gpu_decode_power")
    decode_spec = platform.gpu_decode_power
    workload_spec = load_workload_spec(
        experiment.workload_config, project_root=root)
    architecture_specs = tuple(
        load_architecture_spec(item, project_root=root)
        for item in experiment.architecture_configs
    )
    architecture_input_rows = []
    for spec in architecture_specs:
        case = load_case_config(spec.canonical_case)
        architecture_input_rows.append(_ArchitectureCaseInput(
            spec=spec,
            case=case,
            geometry=resolve_case_geometry(case),
        ))
    architecture_inputs = tuple(architecture_input_rows)
    architecture_ids = tuple(item.architecture_id for item in architecture_specs)
    policies = experiment.scenario.unresolved_logic_background_policy
    if set(policies) != set(architecture_ids):
        raise ValueError(
            "unresolved logic-background policies must exactly match architectures")
    workload = evaluate_llm_decode(workload_spec.decode)
    workload_demand = resolve_llm_decode_demand(workload_spec, workload)
    (matched_bandwidth_bits_per_s,
     matched_bandwidth_capability_bits_per_s) = (
        _resolve_matched_bandwidth_bits_per_second(
            experiment.scenario, architecture_inputs))
    bandwidth_demand_bytes_per_s = matched_bandwidth_bits_per_s / 8.0
    transfers: dict[str, LocalMemoryGPUTransferOperatingPoint | None] = {}
    resolved_items = []
    for item in architecture_inputs:
        transfer = _resolve_m3d_transfer_operating_point(
            item,
            project_root=root,
            bandwidth_demand_bytes_per_s=bandwidth_demand_bytes_per_s,
            gpu_peak_bandwidth_bytes_per_s=(
                decode_spec.peak_memory_bandwidth_bytes_per_s),
        )
        gpu_input_demand = (
            bandwidth_demand_bytes_per_s
            if transfer is None
            else min(
                transfer.bandwidth_demand_bytes_per_s,
                transfer.memory_capability_bytes_per_s,
            )
        )
        gpu_point = resolve_gpu_decode_power(
            static_power_W=decode_spec.static_power_W,
            e_decode_J_per_bit=decode_spec.e_decode_J_per_bit,
            bandwidth_demand_bytes_per_s=gpu_input_demand,
            peak_bandwidth_bytes_per_s=(
                decode_spec.peak_memory_bandwidth_bytes_per_s),
        )
        if transfer is not None and not math.isclose(
            gpu_point.bandwidth_actual_bytes_per_s,
            transfer.bandwidth_actual_bytes_per_s,
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise RuntimeError("GPU and transfer bandwidth resolution diverged")
        transfers[item.spec.architecture_id] = transfer
        resolved_items.append(resolve_architecture_spec(
            item.spec,
            project_root=root,
            gpu_operating_point=gpu_point,
            transfer_operating_point=transfer,
        ))
    resolved_architectures = tuple(resolved_items)

    capacities = []
    performances = []
    energies = []
    powers = []
    thermals = []
    rows = []
    gpu_energies = []
    for resolved in resolved_architectures:
        reference_system = resolved.system_power
        capacity = evaluate_architecture_capacity_feasibility(
            workload_demand,
            resolved.packing,
            reserved_capacity_bytes=experiment.scenario.reserved_capacity_bytes,
        )
        performance = evaluate_llm_decode_performance(
            workload,
            capacity,
            batch_size=workload_spec.decode.batch_size,
            matched_payload_bandwidth_bits_per_second=(
                matched_bandwidth_bits_per_s),
            effective_compute_flops_per_second=(
                experiment.scenario.effective_compute_flops_per_second),
            bandwidth_status=experiment.scenario.bandwidth_status,
            compute_throughput_status=experiment.scenario.compute_status,
        )
        capacities.append(capacity)
        performances.append(performance)
        for rho in experiment.scenario.rho_values:
            active_transfer = (
                transfers[resolved.spec.architecture_id]
                if performance.bottleneck == "MEMORY"
                else None
            )
            energy = evaluate_architecture_decode_memory_energy(
                workload, capacity, reference_system, rho=rho)
            gpu_energy = evaluate_gpu_decode_energy(
                performance, energy, decode_spec,
                platform.gpu_compute_power,
                transfer_operating_point=active_transfer)
            system = (
                resolve_system_power(
                    resolved.case,
                    project_root=root,
                    geometry=resolved.geometry,
                    gpu_operating_point=gpu_energy.gpu_power_operating_point,
                    transfer_operating_point=active_transfer,
                )
                if gpu_energy.evaluation_status
                == "EVALUATED_ANALYTICAL_GPU_DECODE_ENERGY"
                else reference_system
            )
            power = evaluate_llm_decode_workload_power(
                energy,
                performance,
                system,
                unresolved_logic_background_policy=policies[
                    resolved.spec.architecture_id],
                gpu_decode_energy=gpu_energy,
            )
            mapping = map_workload_power_to_thermal(
                resolved.case, system, power)
            thermal = run_llm_decode_workload_thermal(mapping)
            row = assemble_conditional_llm_decode_e2e_row(
                workload_spec.decode,
                workload,
                capacity,
                performance,
                energy,
                power,
                thermal,
                workload_identifier=workload_spec.workload_id,
                architecture_display_name=resolved.spec.display_name,
            )
            energies.append(energy)
            powers.append(power)
            thermals.append(thermal)
            rows.append(row)
            gpu_energies.append(gpu_energy)

    validated_rows = validate_conditional_llm_decode_e2e_rows(
        rows,
        expected_architecture_ids=architecture_ids,
        expected_rhos=experiment.scenario.rho_values,
    )
    sensitivity_result = None
    sensitivity = experiment.scenario.m3d_parameter_sensitivity
    if sensitivity is not None:
        if sensitivity.architecture_id not in architecture_ids:
            raise ValueError(
                "M3D sensitivity architecture is absent from experiment")
        index = architecture_ids.index(sensitivity.architecture_id)
        resolved = resolved_architectures[index]
        sensitivity_result = run_m3d_parameter_sensitivity(
            case=resolved.case,
            system=resolved.system_power,
            workload=workload,
            capacity=capacities[index],
            performance=performances[index],
            interface_energy_values_pj_per_bit=(
                sensitivity.interface_energy_pj_per_bit),
            logic_background_values_W=sensitivity.logic_background_w,
            thermal_runner=run_llm_decode_workload_thermal,
            gpu_decode_power=platform.gpu_decode_power,
            transfer_operating_point=transfers[sensitivity.architecture_id],
        )

    input_paths = {
        "experiment": path,
        "platform": experiment.platform_config,
        "workload": experiment.workload_config,
        **{
            f"architecture:{spec.architecture_id}": config_path
            for spec, config_path in zip(
                architecture_specs, experiment.architecture_configs)
        },
        **{
            f"canonical_case:{spec.architecture_id}": spec.canonical_case
            for spec in architecture_specs
        },
    }
    git = _git_metadata(root)
    provenance = RunProvenance(
        main_repo_commit=git.get("main_repo_commit") or "UNKNOWN",
        main_repo_branch=git.get("main_repo_branch"),
        main_repo_tracked_clean=_tracked_clean(root),
        dreamram_commit=git.get("dreamram_commit"),
        dreamram_branch=git.get("dreamram_branch"),
        python_version=platform_module.python_version(),
        platform=sys.platform,
        executable=sys.executable,
        experiment_config_path=str(path),
        input_sha256={name: _sha256(item) for name, item in input_paths.items()},
        execution_started_utc=started,
        execution_finished_utc=datetime.now(timezone.utc).isoformat(),
        environment={
            "thermal_backend": thermals[0].thermal_backend,
            "matched_bandwidth_bits_per_second": matched_bandwidth_bits_per_s,
            "matched_bandwidth_capability_bits_per_second": (
                matched_bandwidth_capability_bits_per_s),
            "matched_bandwidth_resolution": (
                "DERIVED_FROM_ORTHOGONAL_SLAB_IO"
                if experiment.scenario.matched_bandwidth_derivation is not None
                else "LITERAL"),
            "bandwidth_capability_status": "NOT_VALIDATED",
            "memory_gpu_transfer_operating_points": {
                architecture_id: (
                    None if point is None else point.model_dump())
                for architecture_id, point in transfers.items()
            },
            "write_energy_model_status": "NOT_VALIDATED",
            "gpu_energy_model_status": (
                "ANALYTICAL_AFFINE_UTILIZATION_MODEL"
                if gpu_energies else "NOT_AVAILABLE"),
            "system_j_token_status": "NOT_AVAILABLE",
            **({
                "gpu_decode_energy_stage_status": (
                    "EVALUATED_ANALYTICAL_GPU_DECODE_ENERGY"),
                "gpu_thermal_power_status": "SHARED_WITH_GPU_DECODE_ENERGY",
            } if gpu_energies else {}),
        },
    )

    if write_bundle:
        write_result_bundle(
            output_dir,
            resolved_config={
                "experiment": experiment,
                "platform": platform,
                "workload": workload_spec,
                "architectures": architecture_specs,
            },
            architecture=[
                _resolved_architecture_payload(item)
                for item in resolved_architectures],
            workload={
                "spec": workload_spec,
                "metrics": workload,
                "demand": workload_demand,
            },
            capacity=capacities,
            performance=performances,
            energy=energies,
            power=powers,
            thermal=thermals,
            provenance=provenance,
            summary={
                "experiment_id": experiment.experiment_id,
                "status": "PASS",
                "rows": validated_rows,
                "m3d_parameter_sensitivity": sensitivity_result,
                "gpu_decode_energy": gpu_energies or None,
            },
        )
    else:
        output_dir = None
    return ExperimentRunResult(
        experiment=experiment,
        rows=validated_rows,
        output_dir=output_dir,
        provenance=provenance,
        m3d_parameter_sensitivity=sensitivity_result,
        gpu_decode_energy=tuple(gpu_energies) if gpu_energies else None,
    )
