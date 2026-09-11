"""Shared M3D capacity/backend resolution (obsolete Decode API removed)."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from om3dthermal.power import calculate_memory_power, calculate_physical_access_latency, load_case_config, resolve_case_geometry
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.resident_pages import ResidentDataObject

@dataclass(frozen=True)
class M3DArchitectureBackend:
    case: object
    geometry: object
    memory: object
    topology: object
    feol: object
    physical_latency: object
    layout: object
    bandwidth: object
    gpu_compute_flops_per_s: float


@lru_cache(maxsize=2)
def resolve_m3d_architecture_backend(project_root: str | Path) -> M3DArchitectureBackend:
    project_root = Path(project_root).resolve()
    from om3dthermal.experiment import load_experiment_spec

    case = load_case_config(project_root / "configs/cases/orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    memory = calculate_memory_power(
        case, read_bandwidth_gbps=case.workload.read_bandwidth_gbps,
        project_root=project_root, geometry=geometry)
    topology = calculate_m3d_subarray(case.architecture.m3d_subarray, geometry.m3d)
    feol = calculate_feol_route(case.architecture, topology)
    physical_latency = calculate_physical_access_latency(
        case.architecture.physical_access_latency,
        feol_route=feol,
        miv_length_per_layer_um=memory.diagnostics["miv_length_per_layer_um"],
        miv_delay_per_layer_ns=memory.diagnostics["miv_delay_per_layer_ns"],
        miv_status=memory.diagnostics["miv_latency_status"],
        miv_parameter_status=memory.diagnostics["miv_resistance_parameter_status"],
        miv_provenance=memory.diagnostics["miv_resistance_provenance"],
    )
    layout = memory.physical_capacity_layout
    bandwidth = memory.architecture_bandwidth_closure
    if layout is None or bandwidth is None:
        raise ValueError("canonical M3D capacity/bandwidth did not resolve")
    experiment = load_experiment_spec(
        project_root / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml",
        project_root=project_root)
    return M3DArchitectureBackend(
        case, geometry, memory, topology, feol, physical_latency, layout,
        bandwidth, experiment.scenario.effective_compute_flops_per_second)


def rounded_capacity_bytes(objects: tuple[ResidentDataObject, ...], page_bytes: int) -> int:
    return sum(
        ((item.size_bytes + page_bytes - 1) // page_bytes) * page_bytes
        for item in objects)
