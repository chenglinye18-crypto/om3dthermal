"""Frozen No-NMP steady-state bandwidth sweep for HBM and M3D."""

from __future__ import annotations

import csv
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .architecture_comparison import (
    _resolve_case_power_operating_point_kwargs,
    compile_canonical_thermal_case,
)
from .case_runner import run_steady_pipeline
from .config import CellSizeConfig, ThermalPowerSourcesConfig
from .power import load_case_config, resolve_case_geometry, resolve_system_power
from .thermal.setup_cache import load_setup_cache


GPU_STATIC_W = 74.0
GPU_DYNAMIC_PJ_PER_BIT = 11.68
M3D_MESH_M = (0.5e-3, 1.0e-3, 0.25e-3)


@dataclass(frozen=True)
class SweepArchitecture:
    architecture: str
    case_file: str
    bandwidths_TBps: tuple[float, ...]
    cache_file: str


ARCHITECTURES = (
    SweepArchitecture(
        architecture="conventional_hbm_2x1",
        case_file="conventional_hbm_2x1.yaml",
        bandwidths_TBps=tuple(round(0.2 * index, 10) for index in range(25)),
        cache_file="conventional_hbm_2x1.pkl",
    ),
    SweepArchitecture(
        architecture="orthogonal_m3d_igzo",
        case_file="orthogonal_m3d_igzo.yaml",
        bandwidths_TBps=(
            *tuple(round(0.2 * index, 10) for index in range(34)), 6.7),
        cache_file="orthogonal_m3d_igzo_cu.pkl",
    ),
)


def _base_case(project_root: Path, spec: SweepArchitecture):
    case = load_case_config(project_root / "configs" / "cases" / spec.case_file)
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(
        case,
        project_root=project_root,
        geometry=geometry,
        **_resolve_case_power_operating_point_kwargs(case, project_root),
    )
    simulation = compile_canonical_thermal_case(case, system)
    if spec.architecture == "orthogonal_m3d_igzo":
        discretization = simulation.discretization
        assert discretization is not None
        simulation = simulation.model_copy(update={
            "discretization": discretization.model_copy(update={
                "max_cell_size": CellSizeConfig(
                    x=M3D_MESH_M[0], y=M3D_MESH_M[1], z=M3D_MESH_M[2])})})
    memory = system.memory_result
    if memory is None:
        raise ValueError(f"{spec.architecture} has no resolved memory energy")
    return case, simulation, memory


def _point_simulation(simulation, architecture: str, memory, bandwidth_TBps):
    bandwidth_bytes_per_s = bandwidth_TBps * 1e12
    gpu_dynamic_W = (
        bandwidth_bytes_per_s * 8.0 * GPU_DYNAMIC_PJ_PER_BIT * 1e-12)
    gpu_power_W = GPU_STATIC_W + gpu_dynamic_W
    memory_power_W = (
        bandwidth_bytes_per_s * 8.0
        * memory.E_access_total_pj_bit * 1e-12)
    if architecture == "conventional_hbm_2x1":
        base_fraction = (
            memory.E_base_route_pj_bit / memory.E_access_total_pj_bit)
        dram_fraction = 1.0 - base_fraction
        powers = {
            "gpu": gpu_power_W,
            "dram_group_0": memory_power_W * dram_fraction / 2.0,
            "base_route_group_0": memory_power_W * base_fraction / 2.0,
            "dram_group_1": memory_power_W * dram_fraction / 2.0,
            "base_route_group_1": memory_power_W * base_fraction / 2.0,
        }
    else:
        powers = {
            "gpu": gpu_power_W,
            "m3d_memory_bitcell_beol": memory_power_W,
        }
    sources = simulation.thermal_power_sources
    assert sources is not None
    updated = [
        source.model_copy(update={"total_power": powers[source.name]})
        for source in sources.sources
    ]
    point = simulation.model_copy(update={
        "thermal_power_sources": ThermalPowerSourcesConfig(sources=updated)})
    return point, gpu_power_W, memory_power_W


def _row(spec, bandwidth, gpu_power, memory_power, pipeline):
    if not pipeline.result.converged:
        raise RuntimeError(
            f"{spec.architecture} at {bandwidth} TB/s did not converge")
    x, y, z = (value * 1e3 for value in pipeline.hottest_cell_xyz_m)
    hotspot = (
        f"{pipeline.hottest_cell_component}/{pipeline.hottest_cell_material}"
        f"@({x:.6f},{y:.6f},{z:.6f})mm")
    return {
        "architecture": spec.architecture,
        "bandwidth_TBps": bandwidth,
        "GPU_power_W": gpu_power,
        "memory_power_W": memory_power,
        "total_power_W": gpu_power + memory_power,
        "Tmax_C": pipeline.result.max_temperature_K - 273.15,
        "hotspot": hotspot,
        "iterations": pipeline.result.iterations,
        "residual": pipeline.result.final_relative_residual,
        "solve_time_s": pipeline.solve_seconds,
    }


def run_no_nmp_bandwidth_thermal_sweep(
    output_dir: str | Path,
    *,
    project_root: str | Path,
) -> dict:
    """Run the frozen two-architecture sweep and write CSV/PNG/timings."""
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output / "bandwidth_thermal_sweep.csv"
    fields = [
        "architecture", "bandwidth_TBps", "GPU_power_W", "memory_power_W",
        "total_power_W", "Tmax_C", "hotspot", "iterations", "residual",
        "solve_time_s",
    ]
    rows = []
    timings = {}
    sweep_started = time.perf_counter()
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for spec in ARCHITECTURES:
            architecture_started = time.perf_counter()
            _case, simulation, memory = _base_case(root, spec)
            cache_path = cache_dir / spec.cache_file
            first_bandwidth = spec.bandwidths_TBps[0]
            point, gpu_power, memory_power = _point_simulation(
                simulation, spec.architecture, memory, first_bandwidth)
            first = run_steady_pipeline(
                point, backend="gpu_pcg", setup_cache_path=cache_path,
                rtol=1e-3, max_delta_t_K=1e-2, max_iterations=100_000,
                check_interval=10, initial_temperature_K=293.15)
            row = _row(spec, first_bandwidth, gpu_power, memory_power, first)
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            signature = first.cache_physical_signature
            if signature is None:
                raise RuntimeError("sweep cache did not produce a signature")
            architecture_timing = {
                "setup_build_time_s": first.setup_build_seconds,
                "cache_save_time_s": first.cache_serialization_seconds,
                "initial_point_solve_time_s": first.solve_seconds,
                "cache_file_size_bytes": first.cache_size_bytes,
                "cache_status": first.cache_status,
                "physical_signature": signature,
            }
            del first
            gc.collect()
            reusable, load_seconds, cache_status = load_setup_cache(
                cache_path, signature)
            if reusable is None or cache_status != "HIT":
                raise RuntimeError(f"failed to reload {spec.architecture} cache")
            architecture_timing["cache_load_time_s"] = load_seconds
            for bandwidth in spec.bandwidths_TBps[1:]:
                point, gpu_power, memory_power = _point_simulation(
                    simulation, spec.architecture, memory, bandwidth)
                pipeline = run_steady_pipeline(
                    point, backend="gpu_pcg", reusable_setup=reusable,
                    rtol=1e-3, max_delta_t_K=1e-2,
                    max_iterations=100_000, check_interval=10,
                    initial_temperature_K=293.15)
                row = _row(
                    spec, bandwidth, gpu_power, memory_power, pipeline)
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                del pipeline
            architecture_timing["architecture_wall_clock_time_s"] = (
                time.perf_counter() - architecture_started)
            timings[spec.architecture] = architecture_timing
            del reusable
            gc.collect()

    timings["sweep_wall_clock_time_s"] = time.perf_counter() - sweep_started
    timing_path = output / "timings.json"
    timing_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for spec in ARCHITECTURES:
        selected = [row for row in rows
                    if row["architecture"] == spec.architecture]
        axis.plot(
            [row["bandwidth_TBps"] for row in selected],
            [row["Tmax_C"] for row in selected], marker="o", markersize=3,
            label=spec.architecture)
    axis.set_xlabel("Bandwidth (TB/s)")
    axis.set_ylabel("Tmax (°C)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plot_path = output / "tmax_vs_bandwidth.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return {
        "csv": str(csv_path),
        "plot": str(plot_path),
        "timings": str(timing_path),
        "row_count": len(rows),
    }
