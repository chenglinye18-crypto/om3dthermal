"""Nominal power/capacity/density/thermal comparison for canonical cases."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .case_runner import run_steady_pipeline
from .config import (
    PowerSelector,
    PowerSourceConfig,
    SimulationConfig,
    ThermalPowerSourcesConfig,
    compile_user_config,
)
from .power import (
    load_case_config,
    map_system_power_to_thermal,
    resolve_case_geometry,
    resolve_system_power,
)
from .power.config import CanonicalCaseConfig, find_project_root
from .power.geometry import ResolvedGeometry
from .power.system import ResolvedSystemPower


@dataclass(frozen=True)
class ArchitectureMetrics:
    architecture: str
    access_energy_pJ_per_bit: float
    memory_power_W: float
    package_power_W: float
    system_capacity_GiB: float
    memory_plane_density_Mb_mm2: float
    architecture_footprint_density_Gb_mm2: float
    memory_Tmax_degC: float
    gpu_Tmax_degC: float
    package_Tmax_degC: float
    instance_count: int
    capacity_per_instance_GiB: float
    refresh_power_W: float
    memory_plane_area_mm2: float
    architecture_footprint_area_mm2: float
    resolved_package_power_W: float
    mapped_package_power_W: float
    power_closure_absolute_error_W: float
    power_closure_relative_error: float
    ambient_degC: float
    delta_Tmax_K: float
    converged: bool
    iterations: int
    final_relative_residual: float
    cell_count: int
    internal_edge_count: int


def _resolved_capacity(
        case: CanonicalCaseConfig, geometry: ResolvedGeometry,
        system: ResolvedSystemPower) -> dict[str, float | int]:
    result = system.memory_result
    if result is None:
        raise ValueError("comparison requires validated analytical memory power")
    d = result.diagnostics
    if geometry.memory_region == "hbm_dram_die":
        instances = geometry.memory_region_count
        total_bits = int(d["total_stored_bits"])
        bits_per_instance = int(d["bits_per_stack"])
        bits_per_plane = int(d["bits_per_die"])
        plane_area = geometry.configured_x_mm * geometry.configured_y_mm
        layout = case.geometry.layout
        footprint_area = (
            int(layout["visible_group_count"])
            * float(layout["visible_group_footprint_mm"][0])
            * float(layout["visible_group_footprint_mm"][1]))
    elif geometry.memory_region == "orthogonal_memory_slab":
        instances = geometry.memory_region_count
        bits_per_instance = int(d["bits_per_slab"])
        total_bits = int(d["total_stored_bits"])
        bits_per_plane = bits_per_instance
        plane_area = geometry.configured_x_mm * geometry.configured_y_mm
        orth = case.geometry.orthogonal
        assert orth is not None
        footprint_area = orth.cube_length_x_mm * orth.slab_plane_y_mm
    else:
        instances = geometry.memory_region_count
        layers = int(d["memory_layer_count"])
        bits_per_plane = int(d["bits_per_layer"])
        bits_per_instance = bits_per_plane * layers
        total_bits = int(d["total_stored_bits"])
        plane_area = geometry.configured_x_mm * geometry.configured_y_mm
        orth = case.geometry.orthogonal
        assert orth is not None
        footprint_area = orth.cube_length_x_mm * orth.slab_plane_y_mm
    if total_bits != bits_per_instance * instances:
        raise RuntimeError("system capacity does not close over physical instances")
    return {
        "instance_count": instances,
        "bits_per_instance": bits_per_instance,
        "total_bits": total_bits,
        "capacity_per_instance_GiB": bits_per_instance / 8 / 2**30,
        "system_capacity_GiB": total_bits / 8 / 2**30,
        "memory_plane_area_mm2": plane_area,
        "memory_plane_density_Mb_mm2": bits_per_plane / 1e6 / plane_area,
        "architecture_footprint_area_mm2": footprint_area,
        "architecture_footprint_density_Gb_mm2": (
            total_bits / 1e9 / footprint_area),
    }


def _common_compact(case: CanonicalCaseConfig) -> dict[str, Any]:
    thermal = case.thermal
    common = thermal["common_stack"]
    gpu = common["gpu"]
    return {
        "name": case.name,
        "materials": thermal["materials"],
        "geometry": {
            "package": {"size": [f"{thermal['package_size_mm'][0]} mm",
                                     f"{thermal['package_size_mm'][1]} mm"]},
            "gpu": {"size": [f"{thermal['gpu_footprint_mm'][0]} mm",
                                 f"{thermal['gpu_footprint_mm'][1]} mm"]},
        },
        "stacks": {
            "foundation": {"layers": [["Laminate", f"{common['foundation_um']} um"]]},
            "gpu": {"layers": [
                ["Cu_Pillar_Bump", f"{gpu['cu_pillar_um']} um"],
                ["BSPDN", f"{gpu['bspdn_um']} um"],
                ["FEOL", f"{gpu['feol_um']} um"],
                ["BEOL_MXY", f"{gpu['beol_um']} um"],
            ]},
            "top": {"layers": [
                ["TIM", f"{common['tim_um']} um"],
                ["Lid", f"{common['lid_um']} um"],
            ]},
        },
        "mesh": {
            "dx": f"{thermal['mesh']['dx_mm']} mm",
            "dy": f"{thermal['mesh']['dy_mm']} mm",
            "dz_max": f"{thermal['mesh']['dz_max_um']} um",
        },
        "boundary": {
            "ambient": f"{thermal['boundary']['ambient_degC']} degC",
            "lid_top_htc": f"{thermal['boundary']['lid_top_htc_W_m2K']} W/m^2/K",
            "laminate_bottom_htc": (
                f"{thermal['boundary']['laminate_bottom_htc_W_m2K']} W/m^2/K"),
        },
        "power": {"model": "uniform", "gpu": "1 W"},
        "solver": {"method": "pcg", "rtol": thermal["solver"]["rtol"]},
        "metadata": {"case_id": case.name, "solver": {"backend": "cpu"}},
    }


def compile_case_thermal(
        case: CanonicalCaseConfig, system: ResolvedSystemPower,
        ) -> SimulationConfig:
    """Compile thermal geometry from the same canonical case object."""
    raw = _common_compact(case)
    if case.geometry.type == "dreamram_hbm":
        layout = case.geometry.layout
        stack = case.thermal["stack"]
        repeated = stack["repeated_dram"]
        top = stack["top_dram"]
        dram_die_count = int(layout["dram_dies_per_stack"])
        repeated_dram_count = dram_die_count - 1
        group_names = ["hbm_left", "hbm_right"]
        centers = layout["group_centers_mm"]
        raw["geometry"].update({
            "memory_zone": {"size": ["30 mm", "22 mm"]},
            "thermal_silicon": {"size": ["8 mm", "22 mm"]},
            "hbm": {
                "size": [f"{layout['visible_group_footprint_mm'][0]} mm",
                         f"{layout['visible_group_footprint_mm'][1]} mm"],
                "dram_size": [f"{case.geometry.memory_region.width_mm} mm",
                              f"{case.geometry.memory_region.height_mm} mm"],
                "centers": {name: [f"{center[0]} mm", f"{center[1]} mm"]
                            for name, center in zip(group_names, centers)},
            },
        })
        raw["stacks"].update({
            "hbm": {
                "base": {"layers": [["GPU_HBM_uBump",
                                      f"{stack['gpu_hbm_ubump_um']} um"]]},
                "dram": {"repeat": repeated_dram_count, "layers": [
                    ["Hybrid_Bonding", f"{repeated['hybrid_bonding_um']} um"],
                    ["DRAM_BEOL", f"{repeated['beol_um']} um"],
                    ["Silicon", f"{repeated['si_um']} um"],
                ]},
                "top": {"layers": [
                    ["Hybrid_Bonding", f"{top['hybrid_bonding_um']} um"],
                    ["DRAM_BEOL", f"{top['beol_um']} um"],
                    ["Silicon", f"{top['si_um']} um"],
                ]},
            },
        })
        hbm_height = (
            stack["gpu_hbm_ubump_um"]
            + repeated_dram_count * sum(repeated[k] for k in (
                "hybrid_bonding_um", "beol_um", "si_um"))
            + sum(top[k] for k in (
                "hybrid_bonding_um", "beol_um", "si_um")))
        raw["stacks"]["thermal_silicon"] = {"layers": [
            ["Oxide", "1 um"],
            ["Thermal_Silicon", f"{hbm_height - 1.0} um"],
        ]}
    else:
        orth = case.geometry.orthogonal
        assert orth is not None
        raw["orthogonal_hbm"] = {
            "cube_size": [f"{orth.slab_plane_y_mm} mm",
                          f"{orth.cube_length_x_mm} mm",
                          f"{orth.slab_height_z_mm} mm"],
            "background_material": "Mold",
            "adhesive": {"material": "Adhesive", "thickness": "3 um"},
            "memory_die": {
                "count": orth.slab_count,
                "width": f"{orth.slab_plane_y_mm} mm",
                "height": f"{orth.slab_height_z_mm} mm",
                # Required only by the compact legacy compiler; analytical
                # sources replace this generated placeholder before meshing.
                "power_per_die": "0 W",
            },
        }
        if case.geometry.type == "orthogonal_si":
            s = case.geometry.orthogonal_si_stack
            assert s is not None
            raw["orthogonal_hbm"]["memory_die"]["stack"] = [
                {"MOSAIC_Si": f"{s.si_substrate_um} um"},
                {"MOSAIC_BEOL": f"{s.beol_um} um"},
                {"MOSAIC_DAA": f"{s.daa_um} um"},
            ]
        else:
            s = case.geometry.m3d_stack
            assert s is not None
            raw["orthogonal_hbm"]["memory_die"]["stack"] = [
                {"material": "M3D_Si", "thickness": f"{s.si_substrate_um} um",
                 "role": "si_substrate", "name": "si_substrate"},
                {"material": "M3D_FEOL", "thickness": f"{s.feol_um} um",
                 "role": "feol", "name": "feol"},
                {"material": "M3D_Bitcell",
                 "thickness": (
                     f"{s.bitcell_layers * s.bitcell_layer_pitch_nm * 1e-3} um"),
                 "role": "m3d_bitcell_stack", "name": "m3d_bitcell_stack"},
                {"material": "M3D_BEOL",
                 "thickness": f"{s.beol_interconnect_um} um",
                 "role": "beol_interconnect", "name": "beol_interconnect"},
                {"material": "M3D_DAA", "thickness": f"{s.daa_um} um",
                 "role": "daa", "name": "daa"},
            ]

    compiled = SimulationConfig.model_validate(compile_user_config(raw))
    mapping = map_system_power_to_thermal(case, system)
    if mapping.unresolved:
        raise ValueError("unresolved memory power cannot enter thermal solve")
    sources: list[PowerSourceConfig] = []
    for target in mapping.sources:
        if target.name == "gpu":
            selector = PowerSelector(component="gpu", material="FEOL")
        elif case.geometry.type == "dreamram_hbm":
            # Mapping contains one source per visible group. Select each group
            # explicitly so merged 2x1 layout semantics remain visible.
            index = int(target.name.rsplit("_", 1)[1])
            group = ("hbm_left", "hbm_right")[index]
            selector = PowerSelector(
                component=f"memory_column:{group}", material="DRAM_BEOL")
        elif case.geometry.type == "orthogonal_si":
            selector = PowerSelector(material="MOSAIC_BEOL")
        elif target.target_region == "M3D_BEOL_INTERCONNECT":
            selector = PowerSelector(tags={"role": "beol_interconnect"})
        else:
            selector = PowerSelector(tags={"role": "m3d_bitcell_stack"})
        sources.append(PowerSourceConfig(
            name=target.name,
            total_power=target.power_W,
            selector=selector,
            metadata={"mapping_provenance": target.mapping_provenance},
        ))
    return compiled.model_copy(update={
        "thermal_power_sources": ThermalPowerSourcesConfig(sources=sources)})


def _temperature_maxima(pipeline) -> tuple[float, float, float]:
    temperatures_C = pipeline.result.temperature_K - 273.15
    gpu = np.array([
        cell.component == "gpu" for cell in pipeline.cells], dtype=bool)
    memory = np.array([
        (str(cell.component).startswith("memory_column:")
         or str(cell.component).startswith("orthogonal_hbm:"))
        for cell in pipeline.cells], dtype=bool)
    if not np.any(gpu) or not np.any(memory):
        raise RuntimeError("GPU or memory thermal region is absent")
    return (float(np.max(temperatures_C[memory])),
            float(np.max(temperatures_C[gpu])),
            float(np.max(temperatures_C)))


def run_architecture_comparison(
        case_paths: list[Path], output_dir: Path) -> list[ArchitectureMetrics]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[ArchitectureMetrics] = []
    for path in case_paths:
        case = load_case_config(path)
        geometry = resolve_case_geometry(case)
        root = find_project_root(path)
        system = resolve_system_power(case, project_root=root, geometry=geometry)
        capacity = _resolved_capacity(case, geometry, system)
        mapping = map_system_power_to_thermal(case, system)
        assert system.resolved_total_memory_power_W is not None
        resolved_package = system.gpu_power_W + system.resolved_total_memory_power_W
        error = abs(mapping.total_mapped_power_W - resolved_package)
        relative = error / resolved_package
        if error > 1e-10 or mapping.unresolved:
            raise RuntimeError(
                f"thermal power closure failed for {case.name}: {error} W")
        thermal_config = compile_case_thermal(case, system)
        pipeline = run_steady_pipeline(
            thermal_config, method="pcg",
            rtol=float(case.thermal["solver"]["rtol"]),
            max_iterations=10_000, initial_temperature_K=293.15)
        mapped_actual = float(np.sum(pipeline.power.power_W))
        if abs(mapped_actual - resolved_package) > 1e-9:
            raise RuntimeError("cell-level mapped thermal power does not close")
        memory_t, gpu_t, package_t = _temperature_maxima(pipeline)
        ambient = float(case.thermal["boundary"]["ambient_degC"])
        result = pipeline.result
        row = ArchitectureMetrics(
            architecture=case.name,
            access_energy_pJ_per_bit=system.memory_access_energy_pJ_per_bit,
            memory_power_W=system.resolved_total_memory_power_W,
            package_power_W=resolved_package,
            system_capacity_GiB=float(capacity["system_capacity_GiB"]),
            memory_plane_density_Mb_mm2=float(
                capacity["memory_plane_density_Mb_mm2"]),
            architecture_footprint_density_Gb_mm2=float(
                capacity["architecture_footprint_density_Gb_mm2"]),
            memory_Tmax_degC=memory_t, gpu_Tmax_degC=gpu_t,
            package_Tmax_degC=package_t,
            instance_count=int(capacity["instance_count"]),
            capacity_per_instance_GiB=float(
                capacity["capacity_per_instance_GiB"]),
            refresh_power_W=system.refresh_power_W,
            memory_plane_area_mm2=float(capacity["memory_plane_area_mm2"]),
            architecture_footprint_area_mm2=float(
                capacity["architecture_footprint_area_mm2"]),
            resolved_package_power_W=resolved_package,
            mapped_package_power_W=mapped_actual,
            power_closure_absolute_error_W=abs(mapped_actual-resolved_package),
            power_closure_relative_error=(
                abs(mapped_actual-resolved_package)/resolved_package),
            ambient_degC=ambient, delta_Tmax_K=package_t-ambient,
            converged=result.converged, iterations=result.iterations,
            final_relative_residual=result.final_relative_residual,
            cell_count=pipeline.cell_count,
            internal_edge_count=pipeline.internal_edge_count,
        )
        rows.append(row)
        run_dir = output_dir / case.name
        run_dir.mkdir(exist_ok=True)
        run_summary = asdict(row)
        run_summary["thermal_power_by_source_W"] = dict(
            pipeline.power.power_by_source)
        run_summary["thermal_memory_target_regions"] = {
            source.name: source.target_region
            for source in mapping.sources if source.name != "gpu"
        }
        (run_dir / "thermal_summary.json").write_text(
            json.dumps(run_summary, indent=2), encoding="utf-8")

    write_comparison_summary(rows, output_dir)
    return rows


def write_comparison_summary(
        rows: list[ArchitectureMetrics], output_dir: Path) -> None:
    """Write the compact comparison tables without thermal field arrays."""
    if not rows:
        raise ValueError("comparison summary requires at least one row")
    data = [asdict(row) for row in rows]
    (output_dir / "summary.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
