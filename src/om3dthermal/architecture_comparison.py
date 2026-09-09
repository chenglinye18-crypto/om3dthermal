"""Nominal power/capacity/density/thermal comparison for canonical cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any


from .config import (
    PowerSelector,
    PowerSourceConfig,
    SimulationConfig,
    ThermalPowerSourcesConfig,
    compile_user_config,
)
from .power import (
    calculate_memory_power,
    map_system_power_to_thermal,
    resolve_case_geometry,
    resolve_effective_bandwidth,
)
from .power.config import (
    CanonicalCaseConfig,
)
from .power.system import ResolvedSystemPower
from .platform import (
    GPUBandwidthServiceOperatingPoint,
    GPUDecodePowerOperatingPoint,
    LocalMemoryGPUTransferOperatingPoint,
    load_platform_spec_file,
    resolve_gpu_decode_power,
    resolve_gpu_bandwidth_service,
    resolve_local_memory_gpu_transfer,
)


def _resolve_case_power_operating_points(
    case: CanonicalCaseConfig,
    project_root: Path,
) -> tuple[
    GPUDecodePowerOperatingPoint,
    LocalMemoryGPUTransferOperatingPoint | None,
    GPUBandwidthServiceOperatingPoint,
]:
    """Resolve standalone case demand through canonical memory/GPU limits."""
    platform = load_platform_spec_file(
        project_root / "configs/platform/gpu_package_h200_reference.yaml")
    if platform.gpu_decode_power is None:
        raise ValueError("canonical platform is missing gpu_decode_power")
    spec = platform.gpu_decode_power
    demand = (
        (case.workload.read_bandwidth_gbps
         + case.workload.write_bandwidth_gbps) * 1e9 / 8.0)
    transfer = None
    gpu_input_demand = demand
    if (case.geometry.type == "orthogonal_m3d"
            and case.power.memory.model == "analytical"):
        geometry = resolve_case_geometry(case)
        intrinsic = calculate_memory_power(
            case,
            project_root=project_root,
            geometry=geometry,
            read_bandwidth_gbps=0.0,
        )
        closure = intrinsic.architecture_bandwidth_closure
        if closure is None:
            raise ValueError("M3D case is missing raw bandwidth closure")
        raw = resolve_effective_bandwidth(
            closure,
            closure.average_service_cycle_ns / closure.service_cycle_scale,
        )
        transfer = resolve_local_memory_gpu_transfer(
            bandwidth_demand_bytes_per_s=demand,
            memory_capability_bytes_per_s=raw.effective_bandwidth_bytes_per_s,
            gpu_peak_bandwidth_bytes_per_s=(
                spec.peak_memory_bandwidth_bytes_per_s),
        )
        gpu_input_demand = min(
            transfer.bandwidth_demand_bytes_per_s,
            transfer.memory_capability_bytes_per_s,
        )
    ceiling = resolve_gpu_decode_power(
        static_power_W=spec.static_power_W,
        e_decode_J_per_bit=spec.e_decode_J_per_bit,
        bandwidth_demand_bytes_per_s=gpu_input_demand,
        peak_bandwidth_bytes_per_s=(
            spec.peak_memory_bandwidth_bytes_per_s),
    )
    service_spec = platform.gpu_bandwidth_service
    service = resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=ceiling.bandwidth_actual_bytes_per_s,
        service_status=service_spec.service_status,
        provenance=service_spec.provenance,
    )
    gpu = resolve_gpu_decode_power(
        static_power_W=spec.static_power_W,
        e_decode_J_per_bit=spec.e_decode_J_per_bit,
        bandwidth_demand_bytes_per_s=service.sustained_bandwidth_bytes_per_s,
        peak_bandwidth_bytes_per_s=spec.peak_memory_bandwidth_bytes_per_s,
    )
    return gpu, transfer, service


def _resolve_case_gpu_operating_point(
    case: CanonicalCaseConfig,
    project_root: Path,
) -> GPUDecodePowerOperatingPoint:
    """Return only the GPU member for GPU-only consumers."""
    return _resolve_case_power_operating_points(case, project_root)[0]


def _resolve_case_power_operating_point_kwargs(
    case: CanonicalCaseConfig,
    project_root: Path,
) -> dict[str, object]:
    """Explicit keyword bundle for system/architecture resolver calls."""
    gpu, transfer, service = _resolve_case_power_operating_points(
        case, project_root)
    return {
        "gpu_operating_point": gpu,
        "transfer_operating_point": transfer,
        "bandwidth_service_operating_point": service,
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
        "solver": {"rtol": thermal["solver"]["rtol"]},
        "metadata": {"case_id": case.name, "solver": {"backend": "gpu_pcg"}},
    }


def compile_canonical_thermal_case(
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
        # HBM-on-GPU: the memory zone spans the whole GPU die; the thermal
        # silicon bar fills the inter-group gap.  Both are derived from the
        # case geometry so footprint revisions stay single-sourced.
        gpu_fp = [float(v) for v in case.thermal["gpu_footprint_mm"]]
        group_x = float(layout["visible_group_footprint_mm"][0])
        si_width = 2.0 * abs(float(centers[0][0])) - group_x
        if si_width <= 0.0:
            raise ValueError(
                "thermal silicon gap does not close: group centers "
                f"{centers} vs group width {group_x} mm")
        raw["geometry"].update({
            "memory_zone": {"size": [f"{gpu_fp[0]} mm", f"{gpu_fp[1]} mm"]},
            "thermal_silicon": {"size": [f"{si_width} mm", f"{gpu_fp[1]} mm"]},
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
                "base": {"layers": [
                    ["GPU_HBM_uBump", f"{stack['gpu_hbm_ubump_um']} um"],
                    *([] if stack["base_logic_die"] == "removed" else [
                        ["HBM_Base_BEOL", f"{stack['base_beol_um']} um"],
                        ["Silicon", f"{stack['base_si_um']} um"],
                    ]),
                ]},
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
            + (0.0 if stack["base_logic_die"] == "removed" else
               stack["base_beol_um"] + stack["base_si_um"])
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
            "adhesive": {
                "material": "Adhesive",
                "thickness": (
                    f"{case.thermal['adhesive']['thickness_um']} um"),
            },
            "memory_die": {
                "count": orth.slab_count,
                "width": f"{orth.slab_plane_y_mm} mm",
                "height": f"{orth.slab_height_z_mm} mm",
                # Required only by the compact legacy compiler; analytical
                # sources replace this generated placeholder before meshing.
                "power_per_die": "0 W",
            },
        }
        # Optional sidebars occupy the exposed y edges beside the memory cube.
        edge_strip_material = case.thermal.get("edge_strip_material")
        if edge_strip_material is not None:
            raw["orthogonal_hbm"]["edge_strip_material"] = str(
                edge_strip_material)
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
                {"material": "M3D_Bitcell_BEOL",
                 "thickness": (
                     f"{s.bitcell_layers * s.bitcell_layer_pitch_nm * 1e-3 + s.beol_interconnect_um} um"),
                 "role": "m3d_bitcell_beol_stack",
                 "name": "m3d_bitcell_beol_stack"},
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
            material = (
                "HBM_Base_BEOL" if target.name.startswith("base_route_")
                else "DRAM_BEOL")
            selector = PowerSelector(
                component=f"memory_column:{group}", material=material)
        elif case.geometry.type == "orthogonal_si":
            selector = PowerSelector(material="MOSAIC_BEOL")
        elif case.geometry.type == "orthogonal_m3d":
            # Bitcell and BEOL have the same thermal conductivity and their
            # powers are already split in proportion to thickness.  Mapping
            # both sources onto the combined region therefore preserves the
            # original uniform volumetric heat density exactly.
            selector = PowerSelector(
                tags={"role": "m3d_bitcell_beol_stack"})
        else:
            raise AssertionError(
                f"unhandled thermal target {target.target_region!r}")
        sources.append(PowerSourceConfig(
            name=target.name,
            total_power=target.power_W,
            selector=selector,
            metadata={"mapping_provenance": target.mapping_provenance},
        ))
    return compiled.model_copy(update={
        "thermal_power_sources": ThermalPowerSourcesConfig(sources=sources)})
