"""Read-only adapter for the pinned DreamRAM DATE2026 analytical model."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, replace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Iterator

from ..cell_model import ONE_T_ONE_C_SPECIFIC, REUSABLE_STRUCTURE
from ..config import MemoryPowerConfig, resolve_project_path
from ..geometry import (
    ResolvedGeometry,
    evaluate_geometry_fit,
    resolve_legacy_geometry,
)
from ..miv import build_miv_topology, calculate_length_scaled_miv_energy
from ..m3d_subarray import M3DSubarrayResult
from ..result import BackendEnergyResult, EnergyDecomposition
from ..si_packing import pack_si_primitive


DREAMRAM_BRANCH = "DATE2026"
DREAMRAM_COMMIT = "c069ce14dfa85ce1983f3a1274a265d1e7b5494a"

_GROUP_COMPONENTS = {
    "memory_internal": ONE_T_ONE_C_SPECIFIC | REUSABLE_STRUCTURE,
    "vertical": {"row-tsv", "col-tsv", "tsv"},
    "base_route": {"row-base", "col-base", "base"},
    "interface": {"row-dq", "col-dq", "dq"},
}
_COMMAND_TERMS = {
    "pre": {
        "row-dq": 0.5, "row-base": 0.5, "row-tsv": 0.5,
        "row": 0.5, "mwl": 1.0, "bl-pre": 1.0,
    },
    "act": {
        "row-dq": 1.0, "row-base": 1.0, "row-tsv": 1.0,
        "row": 1.0, "lwl": 1.0, "bl-act": 1.0,
    },
    "rd": {
        "col-dq": 1.0, "col-base": 1.0, "col-tsv": 1.0,
        "col": 1.0, "csl": 1.0, "ldl": 1.0, "mdl": 1.0,
        "bgbus+gbus": 1.0, "tsv": 1.0, "base": 1.0, "dq": 1.0,
    },
}


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _verify_pin(repo: Path) -> None:
    branch = _git(repo, "branch", "--show-current")
    commit = _git(repo, "rev-parse", "HEAD")
    changed = _git(repo, "status", "--short", "--untracked-files=no")
    if branch != DREAMRAM_BRANCH or commit != DREAMRAM_COMMIT:
        raise RuntimeError(
            "DreamRAM pin mismatch: expected "
            f"{DREAMRAM_BRANCH}@{DREAMRAM_COMMIT}, got {branch}@{commit}")
    if changed:
        raise RuntimeError("DreamRAM tracked source is modified; refusing to run")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DreamRAM module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _loaded_dreamram(repo: Path) -> Iterator[tuple[ModuleType, ModuleType, ModuleType]]:
    names = ("tech", "hbm", "parse")
    previous = {name: sys.modules.get(name) for name in names}
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        tech_module = _load_module("tech", repo / "tech.py")
        hbm_module = _load_module("hbm", repo / "hbm.py")
        parse_module = _load_module("parse", repo / "parse.py")
        yield tech_module, hbm_module, parse_module
    finally:
        sys.dont_write_bytecode = old_bytecode
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _group_for(component: str) -> str:
    matches = [group for group, names in _GROUP_COMPONENTS.items()
               if component in names]
    if len(matches) != 1:
        raise RuntimeError(f"unclassified DreamRAM component {component!r}")
    return matches[0]


class DreamRAMBackend:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def calculate(
            self, config: MemoryPowerConfig, *,
            m3d_subarray: M3DSubarrayResult | None = None,
            geometry: ResolvedGeometry | None = None,
            ) -> BackendEnergyResult:
        if geometry is None:
            source = config.architecture.geometry_source
            if source is None:
                raise ValueError("DreamRAM backend requires resolved geometry")
            geometry = resolve_legacy_geometry(self.project_root, source)
        dreamram_input = config.memory.dreamram
        if dreamram_input is None:
            raise ValueError("DreamRAM backend requires memory.dreamram")
        repo = self.project_root / "third_party" / "DreamRAM"
        _verify_pin(repo)
        memory_path = resolve_project_path(
            self.project_root, dreamram_input.memory_config).resolve()
        technology_path = resolve_project_path(
            self.project_root, dreamram_input.technology_config).resolve()

        with _loaded_dreamram(repo) as (tech_module, hbm_module, parse_module):
            previous_cwd = Path.cwd()
            try:
                # DreamRAM technology configs reference their baseline relative
                # to the upstream repository, matching its official CLI.
                os.chdir(repo)
                memory = parse_module.mem_baseline(str(memory_path))
                technology = parse_module.tech(str(technology_path))
            finally:
                os.chdir(previous_cwd)
            if memory is None or technology is None:
                raise ValueError("DreamRAM configuration could not be resolved")
            native_tech = tech_module.Tech(**technology)
            cell_geometry = config.memory.cell_model.geometry
            if cell_geometry is None:
                tech = native_tech
                geometry_mapping = "dreamram_native"
            else:
                # Build a per-invocation Tech. Dataclass defaults for pitch_ldl
                # and pitch_mdl were evaluated using the native pitches, so
                # both dependent pitches must be updated explicitly.
                pitch_bl = cell_geometry.pitch_x_um
                pitch_wl = cell_geometry.pitch_y_um
                tech = replace(
                    native_tech,
                    pitch_bl=pitch_bl,
                    pitch_wl=pitch_wl,
                    pitch_ldl=pitch_wl * native_tech.pitch_ldl_to_wl_min,
                    pitch_mdl=pitch_bl * native_tech.pitch_mdl_to_bl_min,
                )
                geometry_mapping = "pitch_x_to_bl__pitch_y_to_wl"
            hbm_fields = {item.name for item in fields(hbm_module.Hbm)}
            hbm_kwargs = {key: value for key, value in memory.items()
                          if key in hbm_fields}
            hbm_kwargs["brv_sa"] = memory["brvsa"]
            dram = hbm_module.Hbm(**hbm_kwargs)
            bank_x_um, bank_y_um, _ = dram.bank_dims(tech)
            row_decoder_width_um, col_decoder_height_um = (
                dram.decoder_dims(tech))
            swd_width_um = dram.swd_width(tech)
            blsa_height_um = dram.blsa_height(tech)
            bank_select_decoder_width_um = 2 * row_decoder_width_um
            bank_tile_width_um = bank_x_um + bank_select_decoder_width_um
            mat_x_um = (
                dram.mat_cols * dram.isolation_cols_overhead * tech.pitch_bl)
            mat_y_um = (
                dram.mat_rows * dram.isolation_rows_overhead * tech.pitch_wl)
            stack_dims = dram.calc_stack_dims(tech)
            dies_stacked = int(stack_dims[0])
            if m3d_subarray is not None:
                required_x_mm = m3d_subarray.placed_width_um * 1e-3
                required_y_mm = m3d_subarray.placed_height_um * 1e-3
            else:
                required_x_mm = float(stack_dims[1]) * 1e-3
                required_y_mm = float(sum(stack_dims[2:])) * 1e-3
            wire_lengths = dram.wire_lengths(tech)
            wire_counts = dram.wire_counts()
            row_bits, col_bits = dram.ch_cmd_bits()
            length_scaled_reference = (
                config.architecture.vertical.electrical_model
                == "dreamram_length_scaled_reference")
            if length_scaled_reference:
                miv_serialization = int(dram.gbus_tsv_sd)
                dreamram_scaled_tsv_capacitance_pF = float(
                    tech.scaled_cap_tsv())
                pitch_scale = float(
                    (tech.tsv_pitch / tech._tsv_pitch)
                    ** tech.tsv_c_pitch_scale_conf)
                miv_vertical_capacitance_pF_per_um = float(
                    tech._c_tsv / tech._tsv_height * pitch_scale)
                dreamram_reference = {
                    "dreamram_reference_tsv_height_um": float(tech._tsv_height),
                    "dreamram_current_tsv_height_um": float(tech.tsv_height),
                    "dreamram_reference_tsv_capacitance_pF": float(tech._c_tsv),
                    "dreamram_reference_tsv_pitch_um": float(tech._tsv_pitch),
                    "dreamram_current_tsv_pitch_um": float(tech.tsv_pitch),
                    "dreamram_reference_load_capacitance_pF": float(tech._c_load),
                    "dreamram_complete_scaled_tsv_capacitance_pF": (
                        dreamram_scaled_tsv_capacitance_pF),
                    "dreamram_tsv_pitch_capacitance_scale": pitch_scale,
                }
            else:
                miv_serialization = None
                miv_vertical_capacitance_pF_per_um = None
                dreamram_reference = {}
            data_pumps = int(dram.pumps_per_atom())
            data_transition_factor = float(dram.dbi_transition_factor_max())
            row_col_voltage_product = float(tech.vcore_int * tech.vdd)
            data_voltage_product = float(tech.vddql_int * tech.vdd)
            command_energy, components = dram.per_cmd_energy(tech)
            atoms_per_page = int(dram.atoms_per_page())
            atom_size = int(dram.atom_size)
            independent_row_pages = int(dram.ind_row_pages())
            refresh_component_energy_pJ = {
                "row": 1.5 * float(components["row"]),
                "mwl": float(components["mwl"]),
                "lwl": float(components["lwl"]),
                "bl-pre": float(components["bl-pre"]),
                "bl-act": float(components["bl-act"]),
            }
            refresh_internal_event_energy_pJ = sum(
                refresh_component_energy_pJ.values())
            refresh_events_per_full_memory_cycle = int(
                dram.ranks * dram.channels * dram.pch
                * dram.horiz_bg * dram.vert_bg * dram.banks
                * dram.subarrays * dram.mat_rows * independent_row_pages)
            refresh_bits_per_event = int(
                dram.mats * dram.mat_cols / independent_row_pages)
            dreamram_total_stored_bits = (
                refresh_events_per_full_memory_cycle * refresh_bits_per_event)
            refresh_organization = {
                "dies": dies_stacked,
                "ranks": int(dram.ranks),
                "channels": int(dram.channels),
                "channels_per_die": int(dram.ch_per_die),
                "pseudochannels": int(dram.pch),
                "horizontal_bankgroups": int(dram.horiz_bg),
                "vertical_bankgroups": int(dram.vert_bg),
                "banks_per_bankgroup": int(dram.banks),
                "subarrays_per_bank": int(dram.subarrays),
                "mats_per_subarray": int(dram.mats),
                "rows_per_mat": int(dram.mat_rows),
                "columns_per_mat": int(dram.mat_cols),
                "independent_row_pages": independent_row_pages,
            }

        if m3d_subarray is None:
            row_policy = config.workload.row_policy
            if row_policy is None:
                raise ValueError("DreamRAM 1T1C path requires row_policy")
            utilization = row_policy.activated_row_data_utilization
            minimum_utilization = 1.0 / atoms_per_page
            effective_rd_per_act = utilization * atoms_per_page
            if effective_rd_per_act < 1.0:
                raise ValueError(
                    "activated_row_data_utilization must be at least "
                    f"1/atoms_per_page={minimum_utilization} for "
                    f"atoms_per_page={atoms_per_page}; got {utilization}")
            access_reuse = effective_rd_per_act
        else:
            if config.workload.control_address_reuse is None:
                raise ValueError("M3D requires workload.control_address_reuse")
            utilization = None
            minimum_utilization = None
            effective_rd_per_act = None
            access_reuse = config.workload.control_address_reuse

        reference_geometry_fit = evaluate_geometry_fit(
            configured_x_mm=geometry.configured_x_mm,
            configured_y_mm=geometry.configured_y_mm,
            required_x_mm=required_x_mm,
            required_y_mm=required_y_mm,
        )
        si_packing_metadata: dict[str, object] = {}
        geometry_fit = reference_geometry_fit
        if (m3d_subarray is None
                and geometry.memory_region == "orthogonal_memory_slab"):
            primitive_bits = int(
                dram.subarrays * dram.mat_rows * dram.mats * dram.mat_cols)
            packing = pack_si_primitive(
                slab_width_um=geometry.configured_x_mm * 1e3,
                slab_height_um=geometry.configured_y_mm * 1e3,
                slab_count=geometry.memory_region_count,
                primitive_type=(
                    "DREAMRAM_BANK_TILE_WITH_DECODERS_SWD_BLSA"),
                primitive_width_um=float(bank_tile_width_um),
                primitive_height_um=float(bank_y_um),
                primitive_bits=primitive_bits,
            )
            geometry_fit = evaluate_geometry_fit(
                configured_x_mm=geometry.configured_x_mm,
                configured_y_mm=geometry.configured_y_mm,
                required_x_mm=packing.packed_width_um * 1e-3,
                required_y_mm=packing.packed_height_um * 1e-3,
            )
            events_per_bank = int(
                dram.subarrays * dram.mat_rows * independent_row_pages)
            refresh_events_per_full_memory_cycle = (
                events_per_bank * packing.primitives_per_slab
                * packing.slab_count)
            dreamram_total_stored_bits = packing.total_system_bits
            if (refresh_events_per_full_memory_cycle * refresh_bits_per_event
                    != dreamram_total_stored_bits):
                raise RuntimeError(
                    "packed Si refresh-event capacity does not close")
            refresh_organization.update({
                "capacity_basis": "INTEGER_PACKED_DREAMRAM_BANKS",
                "packed_banks_per_slab": packing.primitives_per_slab,
                "orthogonal_slab_count": packing.slab_count,
                "refresh_events_per_packed_bank": events_per_bank,
            })
            si_packing_metadata = {
                "capacity_model": "GEOMETRY_DRIVEN_INTEGER_BANK_PACKING",
                **packing.as_dict(),
                "dreamram_reference_required_x_mm": (
                    reference_geometry_fit.required_x_mm),
                "dreamram_reference_required_y_mm": (
                    reference_geometry_fit.required_y_mm),
                "dreamram_reference_x_utilization": (
                    reference_geometry_fit.x_utilization),
                "dreamram_reference_y_utilization": (
                    reference_geometry_fit.y_utilization),
                "dreamram_reference_geometry_feasible": (
                    reference_geometry_fit.geometry_feasible),
                "dreamram_reference_geometry_role": "REFERENCE_ONLY",
                "packing_rotation_tested": True,
                "packing_primitive_scope": (
                    "BANK_TILE_INCLUDES_LOCAL_ROW_COLUMN_DECODERS_"
                    "BANK_SELECT_DECODER_SWD_BLSA_OPEN_BITLINE_AND_"
                    "ECC_FOOTPRINT"),
            }
        miv_metadata: dict[str, object] = {}
        miv_access_energy: float | None = None
        if config.architecture.vertical.type == "miv":
            if m3d_subarray is None:
                raise ValueError(
                    "Orthogonal M3D requires resolved embedded-peripheral "
                    "subarray topology")
            m3d_geometry = geometry.m3d
            if m3d_geometry is None:
                raise ValueError("M3D architecture requires resolved M3D geometry")
            vertical = config.architecture.vertical
            topology = build_miv_topology(
                m3d_layers=m3d_geometry.layers,
                layer_pitch_um=m3d_geometry.layer_pitch_um,
                data_width_before_vertical=(
                    m3d_subarray.data_width_before_vertical),
                vertical_serialization_factor=(
                    miv_serialization
                    if length_scaled_reference
                    else vertical.vertical_serialization_factor
                    if vertical.vertical_serialization_factor is not None
                    else "unresolved"),
                row_miv_count=m3d_subarray.row_address_bits,
                col_miv_count=m3d_subarray.column_address_bits,
                layer_access_probability=(
                    config.workload.layer_access_probability),
                capacitance_fF=(
                    miv_vertical_capacitance_pF_per_um * 1e3
                    if (length_scaled_reference
                        and miv_vertical_capacitance_pF_per_um is not None)
                    else vertical.capacitance_fF
                    if vertical.capacitance_fF is not None else "unresolved"),
            )
            miv_metadata = topology.as_dict()
            miv_metadata["miv_planar_footprint_basis"] = (
                "tang_subarray_cluster")
            miv_metadata["m3d_layers_source"] = (
                "geometry_source.m3d_beol.bitcell_layers")
            miv_metadata["layer_pitch_source"] = (
                "geometry_source.m3d_beol.bitcell_layer_pitch_nm")
            miv_metadata["dies_stacked"] = dies_stacked
            miv_metadata["m3d_layers_independent_of_dies_stacked"] = True
            if length_scaled_reference:
                if miv_vertical_capacitance_pF_per_um is None:
                    raise RuntimeError("length-scaled capacitance was not resolved")
                if vertical.fixed_load_pF is None:
                    raise RuntimeError("MIV fixed endpoint load was not configured")
                miv_energy = calculate_length_scaled_miv_energy(
                    topology,
                    vertical_capacitance_pF_per_um=(
                        miv_vertical_capacitance_pF_per_um),
                    fixed_load_pF=vertical.fixed_load_pF,
                    row_voltage_product_V2=row_col_voltage_product,
                    col_voltage_product_V2=row_col_voltage_product,
                    data_voltage_product_V2=data_voltage_product,
                    data_pumps=data_pumps,
                    data_transition_factor=data_transition_factor,
                    control_address_reuse=(
                        config.workload.control_address_reuse),
                    atom_size_bits=atom_size,
                )
                miv_access_energy = miv_energy.miv_access_energy_pJ_per_bit
                miv_metadata.update(miv_energy.as_dict())
                miv_metadata.update({
                    **dreamram_reference,
                    "miv_electrical_model": (
                        "DREAMRAM_TSV_LENGTH_SCALED_REFERENCE"),
                    "miv_modeling_class": "MODELING_CHOICE",
                    "miv_serialization_factor": miv_serialization,
                    "miv_serialization_source": "DREAMRAM_TSV_EQUIVALENT",
                    "miv_vertical_capacitance_pF_per_um": (
                        miv_vertical_capacitance_pF_per_um),
                    "miv_vertical_capacitance_fF_per_um": (
                        miv_vertical_capacitance_pF_per_um * 1e3),
                    "miv_vertical_capacitance_source": (
                        "DREAMRAM_DATE2026_TSV_REFERENCE"),
                    "miv_vertical_capacitance_classification": (
                        "DERIVED_FROM_REFERENCE"),
                    "miv_fixed_load_pF": vertical.fixed_load_pF,
                    "miv_fixed_load_fF": vertical.fixed_load_pF * 1e3,
                    "miv_fixed_load_classification": (
                        vertical.fixed_load_provenance),
                    "dreamram_scaled_cap_tsv_used_per_m3d_segment": False,
                    "dreamram_reference_load_used_as_miv_fixed_load": False,
                    "miv_segments_energy_role": "GEOMETRY_DIAGNOSTIC_ONLY",
                    "row_miv_energy_pj_per_bit": (
                        miv_energy.row_miv_access_energy_pJ_per_bit),
                    "col_miv_energy_pj_per_bit": (
                        miv_energy.col_miv_access_energy_pJ_per_bit),
                    "data_miv_energy_pj_per_bit": (
                        miv_energy.data_miv_access_energy_pJ_per_bit),
                    "row_miv_voltage_source": "DREAMRAM_TSV_EQUIVALENT",
                    "col_miv_voltage_source": "DREAMRAM_TSV_EQUIVALENT",
                    "data_miv_voltage_source": "DREAMRAM_TSV_EQUIVALENT",
                    "row_miv_voltage_mapping": "row-tsv_voltage_domain",
                    "col_miv_voltage_mapping": "col-tsv_voltage_domain",
                    "data_miv_voltage_mapping": "data-tsv_voltage_domain",
                    "miv_geometry_source": "EXISTING_M3D_GEOMETRY",
                    "miv_electrical_provenance": {
                        "type": "DREAMRAM_TSV_LENGTH_SCALED_REFERENCE",
                        "classification": "MODELING_CHOICE",
                        "serialization_source": "DREAMRAM_TSV_EQUIVALENT",
                        "distributed_capacitance_source": (
                            "DREAMRAM_DATE2026_TSV_REFERENCE"),
                        "distributed_capacitance_classification": (
                            "DERIVED_FROM_REFERENCE"),
                        "fixed_load_classification": "MODELING_CHOICE",
                        "fixed_load_source": "M3D_POWER_CONFIG",
                        "voltage_source": "DREAMRAM_TSV_EQUIVALENT",
                        "geometry_source": "EXISTING_M3D_GEOMETRY",
                    },
                })

        command_groups = {
            command: {group: 0.0 for group in _GROUP_COMPONENTS}
            for command in _COMMAND_TERMS
        }
        for command, terms in _COMMAND_TERMS.items():
            for component, multiplier in terms.items():
                command_groups[command][_group_for(component)] += (
                    multiplier * float(components[component]))
            reconstructed = sum(command_groups[command].values())
            if abs(reconstructed - float(command_energy[command])) > 1e-10:
                raise RuntimeError(
                    f"DreamRAM {command} decomposition does not close: "
                    f"{reconstructed} != {command_energy[command]}")

        denominator = access_reuse * atom_size
        access = {
            group: (
                command_groups["pre"][group]
                + command_groups["act"][group]
                + access_reuse * command_groups["rd"][group]
            ) / denominator
            for group in _GROUP_COMPONENTS
        }
        if m3d_subarray is not None:
            if miv_access_energy is None:
                access["vertical"] = 0.0
            else:
                access["vertical"] = miv_access_energy
            access["memory_internal"] = 0.0
            access["base_route"] = 0.0
            access["interface"] = 0.0
        internal_components = {
            component: (
                _COMMAND_TERMS["pre"].get(component, 0.0)
                * float(components[component])
                + _COMMAND_TERMS["act"].get(component, 0.0)
                * float(components[component])
                + access_reuse * _COMMAND_TERMS["rd"].get(component, 0.0)
                * float(components[component])
            ) / denominator
            for component in _GROUP_COMPONENTS["memory_internal"]
        }
        decomposition = EnergyDecomposition(**access)
        if m3d_subarray is not None:
            internal_components = {}
        reference = (
            decomposition.total
            if m3d_subarray is not None
            else (
                float(command_energy["pre"]) + float(command_energy["act"])
                + access_reuse * float(command_energy["rd"])
            ) / denominator)
        if abs(decomposition.total - reference) > 1e-12:
            raise RuntimeError("DreamRAM access-energy decomposition does not close")
        if abs(sum(internal_components.values())
               - decomposition.memory_internal) > 1e-12:
            raise RuntimeError(
                "DreamRAM internal component partition does not close")

        metadata = {
            "branch": DREAMRAM_BRANCH,
            "commit": DREAMRAM_COMMIT,
            "memory_config": str(memory_path),
            "technology_config": str(technology_path),
            "atom_size_bits": atom_size,
            "atoms_per_page": atoms_per_page,
            "geometry_source_config": geometry.source,
            "memory_region": geometry.memory_region,
            **geometry_fit.as_dict(),
            **si_packing_metadata,
            **miv_metadata,
            "unsupported_operations": ["write", "background"],
        }
        if m3d_subarray is None:
            metadata.update({
                "activated_row_data_utilization": utilization,
                "activated_row_data_utilization_percent": (
                    100.0 * utilization),
                "minimum_activated_row_data_utilization": (
                    minimum_utilization),
                "minimum_activated_row_data_utilization_percent": (
                    100.0 * minimum_utilization),
                "effective_rd_per_act": effective_rd_per_act,
                "dies_stacked": dies_stacked,
                "cell_pitch_mapping": geometry_mapping,
                "pitch_bl_um": float(tech.pitch_bl),
                "pitch_wl_um": float(tech.pitch_wl),
                "pitch_ldl_um": float(tech.pitch_ldl),
                "pitch_mdl_um": float(tech.pitch_mdl),
                "pitch_ldl_to_wl_min": float(tech.pitch_ldl_to_wl_min),
                "pitch_mdl_to_bl_min": float(tech.pitch_mdl_to_bl_min),
                "mat_x_um": float(mat_x_um),
                "mat_y_um": float(mat_y_um),
                "bank_x_um": float(bank_x_um),
                "bank_y_um": float(bank_y_um),
                "dreamram_mat_core_bits": int(
                    dram.mat_rows * dram.mat_cols),
                "dreamram_mat_local_footprint_width_um": float(
                    mat_x_um + swd_width_um),
                "dreamram_mat_local_footprint_height_um": float(
                    mat_y_um + blsa_height_um),
                "dreamram_subarray_bits": int(
                    dram.mats * dram.mat_rows * dram.mat_cols),
                "dreamram_subarray_standalone_packing_geometry": (
                    "NOT_EXPOSED_SHARED_OPEN_BITLINE_AND_DECODER_OVERHEAD"),
                "dreamram_bank_select_decoder_width_um": float(
                    bank_select_decoder_width_um),
                "dreamram_bank_tile_width_um": float(bank_tile_width_um),
                "dreamram_bank_tile_height_um": float(bank_y_um),
                "wire_lengths_um": {
                    name: float(value) for name, value in wire_lengths.items()},
                "E_PRE_pJ": float(command_energy["pre"]),
                "E_ACT_pJ": float(command_energy["act"]),
                "E_RD_pJ": float(command_energy["rd"]),
                "component_classification": {
                    "1T1C_SPECIFIC": sorted(ONE_T_ONE_C_SPECIFIC),
                    "REUSABLE_STRUCTURE": sorted(REUSABLE_STRUCTURE),
                },
                "dreamram_refresh_included_components": list(
                    refresh_component_energy_pJ),
                "dreamram_refresh_included_component_energy_pJ": (
                    refresh_component_energy_pJ),
                "dreamram_refresh_excluded_components": [
                    "row-tsv", "col-tsv", "tsv",
                    "row-base", "col-base", "base",
                    "row-dq", "col-dq", "dq",
                    "col", "csl", "ldl", "mdl", "bgbus+gbus",
                ],
                "refresh_internal_event_energy_pJ": (
                    refresh_internal_event_energy_pJ),
                "refresh_event_scope": (
                    "ONE_SELECTED_BANK_SUBARRAY_ROW_ACROSS_PARALLEL_MATS"),
                "refresh_events_per_full_memory_cycle": (
                    refresh_events_per_full_memory_cycle),
                "refresh_bits_per_event": refresh_bits_per_event,
                "dreamram_total_stored_bits": dreamram_total_stored_bits,
                "dreamram_refresh_organization": refresh_organization,
                "dreamram_refresh_equations_provenance": (
                    "DERIVED_FROM_REFERENCE"),
            })
        else:
            metadata.update({
                "control_address_reuse": access_reuse,
                "control_address_reuse_definition": (
                    "data accesses sharing one row/column address-selection "
                    "operation for MIV control-energy amortization"),
                "dreamram_role": "miv_electrical_reference_only",
                "dreamram_planar_organization_used": False,
                "dreamram_internal_components_used": False,
            })

        return BackendEnergyResult(
            technology=config.memory.technology,
            backend="dreamram",
            read_default=decomposition,
            native_internal_components=internal_components,
            metadata=metadata,
        )
