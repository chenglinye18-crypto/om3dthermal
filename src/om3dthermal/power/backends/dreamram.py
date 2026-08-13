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
    evaluate_geometry_fit,
    load_m3d_geometry,
    load_memory_region_size,
)
from ..miv import build_miv_topology, calculate_tsv_equivalent_miv_energy
from ..result import BackendEnergyResult, EnergyDecomposition


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

    def calculate(self, config: MemoryPowerConfig) -> BackendEnergyResult:
        dreamram_input = config.memory.dreamram
        if dreamram_input is None:
            raise ValueError("DreamRAM backend requires memory.dreamram")
        if config.workload.row_policy is None:
            raise ValueError("DreamRAM backend requires workload.row_policy.rd_per_act")

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
            mat_x_um = (
                dram.mat_cols * dram.isolation_cols_overhead * tech.pitch_bl)
            mat_y_um = (
                dram.mat_rows * dram.isolation_rows_overhead * tech.pitch_wl)
            stack_dims = dram.calc_stack_dims(tech)
            dies_stacked = int(stack_dims[0])
            if config.architecture.vertical.type == "miv":
                # M3D is a single-die topology. Exclude HBM TSV/KOZ bands from
                # its planar footprint constraint; dies_stacked is diagnostic
                # only and never enters the MIV length calculation.
                required_x_um, required_y_um = dram.bankdie_dims(tech)
                required_x_mm = float(required_x_um) * 1e-3
                required_y_mm = float(required_y_um) * 1e-3
            else:
                required_x_mm = float(stack_dims[1]) * 1e-3
                required_y_mm = float(sum(stack_dims[2:])) * 1e-3
            wire_lengths = dram.wire_lengths(tech)
            wire_counts = dram.wire_counts()
            row_bits, col_bits = dram.ch_cmd_bits()
            tsv_equivalent = (
                config.architecture.vertical.electrical_model
                == "tsv_equivalent_reference")
            if tsv_equivalent:
                miv_serialization = int(dram.gbus_tsv_sd)
                miv_capacitance_pF = float(tech.scaled_cap_tsv())
            else:
                miv_serialization = None
                miv_capacitance_pF = None
            data_pumps = int(dram.pumps_per_atom())
            data_transition_factor = float(dram.dbi_transition_factor_max())
            row_col_voltage_product = float(tech.vcore_int * tech.vdd)
            data_voltage_product = float(tech.vddql_int * tech.vdd)
            command_energy, components = dram.per_cmd_energy(tech)
            atoms_per_page = int(dram.atoms_per_page())
            atom_size = int(dram.atom_size)

        n_read = config.workload.row_policy.rd_per_act
        if n_read > atoms_per_page:
            raise ValueError(
                f"rd_per_act={n_read} exceeds atoms_per_page={atoms_per_page}")

        geometry_path, configured_x_mm, configured_y_mm = (
            load_memory_region_size(
                self.project_root, config.architecture.geometry_source))
        geometry_fit = evaluate_geometry_fit(
            configured_x_mm=configured_x_mm,
            configured_y_mm=configured_y_mm,
            required_x_mm=required_x_mm,
            required_y_mm=required_y_mm,
        )
        miv_metadata: dict[str, object] = {}
        miv_access_energy: float | None = None
        if config.architecture.vertical.type == "miv":
            m3d_geometry = load_m3d_geometry(
                self.project_root, config.architecture.geometry_source)
            vertical = config.architecture.vertical
            topology = build_miv_topology(
                m3d_layers=m3d_geometry.layers,
                layer_pitch_um=m3d_geometry.layer_pitch_um,
                data_width_before_vertical=int(wire_counts["gbus"]),
                vertical_serialization_factor=(
                    miv_serialization
                    if tsv_equivalent
                    else vertical.vertical_serialization_factor
                    if vertical.vertical_serialization_factor is not None
                    else "unresolved"),
                row_miv_count=int(row_bits),
                col_miv_count=int(col_bits),
                layer_access_probability=(
                    config.workload.layer_access_probability),
                capacitance_fF=(
                    miv_capacitance_pF * 1e3
                    if tsv_equivalent and miv_capacitance_pF is not None
                    else vertical.capacitance_fF
                    if vertical.capacitance_fF is not None else "unresolved"),
            )
            miv_metadata = topology.as_dict()
            miv_metadata["m3d_layers_source"] = (
                "geometry_source.m3d_beol.bitcell_layers")
            miv_metadata["layer_pitch_source"] = (
                "geometry_source.m3d_beol.bitcell_layer_pitch_nm")
            miv_metadata["dies_stacked"] = dies_stacked
            miv_metadata["m3d_layers_independent_of_dies_stacked"] = True
            if tsv_equivalent:
                if miv_capacitance_pF is None:
                    raise RuntimeError("TSV-equivalent capacitance was not resolved")
                miv_energy = calculate_tsv_equivalent_miv_energy(
                    topology,
                    capacitance_pF_per_segment=miv_capacitance_pF,
                    row_voltage_product_V2=row_col_voltage_product,
                    col_voltage_product_V2=row_col_voltage_product,
                    data_voltage_product_V2=data_voltage_product,
                    data_pumps=data_pumps,
                    data_transition_factor=data_transition_factor,
                    rd_per_act=n_read,
                    atom_size_bits=atom_size,
                )
                miv_access_energy = miv_energy.miv_access_energy_pJ_per_bit
                miv_metadata.update(miv_energy.as_dict())
                miv_metadata.update({
                    "miv_electrical_model": "TSV_EQUIVALENT_BASELINE",
                    "miv_modeling_class": "MODELING_CHOICE",
                    "miv_serialization_factor": miv_serialization,
                    "miv_serialization_source": "DREAMRAM_TSV_EQUIVALENT",
                    "miv_capacitance_per_segment_pF": miv_capacitance_pF,
                    "miv_capacitance_per_segment": miv_capacitance_pF,
                    "miv_capacitance_source": "DREAMRAM_TSV_EQUIVALENT",
                    "miv_capacitance_physical_interpretation": (
                        "effective_capacitance_per_vertical_segment"),
                    "miv_segment_mapping": "one_m3d_layer_pitch_per_segment",
                    "row_miv_voltage_source": "DREAMRAM_TSV_EQUIVALENT",
                    "col_miv_voltage_source": "DREAMRAM_TSV_EQUIVALENT",
                    "data_miv_voltage_source": "DREAMRAM_TSV_EQUIVALENT",
                    "row_miv_voltage_mapping": "row-tsv_voltage_domain",
                    "col_miv_voltage_mapping": "col-tsv_voltage_domain",
                    "data_miv_voltage_mapping": "data-tsv_voltage_domain",
                    "miv_geometry_source": "EXISTING_M3D_GEOMETRY",
                    "miv_electrical_provenance": {
                        "type": "TSV_EQUIVALENT_BASELINE",
                        "classification": "MODELING_CHOICE",
                        "serialization_source": "DREAMRAM_TSV_EQUIVALENT",
                        "capacitance_source": "DREAMRAM_TSV_EQUIVALENT",
                        "capacitance_interpretation": (
                            "effective_capacitance_per_vertical_segment"),
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

        denominator = n_read * atom_size
        access = {
            group: (
                command_groups["pre"][group]
                + command_groups["act"][group]
                + n_read * command_groups["rd"][group]
            ) / denominator
            for group in _GROUP_COMPONENTS
        }
        if config.architecture.vertical.type == "miv":
            if miv_access_energy is None:
                access["vertical"] = 0.0
            else:
                access["vertical"] = miv_access_energy
            access["base_route"] = 0.0
            access["interface"] = 0.0
        internal_components = {
            component: (
                _COMMAND_TERMS["pre"].get(component, 0.0)
                * float(components[component])
                + _COMMAND_TERMS["act"].get(component, 0.0)
                * float(components[component])
                + n_read * _COMMAND_TERMS["rd"].get(component, 0.0)
                * float(components[component])
            ) / denominator
            for component in _GROUP_COMPONENTS["memory_internal"]
        }
        decomposition = EnergyDecomposition(**access)
        reference = (
            decomposition.total
            if config.architecture.vertical.type == "miv"
            else (
                float(command_energy["pre"]) + float(command_energy["act"])
                + n_read * float(command_energy["rd"])
            ) / denominator)
        if abs(decomposition.total - reference) > 1e-12:
            raise RuntimeError("DreamRAM access-energy decomposition does not close")
        if abs(sum(internal_components.values())
               - decomposition.memory_internal) > 1e-12:
            raise RuntimeError(
                "DreamRAM internal component partition does not close")

        return BackendEnergyResult(
            technology=config.memory.technology,
            backend="dreamram",
            read_default=decomposition,
            native_internal_components=internal_components,
            metadata={
                "branch": DREAMRAM_BRANCH,
                "commit": DREAMRAM_COMMIT,
                "memory_config": str(memory_path),
                "technology_config": str(technology_path),
                "rd_per_act": n_read,
                "atom_size_bits": atom_size,
                "atoms_per_page": atoms_per_page,
                "dies_stacked": dies_stacked,
                "geometry_source_config": str(geometry_path),
                "memory_region": config.architecture.geometry_source.memory_region,
                **geometry_fit.as_dict(),
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
                "wire_lengths_um": {
                    name: float(value) for name, value in wire_lengths.items()},
                **miv_metadata,
                "E_PRE_pJ": float(command_energy["pre"]),
                "E_ACT_pJ": float(command_energy["act"]),
                "E_RD_pJ": float(command_energy["rd"]),
                "component_classification": {
                    "1T1C_SPECIFIC": sorted(ONE_T_ONE_C_SPECIFIC),
                    "REUSABLE_STRUCTURE": sorted(REUSABLE_STRUCTURE),
                },
                "unsupported_operations": [
                    "write", "refresh", "background"],
            },
        )
