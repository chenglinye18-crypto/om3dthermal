"""Read-only adapter for the pinned DreamRAM DATE2026 analytical model."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Iterator

from ..config import MemoryPowerConfig, resolve_project_path
from ..result import BackendEnergyResult, EnergyDecomposition


DREAMRAM_BRANCH = "DATE2026"
DREAMRAM_COMMIT = "c069ce14dfa85ce1983f3a1274a265d1e7b5494a"

_GROUP_COMPONENTS = {
    "memory_internal": {
        "row", "mwl", "lwl", "bl-pre", "bl-act", "col", "csl",
        "ldl", "mdl", "bgbus+gbus",
    },
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
            tech = tech_module.Tech(**technology)
            hbm_fields = {item.name for item in fields(hbm_module.Hbm)}
            hbm_kwargs = {key: value for key, value in memory.items()
                          if key in hbm_fields}
            hbm_kwargs["brv_sa"] = memory["brvsa"]
            dram = hbm_module.Hbm(**hbm_kwargs)
            command_energy, components = dram.per_cmd_energy(tech)
            atoms_per_page = int(dram.atoms_per_page())
            atom_size = int(dram.atom_size)

        n_read = config.workload.row_policy.rd_per_act
        if n_read > atoms_per_page:
            raise ValueError(
                f"rd_per_act={n_read} exceeds atoms_per_page={atoms_per_page}")

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
        decomposition = EnergyDecomposition(**access)
        reference = (
            float(command_energy["pre"]) + float(command_energy["act"])
            + n_read * float(command_energy["rd"])
        ) / denominator
        if abs(decomposition.total - reference) > 1e-12:
            raise RuntimeError("DreamRAM access-energy decomposition does not close")

        return BackendEnergyResult(
            technology=config.memory.technology,
            backend="dreamram",
            read_default=decomposition,
            metadata={
                "branch": DREAMRAM_BRANCH,
                "commit": DREAMRAM_COMMIT,
                "memory_config": str(memory_path),
                "technology_config": str(technology_path),
                "rd_per_act": n_read,
                "atom_size_bits": atom_size,
                "atoms_per_page": atoms_per_page,
                "E_PRE_pJ": float(command_energy["pre"]),
                "E_ACT_pJ": float(command_energy["act"]),
                "E_RD_pJ": float(command_energy["rd"]),
                "unsupported_operations": [
                    "write", "refresh", "background"],
            },
        )
