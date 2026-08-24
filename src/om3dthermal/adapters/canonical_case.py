"""Resolve a new ArchitectureSpec through the existing canonical case path."""

from __future__ import annotations

from pathlib import Path

from om3dthermal.architecture import (
    ArchitectureSpec,
    ResolvedArchitecture,
    resolve_packing_from_legacy_power_result,
)
from om3dthermal.power import (
    load_case_config,
    resolve_case_geometry,
    resolve_system_power,
)


def resolve_architecture_spec(
    spec: ArchitectureSpec,
    *,
    project_root: Path,
) -> ResolvedArchitecture:
    """Resolve without copying or changing any canonical physical parameter."""

    case = load_case_config(spec.canonical_case)
    if case.name != spec.architecture_id:
        raise ValueError(
            "architecture descriptor identity does not match canonical case")
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(
        case, project_root=project_root, geometry=geometry)
    if system.memory_result is None:
        raise ValueError(
            "formal architecture resolution requires analytical packing evidence")
    packing = resolve_packing_from_legacy_power_result(
        case, geometry, system.memory_result)
    return ResolvedArchitecture(
        spec=spec,
        case=case,
        geometry=geometry,
        system_power=system,
        packing=packing,
    )
