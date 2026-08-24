"""Resolve a new ArchitectureSpec through the existing canonical case path."""

from __future__ import annotations

from pathlib import Path

from om3dthermal.architecture import (
    ArchitectureSpec,
    ResolvedArchitecture,
    ResolvedArchitectureFacts,
    ResolvedEnergyPrimitives,
    ResolvedStaticPower,
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


def extract_architecture_facts(
    resolved: ResolvedArchitecture,
) -> ResolvedArchitectureFacts:
    """Expose typed E2E hardware facts from the validated legacy resolution."""

    system = resolved.system_power
    memory = system.memory_result
    if memory is None:
        raise ValueError("resolved architecture has no analytical memory facts")
    logic_background = memory.P_logic_background_W
    completeness = (
        "RESOLVED"
        if logic_background is not None
        else "UNRESOLVED_LOGIC_BACKGROUND"
    )
    return ResolvedArchitectureFacts(
        architecture_id=resolved.spec.architecture_id,
        display_name=resolved.spec.display_name,
        role=resolved.spec.role,
        canonical_case=resolved.spec.canonical_case,
        geometry_type=system.architecture_type,
        packing=resolved.packing,
        energy_primitives=ResolvedEnergyPrimitives(
            read_access_energy_pj_per_bit=(
                system.memory_access_energy_pJ_per_bit
            ),
            memory_internal_pj_per_bit=memory.E_memory_internal_pj_bit,
            vertical_pj_per_bit=memory.E_vertical_pj_bit,
            feol_route_pj_per_bit=memory.E_feol_route_pj_bit,
            base_route_pj_per_bit=memory.E_base_route_pj_bit,
            interface_pj_per_bit=memory.E_interface_pj_bit,
            source_status=system.memory_power_status,
        ),
        static_power=ResolvedStaticPower(
            refresh_power_W=memory.P_refresh_W,
            memory_background_power_W=memory.P_memory_background_W,
            logic_background_power_W=logic_background,
            fixed_gpu_power_W=system.gpu_power_W,
            source_status=system.memory_power_status,
            completeness_status=completeness,
        ),
        provenance=resolved.spec.provenance,
    )
