"""Experiment-facing architecture identities and packing boundaries."""

from .models import (
    ArchitectureSpec,
    ResolvedArchitecture,
    ResolvedArchitectureFacts,
    ResolvedEnergyPrimitives,
    ResolvedPacking,
    ResolvedStaticPower,
)
from .packing import PACKING_SOURCE_STATUS, resolve_packing_from_legacy_power_result

__all__ = [
    "ArchitectureSpec",
    "PACKING_SOURCE_STATUS",
    "ResolvedArchitecture",
    "ResolvedArchitectureFacts",
    "ResolvedEnergyPrimitives",
    "ResolvedPacking",
    "ResolvedStaticPower",
    "resolve_packing_from_legacy_power_result",
]
