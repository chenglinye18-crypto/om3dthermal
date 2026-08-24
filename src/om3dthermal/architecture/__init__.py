"""Experiment-facing architecture identities and packing boundaries."""

from .models import ArchitectureSpec, ResolvedArchitecture, ResolvedPacking
from .packing import PACKING_SOURCE_STATUS, resolve_packing_from_legacy_power_result

__all__ = [
    "ArchitectureSpec",
    "PACKING_SOURCE_STATUS",
    "ResolvedArchitecture",
    "ResolvedPacking",
    "resolve_packing_from_legacy_power_result",
]
