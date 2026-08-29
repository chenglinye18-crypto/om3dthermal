"""Config-driven memory power accounting."""

from .config import (
    CanonicalCaseConfig,
    MemoryPowerConfig,
    load_case_config,
    load_power_config,
)
from .geometry import ResolvedGeometry, resolve_case_geometry
from .cell_model import (
    MissingCellReplacementError,
    apply_component_replacements,
)
from .model import (
    UnresolvedMIVEnergyError,
    calculate_memory_power,
    run_memory_power,
)
from .result import EnergyDecomposition, MemoryPowerResult
from .physical_latency import (
    PhysicalAccessLatency,
    PhysicalLocationLatency,
    calculate_physical_access_latency,
)
from .physical_capacity import (
    CapacityLatencyCutoff,
    PhysicalCapacityLayout,
    PhysicalSlot,
    PhysicalSlotClass,
    calculate_physical_capacity_layout,
    iter_physical_slots,
)
from .memory_bandwidth import (
    ArchitectureBandwidthClosure,
    EffectiveBandwidth,
    InternalBandwidthPrefix,
    derive_architecture_bandwidth,
    resolve_effective_bandwidth,
)
from .system import (
    ResolvedSystemPower,
    ResolvedThermalPowerMapping,
    map_system_power_to_thermal,
    run_case_system_power,
    resolve_system_power,
)

__all__ = [
    "EnergyDecomposition",
    "MemoryPowerConfig",
    "CanonicalCaseConfig",
    "ResolvedGeometry",
    "MemoryPowerResult",
    "PhysicalAccessLatency",
    "PhysicalLocationLatency",
    "CapacityLatencyCutoff",
    "PhysicalCapacityLayout",
    "PhysicalSlot",
    "PhysicalSlotClass",
    "ArchitectureBandwidthClosure",
    "EffectiveBandwidth",
    "InternalBandwidthPrefix",
    "MissingCellReplacementError",
    "UnresolvedMIVEnergyError",
    "apply_component_replacements",
    "calculate_memory_power",
    "calculate_physical_access_latency",
    "calculate_physical_capacity_layout",
    "iter_physical_slots",
    "derive_architecture_bandwidth",
    "resolve_effective_bandwidth",
    "load_power_config",
    "load_case_config",
    "resolve_case_geometry",
    "run_memory_power",
    "ResolvedSystemPower",
    "ResolvedThermalPowerMapping",
    "resolve_system_power",
    "map_system_power_to_thermal",
    "run_case_system_power",
]
