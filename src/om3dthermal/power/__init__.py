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
from .dream_reference_service import (
    DreamInternalStage,
    DreamReferenceLatency,
    DreamReferenceServiceAudit,
    audit_dream_reference_service,
    classify_bottleneck,
)
from .memory_bandwidth import (
    ArchitectureBandwidthClosure,
    EffectiveBandwidth,
    InternalBandwidthPrefix,
    derive_architecture_bandwidth,
    resolve_effective_bandwidth,
)
from .latency_decomposition_audit import (
    DreamLatencyDecomposition,
    FEOLResistanceSensitivityRow,
    LatencyAuditGates,
    LatencyModelRiskItem,
    M3DLatencyDecomposition,
    UnifiedStageMapping,
    audit_dream_latency_decomposition,
    build_m3d_latency_decomposition,
    build_risk_ranking,
    build_unified_taxonomy,
    classify_gates,
    run_feol_resistance_sensitivity,
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
    "DreamInternalStage",
    "DreamReferenceLatency",
    "DreamReferenceServiceAudit",
    "audit_dream_reference_service",
    "classify_bottleneck",
    "DreamLatencyDecomposition",
    "FEOLResistanceSensitivityRow",
    "LatencyAuditGates",
    "LatencyModelRiskItem",
    "M3DLatencyDecomposition",
    "UnifiedStageMapping",
    "audit_dream_latency_decomposition",
    "build_m3d_latency_decomposition",
    "build_risk_ranking",
    "build_unified_taxonomy",
    "classify_gates",
    "run_feol_resistance_sensitivity",
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
