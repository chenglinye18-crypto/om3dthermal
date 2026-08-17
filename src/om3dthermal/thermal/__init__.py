"""Thermal layer: conductance, boundary conditions, power, and
steady-state solver.

Modules:

- :mod:`.errors` — domain error types
- :mod:`.tensors` — material local tensor, rotation matrix, normal
  conductivity
- :mod:`.interfaces` — interface resistance registry
- :mod:`.conductance` — per-edge two-point face conductance
- :mod:`.export` — CSV / NPZ / JSON writers for conductance
- :mod:`.boundary` — boundary condition rules and the columnar
  ``BoundaryLinkTable``
- :mod:`.power` — per-cell power source mapping
- :mod:`.operator` — matrix-free thermal operator
- :mod:`.steady_state` — shared diagnostics and result type for the
  steady-state solver
- :mod:`.thermal_relaxation` — CPU thermal-resistance relaxation
  solver (the only production steady-state solver)
- :mod:`.gpu_relaxation` — GPU implementation of the same solver
- :mod:`.gpu_common` — shared CuPy loader and probe
- :mod:`.solution_export` — solver CSV / NPZ / JSON writers
"""
from .boundary import (
    BoundaryLinkTable,
    build_boundary_link_table,
    select_boundary_rule,
)
from .conductance import ConductanceTable, build_conductance_table
from .errors import (
    InvalidRotationMatrixError,
    MissingThermalConductivityError,
    UnsupportedMaterialRotationError,
)
from .interfaces import InterfaceResistanceQuery, InterfaceResistanceRegistry
from .gpu_common import GPUBackendUnavailableError, require_cupy
from .gpu_relaxation import (
    GPURelaxationState,
    solve_thermal_resistance_relaxation_gpu,
)
from .gpu_pcg import GPUPCGOperator, GPUSolverBreakdownError, solve_pcg_gpu
from .m3d_power import (
    M3DMemoryPowerResolution,
    M3DOperationPowerBreakdown,
    UnresolvedM3DActivityError,
    calculate_array_read_power,
    calculate_operation_energy_power,
    femtojoules_to_joules,
    resolve_m3d_memory_power,
)
from .operator import MatrixFreeThermalOperator, build_matrix_free_operator
from .power import (
    PowerSourceResult,
    PowerVector,
    build_power_breakdown,
    map_power_sources,
)
from .solution_export import (
    build_solver_summary,
    write_boundary_heat_flows_csv,
    write_solver_history_csv,
    write_solver_summary_json,
    write_temperature_csv,
    write_temperature_npz,
)
from .steady_state import (
    SteadyStateResult,
    UnanchoredThermalComponentError,
    validate_anchored_components,
)
from .thermal_relaxation import solve_thermal_resistance_relaxation
from .tensors import (
    canonical_rotation_key,
    global_conductivity_tensor,
    is_signed_axis_permutation,
    normal_conductivity,
    validate_rotation_matrix,
)

__all__ = [
    "BoundaryLinkTable",
    "ConductanceTable",
    "GPURelaxationState",
    "GPUPCGOperator",
    "GPUBackendUnavailableError",
    "GPUSolverBreakdownError",
    "InterfaceResistanceQuery",
    "InterfaceResistanceRegistry",
    "InvalidRotationMatrixError",
    "MatrixFreeThermalOperator",
    "M3DMemoryPowerResolution",
    "M3DOperationPowerBreakdown",
    "MissingThermalConductivityError",
    "PowerSourceResult",
    "PowerVector",
    "SteadyStateResult",
    "UnanchoredThermalComponentError",
    "UnsupportedMaterialRotationError",
    "UnresolvedM3DActivityError",
    "build_boundary_link_table",
    "build_conductance_table",
    "build_matrix_free_operator",
    "build_power_breakdown",
    "build_solver_summary",
    "calculate_array_read_power",
    "calculate_operation_energy_power",
    "canonical_rotation_key",
    "global_conductivity_tensor",
    "femtojoules_to_joules",
    "is_signed_axis_permutation",
    "map_power_sources",
    "normal_conductivity",
    "resolve_m3d_memory_power",
    "require_cupy",
    "select_boundary_rule",
    "solve_thermal_resistance_relaxation",
    "solve_thermal_resistance_relaxation_gpu",
    "solve_pcg_gpu",
    "validate_anchored_components",
    "validate_rotation_matrix",
    "write_boundary_heat_flows_csv",
    "write_solver_history_csv",
    "write_solver_summary_json",
    "write_temperature_csv",
    "write_temperature_npz",
]
