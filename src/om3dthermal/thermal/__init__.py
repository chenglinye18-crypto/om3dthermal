"""Thermal layer: conductance, boundary conditions, power, and
matrix-free steady-state solver.

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
- :mod:`.steady_state` — weighted Jacobi and PCG solvers
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
    solve_pcg,
    solve_weighted_jacobi,
    validate_anchored_components,
)
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
    "InterfaceResistanceQuery",
    "InterfaceResistanceRegistry",
    "InvalidRotationMatrixError",
    "MatrixFreeThermalOperator",
    "MissingThermalConductivityError",
    "PowerSourceResult",
    "PowerVector",
    "SteadyStateResult",
    "UnanchoredThermalComponentError",
    "UnsupportedMaterialRotationError",
    "build_boundary_link_table",
    "build_conductance_table",
    "build_matrix_free_operator",
    "build_power_breakdown",
    "build_solver_summary",
    "canonical_rotation_key",
    "global_conductivity_tensor",
    "is_signed_axis_permutation",
    "map_power_sources",
    "normal_conductivity",
    "select_boundary_rule",
    "solve_pcg",
    "solve_weighted_jacobi",
    "validate_anchored_components",
    "validate_rotation_matrix",
    "write_boundary_heat_flows_csv",
    "write_solver_history_csv",
    "write_solver_summary_json",
    "write_temperature_csv",
    "write_temperature_npz",
]
