"""Matrix-free steady-state thermal operator and GPU-PCG solver.
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
from .gpu_pcg import GPUPCGOperator, GPUSolverBreakdownError, solve_pcg_gpu

from .operator import MatrixFreeThermalOperator, build_matrix_free_operator
from .power import (
    PowerSourceResult,
    PowerVector,
    build_power_breakdown,
    map_power_sources,
)

from .steady_state import (
    SteadyStateResult,
    UnanchoredThermalComponentError,
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
    "GPUPCGOperator",
    "GPUBackendUnavailableError",
    "GPUSolverBreakdownError",
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
    "canonical_rotation_key",
    "global_conductivity_tensor",
    "is_signed_axis_permutation",
    "map_power_sources",
    "normal_conductivity",
    "require_cupy",
    "select_boundary_rule",
    "solve_pcg_gpu",
    "validate_anchored_components",
    "validate_rotation_matrix",
]
