"""OM3D Thermal geometry front-end."""

from .config import (
    OrthogonalM3DTemplateConfig,
    SimulationConfig,
    UnresolvedPhysicalParametersError,
    load_config,
    load_orthogonal_m3d_template,
)

__all__ = [
    "OrthogonalM3DTemplateConfig",
    "SimulationConfig",
    "UnresolvedPhysicalParametersError",
    "load_config",
    "load_orthogonal_m3d_template",
]
__version__ = "0.1.0"
