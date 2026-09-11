"""Low-level M3D write primitive shared by non-NMP energy analyses."""
from .backends.operation_table import OperationTableCellModel
from .config import MemoryPowerConfig
from .result import MemoryPowerResult

def resolve_orthogonal_m3d_write_energy_pj_per_bit(
        config:MemoryPowerConfig,memory:MemoryPowerResult)->float:
    """Resolve a full M3D write path from existing read-path components."""
    device=OperationTableCellModel().calculate(config)
    write=device.weighted_write(p00=.25,p01=.25,p10=.25,p11=.25)
    return (write+memory.E_vertical_pj_bit+memory.E_feol_route_pj_bit
            +memory.E_base_route_pj_bit+memory.E_interface_pj_bit)
