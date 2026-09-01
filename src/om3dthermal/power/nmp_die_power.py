"""Workload-driven per-die NMP memory/compute power map for B preparation."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import math
from om3dthermal.placement.nmp_load_balance import (NMPPerformanceBalancedPlacement,
    external_bytes_per_die_for_ownership)
from .backends.operation_table import OperationTableCellModel
from .config import MemoryPowerConfig
from .feol_route import FEOLRouteResult,wire_cv2_energy_pj_per_bit
from .m3d_subarray import M3DSubarrayResult
from .nmp_die_activity import NMPDieActivitySummary,NMPDiePower
from .result import MemoryPowerResult

@dataclass(frozen=True)
class NMPPowerPrimitives:
    cluster_width_um:float; cluster_height_um:float; local_route_length_um:float
    local_route_provenance:str; feol_capacitance_fF_per_um:float; feol_voltage_V:float
    feol_activity_factor:float; local_route_energy_pj_per_bit:float
    igzo_local_read_and_global_control_pj_per_bit:float; vertical_miv_pj_per_bit:float
    local_read_total_pj_per_bit:float; igzo_weighted_write_pj_per_bit:float
    local_write_total_pj_per_bit:float; long_feol_pj_per_bit:float
    interface_pj_per_bit:float; mac_energy_pj_per_mac:float
    nmp_logic_overhead_factor:float; component_boundary_status:str

@dataclass(frozen=True)
class NMPDiePowerMap:
    primitives:NMPPowerPrimitives; die_powers:tuple[NMPDiePower,...]
    decode_step_interval_ms:float; refresh_total_W:float; refresh_per_die_W:float
    aggregate_memory_read_dynamic_W:float; aggregate_memory_write_dynamic_W:float
    aggregate_mac_dynamic_W:float; aggregate_refresh_W:float
    aggregate_residual_external_W:float; aggregate_total_W:float
    residual_external_bytes:float; residual_external_attribution_status:str
    power_component_double_count_gate:str; thermal_payload_status:str
    def as_dict(self): return asdict(self)

def build_nmp_die_power_map(config:MemoryPowerConfig,memory:MemoryPowerResult,
        topology:M3DSubarrayResult,feol:FEOLRouteResult,activity:NMPDieActivitySummary,
        placement:NMPPerformanceBalancedPlacement)->NMPDiePowerMap:
    local_length=.5*(topology.cluster_width_um+topology.cluster_height_um)
    local_route=wire_cv2_energy_pj_per_bit(activity_factor=feol.feol_wire_activity_factor,
        capacitance_fF_per_um=feol.feol_wire_capacitance_fF_per_um,length_um=local_length,
        voltage_V=feol.feol_wire_voltage_V)
    if local_route>=feol.feol_route_energy_pj_per_bit:
        raise ValueError("cluster-scale local route must be shorter-energy than long-edge FEOL")
    device=OperationTableCellModel().calculate(config)
    write=device.weighted_write(p00=.25,p01=.25,p10=.25,p11=.25)
    read_total=memory.E_memory_internal_pj_bit+memory.E_vertical_pj_bit+local_route
    write_total=write+memory.E_vertical_pj_bit+local_route
    primitives=NMPPowerPrimitives(topology.cluster_width_um,topology.cluster_height_um,local_length,
        "MODELING_CHOICE_TOPOLOGY_DERIVED_CLUSTER_SCALE_LOCAL_ROUTE",feol.feol_wire_capacitance_fF_per_um,
        feol.feol_wire_voltage_V,feol.feol_wire_activity_factor,local_route,memory.E_memory_internal_pj_bit,
        memory.E_vertical_pj_bit,read_total,write,write_total,feol.feol_route_energy_pj_per_bit,
        memory.E_interface_pj_bit,.604,1.0,
        "ZHU_MAT_LOCAL_INCLUDES_LOCAL_RC_SENSING__TANG_GLOBAL_CONTROL_ONCE__MIV_ONCE__LOCAL_BULK_EXCLUDES_LONG_FEOL_AND_INTERFACE")
    die_count=len(activity.activities); refresh_total=float(memory.P_refresh_W or 0.0); refresh_die=refresh_total/die_count
    external_bytes=external_bytes_per_die_for_ownership(placement.unit_loads,placement.ownership,die_count)
    interval_s=activity.decode_step_interval_ms*1e-3; rows=[]
    for item,ext_bytes in zip(activity.activities,external_bytes,strict=True):
        read_bits=8*(item.weight_read_bytes+item.kv_read_bytes); write_bits=8*item.kv_write_bytes
        read_w=read_bits*read_total*1e-12/interval_s; write_w=write_bits*write_total*1e-12/interval_s
        mac_w=(item.nmp_flops/2*.604e-12)/interval_s
        external_w=ext_bytes*8*(memory.E_feol_route_pj_bit+memory.E_interface_pj_bit)*1e-12/interval_s
        total=read_w+write_w+mac_w+refresh_die+external_w
        rows.append(NMPDiePower(item.die_id,mac_w,read_w+write_w,refresh_die,total,
            "WORKLOAD_DRIVEN_B_PREP_POWER_MAP__MAC_DYNAMIC_ONLY",read_w,write_w,mac_w,1.0,mac_w,
            external_w,total,read_w+write_w+refresh_die,mac_w,"MEMORY_TO_M3D_BITCELL_BEOL__NMP_TO_M3D_FEOL__RESIDUAL_EXTERNAL_THERMAL_MAPPING_PENDING"))
    def total(field): return sum(getattr(x,field) or 0.0 for x in rows)
    result=NMPDiePowerMap(primitives,tuple(rows),activity.decode_step_interval_ms,refresh_total,refresh_die,
        total("memory_read_dynamic_W"),total("memory_write_dynamic_W"),total("mac_dynamic_W"),
        total("refresh_W"),total("residual_external_W"),total("total_W"),sum(external_bytes),
        "DETERMINISTIC_PRODUCING_OR_OWNING_PLACEMENT_UNIT_ATTRIBUTION",
        "PASS","B_PREP_POWER_CARRIERS_READY__THERMAL_SOLVER_NOT_CALLED")
    if any(not math.isfinite(x.total_W or -1) or (x.total_W or 0)<0 for x in rows):
        raise ValueError("per-die power must be finite and non-negative")
    return result
