"""GPU-only port-first bijection of legal resident group/slab slots.

No row splitting, route change, hardware multiplier or tile optimization.
Logical cyclic order remains separate from the new physical lane identity.
"""
from dataclasses import dataclass
import numpy as np
from .nmp_load_balance import ResidentOperator, PlacementPolicy


@dataclass
class GPUPortResidentOperator(ResidentOperator):
    logical_lanes: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        self.order = (self.logical_lanes-self.lane_offset) % (self.die_count*70)


def port_first_lanes(floor):
    # Exact same nearest-port rule as GROUP_DIRECT; no ports become reachable
    # through routing that does not already exist in the physical model.
    ports=np.array([min(range(len(floor.ports)), key=lambda p:
        abs(g['center_um'][0]-floor.ports[p][0])+abs(g['center_um'][1]-floor.ports[p][1]))
        for g in floor.groups])
    groups={p:sorted(np.flatnonzero(ports==p),key=lambda g:(floor.sa_edge_ns[g],int(g)))
            for p in sorted(set(ports))}
    # One group per port first, then slabs; only then reuse a (slab,port)
    # through its next group. Route startup breaks equivalent-load ties.
    lanes=[int(groups[p][depth])*floor.layout.slab_count+slab
           for depth in range(max(map(len,groups.values())))
           for slab in range(floor.layout.slab_count)
           for p in groups if depth<len(groups[p])]
    lanes=np.asarray(lanes,dtype=np.int32)
    assert np.array_equal(np.sort(lanes),np.arange(floor.layout.slab_count*70))
    return lanes,ports


def remap_external_ports(placement):
    permutation,ports=port_first_lanes(placement.floorplan)
    old_used=placement.slot_used
    new_used=np.empty_like(old_used)
    new_used[permutation]=old_used
    remapped=0
    # Preserve per-request aliases and shared embedding ownership. Each entry
    # has the same logical atom order, start layers, and complete row/KV atoms.
    for key,old in list(placement.request_operators.items()):
        new_lanes=permutation[old.lanes]
        entry=GPUPortResidentOperator(old.unit,old.atom_bytes,old.atom_count,
            old.lane_offset,new_lanes,old.start_layers,old.tile_ids.copy(),
            old.die_count,logical_lanes=old.lanes)
        # Inert legal tile IDs are required by the shared physical structure.
        # They are never optimized or executed by the GPU-only branch.
        entry.tile_ids[:]=entry.region_ids*8+(entry.group_ids%18)%8
        placement.request_operators[key]=entry
        if old.unit.request_id in (None,0):
            placement.operators[old.unit.layer_id,old.unit.operator_type]=entry
            remapped+=int(old.prefix_counts(old.atom_count)[old.lanes!=new_lanes].sum())
        elif old.unit.operator_type!='TOKEN_EMBED_LOOKUP':
            remapped+=int(old.prefix_counts(old.atom_count)[old.lanes!=new_lanes].sum())
    placement.slot_used=new_used
    placement.policy=PlacementPolicy.GPU_PORT_BALANCED
    placement.gpu_port_audit=dict(atom_remapping_count=remapped,
        logical_atom_identity_changed=False,logical_traffic_changed=False,
        physical_lane_bijection=True,layer_assignment_changed=False,
        group_to_port=ports.tolist(),reachable_ports_per_slab=len(set(ports)),
        objective='PORT_FIRST_ROUND_ROBIN; ROUTE_STARTUP_TIE_BREAK',
        source_placement='UNCHANGED_CAPACITY_LEGAL_UNIFORM_STRIPING',
        CPA_used=False,NMP_tile_optimization=False)
