"""Atomic resident striping into existing cluster-layer physical slots."""
from dataclasses import dataclass, replace
from enum import StrEnum
from copy import copy
import numpy as np

from om3dthermal.workload.dense_decode_ledger import build_dense_decode_placement_units


class PlacementPolicy(StrEnum):
    GPU_PORT_BALANCED = "GPU_PORT_BALANCED"
    COMPACT_FIRST_FIT = "COMPACT_FIRST_FIT"
    UNIFORM_STRIPING = "UNIFORM_STRIPING"
    BALANCED = "BALANCED"
    CRITICAL_PATH_AWARE = "CRITICAL_PATH_AWARE"


@dataclass
class ResidentOperator:
    unit: object
    atom_bytes: int
    atom_count: int
    lane_offset: int
    lanes: np.ndarray
    start_layers: np.ndarray
    tile_ids: np.ndarray
    die_count: int

    def __post_init__(self):
        self.order = (self.lanes-self.lane_offset) % (self.die_count*70)
        self.layer_order = (np.arange(8,dtype=np.uint8)[None, :]-self.start_layers[:, None]) % 8
        self.group_ids = self.lanes//self.die_count
        self.die_ids = self.lanes % self.die_count
        self.region_ids = np.minimum(self.group_ids//18, 3)
        self.region_dest = self.die_ids*4+self.region_ids

    def prefix_counts(self, atoms):
        if not 0 <= atoms <= self.atom_count:
            raise ValueError("active atoms exceed resident allocation")
        return np.maximum(0, (atoms+self.die_count*70-1-self.order)//(self.die_count*70))

    def layer_bytes(self, atoms, *, begin=0, indices=None):
        """Only read/write resident atoms, optionally restricted to active lanes."""
        count=self.prefix_counts(atoms)
        layer=self.layer_order
        if indices is not None:count=count[indices];layer=layer[indices]
        end=np.maximum(0,(count[:,None]+7-layer)//8)
        if begin:
            before=self.prefix_counts(begin)
            if indices is not None:before=before[indices]
            end-=np.maximum(0,(before[:,None]+7-layer)//8)
        return end*self.atom_bytes



class CompactResidentOperator(ResidentOperator):
    """Compressed contiguous atomic ranges, each entirely in one group/layer."""
    def set_segments(self, lane_indices, layers, counts):
        self.segment_lane = np.asarray(lane_indices)
        self.segment_layer = np.asarray(layers)
        self.segment_count = np.asarray(counts)
        self.segment_begin = np.r_[0, np.cumsum(self.segment_count)[:-1]]

    def _counts(self, atoms):
        if not 0 <= atoms <= self.atom_count:
            raise ValueError("active atoms exceed resident allocation")
        return np.clip(atoms-self.segment_begin,0,self.segment_count)

    def prefix_counts(self, atoms):
        return np.bincount(self.segment_lane,weights=self._counts(atoms),minlength=len(self.lanes)).astype(np.int64)

    def layer_bytes(self, atoms, *, begin=0, indices=None):
        counts=self._counts(atoms)-self._counts(begin)
        result=np.bincount(self.segment_lane*8+self.segment_layer,weights=counts*self.atom_bytes,
                           minlength=len(self.lanes)*8).reshape(-1,8).astype(np.int64)
        return result if indices is None else result[indices]


class PhysicalResidentPlacement:
    """Die-fastest cyclic rows/vectors, then capacity-balanced layer striping.

    Each atom stays in one group/layer and is bit-striped across its four
    physical clusters. Arrays describe counts, never materialized tensors.
    """
    def __init__(self, workload, floorplan, policy=PlacementPolicy.BALANCED):
        if PlacementPolicy(policy) == PlacementPolicy.GPU_PORT_BALANCED:
            # Start with the unchanged legal Uniform allocation. A bijection of
            # physical lanes preserves every slot's occupancy and atom identity.
            self.__init__(workload, floorplan, PlacementPolicy.UNIFORM_STRIPING)
            from .gpu_port_balanced import remap_external_ports
            remap_external_ports(self)
            return
        try:
            self._initialize(workload, floorplan, policy)
        except ValueError as exc:
            if not str(exc).startswith('physical slot capacity exceeded:') or PlacementPolicy(policy) not in (PlacementPolicy.UNIFORM_STRIPING, PlacementPolicy.CRITICAL_PATH_AWARE):
                raise
            failed = str(exc).split(': ',1)[1]
            self._initialize(workload, floorplan, policy, legalize=True)
            self.capacity_legalization['failing_operator'] = failed

    def _initialize(self, workload, floorplan, policy, *, legalize=False):
        self.policy = PlacementPolicy(policy)
        if (workload.weight_bits, workload.kv_bits) != (16, 16):
            raise ValueError("physical execution currently requires 16-bit storage")
        self.floorplan = floorplan
        self.workload = workload
        self.dies = floorplan.layout.slab_count
        n = self.dies*70
        self.slot_used = np.zeros((n, 8), dtype=np.int64)  # bytes per member cluster
        self.operators = {}
        self.request_operators = {}
        self.capacity_legalization = dict(fallback_triggered=False, failing_operator='',
            lanes_moved=0, bytes_moved_between_layers=0, die_changes=0, group_changes=0,
            layer_changes=0, route_objective_used=False, latency_objective_used=False,
            CPA_objective_used=False)
        cursor = 0
        paired = {}
        units = build_dense_decode_placement_units(workload)
        embedding = workload.d_model*workload.vocab_size*2
        for original in units:
            u = original
            if u.operator_type == "TOKEN_EMBED_LOOKUP" and u.request_id != 0:
                alias = copy(self.operators[-1,"TOKEN_EMBED_LOOKUP"])
                alias.unit = u
                self.request_operators[u.layer_id,u.operator_type,u.request_id] = alias
                continue
            if u.operator_type == "TOKEN_EMBED_LOOKUP":
                u = replace(u, weight_bytes=embedding, atomic_count=workload.vocab_size)
                atom = workload.d_model*2
            elif u.operator_type == "OTHER_WEIGHT":
                u = replace(u, weight_bytes=u.weight_bytes-embedding)
                atom = 32
            elif u.shard_mode == "KV_ATOMIC":
                atom = workload.d_head*2
            else:
                atom = int(u.weight_bytes/u.output_rows)
            total = int(u.weight_bytes+u.kv_bytes)
            if total == 0:
                continue
            if total % atom or atom % 4:
                raise ValueError("resident atom does not close to four-cluster striping")
            count = total//atom
            offset = paired[u.layer_id,u.request_id] if u.operator_type == "ATTENTION_AV" else cursor % n
            if u.operator_type == "ATTENTION_QK":
                paired[u.layer_id,u.request_id] = offset
            lanes = (np.arange(min(count, n), dtype=np.int32)+offset) % n
            starts = (np.argmin(self.slot_used[lanes], axis=1) if self.policy == PlacementPolicy.BALANCED or legalize
                      else (np.arange(len(lanes))+cursor//n)%8).astype(np.uint8)
            entry = ResidentOperator(u, atom, count, offset, lanes, starts, np.zeros(len(lanes), dtype=np.int32), self.dies)
            if self.policy == PlacementPolicy.COMPACT_FIRST_FIT:
                entry = self._compact(u,atom,count)
                lanes = entry.lanes
            allocated = entry.layer_bytes(count)
            overflow = np.any(self.slot_used[lanes] + allocated//4 > floorplan.layout.slot_capacity_bytes, axis=1)
            if np.any(overflow) and legalize:
                # Capacity-only fallback: keep each atom's die/group and try the
                # next cyclic start layer only on overflowing lanes.
                audit = self.capacity_legalization
                audit['fallback_triggered'] = True
                if not audit['failing_operator']: audit['failing_operator'] = u.unit_id
                before = allocated.copy()
                original_starts = entry.start_layers.copy()
                for shift in range(1, 8):
                    indices = np.flatnonzero(overflow)
                    if not len(indices): break
                    candidate = np.roll(before[indices], shift, axis=1)
                    fits = np.all(self.slot_used[lanes[indices]] + candidate//4 <= floorplan.layout.slot_capacity_bytes, axis=1)
                    chosen = indices[fits]
                    allocated[chosen] = candidate[fits]
                    entry.start_layers[chosen] = (original_starts[chosen]+shift)%8
                    overflow[chosen] = False
                if np.any(overflow):
                    raise ValueError(f"physical slot capacity exceeded after layer-only legalization: {u.unit_id}")
                entry.__post_init__()
                assert np.array_equal(entry.layer_bytes(count), allocated)
            if legalize:
                audit=self.capacity_legalization
                audit['fallback_triggered']=True
                cyclic=(np.arange(len(lanes))+cursor//n)%8
                moved=entry.start_layers!=cyclic
                audit['lanes_moved']+=int(moved.sum())
                audit['layer_changes']+=int(moved.sum())
                audit['bytes_moved_between_layers']+=int(allocated[moved].sum())
                self.initial_placement_provenance = 'UNIFORM_STRIPING_CAPACITY_LEGALIZED'
            self.slot_used[lanes] += allocated//4
            if np.any(self.slot_used[lanes] > floorplan.layout.slot_capacity_bytes):
                raise ValueError(f"physical slot capacity exceeded: {u.unit_id}")
            assert allocated.sum() == total
            self._assign_tiles(entry)
            self.request_operators[u.layer_id,u.operator_type,u.request_id] = entry
            if u.request_id in (None,0): self.operators[u.layer_id, u.operator_type] = entry
            cursor += count
        expected = workload.n_param*2 + 2*workload.batch_size*workload.n_layers*workload.context_length*workload.n_heads_kv*workload.d_head*2
        assert int(self.slot_used.sum())*4 == expected

    def get(self, layer, op, request=None):
        if request is None: return self.operators[layer,op]
        return self.request_operators[layer,op,request]

    def _compact(self,u,atom,count):
        # Capacity-first slabs; cyclic fair filling within each slab, no route/load objective.
        free=((self.floorplan.layout.slot_capacity_bytes-self.slot_used)*4//atom)
        slot_ids=np.arange(self.dies*70*8).reshape(70,self.dies,8)
        chosen=[]; amounts=[]; remaining=count
        for slab in range(self.dies):
            ids=slot_ids[:,slab,:].ravel()
            capacity=free.ravel()[ids].copy()
            need=min(remaining,int(capacity.sum()))
            allocated=np.zeros_like(capacity)
            pending=need
            while pending:
                eligible=np.flatnonzero(capacity>0)
                q,r=divmod(pending,len(eligible))
                demand=np.full(len(eligible),q,dtype=np.int64);demand[:r]+=1
                take=np.minimum(demand,capacity[eligible])
                allocated[eligible]+=take;capacity[eligible]-=take;pending-=int(take.sum())
            selected=allocated>0
            chosen.extend(ids[selected]);amounts.extend(allocated[selected])
            remaining-=need
            if not remaining:break
        if remaining:raise ValueError("compact atomic capacity exceeded")
        ids=np.asarray(chosen);take=np.asarray(amounts)
        lanes,inverse=np.unique(ids//8,return_inverse=True)
        entry=CompactResidentOperator(u,atom,count,0,lanes,np.zeros(len(lanes),dtype=np.uint8),np.zeros(len(lanes),dtype=np.int32),self.dies)
        entry.set_segments(inverse,ids%8,take)
        return entry

    def _assign_tiles(self, entry):
        f = self.floorplan
        if self.policy != PlacementPolicy.BALANCED:
            entry.tile_ids[:] = entry.region_ids*8+(entry.group_ids%18)%8
            return
        # Earliest projected tile completion includes physical route startup.
        loads = np.zeros((self.dies, 32))
        counts = entry.prefix_counts(entry.atom_count)
        flop_atom = entry.unit.local_flops/entry.atom_count
        if entry.unit.shard_mode == "ROW_PARALLEL": flop_atom *= self.workload.batch_size
        for g in range(70):
            indices = np.flatnonzero(entry.lanes//self.dies == g)
            if not len(indices):
                continue
            dies = entry.lanes[indices] % self.dies
            region = f.groups[g]["region_id"]
            tiles = np.arange(region*8, region*8+8)
            work = counts[indices]*flop_atom
            costs = (loads[dies[:, None], tiles]+work[:, None])/f.tile_flops + f.sa_tile_ns[g, tiles]*1e-9
            chosen = tiles[np.argmin(costs, axis=1)]
            entry.tile_ids[indices] = chosen
            loads[dies, chosen] += work

    def transient_layer_bytes(self, total_bytes):
        """Energy-only Prefill workspace mapping into existing unused slots."""
        if total_bytes < 0 or total_bytes != int(total_bytes):
            raise ValueError("invalid transient payload")
        free=(self.floorplan.layout.slot_capacity_bytes-self.slot_used)*4
        eligible=np.flatnonzero(free.ravel() >= 32)
        services,remainder=divmod(int(total_bytes),32)
        counts=np.zeros(free.size,dtype=np.int64)
        counts[eligible]=services//len(eligible)
        counts[eligible[:services % len(eligible)]]+=1
        b=counts*32
        if remainder: b[eligible[services % len(eligible)]]+=remainder
        if np.any(b > free.ravel()):
            # These are cumulative temporary accesses, not simultaneous residency.
            # Reuse legal 32-byte workspace slots over successive streaming passes.
            service_capacity=(free.ravel()//32).astype(np.int64)
            full_passes,pending=divmod(services,int(service_capacity.sum()))
            assigned=np.zeros_like(service_capacity)
            remaining_capacity=service_capacity.copy()
            while pending:
                eligible=np.flatnonzero(remaining_capacity)
                q,r=divmod(pending,len(eligible))
                demand=np.full(len(eligible),q,dtype=np.int64);demand[:r]+=1
                take=np.minimum(demand,remaining_capacity[eligible])
                assigned[eligible]+=take;remaining_capacity[eligible]-=take;pending-=int(take.sum())
            b=(full_passes*service_capacity+assigned)*32
            if remainder:
                eligible=np.flatnonzero(remaining_capacity)
                b[eligible[0]]+=remainder
            assert int(b.sum())==int(total_bytes)

        return b.reshape(free.shape)

    def audit(self):
        group = self.slot_used.sum(axis=1)*4
        dies = group.reshape(70, self.dies).sum(axis=0)
        f = self.floorplan
        return dict(placement_policy=self.policy, batch_size=self.workload.batch_size,
                    shared_weight_bytes=self.workload.n_param*2,
                    total_kv_bytes=int(group.sum())-self.workload.n_param*2,
                    per_slab_resident_bytes=dies.tolist(),
                    mean_slab_resident_bytes=float(dies.mean()),
                    active_resident_slabs=int(np.count_nonzero(dies)),
                    resident_bytes=int(group.sum()), max_slot_bytes=int(self.slot_used.max()),
                    slot_capacity_bytes=f.layout.slot_capacity_bytes,
                    max_group_bytes=int(group.max()), group_capacity_bytes=32*f.layout.slot_capacity_bytes,
                    max_die_bytes=int(dies.max()), die_capacity_bytes=f.layout.capacity_per_slab_bytes,
                    active_resident_groups=int(np.count_nonzero(group)),
                    physical_slots_used=int(np.count_nonzero(self.slot_used))*4,
                    slot_capacity_violations=int(np.count_nonzero(self.slot_used > f.layout.slot_capacity_bytes)),
                    group_resident_bytes=group.reshape(70, self.dies).T.tolist(),
                    group_active_layer_slots=(self.slot_used > 0).sum(axis=1).reshape(70, self.dies).T.tolist(),
                    atomic_locality="ONE_GROUP_ONE_LAYER_FOUR_MEMBER_CLUSTER_BIT_STRIPES",
                    algorithm=("DIE_FASTEST_CYCLIC_ATOMS__LEAST_OCCUPIED_START_LAYER__CYCLIC_LAYER_STRIPING" if self.policy == PlacementPolicy.BALANCED else self.policy.value))
