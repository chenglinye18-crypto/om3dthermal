"""Atomic resident striping into existing cluster-layer physical slots."""
from dataclasses import dataclass, replace
import numpy as np

from om3dthermal.workload.dense_decode_ledger import build_dense_decode_placement_units


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
        self.layer_order = (np.arange(8)[None, :]-self.start_layers[:, None]) % 8
        self.group_ids = self.lanes//self.die_count
        self.die_ids = self.lanes % self.die_count
        self.region_ids = np.minimum(self.group_ids//18, 3)
        self.region_dest = self.die_ids*4+self.region_ids

    def prefix_counts(self, atoms):
        if not 0 <= atoms <= self.atom_count:
            raise ValueError("active atoms exceed resident allocation")
        return np.maximum(0, (atoms+self.die_count*70-1-self.order)//(self.die_count*70))

    def layer_bytes(self, atoms, *, begin=0):
        """Only read/write already resident atoms; four clusters share a layer."""
        count = self.prefix_counts(atoms)
        before = self.prefix_counts(begin)
        layer = self.layer_order
        end = np.maximum(0, (count[:, None]+7-layer)//8)
        start = np.maximum(0, (before[:, None]+7-layer)//8)
        return (end-start)*self.atom_bytes


class PhysicalResidentPlacement:
    """Die-fastest cyclic rows/vectors, then capacity-balanced layer striping.

    Each atom stays in one group/layer and is bit-striped across its four
    physical clusters. Arrays describe counts, never materialized tensors.
    """
    def __init__(self, workload, floorplan):
        if workload.batch_size != 1 or (workload.weight_bits, workload.kv_bits) != (16, 16):
            raise ValueError("physical execution currently requires B1, 16-bit storage")
        self.floorplan = floorplan
        self.workload = workload
        self.dies = floorplan.layout.slab_count
        n = self.dies*70
        self.slot_used = np.zeros((n, 8), dtype=np.int64)  # bytes per member cluster
        self.operators = {}
        cursor = 0
        paired = {}
        units = build_dense_decode_placement_units(workload)
        embedding = workload.d_model*workload.vocab_size*2
        for original in units:
            u = original
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
            offset = paired[u.layer_id] if u.operator_type == "ATTENTION_AV" else cursor % n
            if u.operator_type == "ATTENTION_QK":
                paired[u.layer_id] = offset
            lanes = (np.arange(min(count, n), dtype=np.int32)+offset) % n
            starts = np.argmin(self.slot_used[lanes], axis=1).astype(np.uint8)
            entry = ResidentOperator(u, atom, count, offset, lanes, starts, np.zeros(len(lanes), dtype=np.int32), self.dies)
            allocated = entry.layer_bytes(count)
            self.slot_used[lanes] += allocated//4
            if np.any(self.slot_used[lanes] > floorplan.layout.slot_capacity_bytes):
                raise ValueError(f"physical slot capacity exceeded: {u.unit_id}")
            assert allocated.sum() == total
            self._assign_tiles(entry)
            self.operators[u.layer_id, u.operator_type] = entry
            cursor += count
        expected = workload.n_param*2 + 2*workload.n_layers*workload.context_length*workload.n_heads_kv*workload.d_head*2
        assert int(self.slot_used.sum())*4 == expected

    def _assign_tiles(self, entry):
        f = self.floorplan
        # Earliest projected tile completion includes physical route startup.
        loads = np.zeros((self.dies, 32))
        counts = entry.prefix_counts(entry.atom_count)
        flop_atom = entry.unit.local_flops/entry.atom_count
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

    def audit(self):
        group = self.slot_used.sum(axis=1)*4
        dies = group.reshape(70, self.dies).sum(axis=0)
        f = self.floorplan
        return dict(resident_bytes=int(group.sum()), max_slot_bytes=int(self.slot_used.max()),
                    slot_capacity_bytes=f.layout.slot_capacity_bytes,
                    max_group_bytes=int(group.max()), group_capacity_bytes=32*f.layout.slot_capacity_bytes,
                    max_die_bytes=int(dies.max()), die_capacity_bytes=f.layout.capacity_per_slab_bytes,
                    active_resident_groups=int(np.count_nonzero(group)),
                    physical_slots_used=int(np.count_nonzero(self.slot_used))*4,
                    slot_capacity_violations=int(np.count_nonzero(self.slot_used > f.layout.slot_capacity_bytes)),
                    group_resident_bytes=group.reshape(70, self.dies).T.tolist(),
                    group_active_layer_slots=(self.slot_used > 0).sum(axis=1).reshape(70, self.dies).T.tolist(),
                    atomic_locality="ONE_GROUP_ONE_LAYER_FOUR_MEMBER_CLUSTER_BIT_STRIPES",
                    algorithm="DIE_FASTEST_CYCLIC_ATOMS__LEAST_OCCUPIED_START_LAYER__CYCLIC_LAYER_STRIPING")
