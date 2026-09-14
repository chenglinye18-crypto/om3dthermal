"""Bounded greedy refinement of uniform physical execution ownership.

Resident chunks contain an integer eighth of a group stream (at least one
atom). Compute chunks are whole group streams. KV migrations are paired.
Eight candidate tile pools derive from the eight
physical tiles/region, not an efficiency parameter. A synchronized proposal
handles tied critical resources across identical slabs. Every proposal is
accepted against the unmodified complete physical stage equation.
"""
from copy import copy
from time import perf_counter
import numpy as np
from om3dthermal.power.nmp_die_activity import PhysicalStageModel
from om3dthermal.placement.nmp_load_balance import ResidentOperator

RELATIVE_STOP = 1e-4  # Optimizer convergence tolerance, not a physical coefficient.
MAX_ITERATIONS = 2
OPERATORS = {"Q", "K", "V", "O", "FFN_GATE", "FFN_UP", "FFN_DOWN",
             "LM_HEAD", "ATTENTION_QK", "ATTENTION_AV"}


class MigratedOperator(ResidentOperator):
    """Move whole prefix atoms between existing groups in the same slab.

    Original atom identity/order and layer are retained, including partial
    context reads and KV appends. Transfers never split a row or KV vector.
    """
    irregular_layers = True

    @classmethod
    def build(cls, base, source, destination, count):
        result = copy(base)
        result.__class__ = cls
        result.base = base
        result.source = np.asarray(source,dtype=int)
        result.destination = np.asarray(destination,dtype=int)
        result.move_count = np.asarray(count,dtype=np.int64)
        assert np.all(base.die_ids[result.source] == base.die_ids[result.destination])
        assert len(np.unique(result.source)) == len(result.source)
        return result

    def prefix_counts(self, atoms):
        counts = self.base.prefix_counts(atoms).copy()
        moved = np.minimum(counts[self.source],self.move_count)
        np.add.at(counts,self.source,-moved)
        np.add.at(counts,self.destination,moved)
        return counts

    def layer_bytes(self, atoms, *, begin=0, indices=None):
        result = self.base.layer_bytes(atoms,begin=begin).copy()
        end = np.minimum(self.base.prefix_counts(atoms)[self.source],self.move_count)
        start = np.minimum(self.base.prefix_counts(begin)[self.source],self.move_count)
        moved = (np.maximum(0,(end[:,None]+7-self.base.layer_order[self.source])//8)
                 - np.maximum(0,(start[:,None]+7-self.base.layer_order[self.source])//8))*self.atom_bytes
        np.add.at(result,self.source,-moved)
        np.add.at(result,self.destination,moved)
        assert np.all(result >= 0)
        return result if indices is None else result[indices]


def resident_proposal(entry, stage, f, rank):
    """Rank tied regions and groups by the current limiting resource."""
    r=stage['resources']
    core=max(('ARRAY','LOCAL_FABRIC','MAC'),key=stage['components'].get)
    group_cost=r['group_times']
    if core=='LOCAL_FABRIC':
        cost=r['fabric_bytes']/f.fabric_Bps+r['fabric_startup']
    elif core=='ARRAY':
        cost=np.column_stack([group_cost[:,a:b].max(axis=1) for a,b in ((0,18),(18,36),(36,54),(54,70))])
    else:
        cost=r['tile_loads'].reshape(entry.die_count,4,8).max(axis=2)/f.tile_flops
        group_cost=np.zeros_like(group_cost)
        group_cost[entry.die_ids,entry.group_ids]=r['tile_loads'][entry.die_ids,entry.tile_ids]/f.tile_flops
    counts=entry.prefix_counts(entry.atom_count)
    sources=[];destinations=[];amounts=[]
    for die in range(entry.die_count):
        ids=np.flatnonzero((entry.die_ids==die)&(counts>0))
        if not len(ids):continue
        active_regions=np.unique(entry.region_ids[ids])
        target_region=min(active_regions,key=lambda region:(cost[die,region],int(region)))
        candidates=ids[entry.region_ids[ids]==target_region]
        candidates=sorted(candidates,key=lambda i:(group_cost[die,entry.group_ids[i]],int(entry.group_ids[i])))[:8]
        target=candidates[rank%len(candidates)]
        peak=cost[die,active_regions].max()
        for region in active_regions:
            if region==target_region or cost[die,region]<peak*(1-RELATIVE_STOP):continue
            eligible=ids[entry.region_ids[ids]==region]
            source=max(eligible,key=lambda i:(group_cost[die,entry.group_ids[i]],-int(entry.group_ids[i])))
            sources.append(source);destinations.append(target);amounts.append(max(1,int(counts[source])//8))
    return MigratedOperator.build(entry,sources,destinations,amounts)

def diagnostic(stage, floorplan):
    r = stage["resources"]
    arrays = r["group_times"]
    fabric = r["fabric_bytes"] / floorplan.fabric_Bps + r["fabric_startup"]
    mac = r["tile_loads"] / floorplan.tile_flops
    def ratio(a):
        active = a[a > 0]
        return float(active.max()/active.mean()) if len(active) else 0.0
    core = max(("ARRAY", "LOCAL_FABRIC", "MAC"), key=stage["components"].get)
    # Optimistic relaxation within fixed active topology: free redistribution,
    # no route startup, unchanged external/NoC/reduction. Not a global oracle.
    ideal_core = max(float(arrays.sum(axis=1).max()/70),
                     float(r["fabric_bytes"].sum(axis=1).max()/4/floorplan.fabric_Bps),
                     float(r["tile_loads"].sum(axis=1).max()/32/floorplan.tile_flops))
    fixed = stage["latency_s"] - max(stage["components"][k] for k in ("ARRAY", "LOCAL_FABRIC", "MAC"))
    return dict(core_bottleneck=core, slowest_group=list(map(int,np.unravel_index(arrays.argmax(),arrays.shape))),
                slowest_region=list(map(int,np.unravel_index(fabric.argmax(),fabric.shape))),
                slowest_tile=list(map(int,np.unravel_index(mac.argmax(),mac.shape))),
                region_fabric_max_mean=ratio(r["fabric_bytes"]), tile_MAC_max_mean=ratio(r["tile_loads"]),
                fixed_topology_ideal_s=fixed+ideal_core)


def proposal(entry, stage, f, tile_count, *, batch_size=1):
    """Critical-resource-guided list scheduling over legal region tile pools."""
    result = entry.tile_ids.copy()
    if stage["bottleneck"] == "EXTERNAL_BOUNDARY":
        return result  # Tile reassignment cannot alter this boundary payload/path.
    counts = entry.prefix_counts(entry.atom_count)
    loads = np.zeros((entry.die_count,32))
    core = max(("ARRAY","LOCAL_FABRIC","MAC"),key=stage["components"].get)
    # Actual current bottleneck load orders groups, independently per slab.
    group_cost = stage["resources"]["group_times"] if core == "ARRAY" else stage["resources"]["group_bytes"]
    for region in range(4):
        gs = np.arange(region*18,min(region*18+18,70))
        # Stable aggregate ranking keeps identical physical resources symmetric.
        gs = gs[np.argsort(-group_cost[:,gs].max(axis=0),kind="stable")]
        tiles = np.arange(region*8,region*8+8)
        # Geometry-only pool ordering; full physical evaluation decides acceptance.
        pool = tiles[np.argsort(f.sa_tile_ns[np.ix_(gs,tiles)].max(axis=0),kind="stable")[:tile_count]]
        for g in gs:
            ids = np.flatnonzero(entry.group_ids == g)
            if not len(ids): continue
            dies = entry.die_ids[ids]
            work = counts[ids].astype(float)*(entry.unit.local_flops/entry.atom_count)
            if entry.unit.shard_mode == 'ROW_PARALLEL': work *= batch_size
            costs = (loads[dies[:,None],pool]+work[:,None])/f.tile_flops + f.sa_tile_ns[g,pool]*1e-9
            target = pool[np.argmin(costs,axis=1)]
            result[ids] = target
            loads[dies,target] += work
    return result


class AggregateCandidateModel:
    """Evaluate one legal request-local move against concurrent stage demand."""
    def __init__(self, placement, model):
        self.placement, self.model = placement, model
        self.cache = {}
        self.scope = None

    def evaluate(self, entry, atoms, **kwargs):
        from om3dthermal.power.batched_physical import merge_request_stages
        u = entry.unit
        if u.request_id is None:
            return self.model.evaluate(entry, atoms, **kwargs)
        scope = (u.layer_id,u.operator_type)
        if u.layer_id != self.scope:
            self.cache.clear(); self.scope = u.layer_id
        rows=[]
        for request in range(self.placement.workload.batch_size):
            peer = entry if request == u.request_id else self.placement.get(*scope,request)
            options={**kwargs,'resources':True}
            if peer is entry:
                rows.append(self.model.evaluate(peer,atoms,**options))
                continue
            key=(id(peer),peer.tile_ids.tobytes(),atoms,tuple(sorted(options.items())))
            if key not in self.cache:
                self.cache[key]=(peer,self.model.evaluate(peer,atoms,**options))
            rows.append(self.cache[key][1])
        return merge_request_stages(self.placement.floorplan,rows)


def refine(placement, platform, *, first_context=126000, last_context=126999):
    model = PhysicalStageModel(placement.floorplan,platform,placement.workload)
    batched = placement.workload.batch_size > 1
    if batched: model = AggregateCandidateModel(placement,model)
    audit = []
    entries=list(placement.request_operators.values()) if batched else list(placement.operators.values())
    for entry in entries:
        layer,op,request=entry.unit.layer_id,entry.unit.operator_type,entry.unit.request_id
        if op not in OPERATORS: continue
        entry = placement.get(layer,op,request)
        started = perf_counter()
        # Optimize the final resident prefix, and reject endpoint regressions.
        atoms = last_context*placement.workload.n_heads_kv if op.startswith("ATTENTION") else entry.atom_count
        before = model.evaluate(entry,atoms,nmp=True,resources=True)
        resident_moved = 0
        resident_evaluations = 0
        # A single bounded resident pass; AV ownership follows the QK move.
        if op != "ATTENTION_AV":
            partner = placement.get(layer,"ATTENTION_AV",request) if op == "ATTENTION_QK" else None
            partner_before = model.evaluate(partner,atoms,nmp=True) if partner is not None else None
            best_resident = None
            for rank in range(8):
                candidate = resident_proposal(entry,before,placement.floorplan,rank)
                if not len(candidate.source): break
                cp = (MigratedOperator.build(partner,candidate.source,candidate.destination,candidate.move_count)
                      if partner is not None else None)
                delta = candidate.layer_bytes(entry.atom_count)-entry.layer_bytes(entry.atom_count)
                if cp is not None: delta += cp.layer_bytes(cp.atom_count)-partner.layer_bytes(partner.atom_count)
                if np.any(placement.slot_used[entry.lanes]+delta//4 > placement.floorplan.layout.slot_capacity_bytes): continue
                trial = model.evaluate(candidate,atoms,nmp=True,resources=True)
                resident_evaluations += 1
                if trial["latency_s"] >= before["latency_s"]*(1-RELATIVE_STOP): continue
                if cp is not None:
                    other = model.evaluate(cp,atoms,nmp=True)
                    resident_evaluations += 1
                    if other["latency_s"] > partner_before["latency_s"]: continue
                    # Keep append completion and the first attention step safe.
                    valid = True
                    for old,new in ((entry,candidate),(partner,cp)):
                        for context in (first_context,last_context):
                            n = context*placement.workload.n_heads_kv
                            for write in (False,True):
                                kw = dict(nmp=not write,write=write,begin=n if write else 0)
                                count = n+placement.workload.n_heads_kv if write else n
                                valid &= model.evaluate(new,count,**kw)["latency_s"] <= model.evaluate(old,count,**kw)["latency_s"]
                                resident_evaluations += 2
                    if not valid: continue
                if best_resident is None or trial["latency_s"] < best_resident[0]["latency_s"]:
                    best_resident = trial,candidate,cp,delta
            if best_resident is not None:
                trial,candidate,cp,delta = best_resident
                placement.slot_used[entry.lanes] += delta//4
                if request in (None,0): placement.operators[layer,op] = candidate
                placement.request_operators[layer,op,entry.unit.request_id] = candidate
                if cp is not None:
                    if request in (None,0): placement.operators[layer,"ATTENTION_AV"] = cp
                    placement.request_operators[layer,"ATTENTION_AV",partner.unit.request_id] = cp
                resident_moved = int(candidate.move_count.sum())
                entry = candidate
        original = entry.tile_ids.copy()
        current = model.evaluate(entry,atoms,nmp=True,resources=True) if resident_moved else before
        evaluations = resident_evaluations
        accepted = int(resident_moved > 0)
        history = [[before["latency_s"],current["latency_s"]]] if resident_moved else []
        for iteration in range(MAX_ITERATIONS):
            best = None
            for size in range(1,9):
                candidate = copy(entry)
                candidate.tile_ids = proposal(entry,current,placement.floorplan,size,batch_size=placement.workload.batch_size)
                if np.array_equal(candidate.tile_ids,entry.tile_ids): continue
                trial = model.evaluate(candidate,atoms,nmp=True,resources=True)
                evaluations += 1
                if trial["latency_s"] >= current["latency_s"]*(1-RELATIVE_STOP): continue
                if best is None or trial["latency_s"] < best[0]["latency_s"]:
                    best = trial,candidate
            if best is None: break
            trial,candidate = best
            if op.startswith("ATTENTION"):
                first = first_context*placement.workload.n_heads_kv
                evaluations += 2
                if model.evaluate(candidate,first,nmp=True)["latency_s"] > model.evaluate(entry,first,nmp=True)["latency_s"]:
                    break
            history.append([current["latency_s"],trial["latency_s"]])
            entry.tile_ids = candidate.tile_ids
            current = trial
            accepted += 1
        moved = entry.tile_ids != original
        counts = entry.prefix_counts(entry.atom_count)
        audit.append(dict(layer=entry.unit.layer_id,operator=op,candidate_evaluations=evaluations,
                          accepted_moves=accepted,compute_chunks_moved=int(moved.sum()),
                          moved_atom_fraction=float(counts[moved].sum()/entry.atom_count),
                          resident_atoms_moved=resident_moved,optimizer_runtime_s=perf_counter()-started,
                          before_s=before["latency_s"],after_s=current["latency_s"],
                          before_bottleneck=before["bottleneck"],after_bottleneck=current["bottleneck"],
                          before_components=before["components"],after_components=current["components"],
                          before_diagnostic=diagnostic(before,placement.floorplan),
                          after_diagnostic=diagnostic(current,placement.floorplan),accepted_history=history))
        if batched: audit[-1]['request_id'] = request
    return audit
