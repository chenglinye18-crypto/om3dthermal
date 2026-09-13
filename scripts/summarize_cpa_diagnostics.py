"""Audit completed CPA results; representative resource loads are labeled."""
import json
import numpy as np
import compare_critical_path_placement as run
from om3dthermal.serving.decode_policy import DecodePolicyModel,ExecutionPolicy,llama31_models


def ratio(a):
    a=np.asarray(a)
    return float(a.max()/a.mean()) if a.mean() else 0.0


def main():
    diagnostics=[];mechanisms=[]
    for name,w in llama31_models().items():
        result=json.loads((run.OUT/f'{name}.json').read_text())
        e=run.engine(name,1,'CRITICAL_PATH_AWARE')
        for policy in ('UNIFORM_STRIPING','CRITICAL_PATH_AWARE'):
            model=(e if policy=='CRITICAL_PATH_AWARE' else DecodePolicyModel(w,project_root=run.ROOT,
                   placement_policy=policy,record_energy=True,record_slabs=True))
            s=model.step(126499,ExecutionPolicy.MAC_NMP,include_stages=True)
            stages=[p for p in s['stages'] if 'resources' in p]
            fabric=sum(p['resources']['fabric_bytes'] for p in stages)
            tiles=sum(p['resources']['tile_loads'] for p in stages)
            data=(result if policy=='CRITICAL_PATH_AWARE' else json.loads((run.REFERENCE/'case_results'/f'{name}_B1_UNIFORM_STRIPING.json').read_text()))
            diagnostics.append(dict(model=name,placement=policy,resource_context=126499,
                active_slabs=s['slab_events']['array_read_bits'].astype(bool).sum(),
                slab_read_activity_max_mean=ratio([p['array_read_bytes'] for p in data['slabs']]),
                slab_MAC_activity_max_mean=ratio([p['NMP_FLOPs'] for p in data['slabs']]),
                region_fabric_max_mean=ratio(fabric),tile_MAC_max_mean=ratio(tiles),
                max_NoC_link_utilization=s['noc_max_link_utilization'],
                max_NoC_link_id=json.dumps([int(x) for x in np.unravel_index(np.argmax(s['noc_link_busy_s']),(318,3,2))]),
                route_wire_J_per_token=data['energy']['components']['feol_wire_J']/1000,
                slab_power_max_mean=data['summary']['slab_power_max_mean_ratio']))
        for op in sorted({a['operator'] for a in result['optimizer_audit']}):
            rows=[a for a in result['optimizer_audit'] if a['operator']==op]
            before=sum(a['before_s'] for a in rows);after=sum(a['after_s'] for a in rows)
            atoms=sum(e.placement.operators[a['layer'],op].atom_count for a in rows)
            mechanisms.append(dict(model=name,operator=op,accepted_moves=sum(a['accepted_moves'] for a in rows),
                candidate_evaluations=sum(a['candidate_evaluations'] for a in rows),
                before_stage_s=before,after_stage_s=after,stage_reduction_fraction=(before-after)/before,
                resident_moved_fraction=sum(a['resident_atoms_moved'] for a in rows)/atoms,
                compute_reassigned_atom_fraction=sum(a['moved_atom_fraction']*e.placement.operators[a['layer'],op].atom_count for a in rows)/atoms,
                before_bottlenecks=json.dumps(sorted({a['before_bottleneck'] for a in rows})),
                after_bottlenecks=json.dumps(sorted({a['after_bottleneck'] for a in rows}))))
    run.base.write_csv('placement_diagnostics.csv',diagnostics)
    run.base.write_csv('mechanism.csv',mechanisms)
    print('Physical load and operator mechanism audits written')


if __name__=='__main__':main()
