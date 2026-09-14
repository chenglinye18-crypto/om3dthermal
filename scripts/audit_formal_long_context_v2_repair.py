"""Direct assertions for the three-row capacity repair; no simulations/tests."""
import hashlib
import json
import math
import formal_long_context_v2_support as f


def main():
    snapshot=json.loads((f.OUT/'capacity_repair_preservation.json').read_text())
    rows=f.read_csv(f.OUT/'final_e2e_metrics.csv')
    key=lambda r:tuple(str(r[k]) for k in ('model','context','batch','path'))
    lookup={key(r):r for r in rows}
    assert len(rows)==len(lookup)==72
    for row in snapshot['rows']:
        for field in ('E2E_s','tokens_per_s','E2E_J','tokens_per_J','Tmax_C'):
            assert float(row[field])==float(lookup[key(row)][field])
    for name,digest in snapshot['files'].items():
        assert hashlib.sha256((f.ROOT/name).read_bytes()).hexdigest()==digest,name
    for row in rows:
        assert row['execution_status']=='EVALUATED'
        assert all(math.isfinite(float(row[k])) and float(row[k])>0 for k in ('tokens_per_s','tokens_per_J','Tmax_C'))
    results=[]
    for path in f.PATHS[1:]:
        r=json.loads((f.OUT/'candidates'/f'Llama-3.1-405B_LC64K_B8_{path}.json').read_text())
        if path!='M3D_GPU':
            a=r['physical_capacity_audit'];l=r['capacity_legalization']
            assert a['slot_capacity_violations']==0 and a['max_slot_bytes']<=a['slot_capacity_bytes']
            assert a['resident_bytes']<=a['total_capacity_bytes']
            assert a['max_group_bytes']<=a['group_capacity_bytes'] and a['max_die_bytes']<=a['die_capacity_bytes']
            assert l['fallback_triggered'] and l['die_changes']==l['group_changes']==0
            assert not any(l[k] for k in ('route_objective_used','latency_objective_used','CPA_objective_used'))
            assert r['summary']['initial_placement_provenance']=='UNIFORM_STRIPING_CAPACITY_LEGALIZED'
        seq=[float(lookup[('Llama-3.1-405B',c,'8',path)]['tokens_per_s']) for c in f.CONTEXTS]
        results.append(dict(path=path,throughput_by_context=seq,monotonic=all(a>b for a,b in zip(seq,seq[1:])),
            capacity=r.get('physical_capacity_audit',{}),legalization=r.get('capacity_legalization',{})))
    reps=json.loads((f.OUT/'capacity_legalization_audit.json').read_text())
    assert len(reps)==4 and all(not r['legalization']['fallback_triggered'] for r in reps[1:])
    # The aggregate GPU gate must never invoke the NMP physical-slot helper.
    gate=f.legacy.physical_capacity_gate
    def forbidden(*args):raise AssertionError('GPU called NMP capacity gate')
    f.legacy.physical_capacity_gate=forbidden
    try:assert f.cap('Llama-3.1-405B','LC64K',8,'M3D_GPU')['status']=='EVALUATED'
    finally:f.legacy.physical_capacity_gate=gate
    audit=dict(existing_69_unchanged='PASS',physical_artifact_hashes='PASS',matrix_72_valid='PASS',
        successful_uniform_initialization_unchanged='PASS',GPU_capacity_independent_of_NMP='PASS',
        pytest_run=False,results=results)
    f.save(f.OUT/'capacity_repair_validation.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
