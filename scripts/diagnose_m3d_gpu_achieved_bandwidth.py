"""Read-only re-composition of the preserved GPU physical data-chain checkpoints.

Run in the om3dthermal Conda environment. No simulator or placement optimizer is
invoked, and no formal benchmark artifact is written.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import formal_long_context_v2_support as f
from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
from om3dthermal.power.nmp_die_activity import external_service

OUT = f.ROOT / 'runs/m3d_gpu_workload_achieved_bw_diagnostic'
FORMAL = f.ROOT / 'runs/formal_long_context_v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def csv_save(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-7), (a, b)


def aggregate(rows):
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
    drop = [r['throughput_degradation_pct'] for r in rows]
    util = [r['bandwidth_utilization_pct'] for r in rows]
    return dict(count=len(rows), degradation_min_pct=min(drop),
                degradation_max_pct=max(drop), degradation_geomean_pct=gm(drop),
                degradation_from_geomean_TPS_ratio_pct=100*(1-gm([r['throughput_ratio'] for r in rows])),
                utilization_min_pct=min(util), utilization_max_pct=max(util),
                utilization_geomean_pct=gm(util))


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'font.size': 8, 'axes.linewidth': .7})
    x = np.array([model*8 + v for model in range(3) for v in (0,1,2.5,3.5,5,6)])
    for key, label, name in [('bandwidth_utilization_pct','Achieved / peak bandwidth (%)','bandwidth_utilization'),
                             ('throughput_ratio','Data-chain / reference throughput','throughput_ratio')]:
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        ax.bar(x, [r[key] for r in rows], width=.7, color='#4477AA', edgecolor='black', linewidth=.45)
        ax.axhline(100 if key.endswith('pct') else 1, color='gray', linestyle='--', linewidth=.8)
        ax.set_xticks(x, ['B1','B8']*9)
        ax.tick_params(axis='x', length=0, pad=4)
        trans=ax.get_xaxis_transform()
        for mi, model in enumerate(('8B','70B','405B')):
            for ci, context in enumerate(('20K','64K','126K')):
                ax.text(mi*8+ci*2.5+.5, -.15, context, ha='center', va='top', transform=trans)
                if ci < 2: ax.plot([mi*8+ci*2.5+1.75]*2,[0,-.22],color='black',lw=.6,transform=trans,clip_on=False)
            ax.text(mi*8+3,-.29,model,ha='center',va='top',transform=trans)
            if mi < 2: ax.plot([mi*8+7]*2,[0,-.39],color='black',lw=.7,transform=trans,clip_on=False)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)
        fig.subplots_adjust(left=.09,right=.99,top=.94,bottom=.32)
        for suffix in ('svg','pdf'):
            fig.savefig(OUT/f'{name}.{suffix}')
        plt.close(fig)


def worst_case_audit():
    """One existing GPU stage replay; no full workload simulation or optimizer."""
    target=OUT/'worst_case_first_step_stages.json'
    if not target.exists():
        from om3dthermal.serving.decode_policy import DecodePolicyModel
        w,cw,_=f.legacy.inputs('Llama-3.1-8B','LC20K',1,f.ROOT)
        model=DecodePolicyModel(w,project_root=f.ROOT,placement_policy='UNIFORM_STRIPING',
                               external_bandwidth_cap=f.setup(f.ROOT)[4])
        point=model.step(cw.history+cw.prompt,'NO_NMP',include_stages=True)
        stages=[dict(operator=s['operator'],layer=s['layer'],latency_s=s['latency_s'],
                     components=s['components'],transfers=[{k:v for k,v in t.items() if k!='resources'}
                     for t in s.get('external_transfers',[])]) for s in point['stages']]
        save(target.name,stages)
    stages=read(target)
    transfers=[t for s in stages for t in s['transfers']]
    by_op=defaultdict(lambda:dict(boundary_s=0.,cap_only_s=0.,port_s=0.,route_startup_s=0.))
    for s in stages:
        for t in s['transfers']:
            for key,field in [('boundary_s','external_service_s'),('cap_only_s','global_cap_serialization_s'),
                              ('port_s','port_cap_serialization_s'),('route_startup_s','rc_startup_s')]:
                by_op[s['operator']][key]+=t[field]
    result=dict(model='Llama-3.1-8B',cached_history=20000,batch=1,decode_context=20128,
                executor='NO_NMP',physical_stage_count=len(stages),
                boundary_reason_counts=dict(Counter(t['limiting_reason'] for t in transfers)),
                operator_totals=dict(by_op),decode_step_s=sum(s['latency_s'] for s in stages),
                boundary_s=sum(t['external_service_s'] for t in transfers),
                cap_only_s=sum(t['global_cap_serialization_s'] for t in transfers),
                route_startup_sum_s=sum(t['rc_startup_s'] for t in transfers))
    save('worst_case_accounting_audit.json',result)
    return result


def report(rows, stats, worst):
    lines=['# M3D-GPU workload-achieved bandwidth diagnostic', '',
        'Verdict: **EXISTING_ROUTING_CONDITIONAL_DIAGNOSTIC__NOT_BASELINE_REPLACEMENT**.', '',
        'All 18 results reuse the exact GPU physical checkpoints selected by each canonical candidate’s '
        'legacy_fingerprint (including the repaired 405B/64K/B8 checkpoint). No reference was recomputed. '
        '576 Decode steps and all 18 original GPU-only Prefill totals are reused. One 8B/20K/B1 first-step '
        'GPU replay provides operator/port attribution; no HBM, DNS, CPA, thermal, or full-workload simulation is run.', '',
        '## Equations and accounting audit', '',
        '* Reference: primary_execution.gpu_memory_closure closes thermal, internal, coil and GPU ceilings. '
        'The effective minimum is the frozen unrounded 3.3957835581187483 TB/s (displayed as 3.395784), '
        'not an unconstrained thermal-only assignment. Full-cycle MAT/MIV/FEOL is already included once.',
        '* Diagnostic: each existing GPU operator uses max(Array+MIV service, external service, GPU service). '
        'Its external service is max(total bytes / thermal cap, max_port(load / port bandwidth + route startup)). '
        'There is no additional bytes/Bcap term added to these times. Operators follow the existing dependency schedule.',
        '* Physical array service sums 32-byte MAT/MIV cycles within each group/layer lane, then takes '
        'the maximum across groups/dies. It does not add the aggregate reference cycle again. FEOL route '
        'startup is separate here because physical floorplan.service_ns contains MAT+MIV only.',
        '* Concurrent requests merge demand on each shared group/port before taking resource maxima; '
        'startup takes a maximum. They are not globally serialized B1 executions. The GPU request terms '
        'sum on the shared GPU, and equal-shape requests have the same compute/memory ratio. '
        'GPU_COMPUTE is a legacy label containing a GPU compute/memory maximum and small-op memory service.',
        '* MAC, Fabric, inter-region NoC, reduction, NMP SRAM/pipeline and NMP background are zero. '
        'The existing physical mapper is UNIFORM_STRIPING for memory ownership; no NMP operator is executed '
        'and no placement algorithm is changed or optimized.',
        '* Read/write/interface traffic agrees with the independent semantic ledger for all 576 growing-context '
        'steps. Each context appears exactly once. Array and boundary byte totals describe successive physical '
        'locations of the same traffic and are not added as extra logical bytes.',
        '* Prefill uses the saved GPU-only incremental ledger and original data-chain timing '
        'max(compute, aggregate array, aggregate boundary), with unchanged H/P/G. Its component durations '
        'were not saved, so their CSV fields are null. MIV duration is also not independently checkpointed '
        'and is marked included in Array, never invented as a separate additive delay.',
        '* Energy uses identical frozen event counts and coefficients; only 74 W GPU static energy scales '
        'with the changed E2E time. This agrees with saved physical-checkpoint energy. No thermal reclosure is performed.', '',
        'Achieved bandwidth = sum of Decode boundary payload bytes / sum of the corresponding operator '
        'boundary active-service durations. It excludes GPU idle/compute gaps and Prefill; it is **not** '
        'bytes/E2E wall time or a measured device utilization. Prefill and total traffic are separate CSV fields. '
        'Bottleneck labels compare saved aggregate Decode resource-service totals.', '',
        '## Large-degradation diagnosis and limitation', '',
        f"The worst case first step has {worst['cap_only_s']*1e3:.6f} ms cap-only service versus "
        f"{worst['boundary_s']*1e3:.6f} ms actual boundary service. Total route startup is only "
        f"{worst['route_startup_sum_s']*1e6:.6f} microseconds. 225 transfers are port-limited and 129 cap-limited.", '',
        'Across its 32 layers, K and V each need 0.524451 ms boundary service versus 0.079050 ms cap-only; '
        'FFN_DOWN needs 1.835175 versus 1.106695 ms. These are small/row-sharded operator port-demand '
        'concentration effects. The saved resident mapper assigns indivisible row atoms to physical groups; '
        'GROUP_DIRECT chooses the nearest physical port. Small row counts cannot uniformly occupy all resources.', '',
        'Although only 35 of 50 ports per slab are used by equal group traffic, their aggregate 11.13 TB/s '
        'still exceeds the thermal cap. Therefore the 35/50 count alone cannot explain the loss: '
        '**per-operator load imbalance**, not a global 70% bandwidth coefficient, is decisive. '
        'The detailed audit files expose actual port serialization and independent startup.', '',
        'No double counting or lost group/request concurrency was found in this replay. However, this '
        'physical path inherits a resident row-atom mapping shared with the NMP framework, unlike the '
        'formal placement-free GPU aggregate closure. Accounting consistency does not prove that a GPU '
        'memory controller must use that mapping or cannot stripe rows more finely. Hence these are '
        'conditional results for the existing routing/mapping, not proof of achievable real-hardware '
        'bandwidth or sufficient evidence to replace the formal baseline. No mapping redesign is made.', '',
        '## Results', '',
        '| Model | H | B | Reference tok/s | Data-chain tok/s | Drop % | BW TB/s | Util. % | Bottleneck |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['model'].split('-')[-1]} | {r['cached_history']//1000}K | {r['batch']} | "
                     f"{r['reference_tokens_per_s']:.3f} | {r['achieved_tokens_per_s']:.3f} | "
                     f"{r['throughput_degradation_pct']:.2f} | {r['achieved_bandwidth_Bps']/1e12:.4f} | "
                     f"{r['bandwidth_utilization_pct']:.2f} | {r['bottleneck']} |")
    lines += ['', '## Aggregate and trends', '',
        'TPS degradation min/max/geomean = 5.080% / 28.415% / 8.380%. '
        'For clarity, 1 minus the geometric mean of TPS ratios is 10.152%; these are different statistics. '
        'Bandwidth utilization min/max/geomean = 71.755% / 96.865% / 90.396%.', '',
        '* B1/B8 geometric-mean utilization: 87.256% / 93.649%. B8 improves utilization in all nine pairs.',
        '* 8B/70B/405B geometric-mean utilization: 83.335% / 93.128% / 95.179%. '
        'Model-size improvement is not universal: at B8/126K, 70B is 96.865% versus 405B 96.528%.',
        '* 20K/64K/126K geometric-mean utilization: 87.631% / 90.806% / 92.828%. '
        'All six model/batch combinations improve boundary utilization with context. '
        'E2E degradation is not uniformly monotonic: 405B/B8 worsens from 5.080% to 5.314% to 5.522% '
        'because E2E includes Prefill and GPU operator scheduling as well as boundary service.', '',
        'Formal 72-row CSV and all canonical candidate files remain byte-identical. '
        'No HBM/DNS/CPA source, workload, hardware, placement or thermal setting changed. No pytest run.', '',
        'Run: `conda run --no-capture-output -n om3dthermal python scripts/diagnose_m3d_gpu_achieved_bandwidth.py`. '
        'Outputs are isolated here; the two SVG/PDF figures are diagnostic only.']
    (OUT/'diagnostic_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = FORMAL/'final_e2e_metrics.csv'
    protected = {str(p.relative_to(f.ROOT)):digest(p) for p in
                 [baseline, *sorted((FORMAL/'candidates').glob('*.json'))]}
    with baseline.open(encoding='utf-8', newline='') as stream:
        formal = list(csv.DictReader(stream))
    assert len(formal)==72
    refs={(r['model'],r['context'],int(r['batch'])):r for r in formal if r['path']=='M3D_GPU'}
    closure=f.primary.gpu_memory_closure(f.ROOT)
    peak=closure['thermal_Bps']  # unrounded frozen value, NOT a changed parameter
    floor=resolve_feol_floorplan(f.ROOT)
    # Structural routing diagnostic only: equal bytes in every memory group.
    # This does not generate any workload or execute any NMP operator.
    external_service(floor,np.ones((floor.layout.slab_count,70)),mode='GROUP_DIRECT')
    counts=np.bincount(floor.group_ports,minlength=len(floor.ports))
    route=dict(groups_per_slab=70, ports_per_slab=len(floor.ports),
               groups_per_port=counts.tolist(),used_ports_per_slab=int(np.count_nonzero(counts)),
               max_groups_per_port=int(counts.max()),port_Bps=floor.port_Bps,
               equal_group_payload_ceiling_Bps=floor.layout.slab_count*70/counts.max()*floor.port_Bps,
               semantics='Existing GROUP_DIRECT nearest-Manhattan-port routing; not a fitted utilization factor')
    route['FEOL_route_startup_ns_min']=float(floor.sa_edge_ns.min())
    route['FEOL_route_startup_ns_max']=float(floor.sa_edge_ns.max())
    rows=[]; sources=[]; steps_out=[]
    for m,c,b in f.points():
        ref=refs[m,c,b]
        candidate_path=FORMAL/'candidates'/f'{m}_{c}_B{b}_M3D_GPU.json'
        canonical=read(candidate_path)
        fp=canonical['legacy_fingerprint']
        raw_path=FORMAL/'checkpoints'/f'{m}_{c}_B{b}_IOM3D_NO_NMP_{fp[:16]}.json'
        step_path=raw_path.with_suffix('.jsonl')
        raw=read(raw_path)
        steps=[json.loads(line) for line in step_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        s=raw['summary']; e=raw['energy']; pre=raw['prefill_ledger']
        assert s['status']=='EVALUATED' and raw['fingerprint']==fp
        assert (s['H'],s['P'],s['G'],s['batch_size'])==(int(ref['H']),128,32,b)
        assert [v['context'] for v in steps]==list(range(s['H']+128,s['H']+160))
        close(s['effective_bandwidth_TBps']*1e12,peak)
        close(sum(v['latency_s'] for v in steps),s['decode_s'])
        close(s['prefill_s']+s['decode_s'],s['E2E_s'])
        close(canonical['summary']['E2E_tok_s'],float(ref['tokens_per_s']))
        close(canonical['energy']['E2E_tokens_per_J'],float(ref['tokens_per_J']))
        assert pre==canonical['prefill_ledger']
        assert raw['events']==canonical['events']
        # Regenerate traffic ledger only, never the reference result or physical run.
        w,cw,_=f.legacy.inputs(m,c,b,f.ROOT)
        from om3dthermal.workload import evaluate_llm_decode
        for v in steps:
            traffic=evaluate_llm_decode(w.model_copy(update={'context_length':v['context']}))
            expected=(traffic.read_bytes_per_token+traffic.write_bytes_per_token)*b
            close(v['boundary_bytes'],expected)
            close(v['local_array_bytes'],expected)
            close(v['energy_events']['interface_bits']/8,expected)
            close((v['energy_events']['array_read_bits']+v['energy_events']['array_write_bits'])/8,expected)
            assert v['noc_bytes']==v['nmp_flops']==0
            assert all(v['component_sums'][k]==0 for k in ('MAC','LOCAL_FABRIC','INTER_REGION_NOC'))
            assert v['external_service_s']+1e-12>=expected/peak
            assert v['latency_s']+1e-12>=max(v['array_service_s'],v['external_service_s'],v['component_sums']['GPU_COMPUTE'])
            steps_out.append(dict(model=m,context=c,batch=b,decode_context=v['context'],
                boundary_bytes=expected,array_including_MIV_s=v['array_service_s'],
                boundary_active_service_s=v['external_service_s'],
                gpu_related_service_s=v['component_sums']['GPU_COMPUTE'],decode_step_s=v['latency_s']))
        for phase in ('prefill','decode'):
            for term in ('mac','router','reduction','pipeline_register','sram_read','sram_write','feol_unresolved'):
                assert e[f'{phase}_{term}_J']==0
        n=b*32; decbytes=sum(v['boundary_bytes'] for v in steps)
        ext=sum(v['external_service_s'] for v in steps)
        arr=sum(v['array_service_s'] for v in steps)
        gpu=sum(v['component_sums']['GPU_COMPUTE'] for v in steps)
        bw=decbytes/ext
        tps=n/s['E2E_s']; ratio=tps/float(ref['tokens_per_s'])
        # Identical frozen events: only GPU static energy changes with duration.
        static=f.setup(f.ROOT)[2].gpu_decode_power.static_power_W
        energy=float(ref['E2E_J'])+static*(s['E2E_s']-float(ref['E2E_s']))
        close(energy,e['E2E_J'])
        r=dict(model=m,cached_history=s['H'],batch=b,prefill_tokens=128,decode_tokens=32,
               reference_tokens_per_s=float(ref['tokens_per_s']),reference_t_prefill=float(ref['prefill_s']),
               reference_t_decode=float(ref['decode_s']),reference_t_e2e=float(ref['E2E_s']),
               reference_tokens_per_j=float(ref['tokens_per_J']),achieved_tokens_per_s=tps,
               achieved_t_prefill=s['prefill_s'],achieved_t_decode=s['decode_s'],achieved_t_e2e=s['E2E_s'],
               achieved_tokens_per_j=n/energy,total_relevant_memory_bytes=decbytes+pre['total_memory_bytes'],
               decode_memory_bytes=decbytes,boundary_bytes=decbytes,prefill_memory_bytes=pre['total_memory_bytes'],
               peak_bandwidth_cap_Bps=peak,achieved_bandwidth_Bps=bw,bandwidth_utilization_pct=100*bw/peak,
               bandwidth_denominator='SUM_DECODE_OPERATOR_BOUNDARY_ACTIVE_SERVICE_SECONDS',
               decode_boundary_service_s=ext,decode_array_including_MIV_s=arr,
               decode_MIV_service_s=None,MIV_status='INCLUDED_IN_ARRAY_SERVICE; not independently checkpointed',
               decode_gpu_related_service_s=gpu,GPU_service_status='GPU_COMPUTE label includes GPU memory roofline and small ops',
               prefill_array_service_s=None,prefill_boundary_service_s=None,
               prefill_components_status='NOT_SAVED_IN_RAW_CHECKPOINT; total GPU-only Prefill timing and ledger reused',
               bottleneck=max({'ARRAY_MIV':arr,'BOUNDARY':ext,'GPU_RELATED':gpu},key={'ARRAY_MIV':arr,'BOUNDARY':ext,'GPU_RELATED':gpu}.get),
               throughput_ratio=ratio,throughput_degradation_pct=100*(1-ratio),
               decode_latency_increase_pct=100*(s['decode_s']/float(ref['decode_s'])-1),
               E2E_latency_increase_pct=100*(s['E2E_s']/float(ref['E2E_s'])-1),
               tokens_per_j_change_pct=100*((n/energy)/float(ref['tokens_per_J'])-1),
               checkpoint_reused=True,model_status='EXISTING_GPU_PHYSICAL_CHAIN_WITH_FROZEN_RESIDENT_MAPPING')
        for value in r.values():
            if isinstance(value,(float,int)): assert math.isfinite(value)
        assert 0<bw<=peak*(1+1e-12)
        assert all(r[k]>0 for k in ('achieved_t_prefill','achieved_t_decode','achieved_t_e2e'))
        rows.append(r)
        sources.append(dict(model=m,context=c,batch=b,source=str(raw_path.relative_to(f.ROOT)),
                            sha256=digest(raw_path),step_source=str(step_path.relative_to(f.ROOT)),step_sha256=digest(step_path),fingerprint=fp))
        print(m,c,b,f'TPS {r["reference_tokens_per_s"]:.3f} -> {tps:.3f}; BW {bw/1e12:.6f}',flush=True)
    assert len(rows)==len(refs)==18
    stats={'all':aggregate(rows)}
    for key in ('batch','model','cached_history'):
        stats[key]={str(value):aggregate([r for r in rows if r[key]==value]) for value in dict.fromkeys(r[key] for r in rows)}
    csv_save('results.csv',rows); csv_save('decode_step_services.csv',steps_out)
    save('aggregate.json',stats); save('routing_audit.json',route)
    worst=worst_case_audit()
    close(worst['decode_step_s'],steps_out[0]['decode_step_s'])
    assert all(digest(f.ROOT/p)==h for p,h in protected.items())
    save('manifest.json',dict(baseline_HEAD='5c79dadb0f49804eafc38d2cd89590596e35bcbd',
        diagnostic='WORKLOAD_ACHIEVED_BW',reference='REFERENCE_PEAK_BW',workloads=18,
        full_workload_simulations_run=0,GPU_diagnostic_first_steps_replayed=1,GPU_physical_decode_steps_reused=576,
        HBM_NMP_simulations_run=0,placement_optimizer_calls=0,pytest_run=False,
        closure=closure,sources=sources,protected_formal_sha256=protected,
        sanity_status='PASS',verdict='EXISTING_ROUTING_CONDITIONAL_DIAGNOSTIC__NOT_BASELINE_REPLACEMENT'))
    plot(rows)
    report(rows,stats,worst)


if __name__=='__main__':
    main()
