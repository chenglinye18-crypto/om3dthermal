"""Select complete thermally feasible frequency rows without retuning the model."""
from collections import Counter
import csv
import json
import math
import numpy as np
from run_cpa_frequency_sweep_v3 import BASE, OUT, ROOT, FREQUENCIES, POINTS, baseline, key, read, save, sha, validate_baseline


def select_feasible(rows):
    return max((r for r in rows if r['thermal_feasible']),
               key=lambda r:(r['tokens_per_s'], -r['feol_frequency_ghz']))


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w', encoding='utf-8', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    validate_baseline()
    response, optimum, anomalies = [], [], []
    for index,(m,c,b) in enumerate(POINTS):
        nominal=baseline(index)
        t=read(BASE/'thermal_rows'/f'{key(index)}_M3D_NMP_CPA.json')
        component={k:sum(s['component_sums'][k] for s in nominal['steps']) for k in nominal['steps'][0]['component_sums']}
        pre_j=sum(v for k,v in nominal['energy'].items() if k.startswith('prefill_'))
        base=dict(model=m,context=c,batch=b,H=nominal['H'],P=nominal['P'],G=nominal['G'],
            feol_frequency_ghz=1.,t_prefill=nominal['prefill']['latency_s'],t_decode=nominal['decode_s'],
            t_e2e=nominal['E2E_s'],e_prefill=pre_j,e_decode=nominal['E2E_J']-pre_j,e_e2e=nominal['E2E_J'],
            tokens_per_s=nominal['tokens_per_s'],tokens_per_j=nominal['tokens_per_J'],
            tmax_c=t['Tmax_C'],thermal_feasible=t['Tmax_C']<=85,
            dominant_bottleneck=max(component,key=component.get),
            bottleneck_definition='LARGEST_SERVICE_SUM; NOMINAL_STAGE_CLASSIFICATION_NOT_STORED',
            source='FROZEN_V3_NOMINAL',**{k+'_service_s':v for k,v in component.items()})
        group=[base]; raw=[]
        for frequency in FREQUENCIES:
            name=f'{key(index)}_f{frequency:g}.json'
            x=read(OUT/'points'/name); thermal=read(OUT/'thermal'/name)
            assert x['t_prefill']==base['t_prefill'] and x['e_prefill']==base['e_prefill']
            assert x['hardware']['external_Bps']==3.4e12
            assert x['hardware']['fabric_Bps']==64e9*frequency
            assert x['hardware']['noc_Bps']==32e9*frequency
            np.testing.assert_allclose(sum(x['energy_components'].values()),x['e_decode'],rtol=1e-12,atol=1e-7)
            assert math.isclose(x['tokens_per_s']*x['t_e2e'],b*32,rel_tol=1e-12)
            assert math.isclose(x['tokens_per_j']*x['e_e2e'],b*32,rel_tol=1e-12)
            raw.append(x)
            row={k:v for k,v in x.items() if not isinstance(v,(dict,list))}
            row.update(tmax_c=thermal['Tmax_C'],thermal_feasible=thermal['thermal_feasible'],
                source='NEW_PHYSICAL_EXECUTION',bottleneck_definition='EXISTING_PER_STAGE_BOTTLENECK_TIME',
                **{k+'_service_s':v for k,v in x['component_sums'].items()})
            group.append(row)
        for x in raw:
            for field in ('ARRAY','EXTERNAL_BOUNDARY','GPU_COMPUTE'):
                assert x['component_sums'][field]==raw[0]['component_sums'][field]
            for field in ('local_array_bytes','boundary_bytes'):
                assert x[field]==raw[0][field]
            assert x['hardware']['tile_flops']/x['feol_frequency_ghz']==raw[0]['hardware']['tile_flops']/raw[0]['feol_frequency_ghz']
            for link,old in zip(x['hardware']['noc_links'],raw[0]['hardware']['noc_links']):
                assert link['rc_ns']==old['rc_ns'] and link['wire_pipeline_cycles']==old['wire_pipeline_cycles']
            for name,value in x['energy_components'].items():
                if name not in ('gpu_static_J','feol_unresolved_J'):
                    np.testing.assert_allclose(value,raw[0]['energy_components'][name],rtol=1e-12,atol=1e-7)
        for row in group:
            for field in ('tokens_per_s','tokens_per_j','tmax_c','t_e2e','e_e2e'):
                assert math.isfinite(row[field]) and row[field]>0
            row['normalized_throughput']=row['tokens_per_s']/base['tokens_per_s']
            row['normalized_energy_efficiency']=row['tokens_per_j']/base['tokens_per_j']
        for a,z in zip(group,group[1:]):
            for metric in ('tokens_per_s','tmax_c'):
                if z[metric]<a[metric]:
                    anomalies.append(dict(workload=key(index),metric=metric,from_ghz=a['feol_frequency_ghz'],
                        to_ghz=z['feol_frequency_ghz'],previous=a[metric],current=z[metric]))
        best=select_feasible(group)
        bad=[r['feol_frequency_ghz'] for r in group if not r['thermal_feasible']]
        optimum.append(dict(model=m,context=c,batch=b,f_opt_measured_ghz=best['feol_frequency_ghz'],
            tps_1ghz=base['tokens_per_s'],tps_opt=best['tokens_per_s'],
            throughput_headroom_ratio=best['normalized_throughput'],
            throughput_headroom_percent=100*(best['normalized_throughput']-1),
            tokens_j_1ghz=base['tokens_per_j'],tokens_j_opt=best['tokens_per_j'],
            tokens_j_ratio=best['normalized_energy_efficiency'],
            tokens_j_change_percent=100*(best['normalized_energy_efficiency']-1),
            tmax_opt=best['tmax_c'],dominant_bottleneck=best['dominant_bottleneck'],
            bottleneck_definition=best['bottleneck_definition'],first_infeasible_ghz=min(bad) if bad else '',
            thermal_limited=bool(bad)))
        response.extend(group)
    assert len(response)==108 and len(optimum)==18
    preserved=read(OUT/'preservation_sha256.json')
    changed=[p for p,h in preserved.items() if sha(ROOT/p)!=h]
    assert not changed, changed
    save(OUT/'preservation_check.json',dict(files=len(preserved),changed=changed,byte_identical=True))
    stats={}
    for label in ('Llama-3.1-8B','Qwen2.5-32B','ALL'):
        subset=[r for r in optimum if label=='ALL' or r['model']==label]
        stats[label]=dict(cases=len(subset),throughput_headroom_geomean=math.exp(sum(math.log(r['throughput_headroom_ratio']) for r in subset)/len(subset)),
            energy_efficiency_geomean=math.exp(sum(math.log(r['tokens_j_ratio']) for r in subset)/len(subset)),
            selected_frequency_distribution=dict(Counter(r['f_opt_measured_ghz'] for r in subset)))
    stats.update(total_points=108,new_runs=90,infeasible_points=sum(not r['thermal_feasible'] for r in response),
        nonmonotonic_transitions=len(anomalies))
    write_csv('frequency_response.csv',response);write_csv('measured_optimum.csv',optimum)
    save(OUT/'analysis_summary.json',stats);save(OUT/'nonmonotonic_diagnostics.json',anomalies)
    save(OUT/'sanity_checks.json',dict(status='PASS',new_runs=90,nominal_rows=18,
        baseline_and_legacy_byte_identical=True,prefill_reused=True,fixed_array_boundary_gpu=True,
        fixed_dynamic_event_energy=True,traffic_conservation=True,energy_conservation=True))
    manifest=read(OUT/'manifest.json');manifest.update(completed_new_runs=90,total_points=108,sanity_checks='PASS')
    save(OUT/'manifest.json',manifest)
    lines=['# CPA FEOL frequency sweep on formal long-context v3','',
        'Models: Llama-3.1-8B / Qwen2.5-32B; H=20000/64000/126000; P=128; G=32; B=1/8/32.',
        '18 frozen nominal v3 CPA rows and 90 new physical frequency executions. Old sweep values are not reused.',
        'The existing set_frequency implementation is reused unchanged: 1.25/1.5/2/2.5/3 GHz at constant voltage. Each workload has one rebuilt 1 GHz CPA placement, frozen across frequency; its nominal first physical step must match v3.',
        'MAC, Fabric, NoC and reduction retain the existing clock scaling; array/MIV, GPU and external 3.4 TB/s are fixed. NoC hop = fixed RC + (baseline hop - RC)/frequency ratio. Routing and pipeline register counts stay fixed.',
        'Prefill time/energy is copied exactly from v3. All 32 growing Decode contexts are evaluated at every new frequency. Physical per-step/per-die events are recorded directly (including Qwen GQA), avoiding unsupported spatial projection rounding; energy equations are unchanged.',
        'Every new frequency point gets a fresh Decode steady-state solve using the canonical FP64 GPU-PCG operator and die-grouped uniform BEOL + GPU FEOL power. No duty-cycle averaging. Tmax >85 C is infeasible; all infeasible samples are retained.',
        'The selected complete row maximizes throughput among sampled thermally feasible frequencies, including nominal 1 GHz; ties prefer lower frequency. Energy is not selected independently. No crossing interpolation or extrapolation.',
        'This is constant-voltage model sensitivity, not demonstrated silicon timing/voltage closure at the selected GHz.',
        'HBM_HOST_OFFLOAD and HBM_RESIDENT_WAVE both remain in the unchanged v3 benchmark. HBM_BEST still chooses one complete policy by E2E throughput; no unnecessary host traffic or waves when the batch fits; infeasible wave retains offload.',
        '', '| Model | H | B | GHz | tok/s | Gain % | tok/J | Energy-efficiency change % | Tmax C | Bottleneck |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in optimum:
        lines.append(f"| {r['model']} | {r['context']} | {r['batch']} | {r['f_opt_measured_ghz']:g} | {r['tps_opt']:.3f} | {r['throughput_headroom_percent']:.3f} | {r['tokens_j_opt']:.4f} | {r['tokens_j_change_percent']:.3f} | {r['tmax_opt']:.3f} | {r['dominant_bottleneck']} |")
    lines+=['','```json',json.dumps(stats,indent=2),'```','',
        'Service sums overlap and are not additive E2E latency. Nominal bottleneck labels use largest stored service sum; new samples use existing per-stage bottleneck time classifications.',
        'Nominal validation: 6 targeted tests passed; the preceding related suites had 657 passed, 11 pre-existing failures and 8 skipped. The missing legacy files/old assertions were not changed to conceal failures.',
        'Frequency-sweep targeted validation: 3 passed, 0 failed in 0.79 s (targeted_tests.xml). Each newly executed point also checks physical-event, traffic and energy conservation.',
        '', 'Run in the native Windows Conda environment `om3dthermal`:', '```powershell',
        'python scripts/run_cpa_frequency_sweep_v3.py --cases 3 --workers 2',
        'python scripts/run_cpa_frequency_sweep_v3.py --thermal',
        'python scripts/analyze_cpa_frequency_sweep_v3.py',
        'python scripts/plot_cpa_frequency_response_v3.py','```',
        'Run thermal after performance to avoid memory pressure. Persistent placement caches are on F: and are not committed; completed result files are reused.']
    (OUT/'frequency_sweep_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
