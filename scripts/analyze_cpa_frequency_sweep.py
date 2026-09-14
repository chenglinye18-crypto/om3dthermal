"""Lightweight completeness, frozen-resource and response analysis; no pytest."""
import csv
import json
import math
from collections import Counter
import numpy as np
from run_cpa_frequency_sweep import BASE, OUT, CACHE, FREQUENCIES, baseline_rows, read, key, save


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    response, optimum, anomalies = [], [], []
    assert len(list((OUT / 'points').glob('*.json'))) == 90
    assert len(list((OUT / 'thermal').glob('*.json'))) == 90
    original = baseline_rows()
    assert len(original) == 18
    save(OUT / 'placement_rebuild_manifest.json', [read(CACHE / key(r['model'], r['context'], int(r['batch'])) / 'ready.json') for r in original])
    for base in original:
        m, c, b = base['model'], base['context'], int(base['batch'])
        tag = key(m, c, b)
        candidate = read(BASE / 'candidates' / f'{tag}_M3D_NMP_CPA.json')
        row = dict(model=m, context=c, cached_history=int(base['H']), batch=b,
            prefill_tokens=int(base['P']), decode_tokens=int(base['G']), policy='M3D_NMP_CPA',
            feol_frequency_ghz=1., t_prefill=float(base['prefill_s']), t_decode=float(base['decode_s']),
            t_e2e=float(base['E2E_s']), tokens_per_s=float(base['tokens_per_s']),
            e_prefill=candidate['energy']['prefill_J'], e_decode=candidate['energy']['decode_J'],
            e_e2e=float(base['E2E_J']), tokens_per_j=float(base['tokens_per_J']),
            tmax_c=float(base['Tmax_C']), thermal_feasible=float(base['Tmax_C']) <= 85,
            source='FROZEN_31c9f324_BASELINE', dominant_bottleneck=max(candidate['component_sums'], key=candidate['component_sums'].get),
            baseline_bottleneck_status='LARGEST_SERVICE_SUM; STAGE_CLASSIFICATION_NOT_STORED',
            **{k+'_service_s': v for k,v in candidate['component_sums'].items()})
        group = [row]
        raw = []
        for ghz in FREQUENCIES:
            name = f'{tag}_f{ghz:g}.json'
            x = read(OUT / 'points' / name)
            t = read(OUT / 'thermal' / name)
            raw.append(x)
            r = {k:v for k,v in x.items() if not isinstance(v, (dict, list))}
            r.update(tmax_c=t['Tmax_C'], thermal_feasible=t['thermal_feasible'],
                source='NEW_PHYSICAL_FREQUENCY_EXECUTION',
                reduction_service_status='LOCAL_REDUCTION_ONLY; INTER_REGION_REDUCTION_INCLUDED_IN_NOC',
                **{k+'_service_s':v for k,v in x['component_sums'].items()})
            group.append(r)
        # Required frozen demand/service checks across new frequencies only.
        # The user explicitly waived comparison to the old reconstructed plan.
        for x in raw:
            assert x['t_prefill'] == row['t_prefill'] and x['e_prefill'] == row['e_prefill']
            assert x['hardware']['external_Bps'] == 3.4e12
            assert x['hardware']['fabric_Bps'] == 64e9*x['feol_frequency_ghz']
            assert x['hardware']['noc_Bps'] == 32e9*x['feol_frequency_ghz']
            assert x['hardware']['tile_flops']/x['feol_frequency_ghz'] == raw[0]['hardware']['tile_flops']/raw[0]['feol_frequency_ghz']
            for link, reference in zip(x['hardware']['noc_links'], raw[0]['hardware']['noc_links']):
                assert link['rc_ns'] == reference['rc_ns']
                assert link['wire_pipeline_cycles'] == reference['wire_pipeline_cycles']
            for field in ('ARRAY', 'EXTERNAL_BOUNDARY', 'GPU_COMPUTE'):
                assert x['component_sums'][field] == raw[0]['component_sums'][field]
            for field in ('local_array_bytes', 'boundary_bytes'):
                assert x[field] == raw[0][field]
            for name,value in x['energy_components'].items():
                if name not in ('gpu_static_J', 'feol_unresolved_J'):
                    np.testing.assert_allclose(value, raw[0]['energy_components'][name], rtol=1e-12, atol=1e-7)
        for r in group:
            for field in ('t_prefill','t_decode','t_e2e','tokens_per_s','e_prefill','e_decode','e_e2e','tokens_per_j','tmax_c'):
                assert math.isfinite(r[field]) and r[field] > 0, (tag, field)
            r['normalized_throughput'] = r['tokens_per_s']/row['tokens_per_s']
            r['normalized_energy_efficiency'] = r['tokens_per_j']/row['tokens_per_j']
        for a,z in zip(group, group[1:]):
            for metric in ('tokens_per_s','tmax_c'):
                if z[metric] < a[metric]:
                    anomalies.append(dict(workload=tag, metric=metric, from_ghz=a['feol_frequency_ghz'],
                        to_ghz=z['feol_frequency_ghz'], previous=a[metric], current=z[metric],
                        note='REPORTED_UNMODIFIED; inspect service and power rows; baseline uses original placement'))
        best = max((r for r in group if r['thermal_feasible']), key=lambda r:(r['tokens_per_s'], -r['feol_frequency_ghz']))
        bad = [r['feol_frequency_ghz'] for r in group if not r['thermal_feasible']]
        optimum.append(dict(model=m, context=c, batch=b, f_opt_measured_ghz=best['feol_frequency_ghz'],
            tps_1ghz=row['tokens_per_s'], tps_opt=best['tokens_per_s'],
            throughput_headroom_ratio=best['normalized_throughput'],
            throughput_headroom_percent=100*(best['normalized_throughput']-1),
            tokens_j_1ghz=row['tokens_per_j'], tokens_j_opt=best['tokens_per_j'],
            tokens_j_ratio=best['normalized_energy_efficiency'],
            tokens_j_change_percent=100*(best['normalized_energy_efficiency']-1),
            tmax_opt=best['tmax_c'], first_infeasible_ghz=min(bad) if bad else '',
            thermal_limited=bool(bad), dominant_bottleneck=best['dominant_bottleneck'],
            increment_2p5_to_3ghz_percent=100*(group[-1]['tokens_per_s']/group[-2]['tokens_per_s']-1),
            bottleneck_3ghz=group[-1]['dominant_bottleneck']))
        response.extend(group)
    assert len(response) == 108
    write_csv('frequency_response.csv', response)
    write_csv('measured_optimum.csv', optimum)
    save(OUT / 'nonmonotonic_diagnostics.json', anomalies)
    stats = {}
    for field in ('throughput_headroom_ratio','tokens_j_ratio'):
        values = [r[field] for r in optimum]
        stats[field] = dict(min=min(values), max=max(values), geomean=math.exp(sum(map(math.log, values))/len(values)))
    stats['f_opt_distribution'] = dict(Counter(r['f_opt_measured_ghz'] for r in optimum))
    stats['new_runs_complete'] = 90
    stats['total_points'] = 108
    stats['nonmonotonic_transitions'] = len(anomalies)
    crossings = [r['first_infeasible_ghz'] for r in optimum if r['thermal_limited']]
    stats['earliest_measured_thermal_crossing_ghz'] = min(crossings) if crossings else None
    stats['earliest_crossing_workloads'] = [key(r['model'], r['context'], r['batch']) for r in optimum
        if r['first_infeasible_ghz'] == stats['earliest_measured_thermal_crossing_ghz']]
    stats['unconstrained_3ghz_increment_percent'] = {key(r['model'],r['context'],r['batch']):r['increment_2p5_to_3ghz_percent']
        for r in optimum if not r['thermal_limited']}
    save(OUT / 'analysis_summary.json', stats)
    save(OUT / 'sanity_checks.json', dict(status='PASS', new_runs=90, total_points=108,
        baseline_unchanged=True, frozen_array_boundary_gpu=True, fixed_dynamic_energy=True,
        pytest_run=False, baseline_placement_audit_compared=False))
    manifest = read(OUT / 'manifest.json')
    manifest.update(completed_new_runs=90, total_frequency_response_points=108,
        baseline_unchanged=True, sanity_checks='PASS', nonmonotonic_transitions=len(anomalies))
    save(OUT / 'manifest.json', manifest)
    lines = ['# CPA FEOL frequency sensitivity', '',
        'Constant-voltage, measured discrete frequencies; no voltage or new clock/leakage model.',
        '18 original 1 GHz CPA rows reused verbatim. 90 new physical executions; all 32 Decode contexts evaluated.',
        'One CPA placement rebuilt at 1 GHz per workload and reused at all five new frequencies. Old placement audit comparison waived by the user.',
        'Prefill latency/energy reused. Thermal uses Decode-only per-die BEOL uniform power plus GPU FEOL power, with the existing FP64 GPU-PCG operator.',
        'MAC/Fabric/NoC/Reduction scale with clock. Array/MIV/GPU/external 3.4 TB/s and energy coefficients stay fixed.',
        'NoC hop = fixed RC + (baseline hop - RC)/frequency ratio. Baseline pipeline register count and routing are retained; no new pipeline hardware.',
        'This is model frequency sensitivity, not a demonstrated silicon timing/voltage operating guarantee.', '',
        '| Model | Context | B | Best GHz | TPS gain % | Tmax °C | tokens/J change % | First infeasible GHz | Bottleneck |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in optimum:
        lines.append(f"| {r['model']} | {r['context']} | {r['batch']} | {r['f_opt_measured_ghz']:g} | {r['throughput_headroom_percent']:.3f} | {r['tmax_opt']:.3f} | {r['tokens_j_change_percent']:.3f} | {r['first_infeasible_ghz']} | {r['dominant_bottleneck']} |")
    lines += ['', 'Aggregate statistics:', '```json', json.dumps(stats, indent=2), '```', '',
        'Service sums are resource diagnostics, not additive components of end-to-end latency: Array/Fabric/MAC overlap through max().',
        'For reused 1 GHz rows, bottleneck labels use the largest stored service sum; new points use the existing per-stage bottleneck time classification.',
        'MIV is embedded in Array group service; reduction_service_s separates local reduction from stored INTER_REGION_NOC. No invented utilization metric.',
        'All infeasible samples retained. No saturation threshold or crossing interpolation applied.',
        f'Nonmonotonic transitions: {len(anomalies)}; see nonmonotonic_diagnostics.json (values are not corrected).',
        '', 'Reproduce/resume in the om3dthermal Conda environment:', '```powershell',
        'python scripts/run_cpa_frequency_sweep.py --cases 2 --workers 5',
        'python scripts/run_cpa_frequency_sweep.py --thermal',
        'python scripts/analyze_cpa_frequency_sweep.py', '```',
        'Run thermal after performance to avoid concurrent operator/placement memory pressure. Completed points are reused.']
    (OUT / 'frequency_sweep_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    from plot_cpa_frequency_response import main as plot
    plot()
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
