"""Summarize completed GPU-only placement diagnostics and archive telemetry."""
import gzip
import json
import math
from pathlib import Path
from diagnose_m3d_gpu_port_balanced import OUT, read, loadcsv, save, sha, f


def main():
    rows=loadcsv(OUT/'comparison.csv');a=read(OUT/'aggregate.json')
    ports=loadcsv(OUT/'port_comparison.csv');worst=loadcsv(OUT/'worst_case_operator_attribution.csv')
    regressions=a['regressions']
    improved=sum(float(r['TPS_gain_pct'])>0 for r in rows)
    verdict=('PORT_BALANCED_PLACEMENT_EFFECTIVE' if improved==18 and not regressions
             else 'PARTIAL_PORT_BALANCE_IMPROVEMENT' if improved else 'PLACEMENT_NOT_PRIMARY_LIMIT')
    lines=['# M3D-GPU port-balanced placement diagnostic','','Status: **DIAGNOSTIC_ONLY**. Verdict: **'+verdict+'**.','',
        '## Method','',
        'The old policy is unchanged UNIFORM_STRIPING: lane = group × 318 + slab, die-fastest cyclic '
        'ownership, GROUP_DIRECT nearest-Manhattan external port routing. The new GPU_PORT_BALANCED '
        'policy starts with that exact capacity-legal allocation and applies one bijective permutation '
        'to physical group/slab lanes. Consecutive logical atoms first visit distinct reachable ports, '
        'then slabs, then the second group sharing each port. Equal-load group choices prefer shorter '
        'existing FEOL startup. Each operator retains its logical cyclic atom sequence and allocation offset.', '',
        'The optimization is a deterministic port-first round robin, not a claim of globally optimal '
        'placement. Any short contiguous operator allocation now spreads over distinct (slab,port) '
        'resources instead of repeatedly using a small set of groups. The global bijection preserves '
        'every slot occupancy exactly, so no capacity heuristic or atom splitting is needed. '
        'QK/AV pairing, row/KV atom granularity, total bytes and logical prefix/append identity remain intact.', '',
        'No service equation is changed: external = max(total bytes / thermal cap, '
        'max_port(port bytes / port_Bps + route startup)); GPU operator time = max(array, external, GPU). '
        'The exact frozen thermal cap is 3.3957835581187483 TB/s. Physical port rate, GPU compute and '
        'bandwidth, geometry, MAT/MIV, energy coefficients, NMP and CPA are unchanged. '
        'GPU_PORT_BALANCED rejects ATTENTION_NMP and MAC_NMP before execution. Tile IDs are inert '
        'legal bookkeeping, never an optimization target.', '',
        'All 18 existing data-chain Prefill totals and Prefill energy are reused exactly. Only Decode '
        'is replayed under the new placement. Old Uniform Decode is also replayed solely to fill the '
        'missing port telemetry; its timing matches the existing checkpoints. Old E2E reference values '
        'are read from the preceding diagnostic. No Prefill, HBM, NMP, CPA optimizer, or thermal benchmark '
        'run occurs. Related regression tests exercise the existing physical/NMP implementations.', '',
        'Optional energy observation was enabled for early completed cases, then disabled because '
        'this task requires performance/port telemetry only. Three unfinished energy-observed B8 '
        'attempts (70B/20K/B8, 405B/64K/B8, 405B/126K/B8) were stopped and restarted without that '
        'observer. Completed cases were reused. The observer does not enter any service equation; '
        'old replay timings match their original checkpoints in both modes. No energy comparison '
        'is claimed in the final table.', '',
        '## Root cause','',
        'Old small weight operators occupy consecutive die-fastest lanes, concentrating complete '
        'rows in a few physical groups and their nearest ports. The replay preserves the original '
        '28.41% worst-case loss. The new permutation changes ownership only. The 35 reachable '
        'GROUP_DIRECT read ports per slab are unchanged; the other 15 are not made reachable for reads '
        'by an invented route. KV writes retain existing REGION_DIRECT routing across region ports.', '',
        'Remaining small-operator limits are real within the atom definition: an 8B K/V projection '
        'has 1,024 indivisible 8,192-byte rows. Even assigning each row a distinct port needs at least '
        '8.192 microseconds at the frozen 1 GB/s port rate, while its aggregate thermal-cap time is '
        'only about 2.470 microseconds. Full utilization is therefore not a valid requirement for '
        'this operator without changing atom granularity, which this diagnostic does not do.', '',
        '## 18-case results','',
        '| Model | H | B | Reference TPS | Old TPS | New TPS | Old drop % | New drop % | Old util. % | New util. % | Gain pp |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+ ' | '.join([r['model'].split('-')[-1],r['context'].replace('LC',''),r['batch'],
            *[f'{float(r[k]):.3f}' for k in ('reference_tokens_per_s','old_tokens_per_s','new_tokens_per_s',
                'old_throughput_drop_pct','new_throughput_drop_pct','old_utilization_pct','new_utilization_pct','utilization_improvement_pp')]])+' |')
    lines+=['','All comparison fields, including old/new achieved TB/s and bottlenecks, '
        'are in comparison.csv. Achieved bandwidth is Decode boundary payload / summed corresponding '
        'active boundary service time; it excludes compute gaps and Prefill. Both E2E timings include '
        'the same reused Prefill.', '', '## Aggregate','',
        '| Statistic | Old | New |','|---|---:|---:|']
    for key in ('utilization_min','utilization_max','utilization_geomean','TPS_drop_min','TPS_drop_max','TPS_drop_geomean','drop_from_TPS_ratio_geomean'):
        lines.append(f"| {key} (%) | {a['old']['all'][key]:.4f} | {a['new']['all'][key]:.4f} |")
    for field in ('batch','model','context'):
        for key in a['old'][field]:
            lines.append(f"| Utilization geometric mean: {field}={key} (%) | "
                f"{a['old'][field][key]['utilization_geomean']:.4f} | {a['new'][field][key]['utilization_geomean']:.4f} |")
    lines+=['','The geometric mean of percentage drops differs from 1 minus the geometric mean of '
        'throughput ratios; both are reported explicitly.','', '## Port telemetry','',
        '| Metric across all Decode transfers | Old | New |','|---|---:|---:|']
    totals={}
    for label in ('old','new'):
        subset=[r for r in ports if r['placement']==label]
        totals[label]={k:sum(float(r[k]) for r in subset) for k in ('transfer_count','PORT_SERIALIZATION',
            'GLOBAL_THERMAL_CAP','ROUTE_STARTUP','boundary_s','cap_only_s','PORT_SERIALIZATION_boundary_s')}
        for k in ('active_ports','max_port_bytes','mean_active_port_bytes','max_mean_load_ratio','max_port_utilization','mean_active_port_utilization'):
            totals[label][k+'_transfer_mean']=sum(float(r[k]) for r in subset)/totals[label]['transfer_count']
    for key in totals['old']:
        lines.append(f"| {key} | {totals['old'][key]:.6f} | {totals['new'][key]:.6f} |")
    lines+=['','The transfer-averaged max/mean load ratio does not improve globally '
        '(1.160169 to 1.173107). It equally weights large reads and tiny appends and excludes '
        'inactive ports from each mean; it is not the optimized maximum completion time. '
        'The report therefore does not claim every fairness statistic improves. Actual boundary '
        'time and port-limited service time decrease, while the append tradeoff is shown below.']
    lines+=['','Port-count/load/utilization means are arithmetic means over individual nonempty '
        'transfers, not utilization of a hypothetical aggregated concurrent workload. Limiting-resource '
        'counts and boundary durations are sums. Per-case statistics are in port_comparison.csv, '
        'including phase-wide active-port unions and cumulative port-load ratios separately. '
        'Phase port utilization is that port’s cumulative bytes / port_Bps / Decode elapsed time; '
        'transfer means and whole-phase utilization are explicitly separate columns. '
        'Per-operator first-step atom counts, groups, active ports and peak/mean load ratios are in '
        'first_step_operator_ports.csv.gz. Full 32-step resident/service per-port distributions are in '
        'the compressed case telemetry.', '', '## Worst-case analysis: 8B / 20K / B1','',
        '| Operator, summed across 32 Decode steps and all layers | Old boundary ms | New boundary ms | Cap-only ms |',
        '|---|---:|---:|---:|']
    for r in worst:
        lines.append(f"| {r['operator']} | {float(r['old_boundary_s'])*1000:.6f} | "
            f"{float(r['new_boundary_s'])*1000:.6f} | {float(r['cap_only_s'])*1000:.6f} |")
    kv=next(r for r in worst if r['operator']=='KV_APPEND')
    lines+=['',f"KV_APPEND is a local tradeoff: its cumulative boundary time increases from "
        f"{float(kv['old_boundary_s'])*1000:.6f} to {float(kv['new_boundary_s'])*1000:.6f} ms. "
        'The same token/head atoms now occupy fewer slabs and a wider set of read ports; writes '
        'still use the unchanged REGION_DIRECT region-port striping, so a small append can have '
        'more bytes per active write port. This is retained and reported, not corrected away. '
        'The weight-read service savings exceed that small append penalty. The policy is not '
        'claimed to improve every individual transfer.']
    lines+=['','## Safety and verdict','',
        f'{improved}/18 workloads improve throughput; {len(regressions)} regressions. '
        'All 18 have legal slot/group/die capacity, conserved semantic traffic and no NMP stages. '
        'The physical-lane bijection also proves capacity conservation relative to the original '
        'Uniform allocation. Existing Uniform/Balanced/CPA placement signatures are checked against '
        'pre-change captures; the frozen Uniform execution tests cover unchanged NMP execution.', '',
        'Related placement, Decode, physical service and energy/traffic tests are recorded in targeted_tests.log. '
        'No full pytest is run. Protected formal files and the prior diagnostic are unchanged; see preservation.json.', '',
        '**'+verdict+'**. '+('The result supports considering this GPU-only mapping for a subsequent formal integration review. '
        'It is not automatically adopted: the formal GPU baseline and Prefill remain unchanged, '
        'and a future integration must explicitly decide how the GPU physical resident mapping becomes '
        'part of the architecture semantics.' if not regressions else
        'See aggregate.json for every regression before considering formal integration.'), '',
        'Reproduce: `conda run --no-capture-output -n om3dthermal python scripts/diagnose_m3d_gpu_port_balanced.py`; '
        'then `conda run --no-capture-output -n om3dthermal python scripts/report_m3d_gpu_port_balanced.py`.']
    (OUT/'diagnostic_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'verdict.json',dict(verdict=verdict,improved_cases=improved,regressions=len(regressions),port_totals=totals,
        old_HEAD='ea5aa97867214f9a73bff7563b683813a9f2a3c5',status='DIAGNOSTIC_ONLY'))
    sources=[f.ROOT/p for p in ('src/om3dthermal/placement/nmp_load_balance.py',
        'src/om3dthermal/placement/gpu_port_balanced.py','src/om3dthermal/serving/decode_policy.py',
        'src/om3dthermal/power/nmp_die_activity.py','src/om3dthermal/power/batched_physical.py')]
    save(OUT/'manifest.json',dict(baseline_HEAD='ea5aa97867214f9a73bff7563b683813a9f2a3c5',
        status='DIAGNOSTIC_ONLY',policy='GPU_PORT_BALANCED',models=3,contexts=3,batches=[1,8],P=128,G=32,
        completed_new_GPU_cases=18,completed_old_GPU_port_telemetry_cases=18,
        optional_energy_observer_abandoned_partial_cases=['70B/20K/B8','405B/64K/B8','405B/126K/B8'],
        Prefill='EXACT_OLD_DIAGNOSTIC_TOTALS_REUSED',formal_replaced=False,
        full_pytest_run=False,related_tests_passed=165,prechange_signature_test_passed=1,
        source_sha256={str(p.relative_to(f.ROOT)):sha(p) for p in sources}))
    # Lossless result archive, not simulator caches. Keep repository artifacts
    # small without dropping required per-port resident/service distributions.
    for path in (OUT/'cases').glob('*.json'):
        packed=path.with_suffix('.json.gz')
        with gzip.GzipFile(filename=str(packed),mode='wb',mtime=0) as stream:
            stream.write(path.read_bytes())
        assert path.resolve().parent==(OUT/'cases').resolve()
        path.unlink()
    detail=OUT/'first_step_operator_ports.csv'
    if detail.exists():
        with gzip.GzipFile(filename=str(detail)+'.gz',mode='wb',mtime=0) as stream:
            stream.write(detail.read_bytes())
        assert detail.resolve().parent==OUT.resolve()
        detail.unlink()


if __name__=='__main__':main()
