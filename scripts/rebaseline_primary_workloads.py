"""Audit/recompose only completed B1/B8 results; never execute physical Decode.

The reviewed integrity manifest binds legacy source and checkpoint bytes. This
entry point has no batch selector and never invokes the legacy sweep/output
writer. Its output allowlist cannot overwrite stress or legacy result tables.
"""
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from om3dthermal.serving.primary_execution import (
    ARCHITECTURES, SELECTOR_STATUS, candidate_timing, corrected_gpu_result,
    adaptive_result, gpu_memory_closure)
from om3dthermal.serving.workload_matrix import setup
from om3dthermal.power.feol_energy import sum_events, FEOLEnergyModel
from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/formal_iom3d_workload_sweep_v1'
OUTPUTS = frozenset(('primary_summary.csv', 'primary_capacity.csv', 'primary_traffic.csv',
    'primary_energy.csv', 'primary_normalized.csv', 'gpu_only_consistency_audit.csv',
    'adaptive_selection_b1_b8.csv', 'placement_ablation_b1_b8.csv', 'invalidation_audit.csv',
    'manifest.json', 'current_results_report.md', 'primary_validation.json'))


def sha(path, *, text=False):
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b'\r\n',b'\n') if text else data).hexdigest()


def check_integrity(root, integrity):
    reviewed = integrity.get('reviewed_non_decode_source_changes', {})
    # This diagnostic is not imported by the physical Decode/CPA path. Keep
    # the original digest and explicit reviewed postimage; never broadly
    # waive source validation to reuse a checkpoint.
    if not set(reviewed) <= {'src/om3dthermal/thermal_sensitivity.py'}:
        raise ValueError('Unreviewed physical source exception')
    for path, expected in integrity['frozen_sources'].items():
        if path in reviewed:
            if reviewed[path]['original_sha256_LF'] != expected:
                raise ValueError(f'Original source digest mismatch: {path}')
            expected = reviewed[path]['reviewed_sha256_LF']
        if sha(root/path, text=True) != expected:
            raise ValueError(f"Physical source/config dependency changed: {path}")
    for path, expected in integrity['preserved_artifacts'].items():
        if sha(root/path) != expected:
            raise ValueError(f"Preserved checkpoint/legacy artifact changed: {path}")


def write(name, value):
    if name not in OUTPUTS:
        raise ValueError(f"Not a primary output: {name}")
    path = OUT/name
    if name.endswith('.csv'):
        keys = list(dict.fromkeys(k for row in value for k in row))
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerows(value)
    else:
        path.write_text(value if isinstance(value,str) else json.dumps(value,indent=2), encoding='utf-8')


def ident(result):
    return {k:result['summary'][k] for k in ('model','workload_id','batch_size','system','status')}


def validate_checkpoint(result, path):
    """Conservation and full ordered-context audit, without physical replay."""
    s = result['summary']
    assert s['batch_size'] in (1,8)
    if s['status'] != 'EVALUATED' or s['system'] == 'HBM_GRACE_C2C':
        return []
    steps = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    assert [x['context'] for x in steps] == list(range(s['H']+s['P'], s['H']+s['P']+s['G']))
    assert sum(x['latency_s'] for x in steps) == s['decode_s']
    assert sum_events(x['energy_events'] for x in steps) == result['events']
    assert {k:sum(x['component_sums'][k] for x in steps) for k in steps[0]['component_sums']} == result['component_sums']
    for x in steps:
        e = x['energy_events']
        assert e['array_read_bits']+e['array_write_bits'] == x['local_array_bytes']*8
        assert sum(e['miv_read_bits_by_layer']) == e['array_read_bits']
        assert sum(e['miv_write_bits_by_layer']) == e['array_write_bits']
        assert e['interface_bits'] == e['gpu_decode_proxy_bits'] == x['boundary_bytes']*8
        assert e['mac_operations']*2 == x['nmp_flops']
        assert e['row_select_events'] == e['column_select_events'] == e['read_services']+e['write_services']
    policy = 'NO_NMP' if s['system'] == 'IOM3D_NO_NMP' else 'MAC_NMP'
    model = FEOLEnergyModel(resolve_feol_floorplan(ROOT), setup(ROOT)[2])
    energy = model.account(result['events'], s['decode_s'], phase='decode', policy=policy)
    assert energy['total_J'] == result['energy']['decode_J']
    assert all(result['energy']['decode_'+k] == v for k,v in energy['components'].items())
    for entry in result.get('cpa_audit', []):
        assert entry['after_s'] <= entry['before_s']
        assert all(after < before for before,after in entry['accepted_history'])
        assert 0 <= entry['moved_atom_fraction'] <= 1
    for phase in ('prefill','decode'):
        e = result['energy']
        assert sum(v for k,v in e.items() if k.startswith(phase+'_') and k.endswith('_J') and k not in (phase+'_J',phase+'_tokens_per_J')) == e[phase+'_J']
    return steps


def audit_row(old, gpu, hbm, candidate):
    s, hs = gpu['summary'], hbm['summary']
    ledger, c, steps = candidate['ledger'], candidate['closure'], candidate['steps']
    batch = s['batch_size']; hbw = setup(ROOT)[3].sustained_bandwidth_bytes_per_s
    read = sum(x['read_bytes'] for x in steps); write_bytes = sum(x['write_bytes'] for x in steps)
    memory = read+write_bytes
    evaluated = hs['status'] == 'EVALUATED'
    if evaluated:
        assert hbm['prefill_ledger'] == ledger
        t = hbm['traffic']
        hread = sum(t[k] for k in ('HBM_read_GB','Grace_weight_read_GB','Grace_KV_read_GB'))*1e9
        hwrite = sum(t[k] for k in ('HBM_write_GB','Grace_KV_write_GB'))*1e9
        assert abs(hread-read) <= read*1e-12
        assert abs(hwrite-write_bytes) <= write_bytes*1e-12
        if hs['HBM_only_fit']:
            assert s['E2E_s'] <= hs['E2E_s']
    local = evaluated and hs['HBM_only_fit']
    row = dict(model=s['model'],workload=s['workload_id'],B=batch,
        HBM_effective_BW=hbw, M3D_thermal_cap=c['thermal_Bps'], M3D_internal_cap=c['internal_Bps'],
        M3D_boundary_cap=c['boundary_Bps'], M3D_GPU_cap=c['GPU_Bps'], M3D_effective_BW=c['effective_Bps'], BW_unit='byte/s',
        HBM_prefill_compute_s=candidate['prefill_compute_s'],
        HBM_prefill_memory_s=(hbm['traffic']['prefill_HBM_read_GB']+hbm['traffic']['prefill_HBM_write_GB'])*1e9/hbw if evaluated else None,
        M3D_prefill_compute_s=candidate['prefill_compute_s'], M3D_prefill_array_s=ledger['total_memory_bytes']/c['internal_Bps'],
        M3D_prefill_external_s=ledger['total_memory_bytes']/min(c[k] for k in ('thermal_Bps','boundary_Bps','GPU_Bps')),
        M3D_prefill_startup_s=c['startup_s'], HBM_decode_compute_s=sum(x['compute_s'] for x in steps),
        HBM_decode_memory_s=(hbm['traffic']['HBM_read_GB']+hbm['traffic']['HBM_write_GB'])*1e9/hbw if evaluated else None,
        M3D_decode_compute_s=sum(x['compute_s'] for x in steps), M3D_decode_array_s=memory/c['internal_Bps'],
        M3D_decode_external_s=memory/min(c[k] for k in ('thermal_Bps','boundary_Bps','GPU_Bps')),
        M3D_decode_startup_s=len(steps)*c['startup_s'],
        prefill_FLOPs=ledger['total_flops'], prefill_weight_read_bytes=ledger['active_weight_read_bytes'],
        prefill_history_KV_read_bytes=ledger['historical_cached_kv_read_bytes'],
        prefill_other_memory_bytes=ledger['total_memory_bytes']-ledger['active_weight_read_bytes']-ledger['historical_cached_kv_read_bytes'],
        decode_FLOPs=sum(x['flops'] for x in steps), decode_weight_bytes=sum(x['weight_bytes'] for x in steps),
        decode_KV_read_bytes=sum(x['KV_read_bytes'] for x in steps),decode_KV_write_bytes=sum(x['KV_write_bytes'] for x in steps),
        GPU_compute_ceiling_FLOPs_s=setup(ROOT)[2].gpu_compute_power.peak_compute_BF16_dense_flops_per_s,
        old_M3D_prefill_s=old['summary']['prefill_s'], old_M3D_decode_s=old['summary']['decode_s'],
        old_M3D_decode_array_s=old['component_sums']['ARRAY'],
        old_M3D_decode_external_s=old['component_sums']['EXTERNAL_BOUNDARY'],
        old_external_excess_over_streaming_s=old['component_sums']['EXTERNAL_BOUNDARY']-memory/c['effective_Bps'],
        duplicate_array_plus_bandwidth='NOT_FOUND__OLD_STAGE_USES_MAX',
        startup_provenance=c['startup_status'],
        NMP_prefill_verdict='UNCHANGED_VALID_COMPUTE_BOUND',
        HBM_status=hs['status'],HBM_local_paired=local,
        difference_source='LEGACY_NMP_ROW_GROUP_DIRECT_PORT_SERIALIZATION_AND_PER_OPERATOR_GPU_ROOFLINES',
        audit_verdict='PASS_PAIRED_GPU_CONSISTENCY' if local else 'PASS_HOST_OFFLOAD_DIFFERENCE' if evaluated else 'PASS_GPU_LEDGER__HBM_CAPACITY_INFEASIBLE')
    return row


def main():
    integrity = json.loads((OUT/'rebaseline_integrity.json').read_text())
    check_integrity(ROOT, integrity)
    legacy_manifest = json.loads((OUT/'legacy_archive/manifest.json').read_text())
    fp = integrity['legacy_source_fingerprint']
    assert fp == legacy_manifest['source_fingerprint']
    config = setup(ROOT)[0]
    final, selections, ablation, audits, checkpoint_audits = [], [], [], [], []
    physical_steps = 0; nmp_cases = 0
    for name in config['models']:
        for wid in config['workloads']:
            for batch in (1,8):
                old = {}
                for system in ('HBM_GRACE_C2C','IOM3D_NO_NMP','IOM3D_MAC_NMP_UNIFORM','IOM3D_MAC_NMP_CPA'):
                    path = OUT/'checkpoints'/f'{name}_{wid}_B{batch}_{system}_{fp[:16]}.json'
                    relative = path.relative_to(ROOT).as_posix()
                    if relative not in integrity['preserved_artifacts']:
                        raise ValueError(f"Unreviewed checkpoint: {path.name}")
                    r = json.loads(path.read_text())
                    assert r['fingerprint'] == fp
                    assert (r['summary']['model'],r['summary']['workload_id'],r['summary']['batch_size'],r['summary']['system']) == (name,wid,batch,system)
                    steps = validate_checkpoint(r,path)
                    if 'MAC_NMP' in system:
                        nmp_cases += 1; physical_steps += len(steps)
                    old[system] = r
                    checkpoint_audits.append(dict(checkpoint=relative,sha256=sha(path),steps=len(steps),verdict='PASS_REUSE' if system!='IOM3D_NO_NMP' else 'PASS_EVENTS__RECOMPUTE_GPU_TIMING'))
                hbm = deepcopy(old['HBM_GRACE_C2C'])
                hbm['summary'].update(system='HBM_GPU',placement_policy='NOT_APPLICABLE_GPU_EXECUTION')
                candidate = candidate_timing(name,wid,batch,ROOT)
                gpu = corrected_gpu_result(old['IOM3D_NO_NMP'],candidate,ROOT)
                nmp = deepcopy(old['IOM3D_MAC_NMP_CPA'])
                uniform = old['IOM3D_MAC_NMP_UNIFORM']
                # All 18 existing Prefills are compute bound. Do not touch valid
                # Prefill event routing, NMP energy, CPA objective or Decode.
                for r in (old['IOM3D_NO_NMP'],nmp,uniform):
                    assert r['prefill_ledger'] == candidate['ledger']
                    assert r['summary']['prefill_s'] == candidate['prefill_compute_s'] == candidate['timing']['prefill_s']
                for event in ('array_read_bits','array_write_bits','mac_operations','interface_bits','gpu_decode_proxy_bits'):
                    assert nmp['events'][event] == uniform['events'][event]
                nmp['summary'].update(system='NMP_CPA',placement_policy='CRITICAL_PATH_AWARE')
                proposed = adaptive_result(gpu,nmp)
                a, b, p = gpu['summary'], nmp['summary'], proposed['summary']
                assert p['decode_s'] == min(a['decode_s'],b['decode_s'])
                assert p['E2E_s'] == min(a['E2E_s'],b['E2E_s'])
                assert p['E2E_tok_s']/a['E2E_tok_s'] >= 1
                selected = gpu if p['selected_executor']=='GPU' else nmp
                assert proposed['energy'] == selected['energy'] and proposed['traffic'] == selected['traffic']
                audits.append(audit_row(old['IOM3D_NO_NMP'],gpu,hbm,candidate))
                final.extend((hbm,gpu,proposed))
                selections.append(dict(model=name,workload_id=wid,batch_size=batch,
                    GPU_decode_s=a['decode_s'], NMP_CPA_decode_s=b['decode_s'],
                    GPU_E2E_tok_s=a['E2E_tok_s'], NMP_CPA_E2E_tok_s=b['E2E_tok_s'],
                    proposed_E2E_tok_s=p['E2E_tok_s'],selected_executor=p['selected_executor'],
                    speedup_vs_M3D_GPU=p['E2E_tok_s']/a['E2E_tok_s'],
                    GPU_E2E_J=gpu['energy']['E2E_J'],NMP_CPA_E2E_J=nmp['energy']['E2E_J'],
                    proposed_E2E_J=proposed['energy']['E2E_J'],
                    tokens_per_J_ratio_vs_M3D_GPU=proposed['energy']['E2E_tokens_per_J']/gpu['energy']['E2E_tokens_per_J'],
                    selector_objective='MINIMUM_DECODE_LATENCY', selector_overhead_status=SELECTOR_STATUS,
                    selector_latency_s=0,selector_energy_J=0,placement_policy=p['placement_policy'],
                    CPA_moves=p['CPA_moves'],CPA_optimizer_runtime=p['CPA_optimizer_runtime']))
                u = uniform['summary']
                ablation.append(dict(model=name,workload_id=wid,batch_size=batch,
                    comparison_role='NMP_PLACEMENT_ABLATION_ONLY',Uniform_decode_s=u['decode_s'],CPA_decode_s=b['decode_s'],
                    Uniform_E2E_tok_s=u['E2E_tok_s'],CPA_E2E_tok_s=b['E2E_tok_s'],
                    CPA_vs_Uniform_E2E=b['E2E_tok_s']/u['E2E_tok_s'],
                    Uniform_E2E_J=uniform['energy']['E2E_J'],CPA_E2E_J=nmp['energy']['E2E_J'],
                    CPA_moves=b['accepted_moves'],CPA_optimizer_runtime=b['optimizer_runtime_s'],
                    applies_to_selected_execution=p['selected_executor']=='NMP_CPA',checkpoint_reused=True))
                print(name,wid,batch,p['selected_executor'],f"{a['E2E_tok_s']:.6f}",f"{p['E2E_tok_s']:.6f}",flush=True)
    assert len(final)==54 and nmp_cases==36 and len(selections)==18
    check_integrity(ROOT,integrity)
    write('primary_summary.csv',[{**r['summary'],**{k:v for k,v in r['energy'].items() if k in ('E2E_J_per_token','E2E_tokens_per_J','energy_status')}} for r in final])
    capkeys = set(next(iter(old.values()))['summary']) - set(candidate['timing'])
    capkeys -= {'effective_bandwidth_TBps','prefill_host_GB','optimizer_runtime_s','accepted_moves','moved_fraction','resident_moved_fraction'}
    write('primary_capacity.csv',[{k:v for k,v in r['summary'].items() if k in capkeys} for r in final])
    for family in ('traffic','energy'):
        write('primary_'+family+'.csv',[{**ident(r),**r[family]} for r in final])
    normalized = []
    for i in range(0,len(final),3):
        h,g,p = final[i:i+3]
        for r in (h,g,p):
            s = r['summary']; valid = s['status']=='EVALUATED'
            normalized.append(dict(**ident(r),
                speedup_vs_M3D_GPU=s['E2E_tok_s']/g['summary']['E2E_tok_s'] if valid else None,
                speedup_vs_HBM_GPU=s['E2E_tok_s']/h['summary']['E2E_tok_s'] if valid and h['summary']['status']=='EVALUATED' else None,
                HBM_normalization_status='AVAILABLE' if h['summary']['status']=='EVALUATED' else 'SYSTEM_CAPACITY_INFEASIBLE',
                energy_efficiency_vs_M3D_GPU=r['energy']['E2E_tokens_per_J']/g['energy']['E2E_tokens_per_J'] if r['energy'].get('E2E_tokens_per_J') else None))
    write('primary_normalized.csv',normalized)
    write('gpu_only_consistency_audit.csv',audits)
    write('adaptive_selection_b1_b8.csv',selections)
    write('placement_ablation_b1_b8.csv',ablation)
    invalidation = []
    for family,scope,verdict,reason,action,reuse,recomputed in (
        ('M3D_GPU','B1/B8','INVALIDATED_TIMING','NMP row/port operator timing used in aggregate GPU memory benchmark','RECOMPUTE_GPU_TIMING_STATIC_ENERGY',True,18),
        ('NMP_CPA Decode','B1/B8','UNCHANGED_VALID','Physical source/config and events unchanged','REUSE',True,0),
        ('NMP_UNIFORM Decode','B1/B8','UNCHANGED_VALID','Physical source/config and events unchanged','REUSE',True,0),
        ('NMP Prefill','B1/B8','UNCHANGED_VALID','All 18 operating points compute bound; exact Prefill timing and events retained','REUSE',True,0),
        ('HBM_GPU','B1/B8','UNCHANGED_VALID','Unchanged conventional function and full dependency digest','REUSE',True,0),
        ('M3D_MAC_NMP','B1/B8','NEW_ADAPTIVE_COMPOSITION','Minimum modeled Decode latency; selected energy unchanged','COMPOSE',True,18),
        ('Thermal read-stress diagnostic','NOT_A_BENCHMARK_RUN','INDEPENDENT_CORRECTNESS_FIX','Resized floorplan copy inherited warmed slab-indexed route map; Decode never calls this helper','REBUILD_COPY_ROUTE_MAP_ONLY',False,0),
        ('All systems','B32','FROZEN','Outside scope; original files and statuses byte preserved','NO_ACTION',False,0)):
        invalidation.append(dict(result_family=family,batch_scope=scope,old_status='FROZEN' if scope=='B32' else 'EXISTING' if action!='COMPOSE' else 'NOT_PRESENT',
            audit_verdict=verdict,reason=reason,action=action,checkpoint_reused=reuse,recomputed=recomputed))
    write('invalidation_audit.csv',invalidation)
    validation = dict(status='PASS',primary_operating_points=18,architecture_rows=54,
        NMP_physical_Decode_checkpoints_reused=nmp_cases,NMP_physical_Decode_steps_reused=physical_steps,
        Prefill_recomposed_cases=0,Prefill_unchanged_operating_points=18,GPU_only_recomputed_cases=18,
        B8_NMP_selected=sum(x['batch_size']==8 and x['selected_executor']=='NMP_CPA' for x in selections),
        B8_GPU_selected=sum(x['batch_size']==8 and x['selected_executor']=='GPU' for x in selections),
        physical_source_config_unchanged=True, all_legacy_artifacts_byte_identical=True,
        reviewed_non_decode_source_changes=integrity.get('reviewed_non_decode_source_changes',{}),
        CPA_saved_objective_checks='PASS_NON_INCREASING',
        B32_STATUS='FROZEN_NOT_PART_OF_THIS_REBASELINE',checkpoint_audits=checkpoint_audits)
    write('primary_validation.json',validation)
    manifest = dict(legacy_manifest=legacy_manifest,source_fingerprint=fp,
        git_HEAD_at_generation=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        primary_status='AUDITED_B1_B8_ADAPTIVE_REBASELINE',architectures=list(ARCHITECTURES),
        primary_batches=[1,8],primary_operating_points=18,B32_STATUS='FROZEN_NOT_PART_OF_THIS_REBASELINE',
        primary_workloads=config['workloads'],models=config['models'],GPU_memory_closure=gpu_memory_closure(ROOT),
        selector_objective='MINIMUM_DECODE_LATENCY',selector_overhead_status=SELECTOR_STATUS,
        NMP_Decode='UNCHANGED_PHYSICAL_FEOL_CONFIG_AND_CPA__CHECKPOINT_REUSE',
        NMP_Prefill='UNCHANGED_VALID_COMPUTE_BOUND',
        reviewed_non_decode_source_changes=integrity.get('reviewed_non_decode_source_changes',{}),
        thermal='FROZEN_CAP_REUSED__NO_WORKLOAD_THERMAL_SOLVE',
        HBM_energy='INCOMPLETE_HBM_WRITE_AND_HOST_STATIC_UNRESOLVED',
        integrity_manifest='rebaseline_integrity.json',validation='primary_validation.json',
        rebaseline_sources={p:sha(ROOT/p,text=True) for p in (
            'src/om3dthermal/serving/primary_execution.py','scripts/rebaseline_primary_workloads.py')},
        outputs={name:sha(OUT/name) for name in OUTPUTS if name.endswith('.csv')})
    write('manifest.json',manifest)
    report(final,selections,ablation,validation)
    check_integrity(ROOT,integrity)


def report(results,selections,ablation,validation):
    c = gpu_memory_closure(ROOT)
    lines = ['# Formal B=1/8 architecture rebaseline', '',
        'Scope: Llama-3.1-8B/70B/405B; W1=(2000,512,256), W2=(20000,512,256), W3=(126000,512,512); B=1,8. Throughput is aggregate generated B*G / (Prefill + Decode), tok/s.', '',
        '## Root cause and claim boundary', '',
        'BUG_FIXED: the old GPU-only path inherited the NMP whole-row resident mapping, GROUP_DIRECT nearest-port serialization, and a sum of per-operator GPU rooflines. HBM used the full-step active-weight/KV GPU ledger. This was an inconsistent execution abstraction; it was not a proven duplicate MAT/MIV-plus-bandwidth sum.', '',
        'For 8B W1 B1, old Decode=1.6831122578856168 s and external component=1.6813510243409877 s, versus HBM Decode=1.1921522536921827 s. At context 2512, each layer K/V has a maximum per-port payload of 16,384 bytes on a 1 GB/s port under the whole-row mapping; each K/V family totals ~0.524451 ms over 32 layers instead of ~0.079050 ms at the thermal streaming cap. Q/O and FFN_DOWN have similar port concentration. Old stages use max(array,boundary,GPU); component sums overlap and are not additive latency.', '',
        'The corrected GPU-only formal semantics use the existing first-order hierarchical GPU memory service model, not the physical NMP row/tile schedule. This is an aggregate streaming prediction, not a claim that concentrated GROUP_DIRECT traffic can attain the aggregate cap without changing the scheduling abstraction. The physical NMP route constraints remain valid and untouched.', '',
        f"Internal closure = 318 slabs * 50 lanes/slab * 32 bytes / ({c['service_cycle_ns']:.12f} ns * {c['service_cycle_scale']}) = {c['internal_Bps']/1e12:.9f} TB/s. The existing full physical cycle includes MAT, MIV and FEOL routing. Eight layers share lanes. Boundary=15.9 TB/s; GPU=4.8 TB/s; thermal={c['thermal_Bps']/1e12:.12f} TB/s. Effective=min(all four). No additional independent startup is established by this closure; adding its access cycles again would duplicate service.", '',
        'Both GPU backends use the same active weights/KV traffic, FLOPs, 989.5 TFLOP/s Decode ceiling and 700/700 TFLOP/s Prefill family ceilings. For HBM-local pairs the corrected M3D GPU time is no larger; host-offload differences retain the 416.34 GB/s Grace C2C model. All 18 existing M3D/NMP Prefills equal their GPU compute time, so no Prefill latency or NMP energy event was changed.', '',
        'SEMANTIC_RENAME: formal systems are HBM_GPU, M3D_GPU, M3D_MAC_NMP. ADAPTIVE_POLICY: pre-dispatch minimum modeled Decode latency, tie to GPU, zero inference latency/energy. GPU-selected performance, traffic and energy are exact M3D_GPU; CPA and all NMP activity are inactive. NMP-selected results copy the frozen CPA checkpoint. UNCHANGED_VALID_MODEL: all NMP physical equations, events, resource limits and CPA objective.', '',
        '## Primary performance', '',
        '| Model | W | B | HBM_GPU | M3D_GPU | NMP_CPA candidate | M3D_MAC_NMP | Executor | Proposed/GPU |',
        '|---|---|---:|---:|---:|---:|---:|---|---:|']
    for i,row in enumerate(selections):
        hs=results[i*3]['summary']; h=f"{hs['E2E_tok_s']:.3f}" if hs['status']=='EVALUATED' else 'Capacity infeasible'
        lines.append(f"| {row['model']} | {row['workload_id']} | {row['batch_size']} | {h} | {row['GPU_E2E_tok_s']:.3f} | {row['NMP_CPA_E2E_tok_s']:.3f} | {row['proposed_E2E_tok_s']:.3f} | {row['selected_executor']} | {row['speedup_vs_M3D_GPU']:.6f} |")
    lines += ['',f"B8 decisions: NMP={validation['B8_NMP_selected']}; GPU={validation['B8_GPU_selected']}.", '',
        'Selection minimizes latency only. It does not guarantee better tokens/J. Candidate and selected energy appear together in adaptive_selection_b1_b8.csv; HBM absolute tokens/J remains unresolved rather than synthesized.', '',
        '## CPA vs Uniform placement ablation', '',
        '| Model | W | B | CPA/Uniform E2E speed | Used by selected executor |', '|---|---|---:|---:|---|']
    for r in ablation:
        lines.append(f"| {r['model']} | {r['workload_id']} | {r['batch_size']} | {r['CPA_vs_Uniform_E2E']:.6f} | {r['applies_to_selected_execution']} |")
    lines += ['', 'Uniform is an NMP placement ablation, never a fourth architecture. CPA optimizer runtime is preserved as an offline planning diagnostic; a GPU-selected case has zero CPA moves/runtime in the formal result.', '',
        '## Reuse and validation', '',
        f"NMP physical Decode checkpoints reused: {validation['NMP_physical_Decode_checkpoints_reused']}; complete Decode steps reused: {validation['NMP_physical_Decode_steps_reused']}. Prefill recomposed: 0; GPU-only cases recomputed: 18. All original checkpoint files and legacy CSVs retain their original SHA-256. Source/config digest verification, ordered contexts, component totals, memory/interface/MIV/MAC event conservation, and coefficient-only energy reproduction pass for every primary physical checkpoint.", '',
        'Regression tests cover identical GPU workload/compute semantics, internal and boundary bottleneck counterexamples, single-counted service latency, exact adaptive candidate copying, ties and GPU fallback, CPA eligibility, and immutable legacy/B32 artifacts. See primary_validation.json and the final Git/test report for execution evidence.', '',
        'Independent correctness repair: thermal_sensitivity.no_nmp_read_energy now drops slab-indexed route maps inherited by a resized floorplan copy. Warm/cold read-stress events are identical; the canonical floorplan is not mutated. This diagnostic is not called by physical Decode/CPA. Its original and reviewed source digests are recorded separately; no saved thermal or primary result is invalidated. Legacy test goldens retain explicit legacy service/workload/layout fixtures instead of being applied to incompatible current defaults.', '',
        'B32 untouched and frozen in this task.\nNo B32 result was recomputed, selected, extrapolated, or overwritten.', '',
        'Verdict: B1/B8 GPU timing semantics are consistent within the stated aggregate streaming model; NMP physical execution remains unchanged; modeled adaptive latency is the candidate minimum; CPA applies only to NMP execution. These B1/B8 architecture semantics can be frozen. Workload thermal closure and local hardware validation remain outside this benchmark; HBM unresolved energy is retained.']
    write('current_results_report.md','\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
