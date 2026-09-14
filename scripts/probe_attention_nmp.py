"""Strict 8B/W1/B8 probe. No formal results, hardware, or scheduler mutations.

The existing fixed operator graph supplies QK/AV physical stages and events.
GPU operations between those handoffs use the frozen streaming closure. This
is a probe adapter, not a new execution policy or placement algorithm.
"""
from copy import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from om3dthermal.serving.decode_policy import DecodePolicyModel, ATTENTION, ExecutionPolicy, on_nmp
from om3dthermal.serving.primary_execution import gpu_memory_closure
from om3dthermal.serving.workload_matrix import inputs
from om3dthermal.workload import evaluate_llm_decode
from om3dthermal.power.feol_energy import sum_events, FEOLEnergyModel
from om3dthermal.placement.critical_path import refine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/attention_nmp_probe_8b_w1_b8'
FORMAL = ROOT/'runs/formal_iom3d_workload_sweep_v1'
CASE = ('Llama-3.1-8B', 'W1', 8)
BASE_HEAD = '3e0604caf028d7f2006555973ff341e1a855e37c'
FULL = 'M3D_NMP_FULL_CPA'
ATTN = 'M3D_ATTENTION_NMP'


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2), encoding='utf-8')


def csv_out(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def frozen_hashes():
    paths = [*FORMAL.rglob('*'), *(ROOT/'src').rglob('*.py'), *(ROOT/'configs').rglob('*.yaml')]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths) if p.is_file()}


def require_case(name, workload, batch):
    if (name, workload, batch) != CASE:
        raise ValueError('Only Llama-3.1-8B/W1/B8 is authorized')


class AttentionEntries(dict):
    """Limit existing refine's entry enumeration; retain all lookup/capacity data."""
    def values(self):
        return [v for v in super().values() if v.unit.operator_type in ATTENTION]


def attention_cpa(placement, platform, first, last):
    # Existing CPA evaluates operators independently and explicitly supports
    # paired QK/AV resident moves and concurrent request resource accounting.
    # Only enumerate the two NMP operators; no GPU weight placement moves.
    original = placement.request_operators
    placement.request_operators = AttentionEntries(original)
    try:
        audit = refine(placement, platform, first_context=first, last_context=last)
    finally:
        placement.request_operators = dict(placement.request_operators)
    assert {a['operator'] for a in audit} == ATTENTION
    return audit


def gpu_segment(rows, compute_peak, closure):
    flops = sum(r.get('gpu_flops', 0) for r in rows)
    memory = sum(r.get('gpu_memory_bytes', 0) for r in rows)
    # Canonical small GPU ops retain their existing GPU-local memory service.
    local_s = sum(r.get('gpu_local_s', 0) for r in rows)
    compute = flops/compute_peak
    streaming = memory/closure['effective_Bps']
    return dict(latency_s=max(compute, streaming)+local_s,
                gpu_compute_s=compute, gpu_memory_s=streaming+local_s,
                gpu_streaming_s=streaming, gpu_local_s=local_s,
                flops=flops, memory_bytes=memory,
                critical_resource='GPU_COMPUTE' if compute >= streaming else 'GPU_MEMORY')


class ProbeEngine(DecodePolicyModel):
    def step(self, context, policy=ExecutionPolicy.ATTENTION_NMP, *, include_stages=False):
        if policy != ExecutionPolicy.ATTENTION_NMP:
            raise ValueError('Probe cannot execute a reference policy')
        raw = super().step(context, policy, include_stages=True)
        w = self.workload
        stages = raw.pop('stages')
        physical = [r for r in stages if 'local_array_bytes' in r]
        nmp = [r for r in physical if r['executor'] == 'NMP']
        assert {r['operator'] for r in nmp} == ATTENTION
        assert all(on_nmp(r['operator'], policy) == (r['executor']=='NMP') for r in physical)
        segments, pending = [], []
        stage_audit = []
        for r in stages:
            if r['executor'] == 'NMP':
                if pending:
                    segments.append(gpu_segment(pending, self.physical.gpu_compute, self.closure))
                    pending = []
                c = r['components']
                reduction = c['INTER_REGION_NOC']-r['noc_s']
                assert math.isclose(r['latency_s'], r['external_service_s']+r['noc_s']+
                                    reduction+max(c[k] for k in ('ARRAY','LOCAL_FABRIC','MAC')), rel_tol=1e-12)
                stage_audit.append(dict(operator=r['operator'], layer=r['layer'],
                    critical_resource=max(c, key=c.get), core_resource=max(('ARRAY','LOCAL_FABRIC','MAC'), key=c.get),
                    latency_s=r['latency_s'], **c))
            else:
                r = dict(r)
                if 'local_array_bytes' in r:
                    r['gpu_memory_bytes'] = r['local_array_bytes']
                    if r['operator'] not in ('KV_APPEND', 'TOKEN_EMBED_LOOKUP'):
                        r['gpu_flops'] = self.placement.operators[r['layer'], r['operator']].unit.local_flops*w.batch_size
                else:
                    r['gpu_local_s'] = r['latency_s']
                pending.append(r)
        if pending:
            segments.append(gpu_segment(pending, self.physical.gpu_compute, self.closure))
        ledger = evaluate_llm_decode(w.model_copy(update={'context_length': context}))
        gpu_flops = sum(s['flops'] for s in segments)
        nmp_flops = sum(r['nmp_flops'] for r in nmp)
        assert gpu_flops+nmp_flops == ledger.flops_per_token*w.batch_size
        assert sum(s['memory_bytes'] for s in segments) == w.batch_size*(ledger.weight_read_bytes_per_token+ledger.kv_write_bytes_per_token)
        assert raw['energy_events']['interface_bits'] == raw['boundary_bytes']*8
        assert raw['energy_events']['array_read_bits'] == ledger.read_bytes_per_token*w.batch_size*8
        assert raw['energy_events']['array_write_bits'] == ledger.write_bytes_per_token*w.batch_size*8
        assert raw['energy_events']['mac_operations']*2 == nmp_flops
        incoming = sum(r['input_boundary_bytes'] for r in nmp)
        outgoing = sum(r['output_boundary_bytes'] for r in nmp)
        partial = sum(r['output_boundary_bytes'] for r in nmp if r['operator']=='ATTENTION_AV')
        bulk = sum(r['boundary_bytes'] for r in physical if r['executor']=='GPU')
        assert incoming+outgoing+bulk == raw['boundary_bytes']
        lat = dict(gpu_compute_s=sum(s['gpu_compute_s'] for s in segments),
            gpu_memory_s=sum(s['gpu_memory_s'] for s in segments),
            gpu_segment_wall_s=sum(s['latency_s'] for s in segments),
            gpu_local_s=sum(s['gpu_local_s'] for s in segments),
            nmp_array_s=sum(r['array_service_s'] for r in nmp),
            nmp_fabric_s=sum(r['fabric_service_s'] for r in nmp),
            nmp_mac_s=sum(r['components']['MAC'] for r in nmp),
            nmp_noc_s=sum(r['noc_s'] for r in nmp),
            nmp_reduction_s=sum(r['components']['INTER_REGION_NOC']-r['noc_s'] for r in nmp),
            boundary_transfer_s=sum(r['external_service_s'] for r in nmp),
            nmp_core_wall_s=sum(max(r['components'][k] for k in ('ARRAY','LOCAL_FABRIC','MAC')) for r in nmp))
        seconds = lat['gpu_segment_wall_s']+sum(r['latency_s'] for r in nmp)
        raw.update(latency_s=seconds, latency_breakdown=lat, gpu_flops=gpu_flops,
            gpu_to_nmp_bytes=incoming, nmp_to_gpu_bytes=outgoing,
            partial_boundary_bytes=partial, other_boundary_bytes=bulk,
            weight_bytes=ledger.weight_read_bytes_per_token*w.batch_size,
            kv_read_bytes=ledger.kv_read_bytes_per_token*w.batch_size,
            internal_nmp_bytes=sum(r['local_array_bytes'] for r in nmp),
            fabric_bytes=sum(r['fabric_bytes'] for r in nmp), stage_audit=stage_audit)
        # Keep compact checkpoint evidence, without huge physical resource arrays.
        return {k:raw[k] for k in ('context','latency_s','latency_breakdown','energy_events',
            'gpu_flops','gpu_to_nmp_bytes','nmp_to_gpu_bytes','partial_boundary_bytes',
            'other_boundary_bytes','weight_bytes','kv_read_bytes','internal_nmp_bytes',
            'fabric_bytes','boundary_bytes','noc_bytes','nmp_flops','stage_audit')}


def full_directional_ledger(placement, cw, frozen):
    """Count saved model's boundary events, without evaluating any physical stage.

    CPA resident moves are intra-slab and cannot change slab ownership. For
    this case each non-embedding operator spans all slabs at both endpoints;
    monotonic prefix occupancy proves the same for every intervening context.
    """
    w = placement.workload
    incoming = outgoing = partial = 0.
    for entry in placement.request_operators.values():
        u = entry.unit; op = u.operator_type
        if op == 'OTHER_WEIGHT':
            continue
        if op == 'TOKEN_EMBED_LOOKUP':
            outgoing += cw.generated*w.d_model*2
            continue
        first = cw.history+cw.prompt
        last = first+cw.generated-1
        for context in (first, last):
            count = entry.prefix_counts(context*w.n_heads_kv if op in ATTENTION else entry.atom_count)
            assert len(np.unique(entry.die_ids[count>0])) == placement.floorplan.layout.slab_count
        dies = placement.floorplan.layout.slab_count
        if op == 'ATTENTION_QK':
            incoming += cw.generated*dies*w.d_model*2
            outgoing += sum(cw.contexts)*w.n_heads_q*2
        elif op == 'ATTENTION_AV':
            incoming += sum(cw.contexts)*w.n_heads_q*2
            x = cw.generated*dies*w.d_model*4
            outgoing += x; partial += x
        else:
            incoming += cw.generated*dies*u.activation_input_bytes
            outgoing += cw.generated*entry.atom_count*2*w.batch_size
    append = frozen['events']['array_write_bits']/8
    total = frozen['events']['interface_bits']/8
    assert incoming+outgoing+append == total, (incoming+outgoing+append, total)
    return dict(gpu_to_nmp_bytes=incoming, nmp_to_gpu_bytes=outgoing,
                partial_boundary_bytes=partial, other_boundary_bytes=append,
                boundary_bytes=total, D_int_bytes=incoming+outgoing)


def references():
    def selected(name, system):
        with (FORMAL/name).open() as f:
            return next(r for r in csv.DictReader(f) if (r['model'],r['workload_id'],int(r['batch_size']))==CASE and r['system']==system)
    gpu = selected('primary_summary.csv','M3D_GPU')
    energy = selected('primary_energy.csv','M3D_GPU')
    checkpoint = next((FORMAL/'checkpoints').glob('Llama-3.1-8B_W1_B8_IOM3D_MAC_NMP_CPA_*.json'))
    full = json.loads(checkpoint.read_text())
    return gpu, energy, full, checkpoint


def main():
    require_case(*CASE)
    OUT.mkdir(parents=True, exist_ok=True)
    before = frozen_hashes()
    save('frozen_integrity.json', before)
    gpu, gpu_energy, full, full_path = references()
    w, cw, _ = inputs(*CASE, ROOT)
    assert (cw.history,cw.prompt,cw.generated)==(2000,512,256)
    engine = ProbeEngine(w,project_root=ROOT,record_energy=True,placement_policy='UNIFORM_STRIPING',decode_start_context=2512)
    engine.closure = gpu_memory_closure(ROOT)
    direction = full_directional_ledger(engine.placement,cw,full)
    print('FULL_BOUNDARY_LEDGER_CONSERVED',direction,flush=True)
    started = time.perf_counter()
    audit = attention_cpa(engine.placement,engine.platform,2512,2767)
    save('attention_cpa_audit.json',audit)
    print('ATTENTION_CPA_COMPLETE',time.perf_counter()-started,flush=True)
    steps=[]
    with (OUT/'attention_steps.jsonl').open('w') as stream:
        for context in cw.contexts:
            row=engine.step(context)
            steps.append(row)
            stream.write(json.dumps(row)+'\n');stream.flush()
            if len(steps)%8==0:
                print(f'ATTENTION_ONLY {len(steps)}/256 {time.perf_counter()-started:.1f}s',flush=True)
    produce(steps,gpu,gpu_energy,full,direction,engine,cw,audit,before,full_path)


def produce(steps,gpu,gpu_energy,full,direction,engine,cw,audit,before,full_path):
    generated=2048
    pre=float(gpu['prefill_s'])
    assert pre==full['summary']['prefill_s']
    dec=sum(r['latency_s'] for r in steps)
    events=sum_events(r['energy_events'] for r in steps)
    energy=FEOLEnergyModel(engine.floorplan,engine.platform).account(events,dec,phase='decode',policy='ATTENTION_NMP')
    total_J=float(gpu_energy['prefill_J'])+energy['total_J']
    assert math.isclose(energy['total_J'],sum(energy['components'].values()),rel_tol=1e-12)
    rows=[]
    for system,e2e,decode,j in [('M3D_GPU',float(gpu['E2E_s']),float(gpu['decode_s']),float(gpu_energy['E2E_J'])),
            (FULL,full['summary']['E2E_s'],full['summary']['decode_s'],full['energy']['E2E_J']),
            (ATTN,pre+dec,dec,total_J)]:
        rate=generated/e2e
        rows.append(dict(system=system,E2E_s=e2e,Decode_s=decode,E2E_tok_s=rate,
            speedup_vs_M3D_GPU=rate/float(gpu['E2E_tok_s']),
            speedup_vs_full_MAC_CPA=rate/full['summary']['E2E_tok_s'],
            J_per_token=j/generated,tokens_per_J=generated/j,prefill_s=pre))
    csv_out('summary.csv',rows)
    attlat={k:sum(r['latency_breakdown'][k] for r in steps) for k in steps[0]['latency_breakdown']}
    c=engine.closure
    weight=sum(r['weight_bytes'] for r in steps);kv=sum(r['kv_read_bytes'] for r in steps)
    append=events['array_write_bits']/8
    gpucompute=sum(r['gpu_flops']+r['nmp_flops'] for r in steps)/engine.physical.gpu_compute
    fc=full['component_sums']
    latency=[dict(system='M3D_GPU',gpu_compute_s=gpucompute,gpu_memory_s=(weight+kv+append)/c['effective_Bps'],
        nmp_array_s=0,nmp_fabric_s=0,nmp_mac_s=0,nmp_noc_s=0,nmp_reduction_s=0,
        boundary_transfer_s=(weight+kv+append)/min(c[k] for k in ('thermal_Bps','boundary_Bps','GPU_Bps')),
        total_decode_s=float(gpu['decode_s']),accounting='GPU=max(compute,memory); boundary overlaps GPU memory'),
        dict(system=FULL,gpu_compute_s=0,gpu_memory_s=fc['GPU_COMPUTE'],nmp_array_s=fc['ARRAY'],
        nmp_fabric_s=fc['LOCAL_FABRIC'],nmp_mac_s=fc['MAC'],nmp_noc_s='COMBINED_IN_NOC_PLUS_REDUCTION',
        nmp_reduction_s='COMBINED_IN_NOC_PLUS_REDUCTION',nmp_noc_plus_reduction_s=fc['INTER_REGION_NOC'],
        boundary_transfer_s=fc['EXTERNAL_BOUNDARY'],total_decode_s=full['summary']['decode_s'],
        accounting='Frozen checkpoint component sums; ARRAY includes KV append; GPU_COMPUTE field was small-op memory service'),
        dict(system=ATTN,**attlat,total_decode_s=dec,accounting='GPU dependency segments + physical NMP; max(array,fabric,MAC) within NMP core')]
    csv_out('latency_breakdown.csv',latency)
    traffic=[]
    for system,d,internal,noc in [('M3D_GPU',dict(boundary_bytes=weight+kv+append,gpu_to_nmp_bytes=0,nmp_to_gpu_bytes=0,partial_boundary_bytes=0,other_boundary_bytes=weight+kv+append,D_int_bytes=0),0,0),
            (FULL,direction,full['events']['array_read_bits']/8,full['traffic']['NoC_GB_per_token']*generated*1e9),
            (ATTN,{k:sum(r[k] for r in steps) for k in ('boundary_bytes','gpu_to_nmp_bytes','nmp_to_gpu_bytes','partial_boundary_bytes','other_boundary_bytes')},sum(r['internal_nmp_bytes'] for r in steps),sum(r['noc_bytes'] for r in steps))]:
        dint=d['gpu_to_nmp_bytes']+d['nmp_to_gpu_bytes']
        traffic.append(dict(system=system,weight_read_GB_per_token=weight/generated/1e9,KV_read_GB_per_token=kv/generated/1e9,
            boundary_GB_per_token=d['boundary_bytes']/generated/1e9,internal_NMP_GB_per_token=internal/generated/1e9,
            D_int_bytes_per_token=dint/generated,GPU_to_NMP_activation_bytes_per_token=d['gpu_to_nmp_bytes']/generated,
            NMP_to_GPU_activation_excluding_partial_bytes_per_token=(d['nmp_to_gpu_bytes']-d['partial_boundary_bytes'])/generated,
            partial_reduction_boundary_bytes_per_token=d['partial_boundary_bytes']/generated,
            other_boundary_bytes_per_token=d['other_boundary_bytes']/generated,NoC_hop_bytes_per_token=noc/generated,
            accounting='Directional activation columns are disjoint; partial is AV slab partial output; NoC hop bytes are not external D_int'))
    csv_out('traffic_breakdown.csv',traffic)
    # Saved CPA diagnostics identify final per-operator core constraints without
    # invoking Full-MAC again. Endpoint diagnostics are not full-trace timings.
    last={}
    for a in full['cpa_audit']:
        last[a['layer'],a['operator']]=a
    csv_out('full_mac_saved_operator_diagnostics.csv',[dict(layer=l,operator=op,
        endpoint_latency_s=a['after_s'],core_resource=a['after_diagnostic']['core_bottleneck'],
        **a['after_components'],source='FROZEN_CPA_ENDPOINT_AUDIT_NOT_RERUN') for (l,op),a in last.items()])
    op_rows=[]
    for op in sorted(ATTENTION):
        rs=[s for r in steps for s in r['stage_audit'] if s['operator']==op]
        op_rows.append(dict(operator=op,stage_count=len(rs),total_s=sum(r['latency_s'] for r in rs),
            **{k:sum(r[k] for r in rs) for k in ('ARRAY','LOCAL_FABRIC','MAC','INTER_REGION_NOC','EXTERNAL_BOUNDARY')},
            core_resources=','.join(sorted({r['core_resource'] for r in rs}))))
    csv_out('attention_operator_diagnostics.csv',op_rows)
    after=frozen_hashes()
    assert before==after,'Frozen source/config/artifact modified'
    passed=rows[-1]['E2E_tok_s']>rows[0]['E2E_tok_s']
    manifest=dict(base_HEAD=BASE_HEAD,case=dict(model=CASE[0],workload=CASE[1],B=8,H=2000,P=512,G=256),
        new_physical_runs=[ATTN],completed_contexts=[r['context'] for r in steps],
        references_reused=True,full_checkpoint=str(full_path.relative_to(ROOT)),prefill_reused=True,
        placement='EXISTING_CPA_QK_AV_ONLY__UNCHANGED_OBJECTIVE',accepted_moves=sum(a['accepted_moves'] for a in audit),
        gpu_memory_closure=c,GPU_timing='FROZEN_STREAMING_ROOFLINE_PER_FIXED_GRAPH_GPU_SEGMENT_PLUS_EXISTING_SMALL_OP_SERVICE',
        scheduler_modified=False,hardware_modified=False,physical_NMP_modified=False,
        frozen_hashes_unchanged=True,B32_status='UNTOUCHED',other_17_primary_cases='NOT_RUN',
        traffic_conservation='PASS',event_energy_conservation='PASS',
        energy_status='EXISTING_EVENT_PROXY__WORKLOAD_THERMAL_CLOSURE_PENDING',
        full_noc_reduction_split='NOT_RETAINED_IN_FROZEN_CHECKPOINT__NO_RECOMPUTATION',
        verdict='ATTENTION_NMP_RECOVERS_SHORT_CONTEXT_B8' if passed else 'ATTENTION_NMP_DOES_NOT_RECOVER_SHORT_CONTEXT_B8',
        probe_source_sha256=digest(Path(__file__)))
    save('manifest.json',manifest)
    save('attention_energy_events.json',dict(events=events,decode=energy,prefill_J=float(gpu_energy['prefill_J'])))
    print(json.dumps(dict(summary=rows,latency=latency,traffic=traffic,verdict=manifest['verdict']),indent=2),flush=True)


def write_report():
    """Report-only arithmetic on completed probe and frozen reference evidence."""
    def read(name):
        with (OUT/name).open() as f:
            return list(csv.DictReader(f))
    summary=read('summary.csv'); lat=read('latency_breakdown.csv'); traffic=read('traffic_breakdown.csv')
    manifest=json.loads((OUT/'manifest.json').read_text())
    _,_,full,_=references()
    audit=json.loads((OUT/'attention_cpa_audit.json').read_text())
    saved=[r for r in full['cpa_audit'] if r['operator'] in ATTENTION]
    # This deterministic optimizer's entire attention trace matches the saved
    # Full-MAC trace, including every candidate count, move, accepted objective,
    # endpoint component and geometry diagnostic. Weight stages do not share
    # live resource service with QK/AV in the fixed serial operator graph.
    clean=lambda rows:[{k:v for k,v in r.items() if k!='optimizer_runtime_s'} for r in rows]
    same_attention=clean(audit)==clean(saved)
    assert same_attention
    a=next(r for r in lat if r['system']==ATTN)
    f=next(r for r in lat if r['system']==FULL)
    f['nmp_reduction_s']=float(a['nmp_reduction_s'])
    f['nmp_noc_s']=float(f['nmp_noc_plus_reduction_s'])-float(a['nmp_reduction_s'])
    f['split_provenance']='REUSE_IDENTICAL_DETERMINISTIC_QK_AV_CPA_TRACE_AND_PROBE_REDUCTION; ALL_OTHER_NMP_OPERATORS_HAVE_ZERO_LOCAL_REDUCTION'
    resources=dict(gpu_compute_s='GPU_COMPUTE',gpu_memory_s='GPU_MEMORY',nmp_array_s='ARRAY',
        nmp_fabric_s='FABRIC',nmp_mac_s='MAC',nmp_noc_s='NOC',nmp_reduction_s='REDUCTION',boundary_transfer_s='BOUNDARY')
    for r in lat:
        r['dominant_resource']=resources[max(resources,key=lambda k:float(r[k]))]
    csv_out('latency_breakdown.csv',lat)
    names=dict(ARRAY='ARRAY',LOCAL_FABRIC='FABRIC',MAC='MAC',INTER_REGION_NOC='NOC',EXTERNAL_BOUNDARY='BOUNDARY')
    operators=read('attention_operator_diagnostics.csv')
    stage_rows=[s for line in (OUT/'attention_steps.jsonl').read_text().splitlines()
                for s in json.loads(line)['stage_audit']]
    for r in operators:
        selected=[s for s in stage_rows if s['operator']==r['operator']]
        r['physical_core_resources']=','.join(sorted({names[s['core_resource']] for s in selected}))
        r['largest_stage_component_counts']=json.dumps({names[k]:sum(s['critical_resource']==k for s in selected)
            for k in names if any(s['critical_resource']==k for s in selected)})
    csv_out('attention_operator_diagnostics.csv',operators)
    gpu_peak=manifest['gpu_memory_closure']['GPU_Bps']  # Stored for explanatory provenance only.
    c=manifest['gpu_memory_closure']
    # A generous aggregate GPU relaxation eliminates all segment boundaries and
    # small-op costs. No extra candidate is physically simulated or selected.
    all_compute=float(a['gpu_compute_s'])
    t=next(r for r in traffic if r['system']==ATTN)
    append=full['events']['array_write_bits']/8
    weight=float(t['weight_read_GB_per_token'])*1e9*2048
    optimistic_gpu=max(all_compute,(weight+append)/c['effective_Bps'])
    nmp_wall=sum(float(a[k]) for k in ('nmp_core_wall_s','nmp_noc_s','nmp_reduction_s','boundary_transfer_s'))
    relaxed_decode=optimistic_gpu+nmp_wall
    relaxed_rate=2048/(relaxed_decode+float(summary[0]['prefill_s']))
    removed_kv_streaming=float(t['KV_read_GB_per_token'])*1e9*2048/c['effective_Bps']
    manifest.update(full_noc_reduction_split=f['split_provenance'],
        QK_AV_CPA_trace_matches_frozen_full=True,QK_AV_CPA_trace_rows=len(audit),
        optimistic_aggregate_GPU_relaxation=dict(decode_s=relaxed_decode,E2E_tok_s=relaxed_rate,
            status='DIAGNOSTIC_LOWER_LATENCY_BOUND_NOT_AN_EXECUTED_POLICY',
            removes='All GPU segment boundaries and all existing small-op latency'),
        probe_source_sha256=digest(Path(__file__)))
    save('manifest.json',manifest)
    def md(rows, keys):
        def fmt(x):
            if isinstance(x,float): return f'{x:.9g}'
            return str(x)
        return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|',
            *['| '+' | '.join(fmt(r.get(k,'')) for k in keys)+' |' for r in rows]])
    compact_lat=[{k:(r[k] if k=='system' else float(r[k])) for k in ('system','gpu_compute_s','gpu_memory_s',
        'nmp_array_s','nmp_fabric_s','nmp_mac_s','nmp_noc_s','nmp_reduction_s','boundary_transfer_s','total_decode_s')} for r in lat]
    compact_traffic=[{k:(r[k] if k=='system' else float(r[k])) for k in ('system','weight_read_GB_per_token',
        'KV_read_GB_per_token','boundary_GB_per_token','internal_NMP_GB_per_token','D_int_bytes_per_token',
        'GPU_to_NMP_activation_bytes_per_token','NMP_to_GPU_activation_excluding_partial_bytes_per_token',
        'partial_reduction_boundary_bytes_per_token','other_boundary_bytes_per_token')} for r in traffic]
    d_full=float(traffic[1]['D_int_bytes_per_token']);d_att=float(traffic[2]['D_int_bytes_per_token'])
    delta=(1-d_att/d_full)*100
    passed=manifest['verdict']=='ATTENTION_NMP_RECOVERS_SHORT_CONTEXT_B8'
    result='PASS — Attention-NMP recovers this workload' if passed else 'FAIL — GPU remains faster'
    text=f'''# Attention-only NMP: 8B / W1 / B8 single-point probe

{result}. `{manifest['verdict']}`.

Scope: Llama-3.1-8B, H=2000, P=512, G=256, B=8; numerator B*G=2048.
Base frozen HEAD: `{BASE_HEAD}`. All 256 contexts (2512 through 2767) are explicitly executed.
Only Attention-NMP is newly simulated. No other workload, batch, model, reference execution, thermal sweep, scheduler, or hardware change.

## Performance and existing event energy

{md(summary,('system','E2E_s','Decode_s','E2E_tok_s','speedup_vs_M3D_GPU','speedup_vs_full_MAC_CPA','J_per_token','tokens_per_J'))}

Prefill is copied exactly from the frozen GPU result: {summary[0]['prefill_s']} s. Event energy uses existing coefficients and GPU bit proxy; no new energy model. Workload-specific thermal closure remains pending.

## Semantics and implementation audit

Existing ATTENTION_NMP already maps exactly ATTENTION_QK and ATTENTION_AV to NMP. Embedding, Q/K/V projections, O projection, FFN Gate/Up/Down, LM Head, RMSNorm, Softmax, residual and other small operators stay on GPU. KV appends remain GPU-origin memory writes.

The existing ATTENTION_NMP GPU helper inherited legacy GROUP_DIRECT/per-operator physical timing. The isolated probe adapter replaces only GPU timing with the frozen M3D aggregate streaming closure, grouping the fixed graph's contiguous GPU operators between QK/AV handoffs. This is no dynamic scheduler. Small GPU operators retain their existing GPU-local service. Weight reads are shared across the batch; FLOPs and active weight/KV byte totals are independently checked against the canonical ledger for every context. No old GPU physical latency is added to the streaming closure.

GPU effective BW = min(thermal {c['thermal_Bps']/1e12:.9f}, internal {c['internal_Bps']/1e12:.9f}, boundary {c['boundary_Bps']/1e12:.9f}, GPU {gpu_peak/1e12:.9f}) TB/s. MAT/MIV/FEOL are already in the internal service cycle; independent startup is zero.

CPA is the existing physical optimizer, enumerating only QK/AV entries while retaining complete residency/capacity lookup. Its objective, proposals, endpoint guards, request merging, and all equations are unchanged. No GPU weight operator is optimized. All {len(audit)} QK/AV CPA audit rows match the frozen Full-MAC attention trace exactly except wall-clock optimizer runtime. NMP remains 1 GHz with frozen MAC, Fabric, NoC and boundary capacities.

## Decode latency (seconds)

{md(compact_lat,tuple(compact_lat[0]))}

Component demand sums are NOT additive wall time: GPU streaming overlaps compute; NMP core is max(array, Fabric, MAC). Physical NMP stage = boundary + NoC + core + local reduction. NoC includes the existing inter-region tree communication/add rounds; the separate reduction column is the model's local tile-to-region reduction term. Full-MAC ARRAY includes the original KV append stages. Its legacy GPU_COMPUTE field consists of small-op memory service and is correctly labelled GPU memory above.

The Full-MAC checkpoint retained NoC+local reduction together. The split above reuses local AV reduction from the identical deterministic QK/AV CPA trace; all other physical operators have zero local reduction. The frozen combined component and total Decode are unchanged. Full-MAC is not rerun.

Attention GPU segment wall time = {float(a['gpu_segment_wall_s']):.9f} s; NMP core wall time = {float(a['nmp_core_wall_s']):.9f} s. See attention_operator_diagnostics.csv for QK/AV resource constraints and full_mac_saved_operator_diagnostics.csv for saved Full-MAC endpoint constraints (not extrapolated full-trace results).

An optimistic diagnostic aggregates ALL GPU work, removes ALL GPU small-op service, and retains the same physical QK/AV. Its Decode lower bound is {relaxed_decode:.9f} s, corresponding to at most {relaxed_rate:.6f} E2E tok/s under this serial GPU/NMP handoff model. This is algebra on the single run, not a second execution policy or simulation. It tests whether GPU segmentation/small-op penalties alone explain the result.

## Traffic and D_int

The GPU KV streaming service removed by offload is {removed_kv_streaming:.9f} s, while replacement physical QK/AV requires {nmp_wall:.9f} s. Weight-heavy GPU reads remain. This compares the fixed partition's benefit and cost directly, without changing any parameter.

Decimal GB = 10^9 bytes; values are per generated token (divide aggregate step traffic by B). Weight/KV reads count physical memory reads regardless of consumer; internal_NMP counts NMP-local array payload, not replicated Fabric or hop traffic. Internal Fabric and NoC hop traffic are separately observable in checkpoints/CSV.

{md(compact_traffic,tuple(compact_traffic[0]))}

External D_int is GPU→NMP activation + NMP→GPU ordinary activation + AV partial output. These columns are disjoint. Other boundary bytes are bulk GPU weight reads and/or KV append, excluded from D_int. Intra-NMP NoC hop traffic is NOT added to external D_int.

Full-MAC D_int = {d_full:.3f} bytes/token; Attention D_int = {d_att:.3f} bytes/token; reduction = {delta:.6f}%.

Full-MAC directional bytes are reconstructed from actual resident slab ownership and the frozen transfer equations, and match saved interface event totals exactly. CPA resident migration stays within each slab; all non-embedding operators occupy every slab at both context endpoints, preserving ownership throughout this context range. This byte audit does not execute Full-MAC.

Full-MAC external boundary service is {float(f['boundary_transfer_s']):.9f} s of {float(f['total_decode_s']):.9f} s Decode ({100*float(f['boundary_transfer_s'])/float(f['total_decode_s']):.2f}%). Its Fabric demand sum is {float(f['nmp_fabric_s']):.9f} s, MAC demand sum {float(f['nmp_mac_s']):.9f} s, and NoC+local reduction {float(f['nmp_noc_plus_reduction_s']):.9f} s. Thus D_int/boundary is a major cause and the largest reported serial component, not the sole cause: Fabric/MAC core service, reduction, and the GPU batching advantage also matter. A single D_int/B_link scalar does not capture the physical per-port and routing limits.

## Research decision

'''
    if passed:
        text+='Attention-only NMP may merit a third execution regime. Decide in a later task whether to add it to the formal scheduler; this probe does not change the scheduler.\n'
    else:
        text+='8B-W1-B8 is GPU-favorable under the current hardware model and existing fixed execution mechanisms. Do not introduce extra scheduling or overclocking solely to recover this point. The paper can restrict its primary scope to long-context LLM inference rather than introducing additional mechanisms solely to improve this short-context point.\n\nIf the title and central claims explicitly target long-context inference, W1 (H=2K) can move to a short-context boundary/control experiment; W2 (20K) and W3 (126K) cover two distinct long-context regimes, but alone do not prove every context length or application. Retain transparent scope and the W1 fallback evidence. This task does not remove W1 or change any benchmark.\n'
    text+='\n## Integrity and regression gates\n\nAll frozen source/config and formal artifacts are SHA-256 unchanged, including B32 and STOPPED_BY_USER states. Original reference throughput and Prefill are reused. Only this case has new Attention-NMP execution. The probe checks every context for FLOP, weight/KV, interface, directional boundary and MAC-event conservation. GPU CPA optimization is absent. Full-suite validation is recorded after completion.\n'
    (OUT/'attention_nmp_probe_report.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
    write_report()
