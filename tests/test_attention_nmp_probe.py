"""Single-case probe gates; tests never launch benchmark execution."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from probe_attention_nmp import (
    ATTENTION, ATTN, CASE, FULL, OUT, AttentionEntries, digest, gpu_segment,
    require_case, on_nmp, ExecutionPolicy)


def test_attention_nmp_operator_partition():
    for op in ('Q','K','V','O','FFN_GATE','FFN_UP','FFN_DOWN','LM_HEAD',
               'TOKEN_EMBED_LOOKUP','RMSNORM','SOFTMAX','RESIDUAL_ADD','KV_APPEND'):
        assert not on_nmp(op, ExecutionPolicy.ATTENTION_NMP)
    assert all(on_nmp(op, ExecutionPolicy.ATTENTION_NMP) for op in ATTENTION)
    entries = AttentionEntries({op:SimpleNamespace(unit=SimpleNamespace(operator_type=op))
                                for op in ('Q','O','FFN_DOWN',*ATTENTION)})
    assert {e.unit.operator_type for e in entries.values()} == ATTENTION
    assert entries['Q'].unit.operator_type == 'Q'  # Other storage remains accessible.


@pytest.mark.parametrize('case', [('Llama-3.1-8B','W1',1),('Llama-3.1-8B','W1',32),
    ('Llama-3.1-8B','W2',8),('Llama-3.1-8B','W3',8),('Llama-3.1-70B','W1',8),('Llama-3.1-405B','W1',8)])
def test_attention_probe_rejects_other_cases(case):
    with pytest.raises(ValueError):
        require_case(*case)
    require_case(*CASE)


def test_attention_gpu_segment_closure_and_no_serialized_access_latency():
    rows=[dict(gpu_flops=40,gpu_memory_bytes=10,array_service_s=999),
          dict(gpu_flops=10,gpu_memory_bytes=30,external_service_s=999),
          dict(gpu_local_s=.25)]
    result=gpu_segment(rows,100,dict(effective_Bps=20))
    assert result['latency_s']==2.25
    assert result['gpu_compute_s']==.5
    assert result['gpu_memory_s']==2.25
    assert result['critical_resource']=='GPU_MEMORY'
    compute=gpu_segment(rows,10,dict(effective_Bps=20))
    assert compute['latency_s']==5.25
    assert compute['critical_resource']=='GPU_COMPUTE'


def artifacts():
    if not (OUT/'manifest.json').exists():
        pytest.skip('Probe artifact gates require the one authorized completed run')
    return json.loads((OUT/'manifest.json').read_text())


def table(name):
    with (OUT/name).open() as f:
        return list(csv.DictReader(f))


def test_attention_nmp_8b_w1_b8_accounting():
    m=artifacts()
    assert m['new_physical_runs']==[ATTN]
    assert m['completed_contexts']==list(range(2512,2768))
    assert m['other_17_primary_cases']=='NOT_RUN'
    assert m['B32_status']=='UNTOUCHED'
    assert m['references_reused'] and m['prefill_reused']
    assert not any(m[k] for k in ('hardware_modified','scheduler_modified','physical_NMP_modified'))
    s={r['system']:r for r in table('summary.csv')}
    assert set(s)=={'M3D_GPU',FULL,ATTN}
    for r in s.values():
        assert float(r['E2E_s'])==pytest.approx(float(r['Decode_s'])+float(r['prefill_s']),rel=1e-12)
        assert float(r['E2E_tok_s'])==pytest.approx(2048/float(r['E2E_s']),rel=1e-12)
        assert float(r['tokens_per_J'])*float(r['J_per_token'])==pytest.approx(1,rel=1e-12)
    assert float(s['M3D_GPU']['E2E_tok_s'])==1433.3058726305708
    assert float(s[FULL]['E2E_tok_s'])==1244.59727731527
    e=json.loads((OUT/'attention_energy_events.json').read_text())
    assert sum(e['events']['miv_read_bits_by_layer'])==e['events']['array_read_bits']
    assert sum(e['events']['miv_write_bits_by_layer'])==e['events']['array_write_bits']
    assert sum(e['decode']['components'].values())==e['decode']['total_J']
    assert e['decode']['total_J']+e['prefill_J']==pytest.approx(2048*float(s[ATTN]['J_per_token']),rel=1e-12)
    lat=next(r for r in table('latency_breakdown.csv') if r['system']==ATTN)
    assert sum(float(lat[k]) for k in ('gpu_segment_wall_s','nmp_core_wall_s','nmp_noc_s',
        'nmp_reduction_s','boundary_transfer_s'))==pytest.approx(float(lat['total_decode_s']),rel=1e-12)
    if 'optimistic_aggregate_GPU_relaxation' in m:
        bound=m['optimistic_aggregate_GPU_relaxation']
        assert bound['decode_s']<=float(s[ATTN]['Decode_s'])
        assert bound['E2E_tok_s']>=float(s[ATTN]['E2E_tok_s'])
        assert m['QK_AV_CPA_trace_matches_frozen_full']
        assert m['QK_AV_CPA_trace_rows']==512


def test_attention_nmp_boundary_conservation():
    artifacts()
    rows=table('traffic_breakdown.csv')
    for r in rows:
        directional=sum(float(r[k]) for k in ('GPU_to_NMP_activation_bytes_per_token',
            'NMP_to_GPU_activation_excluding_partial_bytes_per_token','partial_reduction_boundary_bytes_per_token'))
        assert directional==pytest.approx(float(r['D_int_bytes_per_token']),rel=1e-12)
        assert directional+float(r['other_boundary_bytes_per_token'])==pytest.approx(float(r['boundary_GB_per_token'])*1e9,rel=1e-12)
    assert len({r['weight_read_GB_per_token'] for r in rows})==1
    assert len({r['KV_read_GB_per_token'] for r in rows})==1
    r=next(r for r in rows if r['system']==ATTN)
    assert float(r['internal_NMP_GB_per_token'])==float(r['KV_read_GB_per_token'])


def test_attention_probe_frozen_sources_and_results_untouched():
    artifacts()
    hashes=json.loads((OUT/'frozen_integrity.json').read_text())
    assert any('b32_stress.csv' in p for p in hashes)
    for path,expected in hashes.items():
        assert digest(ROOT/path)==expected,path
