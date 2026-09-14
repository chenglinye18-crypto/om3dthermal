"""Audit completed matrix and optionally repeat an uncached physical case."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from om3dthermal.serving.workload_matrix import inputs, setup
from om3dthermal.serving.decode_policy import DecodePolicyModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/formal_iom3d_workload_sweep_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rerun():
    manifest = json.loads((OUT/'manifest.json').read_text())
    fp = manifest['source_fingerprint']
    key = f'Llama-3.1-8B_W1_B1_IOM3D_NO_NMP_{fp[:16]}'
    expected = [json.loads(line) for line in (OUT/'checkpoints'/f'{key}.jsonl').read_text().splitlines()]
    w, cw, _ = inputs('Llama-3.1-8B', 'W1', 1, ROOT)
    engine = DecodePolicyModel(w, project_root=ROOT, placement_policy='UNIFORM_STRIPING',
                              record_energy=True, external_bandwidth_cap=setup(ROOT)[-1])
    actual = []
    for context in cw.contexts:
        stage = engine.step(context, 'NO_NMP')
        actual.append({k:stage[k] for k in expected[0]})
    actual = json.loads(json.dumps(actual, default=lambda x:np.asarray(x).tolist()))
    assert actual == expected
    report = dict(case=key, steps=len(actual), cached_hash=digest(expected),
                  uncached_hash=digest(actual), exact_match=True)
    (OUT/'deterministic_rerun.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)


def audit(partial=False):
    manifest = json.loads((OUT/'manifest.json').read_text())
    fp = manifest['source_fingerprint']
    results = [json.loads(p.read_text()) for p in (OUT/'checkpoints').glob(f'*_{fp[:16]}.json')]
    if not partial:assert len(results) == 108
    by = {(r['summary']['model'],r['summary']['workload_id'],r['summary']['batch_size'],r['summary']['system']):r for r in results}
    assert len(by) == len(results)
    physical_steps = 0
    for key, result in by.items():
        s, e, t = (result[k] for k in ('summary','energy','traffic'))
        if s['status'] != 'EVALUATED':
            assert not e and not t and 'decode_s' not in s
            continue
        assert s['decode_generated_tokens'] == s['batch_size']*s['G']
        assert s['E2E_s'] == s['prefill_s']+s['decode_s']
        assert s['E2E_tok_s'] == s['batch_size']*s['G']/s['E2E_s']
        if s['system'] == 'HBM_GRACE_C2C':
            assert e['E2E_tokens_per_J'] is None
            assert s['Grace_resident_GB'] <= 480
            if s['HBM_only_fit']:assert t['total_C2C_GB'] == 0
            continue
        name = f'{key[0]}_{key[1]}_B{key[2]}_{key[3]}_{fp[:16]}.jsonl'
        steps = [json.loads(x) for x in (OUT/'checkpoints'/name).read_text().splitlines()]
        assert [x['context'] for x in steps] == list(range(s['H']+s['P'],s['H']+s['P']+s['G']))
        assert sum(x['latency_s'] for x in steps) == s['decode_s']
        physical_steps += len(steps)
        for phase in ('prefill','decode'):
            components = [v for k,v in e.items() if k.startswith(phase+'_') and k.endswith('_J')
                          and k not in (phase+'_J',phase+'_tokens_per_J')]
            assert sum(components) == e[phase+'_J']
        assert e['E2E_J'] == e['prefill_J']+e['decode_J']
        assert e['E2E_tokens_per_J'] == s['batch_size']*s['G']/e['E2E_J']
        assert e['decode_read_peripheral_J'] == result['events']['array_read_bits']*.04*1e-12
        if s['system'].endswith('_CPA'):
            uniform = by[(*key[:3],'IOM3D_MAC_NMP_UNIFORM')]
            for event in ('array_read_bits','array_write_bits','mac_operations','interface_bits','gpu_decode_proxy_bits'):
                assert result['events'][event] == uniform['events'][event], (key,event)
            assert s['prefill_s'] == uniform['summary']['prefill_s']
            assert result['prefill_ledger'] == uniform['prefill_ledger']
            # A legal resident group migration can change the actual Prefill
            # delivery wire length. GPU work and payload stay identical.
            for k in ('prefill_gpu_dynamic_J','prefill_gpu_static_J','prefill_interface_J',
                      'prefill_mac_J','prefill_sram_read_J','prefill_sram_write_J'):
                assert e[k] == uniform['energy'][k], (key,k)
    report = dict(logical_rows=len(results),evaluated=sum(r['summary']['status']=='EVALUATED' for r in results),
                  physical_steps=physical_steps,source_fingerprint=fp,status='PARTIAL_PASS' if partial else 'PASS')
    (OUT/('partial_result_audit.json' if partial else 'result_audit.json')).write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rerun-only',action='store_true')
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    rerun() if args.rerun_only else audit(args.partial)
