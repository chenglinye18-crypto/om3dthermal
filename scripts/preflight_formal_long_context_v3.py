"""Capacity preflight only; never executes or changes an existing benchmark."""
from pathlib import Path
import csv
import gc
import json
import math
import hashlib
import urllib.request
import numpy as np
import yaml

from om3dthermal.workload.model_registry import load_dense_model_spec
from om3dthermal.serving.decode_policy import llama31_models
from om3dthermal.serving.decode_policy import CachedWorkload
from om3dthermal.serving.cached_history_wave import resident_limit
from om3dthermal.serving.workspace import (
    WorkspaceExecutionConfig, evaluate_prefill_workspace, evaluate_decode_workspace,
)
from om3dthermal.serving.mixed_phase_e2e import resolve_conventional_hbm_backend
from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
from om3dthermal.placement.nmp_load_balance import PhysicalResidentPlacement

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/formal_long_context_v3'


def audit_models():
    config_url='https://huggingface.co/Qwen/Qwen2.5-32B/raw/main/config.json'
    index_url='https://huggingface.co/Qwen/Qwen2.5-32B/raw/main/model.safetensors.index.json'
    config_bytes=urllib.request.urlopen(config_url,timeout=60).read()
    index_bytes=urllib.request.urlopen(index_url,timeout=60).read()
    config,index=json.loads(config_bytes),json.loads(index_bytes)
    spec=load_dense_model_spec(ROOT/'configs/workload/models/qwen25_32b.yaml')
    for key,field in [('hidden_size','d_model'),('intermediate_size','d_ff'),('num_hidden_layers','n_layers'),
        ('num_attention_heads','n_heads_q'),('num_key_value_heads','n_heads_kv'),('vocab_size','vocab_size')]:
        assert config[key]==getattr(spec,field)
    d=spec.d_model;k=spec.n_heads_kv*(d//spec.n_heads_q)
    parameters=spec.n_layers*(2*d*d+2*d*k+3*d*spec.d_ff+2*d+d+2*k)+2*d*spec.vocab_size+d
    assert parameters==spec.n_param==index['metadata']['total_size']//2
    assert not config['tie_word_embeddings'] and not config['use_sliding_window']
    assert config['max_position_embeddings']>=126160
    value=dict(model='Qwen2.5-32B',parameter_count=parameters,official_config=config,
        official_weight_bytes=index['metadata']['total_size'],QKV_bias_tensors=sum(k.endswith('.bias') for k in index['weight_map']),
        sources=[config_url,index_url],config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        index_sha256=hashlib.sha256(index_bytes).hexdigest(),audit_status='PASS',
        llama31_8b_canonical_parameters=llama31_models()['Llama-3.1-8B'].n_param)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'model_parameter_audit.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    return value


def footprint(spec, history, batch, prompt, generated, workspace, *, nmp=False, dies=0):
    pre = evaluate_prefill_workspace(spec, batch_size=batch, context_length=prompt, config=workspace).peak_bytes
    dec = evaluate_decode_workspace(spec, batch_size=batch, context_length=history+prompt+generated-1,
        config=workspace, proposed_nmp=nmp, nmp_die_count=dies).peak_bytes
    scratch = math.ceil(max(pre, dec)/32)*32
    kv = 2*spec.n_layers*spec.n_heads_kv*(spec.d_model//spec.n_heads_q)*(spec.kv_bits//8)
    weights = spec.n_param*(spec.weight_bits//8)
    persistent = weights+batch*(history+prompt+generated)*kv
    return dict(weight_bytes=weights, final_KV_bytes=persistent-weights,
                prefill_workspace_bytes=pre, decode_workspace_bytes=dec,
                workspace_bytes=scratch, peak_bytes=persistent+scratch)


def main():
    audit_models()
    config = yaml.safe_load((ROOT/'configs/experiment/formal_long_context_v3.yaml').read_text())
    old = yaml.safe_load((ROOT/'configs/experiment/formal_iom3d_workload_sweep_v1.yaml').read_text())
    workspace = WorkspaceExecutionConfig(**yaml.safe_load((ROOT/old['workspace_source']).read_text())['workspace'])
    floor = resolve_feol_floorplan(ROOT)
    hbm = resolve_conventional_hbm_backend(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'benchmark_definition.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    rows = []
    for model, source in config['models'].items():
        spec = load_dense_model_spec(ROOT/source)
        if model == 'Llama-3.1-8B':
            spec = spec.model_copy(update={'n_param': llama31_models()[model].n_param})
        for context, history in config['contexts'].items():
            for batch in config['batches']:
                p, g = config['incremental_prefill_tokens'], config['generated_tokens']
                f = footprint(spec, history, batch, p, g, workspace, nmp=True, dies=floor.layout.slab_count)
                safe = resident_limit(spec, CachedWorkload(history,p,g), workspace, hbm.capacity_bytes)
                wave_sizes = [min(safe, batch-i) for i in range(0, batch, safe)] if safe else []
                row = dict(model=model, workload=context, H=history, P=p, G=g, B=batch, **f,
                    HBM_capacity_bytes=int(hbm.capacity_bytes), M3D_capacity_bytes=floor.layout.total_capacity_bytes,
                    HBM_full_batch_fit=f['peak_bytes'] <= hbm.capacity_bytes,
                    safe_resident_batch=safe, wave_sizes=json.dumps(wave_sizes), num_waves=len(wave_sizes),
                    M3D_status='PASS', physical_slot_excess_bytes=0, physical_error='')
                plan = PhysicalResidentPlacement.__new__(PhysicalResidentPlacement)
                if f['peak_bytes'] > floor.layout.total_capacity_bytes:
                    row['M3D_status'] = 'CAPACITY_INFEASIBLE'
                else:
                    try:
                        plan.__init__(spec.decode_input(batch_size=batch, context_length=history+p+g), floor, 'UNIFORM_STRIPING')
                        free = int((((floor.layout.slot_capacity_bytes-plan.slot_used)*4//32)*32).sum())
                        if free < f['workspace_bytes']:
                            row['M3D_status'] = 'CAPACITY_INFEASIBLE'
                    except ValueError as exc:
                        if 'capacity exceeded' not in str(exc):
                            raise
                        row['M3D_status'] = 'CAPACITY_INFEASIBLE'
                        row['physical_error'] = str(exc)
                        if hasattr(plan, 'slot_used'):
                            row['physical_slot_excess_bytes'] = int(np.maximum(plan.slot_used-floor.layout.slot_capacity_bytes, 0).sum())*4
                rows.append(row)
                del plan
                gc.collect()
                with (OUT/'capacity_audit.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                print(model, context, batch, row['M3D_status'], row['physical_error'], flush=True)
    failures = [r for r in rows if r['M3D_status'] != 'PASS']
    print(json.dumps(dict(operating_points=len(rows), M3D_infeasible=len(failures))), flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
