"""Process-local v2 configuration overlay; all v1 source/artifacts stay intact."""
import os
import re
import yaml
import formal_long_context_support as base

CONFIG_PATH=base.ROOT/'configs/experiment/formal_long_context_v2.yaml'
CONFIG=yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
base.OUT=base.ROOT/'runs'/CONFIG['benchmark_id']
RUN_TAG=os.environ.get('OM3DTHERMAL_V2_RUN_TAG','')
if RUN_TAG:
    if not re.fullmatch(r'[A-Za-z0-9_-]+',RUN_TAG):raise ValueError('Invalid isolated run tag')
    base.OUT=base.OUT/RUN_TAG
base.MODELS=tuple(CONFIG['models'])
base.CONTEXTS={k:(k,h,CONFIG['incremental_prefill_tokens'],CONFIG['generated_tokens'])
               for k,h in CONFIG['contexts'].items()}
base.PATHS=tuple(CONFIG['paths'])


def v2_setup(root):
    config,*rest=base.base_setup(root)
    config=base.deepcopy(config)
    config['workloads']={k:dict(history=h,prompt=CONFIG['incremental_prefill_tokens'],generated=CONFIG['generated_tokens'])
                         for k,h in CONFIG['contexts'].items()}
    config['grace_capacity_bytes']=base.math.inf
    return config,*rest


base.setup=base.legacy.setup=base.primary.setup=v2_setup
base.primary.inputs=base.legacy.inputs


def v2_points():
    return [(m,c,b) for m in CONFIG['models'] for c in CONFIG['contexts'] for b in CONFIG['batches']]


base.points=v2_points
from formal_long_context_support import *


def cap(m,c,b,path):
    # Aggregate GPU execution has no NMP physical resident slot constraint.
    # Keep the legacy/B32 capacity implementation untouched.
    if path != 'M3D_GPU':return base.cap(m,c,b,path)
    gate=legacy.physical_capacity_gate
    try:
        legacy.physical_capacity_gate=lambda *args:(float('inf'),'',0)
        row=base.cap(m,c,b,path)
    finally:legacy.physical_capacity_gate=gate
    row['capacity_gate']='GLOBAL_PERSISTENT_PLUS_WORKSPACE'
    return row
