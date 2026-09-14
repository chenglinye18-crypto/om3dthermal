"""Runtime sharing must preserve every physical result, including B32."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from formal_parallel_runtime import ParallelDecodeModel
from om3dthermal.serving.decode_policy import DecodePolicyModel,llama31_models


@pytest.mark.parametrize('batch,placement,policy',[(1,'UNIFORM_STRIPING','NO_NMP'),
    (8,'UNIFORM_STRIPING','MAC_NMP'),(32,'UNIFORM_STRIPING','MAC_NMP'),
    (8,'CRITICAL_PATH_AWARE','MAC_NMP')])
def test_shared_contexts_are_exact(batch,placement,policy):
    w=llama31_models()['Llama-3.1-8B'].model_copy(update={'batch_size':batch,'n_layers':1,'context_length':2520})
    args=dict(project_root=ROOT,placement_policy=placement,record_energy=True,decode_start_context=2512)
    reference=DecodePolicyModel(w,**args)
    parallel=ParallelDecodeModel(w,**args);parallel.workers=2
    normalize=lambda row:json.loads(json.dumps(row,default=lambda a:np.asarray(a).tolist()))
    try:
        for context in range(2512,2520):
            assert normalize(parallel.step(context,policy))==normalize(reference.step(context,policy))
        assert not parallel._directory.exists()
        assert parallel not in ParallelDecodeModel.active
    finally:parallel.close()
