"""Checkpoint/resume only the six new LC64K points; reuse other LC results."""
import argparse
from pathlib import Path
import os
import pickle

import run_formal_iom3d_workloads as runner
from formal_parallel_runtime import ParallelDecodeModel
from formal_long_context_support import ROOT,OUT,MODELS,LEGACY,cap,legacy,fingerprint

PLAN_ROOT=Path(os.environ['LOCALAPPDATA'])/'om3dthermal'/'formal_long_context_plans'


class SavedPlanModel(ParallelDecodeModel):
    plan_file=None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.plan_file:
            self.plan_file.parent.mkdir(parents=True,exist_ok=True)
            with self.plan_file.open('wb') as f:pickle.dump(self.placement,f,protocol=5)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--batch',type=int,choices=(1,8))
    p.add_argument('--path',choices=('M3D_NMP_UNIFORM','M3D_NMP_CPA','M3D_GPU'))
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    runner.OUT=OUT;runner.inputs=legacy.inputs;runner.setup=legacy.setup
    runner.DecodePolicyModel=SavedPlanModel
    SavedPlanModel.workers=a.workers
    fp=fingerprint()
    for m in MODELS:
        if a.model and a.model!=m:continue
        for b in (1,8):
            if a.batch and a.batch!=b:continue
            for path in ('M3D_NMP_UNIFORM','M3D_NMP_CPA','M3D_GPU'):
                if a.path and a.path!=path:continue
                SavedPlanModel.plan_file=PLAN_ROOT/f'{m}_LC64K_B{b}_{path}_{fp[:16]}.pkl'
                try:runner.run_case((cap(m,'LC64K',b,path),fp))
                finally:
                    for e in list(SavedPlanModel.active):e.close()


if __name__=='__main__':main()
