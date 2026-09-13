"""Full 1000-step Uniform regression against the immutable 920efd2 fixture."""
import argparse,json
import compare_placement_ablation as base

base.OUT=base.ROOT/'runs/critical_path_placement_b1_v1/uniform_regression'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--models',nargs='+',choices=list(base.llama31_models()),default=list(base.llama31_models()))
    args=parser.parse_args()
    base.OUT.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((base.ROOT/'tests/data/cpa_uniform_920efd2.json').read_text())['cases']
    for name in args.models:
        result=base.evaluate(name,1,'UNIFORM_STRIPING',args.workers)
        for key in ('summary','events','energy','stage_hash'):
            assert result[key]==frozen[name][key],(name,key,'UNIFORM_REGRESSION_STOP')
        reference=base.ROOT/'runs/placement_ablation_b1_b8_v1/case_results'/f'{name}_B1_UNIFORM_STRIPING.json'
        if reference.exists():
            assert result['slabs']==json.loads(reference.read_text())['slabs'],(name,'UNIFORM_SLAB_POWER_CHANGED_STOP')
        result['thermal']=frozen[name]['thermal']
        (base.OUT/f'{name}.json').write_text(json.dumps(result,default=base.serial)+'\n')
        print('FULL 1000-STEP EXACT UNIFORM GATE PASSED '+name,flush=True)


if __name__=='__main__':main()
