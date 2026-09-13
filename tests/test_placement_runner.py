"""Long-run resume and strict thermal-only orchestration gates."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def test_atomic_checkpoint_retries_transient_lock(runner,monkeypatch):
    calls=[]
    class Temporary:
        def replace(self,target):
            calls.append(target)
            if len(calls)<3:raise PermissionError('sync lock')
    sleeps=[]
    monkeypatch.setattr(runner.time,'sleep',sleeps.append)
    runner.replace_checkpoint(Temporary(),'target')
    assert calls==['target']*3
    assert sleeps==[.5,.5]


def test_atomic_checkpoint_retry_is_bounded(runner,monkeypatch):
    class Temporary:
        def replace(self,target):raise PermissionError('persistent lock')
    sleeps=[]
    monkeypatch.setattr(runner.time,'sleep',sleeps.append)
    with pytest.raises(PermissionError):runner.replace_checkpoint(Temporary(),'target')
    assert sleeps==[.5]*19


def test_step_cache_can_avoid_cloud_synced_output(runner,monkeypatch,tmp_path):
    local=tmp_path/'local_cache'
    monkeypatch.setenv('OM3DTHERMAL_STEP_CACHE',str(local))
    class Engine:
        def step(self,context,policy):
            return dict(context=context,latency_s=1.,boundary_bytes=0.,energy_events={},
                        slab_events={},slab_service_s=np.zeros(1),component_sums={})
    monkeypatch.setattr(runner,'engine',lambda *args:Engine())
    rows=runner.chunk(('model',1,'UNIFORM_STRIPING',126000,126001,'physical'))
    assert len(rows)==1
    assert (local/'model_B1_UNIFORM_STRIPING/126000_126001.pkl').is_file()
    assert not (runner.OUT/'step_checkpoints').exists()


@pytest.fixture
def runner(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/"scripts/compare_placement_ablation.py"
    spec=importlib.util.spec_from_file_location("placement_runner_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,"OUT",tmp_path)
    monkeypatch.setattr(module,"run_fingerprints",lambda:("run","physical"))
    monkeypatch.setattr(sys,"argv",[str(path),"--phase","thermal","--models","Llama-3.1-8B","--batches","1","--placements","BALANCED"])
    return module


def test_chunk_resume_preserves_steps_and_skips_saved_work(runner,monkeypatch):
    calls=[]
    class Engine:
        interrupt=True
        def step(self,context,policy):
            calls.append(context)
            if self.interrupt and context==126025:raise RuntimeError("interrupted")
            return dict(context=context,latency_s=context/1e6,boundary_bytes=context,
                energy_events={},slab_events={"x":np.array([context])},slab_service_s=np.array([1.]),component_sums={})
    monkeypatch.setattr(runner,"engine",lambda *args:Engine())
    args=("model",8,"BALANCED",126000,126030,"physical")
    with pytest.raises(RuntimeError,match="interrupted"):runner.chunk(args)
    Engine.interrupt=False;calls.clear()
    rows=runner.chunk(args)
    assert calls==list(range(126025,126030))
    assert [r["context"] for r in rows]==list(range(126000,126030))
    expected=json.dumps(rows,sort_keys=True,default=runner.serial)
    monkeypatch.setattr(runner,"engine",lambda *args:pytest.fail("completed checkpoint must skip engine initialization"))
    assert json.dumps(runner.chunk(args),sort_keys=True,default=runner.serial)==expected


def test_thermal_only_never_starts_missing_performance(runner,monkeypatch):
    monkeypatch.setattr(runner,"evaluate",lambda *args:pytest.fail("thermal-only started performance"))
    with pytest.raises(RuntimeError,match="Missing validated performance"):runner.main()


@pytest.mark.parametrize("audited",(False,True))
def test_old_runner_requires_explicit_revision_audit(runner,audited):
    folder=runner.OUT/"case_results";folder.mkdir()
    result=dict(computation_fingerprint="physical",fingerprint="old_runner")
    if audited:result["runner_reuse_fingerprint"]="run"
    (folder/"Llama-3.1-8B_B1_BALANCED.json").write_text(json.dumps(result))
    message="setup cache missing" if audited else "Missing validated performance"
    with pytest.raises(RuntimeError,match=message):runner.main()


@pytest.mark.parametrize("cache_exists",(False,True))
def test_thermal_setup_miss_never_rebuilds(runner,monkeypatch,cache_exists):
    from types import SimpleNamespace
    folder=runner.OUT/"case_results";folder.mkdir()
    (folder/"Llama-3.1-8B_B1_BALANCED.json").write_text(json.dumps(dict(computation_fingerprint="physical",fingerprint="run")))
    if cache_exists:(runner.OUT/"thermal_setup.pkl").touch()
    def load(*args):
        assert cache_exists
        return SimpleNamespace(setup=None)
    monkeypatch.setattr(runner,"PlacementThermalDiagnostic",load)
    with pytest.raises(RuntimeError,match="refusing an implicit setup rebuild"):runner.main()


def test_runner_edits_do_not_invalidate_physical_inputs(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/"scripts/compare_placement_ablation.py"
    spec=importlib.util.spec_from_file_location("placement_fingerprint_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/"src/om3dthermal";source.mkdir(parents=True)
    (tmp_path/"configs").mkdir()
    physical=source/"physical.py";physical.write_text("fixed physical model")
    script=tmp_path/"runner.py";script.write_text("progress v1")
    monkeypatch.setattr(module,"ROOT",tmp_path);monkeypatch.setattr(module,"__file__",str(script))
    before=module.run_fingerprints();script.write_text("progress v2")
    after=module.run_fingerprints()
    assert before[0]!=after[0] and before[1]==after[1]
    physical.write_text("changed physical model")
    assert module.run_fingerprints()[1]!=after[1]
