"""Read-only mmap sharing of a frozen engine across context workers.

The scientific model and result cache key are unchanged. This optional runtime
only schedules independent, fully evaluated contexts and restores their order.
"""
from copy import copy
import gc
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from om3dthermal.serving.decode_policy import DecodePolicyModel

ENGINE = None
MAPPING = None


class ArrayWriter(pickle.Pickler):
    def __init__(self, stream, arrays):
        super().__init__(stream, protocol=5)
        self.arrays, self.seen = arrays, {}

    def persistent_id(self, value):
        if not isinstance(value, np.ndarray):return None
        if id(value) not in self.seen:
            assert not value.dtype.hasobject
            padding = (-self.arrays.tell()) % 64
            self.arrays.write(b'\0'*padding)
            offset = self.arrays.tell()
            self.arrays.write(np.ascontiguousarray(value).tobytes())
            self.seen[id(value)] = ('array',offset,value.shape,value.dtype.str)
        return self.seen[id(value)]


class ArrayReader(pickle.Unpickler):
    def persistent_load(self, key):
        tag,offset,shape,dtype = key
        assert tag == 'array'
        return np.ndarray(shape,dtype=dtype,buffer=MAPPING,offset=offset)


def initialize(directory):
    global ENGINE, MAPPING
    directory=Path(directory)
    MAPPING=np.memmap(directory/'arrays.bin',mode='r',dtype=np.uint8)
    with (directory/'engine.pkl').open('rb') as stream:ENGINE=ArrayReader(stream).load()


def evaluate(task):
    context,policy=task
    return ENGINE.step(context,policy)


class ParallelDecodeModel(DecodePolicyModel):
    workers = 4
    active = set()

    def step(self, context, policy, *, include_stages=False):
        assert not include_stages
        if not hasattr(self,'_ordered'):
            self.active.add(self)
            # Warm shared linear stages once. No context is approximated.
            first=super().step(context,policy)
            self.dynamic.clear();self.context=None
            parent=Path(os.environ.get('LOCALAPPDATA',tempfile.gettempdir()))/'om3dthermal'/'formal_shared'
            parent.mkdir(parents=True,exist_ok=True)
            self._directory=Path(tempfile.mkdtemp(prefix='engine_',dir=parent))
            frozen=copy(self);frozen.__class__=DecodePolicyModel
            with (self._directory/'arrays.bin').open('wb') as arrays, (self._directory/'engine.pkl').open('wb') as stream:
                ArrayWriter(stream,arrays).dump(frozen)
            del frozen
            self.placement=None;self.physical=None;self.static={};self.dynamic={}
            gc.collect()
            self._pool=ProcessPoolExecutor(max_workers=self.workers,initializer=initialize,initargs=(str(self._directory),))
            self._ordered=iter(self._pool.map(evaluate,((c,policy) for c in range(context+1,self.workload.context_length))))
            self._next=context+1
            print(f'SHARED_CONTEXT_WORKERS {self.workers}; mmap_bytes={(self._directory/"arrays.bin").stat().st_size}',flush=True)
            if self._next==self.workload.context_length:self.close()
            return first
        assert context==self._next
        self._next+=1
        result=next(self._ordered)
        assert result['context']==context
        if self._next==self.workload.context_length:self.close()
        return result

    def close(self):
        if hasattr(self,'_pool'):
            self._pool.shutdown(wait=True,cancel_futures=True)
            del self._pool
        if hasattr(self,'_directory') and self._directory.exists():
            target=self._directory.resolve()
            expected=Path(os.environ.get('LOCALAPPDATA',tempfile.gettempdir())).resolve()/'om3dthermal'/'formal_shared'
            assert target.parent==expected.resolve() and target.name.startswith('engine_')
            shutil.rmtree(target)
        self.active.discard(self)
