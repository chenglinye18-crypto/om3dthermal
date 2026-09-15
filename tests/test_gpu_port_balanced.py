from pathlib import Path
import hashlib
import numpy as np
import pytest
from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
from om3dthermal.placement.nmp_load_balance import PhysicalResidentPlacement
from om3dthermal.placement.gpu_port_balanced import port_first_lanes
from om3dthermal.serving.decode_policy import DecodePolicyModel, llama31_models

ROOT=Path(__file__).resolve().parents[1]


def test_existing_policy_prechange_signatures():
    # Recorded on ea5aa97 before adding the GPU-only branch. These compare
    # every physical slot, lane, layer, tile and resident prefix count.
    expected={'UNIFORM_STRIPING':'dfb58728b1f925d02001678079eca9eb4572e3b74afa79d5b0a9b276f02ea1b0',
              'BALANCED':'9d8c0f081b2b0da964beb6d06b7414a86d82b7270186ae2678defbbeba89737c',
              'CRITICAL_PATH_AWARE':'dfb58728b1f925d02001678079eca9eb4572e3b74afa79d5b0a9b276f02ea1b0'}
    w=llama31_models()['Llama-3.1-8B'].model_copy(update={'context_length':20160})
    floor=resolve_feol_floorplan(ROOT)
    for policy,signature in expected.items():
        p=PhysicalResidentPlacement(w,floor,policy)
        h=hashlib.sha256(p.slot_used.tobytes())
        for entry in p.request_operators.values():
            for a in (entry.lanes,entry.start_layers,entry.tile_ids,entry.prefix_counts(entry.atom_count)):
                h.update(a.tobytes())
        assert h.hexdigest()==signature


def test_capacity_bijection_and_logical_prefixes():
    w=llama31_models()['Llama-3.1-8B'].model_copy(update={'context_length':20160,'batch_size':8})
    floor=resolve_feol_floorplan(ROOT)
    old=PhysicalResidentPlacement(w,floor,'UNIFORM_STRIPING')
    new=PhysicalResidentPlacement(w,floor,'GPU_PORT_BALANCED')
    perm,ports=port_first_lanes(floor)
    np.testing.assert_array_equal(new.slot_used[perm],old.slot_used)
    assert new.audit()['slot_capacity_violations']==0
    for key,a in old.request_operators.items():
        b=new.request_operators[key]
        for end in (1,min(a.atom_count,20128*8),a.atom_count):
            np.testing.assert_array_equal(a.prefix_counts(end),b.prefix_counts(end))
            np.testing.assert_array_equal(a.layer_bytes(end),b.layer_bytes(end))
        assert a.atom_bytes==b.atom_bytes
        np.testing.assert_array_equal(b.lanes,perm[a.lanes])
    for layer in range(w.n_layers):
        for request in range(8):
            np.testing.assert_array_equal(new.get(layer,'ATTENTION_QK',request).lanes,
                                          new.get(layer,'ATTENTION_AV',request).lanes)
    # First round occupies distinct reachable (slab,port) resources.
    first=perm[:floor.layout.slab_count*len(set(ports))]
    assert len(set(zip(first%floor.layout.slab_count,ports[first//floor.layout.slab_count])))==len(first)


def test_gpu_only_and_boundary_conservation():
    w=llama31_models()['Llama-3.1-8B'].model_copy(update={'context_length':20160})
    a=DecodePolicyModel(w,project_root=ROOT,placement_policy='UNIFORM_STRIPING',record_energy=True)
    b=DecodePolicyModel(w,project_root=ROOT,placement_policy='GPU_PORT_BALANCED',record_energy=True)
    x=a.step(20128,'NO_NMP'); y=b.step(20128,'NO_NMP')
    assert y['latency_s']<x['latency_s']
    assert x['boundary_bytes']==y['boundary_bytes']==y['local_array_bytes']
    for key in ('array_read_bits','array_write_bits','interface_bits','gpu_decode_proxy_bits'):
        assert x['energy_events'][key]==y['energy_events'][key]
    for key in ('MAC','LOCAL_FABRIC','INTER_REGION_NOC'):
        assert y['component_sums'][key]==0
    for policy in ('MAC_NMP','ATTENTION_NMP'):
        with pytest.raises(ValueError,match='only legal for GPU'):
            b.step(20128,policy)
