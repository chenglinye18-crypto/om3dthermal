"""Exact die event projection without rerunning saved physical Decode latency.

Frozen energy primitives are reused. Over these <=512-token windows each
cyclic lane gains at most one atom. Integrating that atom's exact birth context
preserves all integer layer services; no Decode timestep is extrapolated.
"""
from copy import copy
import numpy as np
from om3dthermal.power.feol_energy import slab_stage_events,SCALARS,LAYERS,MAXIMA
from om3dthermal.power.nmp_die_activity import external_service,structured_noc


def integrated_layers(entry,first,last,heads):
    count=last-first+1
    if hasattr(entry,'base'):
        base=entry.base
        result=integrated_layers(base,first,last,heads)
        # Frozen CPA moves a resident prefix. That prefix is already active
        # throughout a long-context timed window, so its contribution is exact.
        assert np.all(entry.move_count<=base.prefix_counts(first*heads)[entry.source])
        moved=np.maximum(0,(entry.move_count[:,None]+7-base.layer_order[entry.source])//8)*base.atom_bytes*count
        np.add.at(result,entry.source,-moved);np.add.at(result,entry.destination,moved)
        assert np.all(result>=0)
        return result
    assert not hasattr(entry,'segment_count'),'Formal path must be Uniform/CPA'
    begin=entry.prefix_counts(first*heads);end=entry.prefix_counts(last*heads)
    assert np.all((end-begin>=0)&(end-begin<=1))
    result=entry.layer_bytes(first*heads)*count
    threshold=(entry.order+begin*(entry.die_count*70)+1+heads-1)//heads
    extra=np.maximum(0,last-np.maximum(threshold,first)+1)
    assert np.all(extra<=count)
    layer=(begin+entry.start_layers)%8
    result[np.arange(len(begin)),layer]+=extra*entry.atom_bytes
    assert result.sum()==sum(range(first,last+1))*heads*entry.atom_bytes
    return result


def stage_projection(f,w,entry,total_layers,*,nmp,write=False,repetitions=1):
    """Apply existing slab event equations to exactly integrated linear demands.

    A single support pattern must hold for repeated NMP events. Rounding of
    services and SRAM words is applied to the original whole-atom counts, not
    to fractional average counts. No service-time model is evaluated.
    """
    lb=np.asarray(total_layers,dtype=float)/repetitions
    counts=lb.sum(axis=1)/entry.atom_bytes
    occupied=counts>0
    counts=counts[occupied];lb=lb[occupied]
    die=entry.die_ids[occupied];group=entry.group_ids[occupied];tiles=entry.tile_ids[occupied]
    rid=entry.region_dest[occupied];n=f.layout.slab_count
    b=counts*entry.atom_bytes
    group_bytes=np.zeros((n,70));group_bytes[die,group]=b
    region=np.bincount(rid,weights=b,minlength=n*4).reshape(n,4)
    active=region>0;dies=active.any(axis=1)
    tile_active=np.zeros((n,32),dtype=bool);tile_active[die,tiles]=True
    tc=tile_active.reshape(n,4,8).sum(axis=2)
    op=entry.unit.operator_type
    flops_atom=2*w.n_heads_q*w.d_head/w.n_heads_kv if op.startswith('ATTENTION') else entry.unit.local_flops/entry.atom_count
    if entry.unit.shard_mode=='ROW_PARALLEL':flops_atom*=w.batch_size
    loads=np.bincount(die*32+tiles,weights=counts*flops_atom,minlength=n*32).reshape(n,32)
    links=np.zeros((n,3,2))
    if nmp:
        if op=='ATTENTION_AV':
            incoming=np.bincount(rid,weights=counts*w.n_heads_q/w.n_heads_kv*2,minlength=n*4).reshape(n,4)
            input_mode='REGION_DIRECT'
            tree=structured_noc(f,active*w.d_model*4,mode='reduce',vector_bytes=w.d_model*4)
            links+=tree['link_bytes']
            outgoing=np.zeros((n,4));outgoing[np.arange(n),tree['root_regions']]=dies*w.d_model*4
            sram=np.bincount(die*32+tiles,weights=counts*w.n_heads_q/w.n_heads_kv*2,minlength=n*32).reshape(n,32)
        else:
            size=w.d_model*2 if op=='ATTENTION_QK' else entry.unit.activation_input_bytes
            incoming=active*size;input_mode='REGION_BROADCAST_INGRESS'
            links+=structured_noc(f,incoming,mode='multicast')['link_bytes']
            output=counts*w.n_heads_q/w.n_heads_kv*2 if op=='ATTENTION_QK' else b if op=='TOKEN_EMBED_LOOKUP' else counts*2*w.batch_size
            outgoing=np.bincount(rid,weights=output,minlength=n*4).reshape(n,4)
            sram=tile_active*size if op!='TOKEN_EMBED_LOOKUP' else np.zeros((n,32))
        transfers=[external_service(f,incoming,mode=input_mode,record_events=True,record_resources=True),
                    external_service(f,outgoing,mode='REGION_DIRECT',record_events=True,record_resources=True)]
    else:
        transfers=[external_service(f,region if write else group_bytes,mode='REGION_DIRECT' if write else 'GROUP_DIRECT',record_events=True,record_resources=True)]
        sram=np.zeros((n,32))
    result=dict(external_transfers=transfers,latency_s=1.,resources=dict(tile_loads=loads if nmp else np.zeros_like(loads)))
    e=slab_stage_events(f,w,entry,0,0,write,nmp,result,occupied,counts,die,group,tiles,tile_active,links,lb)
    for k in SCALARS+list(LAYERS):e[k]*=repetitions
    # Every original layer payload is a multiple of 32 bytes. All original
    # activation payloads are whole 32-bit words. Average-then-ceil is invalid.
    local=np.bincount(die,weights=lb.sum(axis=1)*repetitions,minlength=n)
    services=local/32
    e['write_services' if write else 'read_services']=services
    e['row_select_events']=e['column_select_events']=services.copy()
    if not write:e['sa_sensed_bits']=local*8
    if nmp:
        words=sram.sum(axis=1)*repetitions/4
        assert np.all(words==np.floor(words))
        e['sram_read32_accesses']=words;e['sram_write32_accesses']=words.copy()
    return e


def add(events):
    return {k:sum(e[k] for e in events) for k in SCALARS+list(LAYERS)}


def project_decode(placement,cw,*,nmp):
    f,w=placement.floorplan,placement.workload
    first=cw.history+cw.prompt;last=first+cw.generated-1
    events=[]
    for entry in placement.request_operators.values():
        op=entry.unit.operator_type
        if op=='OTHER_WEIGHT':continue
        if op.startswith('ATTENTION'):
            start=entry.prefix_counts(first*w.n_heads_kv)>0
            end=entry.prefix_counts(last*w.n_heads_kv)>0
            assert np.array_equal(start,end),'Support changes require exact event partitions'
            layers=integrated_layers(entry,first,last,w.n_heads_kv)
            events.append(stage_projection(f,w,entry,layers,nmp=nmp,repetitions=cw.generated))
            layers=entry.layer_bytes((last+1)*w.n_heads_kv,begin=first*w.n_heads_kv)
            events.append(stage_projection(f,w,entry,layers,nmp=False,write=True))
        else:
            begin=entry.unit.request_id if op=='TOKEN_EMBED_LOOKUP' else 0
            atoms=begin+1 if op=='TOKEN_EMBED_LOOKUP' else entry.atom_count
            layers=entry.layer_bytes(atoms,begin=begin)*cw.generated
            events.append(stage_projection(f,w,entry,layers,nmp=nmp,repetitions=cw.generated))
    return add(events)


def project_prefill(placement,cw,ledger):
    f,w=placement.floorplan,placement.workload
    events=[]
    for entry in placement.request_operators.values():
        op=entry.unit.operator_type
        if op=='OTHER_WEIGHT':continue
        begin=entry.unit.request_id if op=='TOKEN_EMBED_LOOKUP' else 0
        atoms=(cw.history*w.n_heads_kv if op.startswith('ATTENTION') else begin+1 if op=='TOKEN_EMBED_LOOKUP' else entry.atom_count)
        events.append(stage_projection(f,w,entry,entry.layer_bytes(atoms,begin=begin),nmp=False))
        if op.startswith('ATTENTION'):
            lb=entry.layer_bytes((cw.history+cw.prompt)*w.n_heads_kv,begin=cw.history*w.n_heads_kv)
            events.append(stage_projection(f,w,entry,lb,nmp=False,write=True))
    known=add(events)
    for write,key in ((False,'total_read_bytes'),(True,'total_write_bytes')):
        remaining=ledger[key]-known['array_write_bits' if write else 'array_read_bits'].sum()/8
        assert remaining>=0
        b=placement.transient_layer_bytes(remaining).reshape(70,318,8)
        e={k:np.zeros((318,8)) if k in LAYERS else np.zeros(318) for k in SCALARS+list(LAYERS)}
        local=b.sum(axis=(0,2));services=((b+31)//32).sum(axis=(0,2))
        e['write_services' if write else 'read_services']=services
        e['array_write_bits' if write else 'array_read_bits']=local*8
        e['miv_write_bits_by_layer' if write else 'miv_read_bits_by_layer']=b.sum(axis=0)*8
        e['row_select_events']=e['column_select_events']=services.copy()
        e['write_driver_bits' if write else 'sa_sensed_bits']=local*8 if write else services*256
        e['sa_to_edge_bit_um']=(b.sum(axis=2)*f.sa_edge_um[:,None]).sum(axis=0)*8
        e['interface_bits']=local*8
        events.append(e)
    result=add(events);result['gpu_decode_proxy_bits'][:]=0
    return result


def audit_global(slab,expected):
    for k in SCALARS:
        np.testing.assert_allclose(slab[k].sum(),expected[k],rtol=1e-12,atol=1e-7,err_msg=k)
    for k in LAYERS:
        np.testing.assert_allclose(slab[k].sum(axis=0),expected[k],rtol=1e-12,atol=1e-7,err_msg=k)
