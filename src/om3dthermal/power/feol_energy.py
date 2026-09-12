"""Observational physical events and coefficient-only energy accounting."""
import numpy as np

SCALARS = ("read_services write_services array_read_bits array_write_bits sa_sensed_bits "
 "row_select_events column_select_events write_driver_bits sa_to_tile_bit_um sa_to_edge_bit_um "
 "root_to_tile_bit_um root_to_port_bit_um root_to_sa_bit_um noc_link_bit_um router_bit_traversals "
 "pipeline_register_bit_stages sram_read32_accesses sram_write32_accesses mac_operations "
 "fp32_reduction_adds interface_bits gpu_decode_proxy_bits activation_sram_bytes "
 "weight_bypass_sram_bytes kv_bypass_sram_bytes").split()
LAYERS = ("miv_read_bits_by_layer", "miv_write_bits_by_layer")
MAXIMA = ("max_tile_activation_chunk_bytes", "max_tile_sram_utilization", "max_tile_sram_port_utilization")

def empty_events():
    return {**dict.fromkeys(SCALARS+list(MAXIMA),0.0), **{k:[0.0]*8 for k in LAYERS}}

def sum_events(events):
    events=list(events)
    out=empty_events()
    for e in events:
        for k in SCALARS: out[k]+=e[k]
        for k in MAXIMA: out[k]=max(out[k],e[k])
        for k in LAYERS: out[k]=[a+b for a,b in zip(out[k],e[k],strict=True)]
    return out

def memory_events(layer_bytes, *, write):
    b=np.asarray(layer_bytes)
    if np.any(b<0): raise ValueError("negative physical memory traffic")
    e=empty_events()
    services=int(((b+31)//32).sum())
    bits=float(b.sum()*8)
    e["write_services" if write else "read_services"]=services
    e["array_write_bits" if write else "array_read_bits"]=bits
    e["miv_write_bits_by_layer" if write else "miv_read_bits_by_layer"]=(b.sum(axis=0)*8).tolist()
    e["row_select_events"]=e["column_select_events"]=services
    e["write_driver_bits" if write else "sa_sensed_bits"]=bits if write else services*256
    return e

def sram_events(tile_bytes, *, capacity_bytes, port_Bps, stage_s):
    b=np.asarray(tile_bytes)
    accesses=int(np.ceil(b/4).sum())
    chunk=min(float(b.max(initial=0)),capacity_bytes)
    utilization=float((2*b).max(initial=0)/stage_s/port_Bps) if stage_s else 0
    assert utilization <= 1+1e-12, "SRAM sanity demand exceeds existing bank capability"
    return dict(sram_read32_accesses=accesses,sram_write32_accesses=accesses,
                activation_sram_bytes=float(b.sum()),max_tile_activation_chunk_bytes=chunk,
                max_tile_sram_utilization=chunk/capacity_bytes,max_tile_sram_port_utilization=utilization)

def stage_events(f,w,entry,atoms,begin,write,nmp,result,occupied,counts,die,group,tiles,tile_active,noc_links):
    e=memory_events(entry.layer_bytes(atoms,begin=begin)[occupied],write=write)
    e["interface_bits"]=e["gpu_decode_proxy_bits"]=result["boundary_bytes"]*8
    for transfer in result["external_transfers"]:
        key="sa_to_edge_bit_um" if transfer["mode"]=="GROUP_DIRECT" else "root_to_port_bit_um"
        e[key]+=transfer["wire_bit_um"]
    b=counts*entry.atom_bytes
    op=entry.unit.operator_type
    if write:
        e["root_to_sa_bit_um"]=float(np.sum(b*8*f.root_sa_um[group]))
    elif nmp:
        e["sa_to_tile_bit_um"]=float(np.sum(b*8*f.sa_tile_um[group,tiles]))
        e["router_bit_traversals"]+=float(b.sum()*8)
        tile_counts=tile_active.reshape(-1,4,8).sum(axis=2)
        region_counts=(tile_counts>0).sum(axis=1)
        if op=="ATTENTION_AV":
            inputs=np.bincount(die*32+tiles,weights=counts*w.n_heads_q/w.n_heads_kv*2,minlength=tile_active.size).reshape(tile_active.shape)
            outputs=tile_active*w.d_model*4
            e["fp32_reduction_adds"]=float((np.maximum(tile_counts-1,0).sum()+np.maximum(region_counts-1,0).sum())*w.d_model)
        else:
            size=w.d_model*2 if op=="ATTENTION_QK" else entry.unit.activation_input_bytes
            inputs=tile_active*size if op!="TOKEN_EMBED_LOOKUP" else np.zeros_like(tile_active,dtype=float)
            out_bytes=(counts*w.n_heads_q/w.n_heads_kv*2 if op=="ATTENTION_QK" else b if op=="TOKEN_EMBED_LOOKUP" else counts*2)
            outputs=np.bincount(die*32+tiles,weights=out_bytes,minlength=tile_active.size).reshape(tile_active.shape)
        e["root_to_tile_bit_um"]=float(np.sum((inputs+outputs)*8*f.root_tile_um))
        e["router_bit_traversals"]+=float((inputs.sum()+outputs.sum()+noc_links.sum())*8)
        e["noc_link_bit_um"]=float(np.sum(noc_links*np.array([l["length_um"] for l in f.links])[None,:,None])*8)
        e["pipeline_register_bit_stages"]=float(np.sum(noc_links*np.array([max(l["wire_pipeline_cycles"]-1,0) for l in f.links])[None,:,None])*8)
        e["mac_operations"]=result["nmp_flops"]/2
        pe_bytes=f.energy_config["pe_sram_bytes"]
        cap=pe_bytes*f.config["macs_per_tile"]
        port=f.config["macs_per_tile"]*f.energy_config["sram_access_width_bits"]/8*f.energy_config["sram_accesses_per_cycle"]*f.config["clock_hz"]
        e.update(sram_events(inputs,capacity_bytes=cap,port_Bps=port,stage_s=result["latency_s"]))
        if op in ("ATTENTION_QK","ATTENTION_AV"): e["kv_bypass_sram_bytes"]=float(b.sum())
        elif op!="TOKEN_EMBED_LOOKUP": e["weight_bypass_sram_bytes"]=float(b.sum())
    return e

class FEOLEnergyModel:
    def __init__(self,floorplan,platform):
        self.f=floorplan; self.platform=platform; self.config=floorplan.energy_config
        self.parameters=self.config["parameters"]
        ops=floorplan.case.memory.cell_model.operations
        self.read_pj=sum(p*v for p,v in zip(self.config["read_probabilities"],(ops.read_0_pj_per_bit,ops.read_1_pj_per_bit)))
        self.write_pj=sum(p*v for p,v in zip(self.config["write_probabilities"],(ops.write_00_pj_per_bit,ops.write_01_pj_per_bit,ops.write_10_pj_per_bit,ops.write_11_pj_per_bit)))
        gpu=platform.gpu_compute_power
        self.prefill_J_per_flop=(gpu.e_compute_dynamic_J_per_FLOP_min+gpu.e_compute_dynamic_J_per_FLOP_max)/2

    def account(self,e,seconds,*,phase,policy,total_flops=0,adverse=False):
        p={k:v["value"] for k,v in self.parameters.items()}
        if adverse: p.update(self.config["NMP_ADVERSE_V1"])
        c={}
        c["array_read_J"]=e["array_read_bits"]*self.read_pj*1e-12
        c["array_write_J"]=e["array_write_bits"]*self.write_pj*1e-12
        for name,event in (("row_select","row_select_events"),("column_select","column_select_events"),
          ("sense_amplifier","sa_sensed_bits"),("write_driver","write_driver_bits"),
          ("router","router_bit_traversals"),("pipeline_register","pipeline_register_bit_stages"),
          ("sram_read","sram_read32_accesses"),("sram_write","sram_write32_accesses"),
          ("mac","mac_operations"),("reduction","fp32_reduction_adds"),("interface","interface_bits")):
            c[name+"_J"]=e[event]*(p[name]*1e-12)
        c["miv_J"]=float(np.dot(np.array(e[LAYERS[0]])+e[LAYERS[1]],self.f.miv_pj_per_bit_by_layer))*1e-12
        wire=self.f.case.architecture.feol_route.wire
        c["feol_wire_J"]=sum(v for k,v in e.items() if k.endswith("_bit_um"))*wire.activity_factor*wire.capacitance_fF_per_um*1e-15*wire.voltage_V**2
        if phase=="prefill":
            c["gpu_dynamic_J"]=total_flops*self.prefill_J_per_flop
            assert e["gpu_decode_proxy_bits"]==0
        elif phase=="decode":
            c["gpu_dynamic_J"]=e["gpu_decode_proxy_bits"]*self.platform.gpu_decode_power.e_decode_J_per_bit
        else: raise ValueError("unknown phase")
        c["gpu_static_J"]=seconds*self.platform.gpu_decode_power.static_power_W
        c["feol_unresolved_J"]=seconds*self.f.layout.slab_count*p["feol_unresolved"] if phase=="decode" and policy!="NO_NMP" else 0.0
        return dict(components=c,total_J=sum(c.values()),average_power_W=sum(c.values())/seconds)

    def audit(self):
        p={k:dict(v) for k,v in self.parameters.items()}
        def add(k,value,unit,classification,source,included,excluded):
            p[k]=dict(value=value,unit=unit,classification=classification,source=source,
                      accounting_boundary=included,included_components=included,excluded_components=excluded)
        add("array_read",self.read_pj,"pJ/payload bit","USER_FROZEN_OPERATION_TABLE_BOUNDARY","case.memory.cell_model.operations; 50:50","BEOL cell/local WL/BL","FEOL SA, selection, routes, interface")
        add("array_write",self.write_pj,"pJ/payload bit","USER_FROZEN_OPERATION_TABLE_BOUNDARY","case.memory.cell_model.operations; equal transitions","BEOL cell/local WL/BL","FEOL write driver, selectors")
        add("gpu_decode",self.platform.gpu_decode_power.e_decode_J_per_bit*1e12,"pJ/boundary bit","EXISTING_FROZEN_DECODE_PROXY",self.config["gpu_source"],"Decode dynamic proxy","Prefill FLOPs; no additional small-op energy")
        add("gpu_static",self.platform.gpu_decode_power.static_power_W,"W","EXISTING_FROZEN_DECODE_PROXY",self.config["gpu_source"],"GPU static during phase wall time","dynamic")
        add("gpu_prefill",self.prefill_J_per_flop,"J/FLOP","SOFTWARE_DERIVED_FROM_EXISTING_H200_COMPUTE_ENERGY_RANGE",self.config["gpu_source"],"midpoint FLOP energy","Decode bit proxy")
        add("miv",self.f.miv_pj_per_bit_by_layer.tolist(),"pJ/bit by actual layer","EXISTING_LAYER_DEPENDENT_ELECTRICAL_MODEL","DreamRAM length-scaled MIV primitive","row/column/data MIV electrical delivery","FEOL selection logic, lateral routing")
        wire=self.f.case.architecture.feol_route.wire
        add("wire",dict(C_fF_um=wire.capacitance_fF_per_um,V=wire.voltage_V,activity=wire.activity_factor),"alpha*C*L*V^2","EXISTING_PHYSICAL_WIRE_MODEL","case.architecture.feol_route.wire","actual physical bit-um","router, interface")
        return dict(parameters=p,config=self.config,pe_count=self.f.layout.slab_count*len(self.f.tiles)*self.f.config["macs_per_tile"],
                    total_sram_bytes=self.f.layout.slab_count*len(self.f.tiles)*self.f.config["macs_per_tile"]*self.config["pe_sram_bytes"])


def power_groups(c,seconds):
    groups=dict(GPU=("gpu_dynamic","gpu_static"),M3D_array=("array_read","array_write"),
       SA_selector=("sense_amplifier","row_select","column_select","write_driver"),
       MIV_wire=("miv","feol_wire"),interface=("interface",),MAC=("mac",),SRAM=("sram_read","sram_write"),
       NoC_reduction=("router","pipeline_register","reduction"),FEOL_unresolved=("feol_unresolved",))
    result={k+"_W":sum(c[x+"_J"] for x in names)/seconds for k,names in groups.items()}
    result["total_W"]=sum(result.values())
    return result


def prefill_events(engine,workload,ledger,bulk_events):
    """Extend existing bulk events only with the ledger's already counted bytes."""
    f=engine.floorplan; w=engine.workload
    events=list(bulk_events)
    for layer in range(w.n_layers):
        for op in ("ATTENTION_QK","ATTENTION_AV"):
            entry=engine.placement.operators[layer,op]
            events.append(engine.physical.evaluate(entry,(workload.history+workload.prompt)*w.n_heads_kv,
               begin=workload.history*w.n_heads_kv,nmp=False,write=True)["energy_events"])
    embed=engine.placement.operators[-1,"TOKEN_EMBED_LOOKUP"]
    events.append(engine.physical.evaluate(embed,1,nmp=False)["energy_events"])
    known=sum_events(events)
    for write,key in ((False,"total_read_bytes"),(True,"total_write_bytes")):
        bit_key="array_write_bits" if write else "array_read_bits"
        remaining=ledger[key]-known[bit_key]/8
        if remaining < 0: raise ValueError("physical Prefill events exceed existing ledger")
        b=engine.placement.transient_layer_bytes(remaining)
        e=memory_events(b,write=write)
        e["sa_to_edge_bit_um"]=float(np.sum(b.sum(axis=1).reshape(70,318)*f.sa_edge_um[:,None])*8)
        e["interface_bits"]=remaining*8
        events.append(e)
    result=sum_events(events)
    result["gpu_decode_proxy_bits"]=0
    assert result["interface_bits"]==ledger["total_memory_bytes"]*8
    assert result["array_read_bits"]==ledger["total_read_bytes"]*8
    assert result["array_write_bits"]==ledger["total_write_bytes"]*8
    return result
