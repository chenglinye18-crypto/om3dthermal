"""Concurrent request stage service on the frozen physical resource capacities."""
import numpy as np
from .feol_energy import sum_events, sum_slab_events


def merge_external(f, transfers):
    loads=sum(t["resources"]["loads"] for t in transfers)
    startup=np.maximum.reduce([t["resources"]["startup"] for t in transfers])
    active=loads>0; serial=loads/f.port_Bps
    total=float(loads.sum()); global_s=total/f.external_Bps
    port_s=float((serial+startup).max())
    seconds=max(global_s,port_s)
    reason="GLOBAL_THERMAL_CAP" if global_s>=port_s else "PORT_SERIALIZATION" if serial.max()>=startup.max() else "ROUTE_STARTUP"
    return dict(mode="REQUEST_AGGREGATE",total_boundary_bytes=total,active_port_count=int(active.sum()),
        max_port_bytes=float(loads.max()),mean_active_port_utilization=float(serial[active].mean()/seconds) if total else 0,
        max_port_utilization=float(serial.max()/seconds) if total else 0,global_cap_serialization_s=global_s,
        port_cap_serialization_s=float(serial.max()),rc_startup_s=float(startup.max()),
        external_service_s=seconds,limiting_reason=reason,resources=dict(loads=loads,startup=startup))


def merge_noc(f, profiles):
    profiles=[p for p in profiles if p is not None]
    if not profiles: return None
    rounds=sum(p["rounds"] for p in profiles)
    startup=np.maximum.reduce([p["startup"] for p in profiles])
    adds=sum(p["adds"] for p in profiles)
    slab_s=(rounds.max(axis=(2,3))/f.link_Bps+startup+adds.max(axis=2)).sum(axis=1)
    return dict(rounds=rounds,startup=startup,adds=adds,slab_s=slab_s)


def slab_service_seconds(f, stage):
    r=stage["resources"]
    array=r["group_times"].max(axis=1)
    ext=sum((t["resources"]["loads"]/f.port_Bps+t["resources"]["startup"]).max(axis=1) for t in stage["external_transfers"])
    if stage["executor"]!="NMP": return np.maximum(array,ext)
    fabric=(r["fabric_bytes"]/f.fabric_Bps+r["fabric_startup"]).max(axis=1)
    compute=r["tile_loads"].max(axis=1)/f.tile_flops
    noc=sum(p["slab_s"] for p in (r["noc_in"],r["noc_out"]) if p is not None)
    return ext+noc+np.maximum.reduce([array,fabric,compute])+r["reduction_add_s"].max(axis=1)


def merge_request_stages(f, stages):
    """Merge demand first: no serial-B1 or constant-B1 latency scaling."""
    if len(stages)==1:return stages[0]
    resources=[s["resources"] for s in stages]
    r={k:sum(x[k] for x in resources) for k in ("group_times","tile_loads","fabric_bytes","group_bytes","reduction_add_s","sram_tile_bytes")}
    r["fabric_startup"]=sum(np.maximum.reduce([x[k] for x in resources]) for k in ("sa_startup","root_startup"))
    r["tile_active"]=np.logical_or.reduce([x["tile_active"] for x in resources])
    for key in ("noc_in","noc_out"):r[key]=merge_noc(f,[x[key] for x in resources])
    transfers=[merge_external(f,[s["external_transfers"][i] for s in stages]) for i in range(len(stages[0]["external_transfers"]))]
    nmp=stages[0]["executor"]=="NMP"
    array=float(r["group_times"].max())
    fabric=float((r["fabric_bytes"]/f.fabric_Bps+r["fabric_startup"]).max()) if nmp else 0.
    compute=float(r["tile_loads"].max()/f.tile_flops) if nmp else 0.
    core=max(array,fabric,compute)
    noc=sum(float(r[k]["slab_s"].max()) for k in ("noc_in","noc_out") if r[k] is not None)
    reduction=float(r["reduction_add_s"].max())
    external=sum(t["external_service_s"] for t in transfers)
    gpu=sum(s["components"]["GPU_COMPUTE"] for s in stages) if not nmp else 0.
    seconds=external+noc+core+reduction if nmp else max(array,external,gpu)
    components=dict(ARRAY=array,LOCAL_FABRIC=fabric,MAC=compute,INTER_REGION_NOC=noc+reduction,EXTERNAL_BOUNDARY=external,GPU_COMPUTE=gpu)
    row=dict(stages[0]);row.update(resources=r,external_transfers=transfers,latency_s=seconds,components=components,
        bottleneck=max(components,key=components.get),array_service_s=array,external_service_s=external,
        fabric_service_s=fabric,noc_s=noc)
    for k in ("boundary_bytes","input_boundary_bytes","output_boundary_bytes","local_array_bytes","fabric_bytes","noc_bytes","nmp_flops"):
        row[k]=sum(s[k] for s in stages)
    region_active=r["group_bytes"].reshape(-1,70)>0
    row["active_groups"]=int(region_active.sum())
    row["active_dies"]=int(region_active.any(axis=1).sum())
    row["active_regions"]=int(sum(region_active[:,a:b].any(axis=1).sum() for a,b in ((0,18),(18,36),(36,54),(54,70))))
    row["active_mac_tiles"]=int(r["tile_active"].sum())
    row["fabric_region_peak_Bps"]=float(r["fabric_bytes"].max()/fabric) if fabric else 0.
    row["fabric_region_average_Bps"]=float(r["fabric_bytes"].sum()/row["active_regions"]/seconds) if nmp else 0.
    row["noc_link_busy_s"]=sum(s["noc_link_busy_s"] for s in stages)
    row["noc_max_link_busy_s"]=float(row["noc_link_busy_s"].max())
    row["nmp_peak_utilization"]=row["nmp_flops"]/core/(f.layout.slab_count*32*f.tile_flops) if nmp else 0.
    row["max_buffer_chunks"]=max(s["max_buffer_chunks"] for s in stages)
    if "energy_events" in row:row["energy_events"]=sum_events(s["energy_events"] for s in stages)
    if "slab_events" in row:row["slab_events"]=sum_slab_events(s["slab_events"] for s in stages)
    sram_util=2*r["sram_tile_bytes"].max(axis=1)/seconds/(f.config["macs_per_tile"]*4*f.config["clock_hz"])
    assert sram_util.max()<=1+1e-12,"aggregate SRAM bank demand exceeds frozen nominal port rate"
    if "energy_events" in row:row["energy_events"]["max_tile_sram_port_utilization"]=float(sram_util.max())
    if "slab_events" in row:row["slab_events"]["max_tile_sram_port_utilization"]=sram_util
    row["request_ids"]=[s.get("request_id") for s in stages]
    return row
