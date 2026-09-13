"""Slab-resolved BEOL-uniform placement diagnostic, not FEOL thermal closure."""
import numpy as np
from om3dthermal.config import PowerSourceConfig, PowerSelector, ThermalPowerSourcesConfig
from om3dthermal.bandwidth_thermal_sweep import _base_case, ARCHITECTURES
from om3dthermal.case_runner import run_steady_pipeline
from om3dthermal.thermal import solve_pcg_gpu
from om3dthermal.thermal.setup_cache import build_scene_and_signature,load_setup_cache

THERMAL_MODEL="SLAB_RESOLVED_BEOL_UNIFORM_DIAGNOSTIC"


def map_slab_power(simulation,gpu_W,slab_W):
    slab_W=np.asarray(slab_W)
    if len(slab_W)!=318 or np.any(slab_W<0) or gpu_W<0:raise ValueError("invalid placement power")
    sources=[PowerSourceConfig(name="gpu",total_power=float(gpu_W),selector=PowerSelector(component="gpu",material="FEOL"))]
    sources.extend(PowerSourceConfig(name=f"slab_{s}",total_power=float(power),
        selector=PowerSelector(tags={"role":"m3d_bitcell_beol_stack","die_index":s+1}),
        metadata={"thermal_model":THERMAL_MODEL}) for s,power in enumerate(slab_W))
    assert np.isclose(sum(s.total_power for s in sources),gpu_W+slab_W.sum(),rtol=1e-13)
    return simulation.model_copy(update={"thermal_power_sources":ThermalPowerSourcesConfig(sources=sources)})


class SlabPowerMapper:
    """Cache geometry-only source selections; preserve uniform-volume mapping."""
    def __init__(self,cells):
        self.source=np.full(len(cells),-1,dtype=np.int32)
        self.slab=np.full(len(cells),-1,dtype=np.int32)
        volume=np.array([c.volume for c in cells])
        for i,c in enumerate(cells):
            if c.component=="gpu" and c.material=="FEOL":self.source[i]=0
            if "die_index" in c.tags:
                self.slab[i]=c.tags["die_index"]-1
                if c.tags.get("role")=="m3d_bitcell_beol_stack":self.source[i]=c.tags["die_index"]
        self.active=self.source>=0
        total=np.bincount(self.source[self.active],weights=volume[self.active],minlength=319)
        if np.any(total<=0):raise ValueError("empty GPU or slab thermal source")
        self.volume=volume[self.active];self.total_volume=total

    def power(self,gpu_W,slab_W):
        values=np.r_[gpu_W,slab_W]
        result=np.zeros(len(self.source))
        ids=self.source[self.active]
        result[self.active]=values[ids]*self.volume/self.total_volume[ids]
        mapped=np.bincount(ids,weights=result[self.active],minlength=319)
        np.testing.assert_allclose(mapped,values,rtol=1e-12,atol=1e-12)
        return result


class PlacementThermalDiagnostic:
    def __init__(self,root,output):
        # Reuse frozen 4.1 geometry, mesh, boundaries and Cu material definitions.
        _,self.simulation,_=_base_case(root,next(s for s in ARCHITECTURES if s.architecture=="orthogonal_m3d_igzo"))
        self.setup=None
        self.cache=output/"thermal_setup.pkl"
        _,signature=build_scene_and_signature(self.simulation)
        self.setup,_,_=load_setup_cache(self.cache,signature)
        self.mapper=None

    def run(self,gpu_W,slab_W):
        simulation=map_slab_power(self.simulation,gpu_W,slab_W)
        if self.setup is None:
            pipeline=run_steady_pipeline(simulation,backend="gpu_pcg",setup_cache_path=self.cache,
                rtol=1e-3,max_delta_t_K=1e-2,max_iterations=100_000,check_interval=10,initial_temperature_K=293.15)
            self.setup=pipeline.reusable_setup
            r=pipeline.result
            power=pipeline.power.power_W
        else:
            if self.mapper is None:self.mapper=SlabPowerMapper(self.setup.cells)
            power=self.mapper.power(gpu_W,slab_W)
            operator=self.setup.operator_template.with_power(power)
            r=solve_pcg_gpu(operator,np.full(operator.cell_count,293.15),self.setup.boundary_table,
                relative_residual_tolerance=1e-3,max_temperature_update_tolerance=1e-2,max_iterations=100_000,check_interval=10)
        if self.mapper is None:self.mapper=SlabPowerMapper(self.setup.cells)
        if not r.converged:raise RuntimeError("placement thermal diagnostic did not converge")
        assert r.solver_info["full_vector_d2h_during_iteration"]==0
        assert r.solver_info["dtype"]=="float64"
        assert np.isclose(power.sum(),gpu_W+sum(slab_W),rtol=1e-12)
        slab_T=np.full(318,-np.inf)
        mask=self.mapper.slab>=0
        np.maximum.at(slab_T,self.mapper.slab[mask],r.temperature_K[mask]-273.15)
        hottest=int(slab_T.argmax())
        hotcell=self.setup.cells[int(r.temperature_K.argmax())]
        return dict(thermal_model=THERMAL_MODEL,Tmax_C=r.max_temperature_K-273.15,
            hotspot=f"{hotcell.component}/{hotcell.material}",hotspot_slab_id=hotcell.tags.get("die_index",0)-1,
            hottest_slab_id=hottest,hottest_slab_Tmax_C=float(slab_T[hottest]),hottest_slab_power_W=float(slab_W[hottest]),
            GPU_power_W=gpu_W,mapped_power_W=float(power.sum()),
            iterations=r.iterations,relative_residual=r.final_relative_residual,
            max_temperature_update_K=r.max_temperature_update,solve_seconds=r.solve_seconds)
