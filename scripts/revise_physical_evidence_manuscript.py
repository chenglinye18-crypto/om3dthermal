"""Preserve the original Word package and edit only presentation paragraphs.

The source is the user-maintained Word manuscript beside the repository.
Build a separate repository copy; Word performs final layout/PDF export.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
from xml.dom import minidom

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/manuscript'


def text(node):
    return ''.join(n.firstChild.data if n.firstChild else '' for n in node.getElementsByTagName('w:t'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);a=parser.parse_args()
    source=a.source or ROOT.parent/'conference-template-letter (1) (已自动恢复).docx'
    expected='2806982b775fced598b44f30b63d1e0c1fe65419da5750859dd8201b004e2bd9'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
        raise ValueError('Source differs from the reviewed manuscript; paragraph-index edits must be reviewed first.')
    with zipfile.ZipFile(source) as z: files={n:z.read(n) for n in z.namelist()}
    doc=minidom.parseString(files['word/document.xml']);body=doc.getElementsByTagName('w:body')[0]
    original=[n for n in body.childNodes if n.nodeType==n.ELEMENT_NODE]
    oldtext={i:text(n) for i,n in enumerate(original)}
    edits=[]
    def replace(i,value):
        node=original[i]
        for child in list(node.childNodes):
            if child.nodeType==child.ELEMENT_NODE and child.tagName!='w:pPr':node.removeChild(child)
        run=doc.createElement('w:r');t=doc.createElement('w:t');t.setAttribute('xml:space','preserve');t.appendChild(doc.createTextNode(value));run.appendChild(t);node.appendChild(run)
        edits.append(dict(index=i,before=oldtext[i],after=value))
    def remove(i):
        body.removeChild(original[i]);edits.append(dict(index=i,before=oldtext[i],after=''))
    def insert_before(i,value,style_index=85):
        node=original[style_index].cloneNode(deep=True)
        for child in list(node.childNodes):
            if child.nodeType==child.ELEMENT_NODE and child.tagName!='w:pPr':node.removeChild(child)
        run=doc.createElement('w:r');t=doc.createElement('w:t');t.appendChild(doc.createTextNode(value));run.appendChild(t);node.appendChild(run)
        body.insertBefore(node,original[i]);edits.append(dict(index='before '+str(i),before='',after=value))
    replace(5,'Keywords—monolithic 3D memory, orthogonal integration, IGZO, near-memory processing, long-context inference')
    replace(4,oldtext[4].replace('remain below 81.4°C','reach at most 81.41°C'))
    replace(9,'IGZO 2T0C memory combines low leakage with low-temperature sequential integration, enabling dense BEOL storage above Si logic [1]. This physical separation leaves the underlying FEOL available for local MAC, SRAM, and control. Orthogonal integration can therefore scale resident capacity while the short BEOL–FEOL path supplies computation beneath the stored weights and KV cache.')
    replace(10,'We present IOM3D-HBM, an architecture integrating orthogonal packaging, monolithic-3D IGZO memory, and underlying Si FEOL near-memory processing (NMP). Its contributions are threefold.')
    replace(11,'First, the integrated physical hierarchy couples thermally scalable orthogonal capacity with dense BEOL storage and local Si MAC execution. Large resident weights and KV data reach FEOL computation through the local BEOL–FEOL path instead of repeatedly traversing the GPU-facing interface.')
    replace(12,'Second, architecture-aware execution and placement exploit this locality. Distributed NMP Scheduling (DNS) keeps Prefill and irregular/global Decode operators on the GPU and maps partitionable MAC operators near resident operands. Critical-Path-Aware Placement (CPA) refines uniform DNS ownership using modeled resource bottlenecks and communication contention.')
    replace(13,'Third, evaluation under physical and thermal constraints demonstrates 1.49 TB capacity, approximately 3.40 TB/s thermally sustainable external bandwidth, and 7.28× E2E throughput and 4.75× energy efficiency over conventional HBM-GPU, with NMP temperatures at most 81.41°C. A fine-grained physical execution model resolves memory service, shared communication resources, NMP execution, energy, and workload-dependent steady-state temperature.')
    remove(14)
    replace(17,'The inference working set comprises model weights plus a KV cache that grows with context length and batch size. For example, Llama-3.1-8B at 128K tokens requires approximately 16 GiB of FP16 KV storage per request. Larger models, longer histories, and greater concurrency progressively exceed local HBM capacity; the evaluated workload matrix includes both HBM-resident and overflowing cases.')
    replace(20,'Fig. 1. Long-context inference demands: (a) model and KV-cache capacity; (b) modeled memory traffic and arithmetic intensity during Decode and Prefill.')
    replace(23,oldtext[23].replace('Fig. X(a)','Fig. 2(a)').replace('Fig. X(b)','Fig. 2(b)'))
    remove(28)
    insert_before(43,'The 50 ports per die are finite physical interface resources. GPU reads use nearest-port group routing with port-balanced resident ownership; NMP transfers stripe across regional ports or enter once before an intra-die broadcast. Shared-port payloads accumulate, and boundary service is the larger of aggregate payload divided by the global ceiling and the slowest port’s serialization plus route startup; nominal port rates are not summed into unconstrained GPU bandwidth.',42)
    replace(44,'The physical execution model maps each LLM operator to resident row/KV partitions, M3D layers, cluster service groups, and regions. Array/MIV service feeds FEOL routes, MAC tiles and SRAM, local fabrics, inter-region NoC, and GPU-facing ports. The fixed operator dependency sequence determines GPU/NMP handoff; concurrent batch requests merge demand on shared groups, tiles, links, and ports before completion times are evaluated.')
    replace(45,'Eight memory layers serialize within each service-group lane; independent groups operate concurrently. An NMP operator takes boundary service plus NoC/reduction service plus the maximum of array, fabric, and MAC service. A GPU-side physical operator takes the maximum of array, boundary, and GPU service. Thus changing placement changes resource loads and completion time. Achieved boundary bandwidth is completed payload divided by summed boundary-active service time, excluding GPU gaps; it is distinct from nominal interface and thermal ceilings.')
    replace(46,'Traffic and energy use the same operator activity: array/peripheral accesses, MIV and FEOL wire movement, boundary bits, MACs, SRAM accesses, and reduction events. Device/cell parameters follow published IGZO measurements [1] and the corresponding MAT SPICE extraction; interconnects use distributed RC with documented literature parameters. HBM retains its configured DreamRAM-derived physical parameters [3]; GPU service and energy retain specification-based ceilings and frozen modeling coefficients. Numerical and traffic/energy-conservation checks verify model consistency. GPU service is modeled at operator level; handoff follows dependencies and boundary transfers, without an independently calibrated software synchronization term.')
    replace(55,'where Tcrit=85°C. A three-dimensional steady-state conductance model resolves material and interface thermal paths under the same package boundary conditions. Workload events determine memory, interconnect, and NMP power, grouped by die and distributed uniformly within each die’s BEOL; GPU power is mapped to its FEOL. FP64 matrix-free PCG solves the spatial temperature field. The bandwidth sweep identifies the sustainable external ceiling.')
    replace(56,'Fig. 5 compares thermal scalability. Doubling conventional HBM from 12-Hi to 24-Hi reduces sustainable bandwidth from 3.09 to 1.96 TB/s (36.5%). The 300-µm and 100-µm IOM3D configurations sustain 3.44 and 3.40 TB/s, respectively: thinning triples capacity with only a 1.2% bandwidth reduction.')
    replace(62,'DNS exploits resident weight and KV locality through a fixed heterogeneous execution policy (Fig. 6).')
    replace(64,'Uniform DNS establishes initial data and compute ownership. CPA then refines legal NMP placements to account for topology-dependent contention and communication; it does not change operator assignment between GPU and NMP.')
    replace(67,'CPA refines uniform DNS placement because communication and shared-resource contention depend on physical topology. Its objective minimizes modeled Decode latency over legal resident-data and tile mappings:')
    remove(68)
    replace(70,'subject to per-region memory capacity and tile-resource limits,')
    replace(72,'CPA identifies the dominant physical component, proposes legal coarse-grained resident or tile remappings, and re-evaluates candidates with the same complete contention-aware physical stage model rather than a placement-cost surrogate. Concurrent request demand is included; paired KV ownership and endpoint/append checks are preserved. Only latency-improving moves are accepted. Search stops when no qualifying move remains or the bounded refinement budget is exhausted; the implementation uses one resident pass and at most two tile-refinement rounds per operator.')
    remove(73);remove(74)
    replace(80,'HBM-GPU uses the closed local bandwidth. On overflow, it accesses sufficiently provisioned Grace LPDDR5X through the frozen 416.34-GB/s NVLink-C2C path with 5.3-pJ/bit transfer energy. M3D-GPU retains GPU execution with port-balanced physical memory service; DNS uses uniform NMP placement and DNS+CPA refines that placement. These M3D paths retain their canonical Prefill and Decode accounting. E2E throughput includes the incremental Prefill and all 32 growing-context Decode steps; temperatures use Decode steady-state power.')
    replace(85,'Performance. Fig. 7 reports per-case HBM-normalized E2E throughput. Geometric-mean improvements are 2.31× for M3D-GPU, 6.68× for DNS, and 7.28× for DNS+CPA (maximum 47.98×); CPA adds 1.09× over DNS. The capacity audit identifies five HBM-resident cases, all 8B configurations except LC126K/B8. Within this subset, DNS and CPA achieve 2.77× and 3.07× over HBM-GPU, and 2.67× and 2.96× over M3D-GPU. The corresponding M3D-GPU/HBM-GPU ratio is 1.04×, separating NMP locality from capacity spill avoidance. The 13 overflow cases yield 10.15× CPA/HBM-GPU and 3.23× CPA/M3D-GPU.')
    insert_before(86,'At LC20K, HBM-GPU/M3D-GPU/CPA throughput (tokens/s) is 169.28/171.38/619.96 for 8B/B1, 65.77/118.06/171.76 for 70B/B8, and 0.600/3.920/28.785 for 405B/B1. CPA mean TPOT is 1.441/37.914/27.075 ms, respectively; one Decode step emits B tokens.')
    # Add the new figure after existing figures, retaining Fig. 1–10 numbering.
    insert_before(100,'PHYSICAL_EXECUTION_FIGURE_PLACEHOLDER')
    insert_before(100,'Fig. 11. Physical Decode Execution Anatomy: Llama-3.1-8B, LC20K, B=1, M3D_NMP_CPA, Decode step 1 (context 20,128), layer 0. (a) Reconstructed dependency-ordered operator intervals, placed in the lane of their dominant service component; these are not individual resource occupancy intervals. (b) Resource service divided by the maximum resource service of each operator; black boxes mark maxima. NoC includes reduction and GPU denotes its modeled local service. Sub-0.1-µs operators are retained in (a) and omitted from (b).',83)
    insert_before(100,'The single-step replay in Fig. 11 matches the canonical 1.441094-ms checkpoint; layer 0 occupies 43.844 µs. Q/K/V and FFN Down are array-dominated, QK and FFN Gate/Up are fabric-dominated, while AV is dominated by NoC/reduction. GPU Softmax and cross-die AV reduction separate NMP stages. The service balance changes across operators even within one layer.')
    replace(96,oldtext[96].replace('85℃∘','85°C ').replace('85℃','85°C'))
    remove(102);remove(103)
    insert_before(104,'REFERENCES',100)
    for i in (104,105,106):
        value=oldtext[i].replace('<sup>2</sup>','²')
        if '[Online]. Available:' in value:value=value.split('[Online]. Available:')[0].rstrip()
        replace(i,value)
        node=original[i];pr=node.getElementsByTagName('w:pPr')[0]
        while pr.firstChild:pr.removeChild(pr.firstChild)
        for tag,attrs in [('w:jc',{'w:val':'left'}),('w:ind',{'w:left':'180','w:hanging':'180'}),
                          ('w:spacing',{'w:after':'40'})]:
            e=doc.createElement(tag)
            for k,v in attrs.items():e.setAttribute(k,v)
            pr.appendChild(e)
        run=node.getElementsByTagName('w:r')[0];rpr=doc.createElement('w:rPr')
        sz=doc.createElement('w:sz');sz.setAttribute('w:val','16');rpr.appendChild(sz);run.insertBefore(rpr,run.firstChild)
    # Retain all equations and original image relationships; remove template-only footers.
    for name in list(files):
        if name.startswith('word/footer') and name.endswith('.xml'):
            xml=files[name].decode('utf-8')
            if 'XXX-X-XXXX' in xml:
                d=minidom.parseString(files[name])
                for t in d.getElementsByTagName('w:t'):
                    if t.firstChild and 'XXX-X-XXXX' in t.firstChild.data:t.firstChild.data=''
                files[name]=d.toxml(encoding='UTF-8')
    files['word/document.xml']=doc.toxml(encoding='UTF-8')
    OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(OUT/'iom3d_hbm_evidence.docx','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    (OUT/'manuscript_edits.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        edits=edits,CPA_original_words=sum(len(oldtext[i].split()) for i in range(67,75)),
        CPA_revised_words=sum(len(e['after'].split()) for e in edits if isinstance(e['index'],int) and 67<=e['index']<75)),indent=2)+'\n',encoding='utf-8')
    print(OUT/'iom3d_hbm_evidence.docx')


if __name__=='__main__':main()
