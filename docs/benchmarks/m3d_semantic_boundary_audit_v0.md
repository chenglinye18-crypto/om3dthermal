# M3D semantic-boundary audit v0

This audit closes only the Conventional-HBM capacity denominator and the M3D
contactless-interface / logic-background parameter boundaries. It does not
change the frozen thermal mesh, operator, solver, or matched 39.2 Tb/s scenario.

## Gate A source of truth

| visible groups | stacks/group | total stacks | dies/stack | capacity/stack | system capacity | provenance |
|---:|---:|---:|---:|---:|---:|---|
| 2 | 2 | 4 | 12 | 27.0 GiB | 108.0 GiB | `exp_conv_2x1_*` metadata derives each continuous 11x22 mm thermal group from two 11x11 mm physical stack equivalents; the parent 2x2 benchmark defines a 10.8x10.8 mm DRAM die per physical stack. Capacity uses DreamRAM integer bank-tile packing on that physical die footprint. |

The former 114.75 GiB value came from packing 306 banks into each full
10.8x21.8 mm thermal-visible group, treating two groups as two physical stacks:
57.375 GiB/group x 2. That interpretation had no physical-stack source and is
retired. The corrected path packs 144 banks/die, 12 dies/stack, and four stacks.

## Gate B contactless-interface boundary

The local MOSAIC paper (`ref/T8.1_Wednesday_Mitarai_0021.pdf`) assumes
4 pJ/b data access and attributes 0.5 pJ/b to I/O, citing Shiba et al.'s
7 nm inductive-coupling interface. It separately reports a 0.18 um MOSAIC
loopback prototype at 6.9 pJ/b. The repository does not prove that 0.5 pJ/b is
a complete TX+RX+coil+clock+SerDes+driver/control boundary.

| component | boundary status |
|---|---|
| inductive/coupling interface aggregate | included only at the cited aggregate-I/O label |
| TX, RX, coil/coupling path, clock, serialization/deserialization, driver/control | individually unconfirmed; no double-counting or completeness claim |
| other memory-internal, MIV, FEOL-route terms | excluded from the interface-only override and retained unchanged |

The formal sensitivity points are 0.25, 0.5, and 1.0 pJ/bit, all marked
`PARAMETRIC_SENSITIVITY`; 0.5 remains the nominal
`CONDITIONAL_ASSUMPTION`.

## Gate C logic/background boundary

The only added model is the explicit parameter set 0, 5, 10, and 20 W:

`P_memory_total = P_memory_dynamic + P_refresh + P_logic_background`

The general accountant retains an explicit memory-background field, which is
zero/disabled in this case and therefore absent from the equation above.
Logic/background is not folded into
the workload-derived dynamic term and is mapped through the existing M3D power
carrier region. Every point is `PARAMETRIC_SENSITIVITY`, not a validated nominal.

The minimal formal configuration is
`configs/experiment/m3d_semantic_boundary_audit_v0.yaml`; its generated bundle
belongs under untracked `results/`.

The completed audit bundle reports 33.5262 W dynamic memory power and
0.03415 W refresh. At 0/5/10/20 W logic/background, memory total power is
33.5604/38.5604/43.5604/53.5604 W and package Tmax is
82.2910/82.8453/83.3996/84.5081 degC. The corrected Conventional baseline is
355.5354 W package input and 81.9256 degC package Tmax with the unchanged
859596 cells and 2531340 edges.
