# exp_conv_2x1_g414_m160 and exp_conv_2x1_g300_m160

This layout is derived from the canonical 2x2 HBM case. The left and right
groups remain separated by the central 8 x 22 mm Thermal Silicon column. Within
each side, the former top and bottom 11 x 11 mm HBM footprints are merged along
global y into one continuous 11 x 22 mm group.

- Parent footprint per group: 11 x 22 mm.
- DRAM footprint per group: 10.8 x 21.8 mm.
- Group centres: (-9.5, 0) mm and (+9.5, 0) mm.
- The DRAM layers cross y=0 continuously; there is no internal Mold seam.
- The 0.1 mm Mold ring remains only around the outer perimeter of each group.
- Vertical 12Hi stack, materials, mesh, cooling, and boundary conditions are
  unchanged from the canonical configuration.
- The canonical four times 40 W HBM power is preserved as two times 80 W.
- Conventional power uses Son23 component-aware vertical placement by default.

Two power variants are supplied: GPU 414 W and GPU 300 W. Their only resolved
physical difference is GPU total power.
