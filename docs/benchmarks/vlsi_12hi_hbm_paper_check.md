# Conventional canonical HBM GPU-300 W power check

This is a strict power-only derivative of `configs/hbm_on_gpu_12hi.yaml`.
Geometry, materials, mesh, boundary conditions, solver settings, HBM power,
and power placement are unchanged. The only physical difference is GPU power:
414 W becomes 300 W.

- Four canonical 11 x 11 mm HBM columns remain in place.
- The central 8 x 22 mm Thermal Silicon column remains in place.
- The conventional HBM stack remains 775 um tall.
- HBM power remains 4 x 40 W = 160 W.
- Total input power is 300 + 160 = 460 W.

This case is an apples-to-apples conventional canonical power comparison. It
does not claim to reproduce the distinct two-cube, 660 um VLSI geometry.
