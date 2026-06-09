#!/usr/bin/env python3
"""
init_tanh.py  –  Initialise H2/air splitter internal field with a tanh profile.

No writeCellCentres needed.  Cell-centre y-coordinates are computed analytically
from the blockMeshDict grading, using OpenFOAM's cell ordering convention:
    i (x) varies fastest, then j (y), then k (z), block 0 before block 1.

Usage:
    blockMesh
    python3 init_tanh.py
    detonationFoam_V2.0

Inlet boundary conditions are handled by codedFixedValue in 0/{U,T,H2,O2,N2}.

Tanh profile
------------
    blend(y) = 0.5*(1 + tanh((y - y_c)/delta))
      → 0 at y << y_c  (pure H2)
      → 1 at y >> y_c  (pure air)

    delta = 1e-4/3 ≈ 33 µm  →  99% transition within ±0.1 mm of y_c
"""

import re, sys
import numpy as np

# ── tuneable parameters ──────────────────────────────────────────────────────
y_c   = 0.0025          # centreline (m)  — must match codedFixedValue in 0/
delta = 1e-4 / 3.0      # half-thickness (m)

U_H2,  U_air  = 427.0, 226.0
T_H2,  T_air  = 350.0, 1500.0
H2_H2, H2_air = 1.0,   0.0
O2_H2, O2_air = 0.0,   0.233
N2_H2, N2_air = 0.0,   0.767

# ── mesh parameters (keep in sync with blockMeshDict) ────────────────────────
Nx = 190    # cells in x per block
Ny =  60    # cells in y per block

# Block 0 (lower, H2 stream):  y = 0 → 2.5 mm, y-grading R = last/first
BLOCK0 = dict(y_start=0.0,    y_end=0.0025, N=Ny, R=0.0917)
# Block 1 (upper, air stream): y = 2.5 → 5 mm, y-grading R = last/first
BLOCK1 = dict(y_start=0.0025, y_end=0.005,  N=Ny, R=10.9)
# ─────────────────────────────────────────────────────────────────────────────


def cell_centres_y(y_start, y_end, N, R):
    """
    y-coordinates of cell centres for N cells over [y_start, y_end]
    with last/first grading ratio R.
    """
    L = y_end - y_start
    if abs(R - 1.0) < 1e-6:
        edges = np.linspace(y_start, y_end, N + 1)
    else:
        r  = R ** (1.0 / (N - 1))
        h1 = L * (r - 1.0) / (r**N - 1.0)
        edges = [y_start]
        h = h1
        for _ in range(N):
            edges.append(edges[-1] + h)
            h *= r
        edges = np.array(edges)
    return 0.5 * (edges[:-1] + edges[1:])


def tanh_blend(y, lo, hi):
    b = 0.5 * (1.0 + np.tanh((y - y_c) / delta))
    return lo + (hi - lo) * b


_ANY_INTERNAL = re.compile(r'internalField\s.*?;', re.DOTALL)


def scalar_block(vals):
    return ("nonuniform List<scalar> \n{}\n(\n".format(len(vals))
            + "\n".join(f"{v:.10e}" for v in vals) + "\n)\n;")


def vector_block(ux, uy, uz):
    return ("nonuniform List<vector> \n{}\n(\n".format(len(ux))
            + "\n".join(f"({u:.10e} {v:.10e} {w:.10e})"
                        for u, v, w in zip(ux, uy, uz)) + "\n)\n;")


def set_internal(text, block_str):
    return _ANY_INTERNAL.sub(f"internalField   {block_str}", text, count=1)


# ── build cell-centre y array ─────────────────────────────────────────────────
# OpenFOAM blockMesh cell ordering: i fastest, then j, then k; block 0 first.
# All cells at the same j-index share the same y-coordinate.
# Each j-row has Nx cells (one per x-column).
yj0 = cell_centres_y(**BLOCK0)   # shape (Ny,)
yj1 = cell_centres_y(**BLOCK1)   # shape (Ny,)

y = np.concatenate([
    np.repeat(yj0, Nx),   # block 0: Ny*Nx cells
    np.repeat(yj1, Nx),   # block 1: Ny*Nx cells
])

ncells = len(y)
print(f"Cell centres computed analytically: {ncells} cells")
print(f"  y ∈ [{y.min()*1e3:.4f}, {y.max()*1e3:.4f}] mm")
print(f"  y_c = {y_c*1e3:.2f} mm,  δ = {delta*1e6:.1f} µm  "
      f"(99% within ±{delta*3*1e3:.2f} mm)\n")

# ── write internalField of each field file ────────────────────────────────────
FIELDS = [
    ("0/T",  False, T_H2,  T_air),
    ("0/H2", False, H2_H2, H2_air),
    ("0/O2", False, O2_H2, O2_air),
    ("0/N2", False, N2_H2, N2_air),
    ("0/U",  True,  U_H2,  U_air),
]

print("Writing tanh-smoothed internalField:")
for path, is_vec, lo, hi in FIELDS:
    text = open(path).read()
    v    = tanh_blend(y, lo, hi)
    blk  = (vector_block(v, np.zeros_like(v), np.zeros_like(v))
            if is_vec else scalar_block(v))
    with open(path, 'w') as f:
        f.write(set_internal(text, blk))
    print(f"  {path}")

print("\nDone.  Boundary conditions enforced by codedFixedValue at runtime.")
