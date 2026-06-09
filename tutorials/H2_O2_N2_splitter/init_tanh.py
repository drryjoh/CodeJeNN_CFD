#!/usr/bin/env python3
"""
init_tanh.py  –  Initialise H2/air splitter fields with a tanh profile in y.

Writes nonuniform List<scalar/vector> values for BOTH the internalField AND
the inlet boundary patches directly into the 0/ field files.  No OpenFOAM
utilities (writeCellCentres, postProcess, etc.) are required.

Cell/face centre y-coordinates are computed analytically from the blockMeshDict
grading.  OpenFOAM cell ordering: i (x) fastest, then j (y), then k (z);
block 0 (lower) before block 1 (upper).  Inlet face ordering: j = 0..N-1.

Usage:
    blockMesh
    python3 init_tanh.py
    detonationFoam_V2.0

Tanh profile
------------
    blend(y) = 0.5*(1 + tanh((y - y_c)/delta))
      → 0  at y << y_c  (pure H2)
      → 1  at y >> y_c  (pure air)

    delta = 1e-4/3 ≈ 33 µm  →  99% within ±0.1 mm of y_c
"""

import re, sys
import numpy as np

# ── tuneable parameters ──────────────────────────────────────────────────────
y_c   = 0.0025          # centreline (m)
delta = 1e-4 / 3.0      # half-thickness (m)  — 99% within ±0.1 mm

U_H2,  U_air  = 427.0, 226.0
T_H2,  T_air  = 350.0, 1500.0
H2_H2, H2_air = 1.0,   0.0
O2_H2, O2_air = 0.0,   0.233
N2_H2, N2_air = 0.0,   0.767

# ── mesh parameters (keep in sync with blockMeshDict) ────────────────────────
Nx = 190    # cells in x per block
Ny =  60    # cells in y per block

# R = last-cell / first-cell expansion ratio along y
BLOCK0 = dict(y_start=0.0,    y_end=0.0025, N=Ny, R=0.0917)  # lower / H2
BLOCK1 = dict(y_start=0.0025, y_end=0.005,  N=Ny, R=10.9)    # upper / air

INLET_PATCHES = {
    "inletFuel": BLOCK0,   # left face of lower block
    "inletAir":  BLOCK1,   # left face of upper block
}
# ─────────────────────────────────────────────────────────────────────────────


def midpoints(y_start, y_end, N, R):
    """
    Midpoints (cell/face centres) for N cells over [y_start, y_end]
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


# ── OpenFOAM field helpers ───────────────────────────────────────────────────

_ANY_INTERNAL = re.compile(r'internalField\s.*?;', re.DOTALL)
_VALUE_ENTRY  = re.compile(
    r'value\s+'
    r'(?:nonuniform\s+List<\w+>\s+\d+\s*\n\s*\(.*?\)\s*;'
    r'|uniform\s+[^;]+;)',
    re.DOTALL)


def scalar_block(vals):
    return ("nonuniform List<scalar> \n{}\n(\n".format(len(vals))
            + "\n".join(f"{v:.10e}" for v in vals) + "\n)\n;")


def vector_block(ux, uy, uz):
    return ("nonuniform List<vector> \n{}\n(\n".format(len(ux))
            + "\n".join(f"({u:.10e} {v:.10e} {w:.10e})"
                        for u, v, w in zip(ux, uy, uz)) + "\n)\n;")


def set_internal(text, blk):
    return _ANY_INTERNAL.sub(f"internalField   {blk}", text, count=1)


def find_patch_braces(text, name):
    """Return (content, open_idx, close_plus1_idx) for named patch block."""
    m = re.search(rf'\b{re.escape(name)}\s*\{{', text)
    if not m:
        return None, -1, -1
    i, depth = m.end() - 1, 0
    while i < len(text):
        if   text[i] == '{': depth += 1
        elif text[i] == '}': depth -= 1
        if depth == 0:
            return text[m.end():i], m.end() - 1, i + 1
        i += 1
    return None, -1, -1


def set_patch_value(text, name, blk):
    content, bs, be = find_patch_braces(text, name)
    if content is None:
        print(f"  WARNING: patch '{name}' not found")
        return text
    new_content = _VALUE_ENTRY.sub(f"value       {blk}", content, count=1)
    return text[:bs + 1] + new_content + text[be - 1:]


# ── build y-coordinate arrays ─────────────────────────────────────────────────
# Internal cells: each j-row has Nx cells with the same y (i varies fastest)
yj0 = midpoints(**BLOCK0)
yj1 = midpoints(**BLOCK1)
y_int = np.concatenate([np.repeat(yj0, Nx), np.repeat(yj1, Nx)])

# Inlet face centres: 1 face per j-row (left face of each block)
patch_y = {name: midpoints(**p) for name, p in INLET_PATCHES.items()}

print(f"Internal: {len(y_int)} cells, y ∈ [{y_int.min()*1e3:.4f}, {y_int.max()*1e3:.4f}] mm")
for name, yf in patch_y.items():
    print(f"Patch '{name}': {len(yf)} faces, "
          f"y ∈ [{yf.min()*1e3:.4f}, {yf.max()*1e3:.4f}] mm")
print(f"\nBlend: y_c={y_c*1e3:.2f} mm, δ={delta*1e6:.1f} µm "
      f"(99% within ±{delta*3*1e3:.2f} mm)\n")

# ── write fields ──────────────────────────────────────────────────────────────
FIELDS = [
    ("0/T",  False, T_H2,  T_air),
    ("0/H2", False, H2_H2, H2_air),
    ("0/O2", False, O2_H2, O2_air),
    ("0/N2", False, N2_H2, N2_air),
    ("0/U",  True,  U_H2,  U_air),
]

print("Writing tanh-smoothed fields:")
for path, is_vec, lo, hi in FIELDS:
    text = open(path).read()

    # internal field
    v = tanh_blend(y_int, lo, hi)
    blk = (vector_block(v, np.zeros_like(v), np.zeros_like(v))
           if is_vec else scalar_block(v))
    text = set_internal(text, blk)

    # inlet patch boundary values
    for name, yf in patch_y.items():
        vp  = tanh_blend(yf, lo, hi)
        blkp = (vector_block(vp, np.zeros_like(vp), np.zeros_like(vp))
                if is_vec else scalar_block(vp))
        text = set_patch_value(text, name, blkp)

    with open(path, 'w') as f:
        f.write(text)
    print(f"  {path}")

print("\nDone.")
