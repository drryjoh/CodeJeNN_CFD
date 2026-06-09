#!/usr/bin/env python3
"""
init_tanh.py  –  Initialise H2/air splitter internal field with a tanh profile.

The inlet boundary conditions are handled by codedFixedValue in 0/{U,T,H2,O2,N2}
and do not need to be touched here.

Usage:
    blockMesh
    writeCellCentres -time 0      # writes 0/Cy
    python3 init_tanh.py
    detonationFoam_V2.0

Tanh profile
------------
    blend(y) = 0.5 * (1 + tanh((y - y_c) / delta))
      → 0  at y << y_c  (pure H2)
      → 1  at y >> y_c  (pure air)

    delta = 1e-4/3 ≈ 33 µm  →  tanh(3) = 0.995  →  99% within ±0.1 mm of y_c
"""

import os, re, sys
import numpy as np

# ── tuneable parameters ──────────────────────────────────────────────────────
y_c   = 0.0025          # centreline (m)  — must match codedFixedValue in 0/
delta = 1e-4 / 3.0      # half-thickness (m)

U_H2,  U_air  = 427.0, 226.0    # Ux (m/s)
T_H2,  T_air  = 350.0, 1500.0   # K
H2_H2, H2_air = 1.0,   0.0
O2_H2, O2_air = 0.0,   0.233
N2_H2, N2_air = 0.0,   0.767
# ─────────────────────────────────────────────────────────────────────────────


def tanh_blend(y, lo, hi):
    b = 0.5 * (1.0 + np.tanh((y - y_c) / delta))
    return lo + (hi - lo) * b


_INT_NONUNIFORM = re.compile(
    r'internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\n\s*\((.*?)\)\s*;',
    re.DOTALL)
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


# ── load cell-centre y-coordinates ───────────────────────────────────────────
for candidate in ["0/Cy", "0/ccy", "0/C_1"]:
    if os.path.exists(candidate):
        cy_path = candidate
        break
else:
    sys.exit(
        "\nERROR: y cell-centre field not found.\n"
        "Run:  blockMesh && writeCellCentres -time 0\n")

cy_text = open(cy_path).read()
m = _INT_NONUNIFORM.search(cy_text)
if not m:
    sys.exit(f"ERROR: cannot parse nonuniform internalField in {cy_path}")
y = np.fromstring(m.group(1), sep='\n')

print(f"Loaded {len(y)} cell centres from '{cy_path}'")
print(f"  y ∈ [{y.min()*1e3:.3f}, {y.max()*1e3:.3f}] mm")
print(f"  y_c = {y_c*1e3:.2f} mm,  δ = {delta*1e6:.1f} µm  "
      f"(99% within ±{delta*3*1e3:.2f} mm)\n")

# ── write internal fields ─────────────────────────────────────────────────────
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

print("\nDone.  Boundary conditions are set by codedFixedValue in 0/.")
