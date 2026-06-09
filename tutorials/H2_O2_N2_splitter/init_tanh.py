#!/usr/bin/env python3
"""
init_tanh.py  –  Initialise H2/air splitter fields with a tanh profile in y.

Reads constant/polyMesh to get the correct cell-centre and face-centre
y-coordinates for any mesh state (original blockMesh or refined).

Usage:
    blockMesh          # (and any mesh refinement steps)
    python3 init_tanh.py
    detonationFoam_V2.0

Tanh profile
------------
    blend(y) = 0.5*(1 + tanh((y - y_c)/delta))
      → 0  at y << y_c  (H2 / fuel state)
      → 1  at y >> y_c  (air state)

    Increase delta for a thicker transition layer.
"""

import os, re, sys
import numpy as np

# ── tuneable parameters ──────────────────────────────────────────────────────
y_c   = 0.0025          # centreline (m)
delta = 1e-4 / 3.0      # half-thickness (m)  →  99% within ±0.1 mm

U_H2,  U_air  = 427.0, 226.0
T_H2,  T_air  = 350.0, 1500.0
H2_H2, H2_air = 1.0,   0.0
O2_H2, O2_air = 0.0,   0.233
N2_H2, N2_air = 0.0,   0.767

MESH_DIR    = "constant/polyMesh"
INLET_NAMES = ["inletFuel", "inletAir"]
# ─────────────────────────────────────────────────────────────────────────────


def tanh_blend(y, lo, hi):
    b = 0.5 * (1.0 + np.tanh((y - y_c) / delta))
    return lo + (hi - lo) * b


# ── polyMesh readers ──────────────────────────────────────────────────────────

def _clean(text):
    text = re.sub(r'//[^\n]*', '', text)
    return re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)


def _list_body(text):
    """Return the text between the outermost ( ) that follows the first integer count."""
    m = re.search(r'\d+\s*\n\s*\(', text)
    if not m:
        raise ValueError("Cannot find list in file")
    start = text.index('(', m.start()) + 1
    depth, i = 1, start
    while depth:
        if   text[i] == '(': depth += 1
        elif text[i] == ')': depth -= 1
        i += 1
    return text[start:i - 1]


def read_points():
    text = _clean(open(f"{MESH_DIR}/points").read())
    body = _list_body(text)
    tuples = re.findall(
        r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)', body)
    return np.array([(float(a), float(b), float(c)) for a, b, c in tuples])


def read_faces():
    """Return list of vertex-index lists, one per face."""
    text = _clean(open(f"{MESH_DIR}/faces").read())
    body = _list_body(text)
    # Match individual face entries: optional_count(v0 v1 ...)  on each line
    face_re = re.compile(r'^\s*\d*\s*\(\s*((?:\d+\s*)+)\)\s*$', re.MULTILINE)
    return [list(map(int, m.group(1).split())) for m in face_re.finditer(body)]


def read_int_list(fname):
    """Read a flat integer list (owner, neighbour, etc.)."""
    text = _clean(open(f"{MESH_DIR}/{fname}").read())
    body = _list_body(text)
    return np.array(list(map(int, body.split())))


def read_boundary():
    text = _clean(open(f"{MESH_DIR}/boundary").read())
    info = {}
    for m in re.finditer(r'\b(\w+)\s*\{([^{}]*)\}', text):
        name, block = m.group(1), m.group(2)
        sf = re.search(r'startFace\s+(\d+)', block)
        nf = re.search(r'nFaces\s+(\d+)', block)
        if sf and nf:
            info[name] = (int(sf.group(1)), int(nf.group(1)))
    return info


# ── compute cell-centre and face-centre y-coords ──────────────────────────────

if not os.path.isdir(MESH_DIR):
    sys.exit(f"ERROR: {MESH_DIR} not found — run blockMesh first.")

print("Reading polyMesh …")
pts      = read_points()
faces    = read_faces()
owner    = read_int_list("owner")
boundary = read_boundary()

n_cells = int(owner.max()) + 1

# Face-centre y (average of vertex y-coords)
face_y = np.array([np.mean(pts[f, 1]) for f in faces])

# Cell-centre y = average of face-centre y for all faces owned by each cell
cell_y_sum   = np.zeros(n_cells)
cell_face_cnt = np.zeros(n_cells, dtype=int)
for fi, fy in enumerate(face_y):
    c = owner[fi]
    cell_y_sum[c]   += fy
    cell_face_cnt[c] += 1
cell_y = cell_y_sum / cell_face_cnt

print(f"  {n_cells} cells,  y ∈ [{cell_y.min()*1e3:.4f}, {cell_y.max()*1e3:.4f}] mm")

# Inlet patch face-centre y (already read in correct polyMesh order)
patch_y = {}
for name in INLET_NAMES:
    if name not in boundary:
        print(f"  WARNING: patch '{name}' not found")
        continue
    sf, nf = boundary[name]
    yf = face_y[sf:sf + nf]
    patch_y[name] = yf
    print(f"  patch '{name}': {nf} faces, "
          f"y ∈ [{yf.min()*1e3:.4f}, {yf.max()*1e3:.4f}] mm")

print(f"\nBlend: y_c={y_c*1e3:.2f} mm, δ={delta*1e6:.1f} µm "
      f"(99% within ±{delta*3*1e3:.2f} mm)\n")

# ── quick sanity check: print Ux at key faces ─────────────────────────────────
for name, yf in patch_y.items():
    Ux = tanh_blend(yf, U_H2, U_air)
    print(f"  {name}: Ux min={Ux.min():.1f}  max={Ux.max():.1f}  "
          f"(at y_min={yf.min()*1e3:.3f} mm, y_max={yf.max()*1e3:.3f} mm)")
print()


# ── OpenFOAM field writers ────────────────────────────────────────────────────

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
        print(f"  WARNING: patch '{name}' not found in field file")
        return text
    new_content = _VALUE_ENTRY.sub(f"value       {blk}", content, count=1)
    return text[:bs + 1] + new_content + text[be - 1:]


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

    v   = tanh_blend(cell_y, lo, hi)
    blk = (vector_block(v, np.zeros_like(v), np.zeros_like(v))
           if is_vec else scalar_block(v))
    text = set_internal(text, blk)

    for name, yf in patch_y.items():
        vp   = tanh_blend(yf, lo, hi)
        blkp = (vector_block(vp, np.zeros_like(vp), np.zeros_like(vp))
                if is_vec else scalar_block(vp))
        text = set_patch_value(text, name, blkp)

    with open(path, 'w') as f:
        f.write(text)
    print(f"  {path}")

print("\nDone.")
