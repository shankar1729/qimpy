"""Generate a 1D line-domain mesh for qimpy.transport (TriSet).

Domain [0, Lx] split into nx interval cells on a line (y = 0): a 1D wire with a
source contact at the left end (x=0) and a drain at the right end (x=Lx).  Cells
are intervals; TriSet's 1D path treats each with two endpoint faces (length is
the cell measure, +/- x the face normals).  The material is unchanged -- the
Fermi surface stays 2D, and only v_x = v.n streams along the line.

    python make_1d_mesh.py --nx 64 --out line.npz
"""
import argparse
import numpy as np
from qimpy.transport.geometry._mesh import save_mesh


def make(nx, path, Lx=1.0):
    x = np.linspace(0.0, Lx, nx + 1)
    vertices = np.column_stack([x, np.zeros(nx + 1)])          # (nx+1, 2), y = 0
    cells = np.column_stack([np.arange(nx), np.arange(1, nx + 1)])  # (nx, 2) intervals
    # Boundary "faces" are the two end vertices, tagged as degenerate (v, v) edges:
    boundary = np.array([[0, 0], [nx, nx]], int)
    markers = ["source", "drain"]
    save_mesh(path, vertices, cells, boundary, markers)
    print(f"wrote {path}: {nx} interval cells on [0,{Lx}], "
          f"source (x=0) / drain (x={Lx})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="line.npz")
    a = ap.parse_args()
    make(a.nx, a.out, a.Lx)
