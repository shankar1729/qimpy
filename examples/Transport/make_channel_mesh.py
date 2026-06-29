"""Generate a channel-domain triangle mesh for qimpy.transport (FiniteVolume).

Domain [0, Lx] x [0, Ly] with a source contact on the left edge (x=0) and a
drain on the right edge (x=Lx), each spanning y in [0, contact_y]; all other
boundary edges are reflective walls. The triangulation is a structured
criss-cross grid (4 triangles/cell about a centre node), perfectly symmetric
about x=Lx/2 so source and drain are exact mirror images.

    python make_channel_mesh.py            # writes channel.npz
"""
import argparse
import numpy as np
from qimpy.transport.geometry._mesh import save_mesh


def make(nx, ny, path, Lx=1.0, Ly=1.25, contact_y=0.25):
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    V = [[x, y] for y in ys for x in xs]
    vid = lambda i, j: j * (nx + 1) + i
    T = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            cen = len(V)
            V.append([0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])])
            T += [[a, b, cen], [b, c, cen], [c, d, cen], [d, a, cen]]   # all CCW
    V = np.array(V, float)

    edges = ([(vid(i, 0), vid(i + 1, 0)) for i in range(nx)] +          # bottom
             [(vid(i, ny), vid(i + 1, ny)) for i in range(nx)] +        # top
             [(vid(0, j), vid(0, j + 1)) for j in range(ny)] +          # left
             [(vid(nx, j), vid(nx, j + 1)) for j in range(ny)])         # right
    be, bm = [], []
    for a, b in edges:
        mx, my = 0.5 * (V[a] + V[b])
        if mx < 1e-9 and my < contact_y:                  # left edge contact
            bm.append("source")
        elif mx > Lx - 1e-9 and my < contact_y:           # right edge contact
            bm.append("drain")
        else:
            bm.append("wall")
        be.append([a, b])
    save_mesh(path, V, np.array(T, int), np.array(be, int), bm)
    print(f"wrote {path}: {len(T)} triangles ({nx}x{ny} criss-cross), "
          f"{bm.count('source')} source / {bm.count('drain')} drain faces")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=24)
    ap.add_argument("--ny", type=int, default=30)
    ap.add_argument("--out", type=str, default="channel.npz")
    a = ap.parse_args()
    make(a.nx, a.ny, a.out)
