"""Generate the external triangle mesh for rect-domain-tri.yaml.

qimpy does not mesh; this is the external tooling step. Produces a rect-domain
[5,105]x[5,55] mesh with source/drain contact faces, written in the format
qimpy.transport.geometry.TriSet consumes (see _dg_mesh.load_mesh).

    python make_rect_mesh.py            # writes rect-domain-tri.npz (h=1.0)
    python make_rect_mesh.py --h 0.5
"""
import argparse
from collections import Counter
import numpy as np
import triangle as tr
from qimpy.transport.geometry._dg_mesh import save_mesh

SRC, DRN = (10.0, 55.0, 5.0), (10.0, 5.0, 5.0)   # (cx, cy, r) contact circles


def make(grid_spacing: float, path: str) -> None:
    pts = np.array([[5, 5], [105, 5], [105, 55], [5, 55]], float)
    seg = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    m = tr.triangulate({"vertices": pts, "segments": seg},
                       f"pq30a{grid_spacing ** 2:g}")
    V, T = m["vertices"], m["triangles"]
    ec: Counter = Counter()
    for t in T:
        for x, y in [(0, 1), (1, 2), (2, 0)]:
            ec[tuple(sorted((int(t[x]), int(t[y]))))] += 1
    be = [e for e, c in ec.items() if c == 1]
    bm = []
    for a, b in be:
        mx, my = 0.5 * (V[a] + V[b])
        if (mx - SRC[0]) ** 2 + (my - SRC[1]) ** 2 <= SRC[2] ** 2:
            bm.append("source")
        elif (mx - DRN[0]) ** 2 + (my - DRN[1]) ** 2 <= DRN[2] ** 2:
            bm.append("drain")
        else:
            bm.append("wall")
    save_mesh(path, V, T, np.array(be), bm)
    print(f"wrote {path}: {len(T)} triangles, "
          f"{bm.count('source')} source / {bm.count('drain')} drain faces")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=1.0, help="target triangle size")
    ap.add_argument("--out", type=str, default="rect-domain-tri.npz")
    a = ap.parse_args()
    make(a.h, a.out)
