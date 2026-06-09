"""Generate the external triangle mesh for mixer-tri.yaml.

qimpy does not mesh; this is the external tooling step. Builds a D4-shaped
cross/mixer domain (four arms of half-width Wc meeting at a narrow junction),
graded fine at the central junction and coarse out in the arms, and tags the
physical faces:
    source = bottom arm end (y = -R)      drain = left arm end (x = -R)
    wall   = everything else (incl. the top/right reflective probe arms)
Written in the format qimpy.transport.geometry.FiniteVolume consumes (_dg_mesh).

    python make_mixer_mesh.py                 # -> mixer-tri.npz
    python make_mixer_mesh.py --h0 0.6        # coarser junction
"""
import argparse
from collections import Counter
import numpy as np
import triangle as tr
from qimpy.transport.geometry._dg_mesh import save_mesh

R = 20.0                                   # arm length (half-span)
a = 0.75                                   # junction half-size
Wc = a + (R - a) * np.tan(np.radians(34.0))  # arm half-width ~= 13.73

# cross polygon, CCW (right, top, left, bottom arms)
CROSS = np.array([[R, -Wc], [R, Wc], [a, a], [Wc, R], [-Wc, R], [-a, a],
                  [-R, Wc], [-R, -Wc], [-a, -a], [-Wc, -R], [Wc, -R], [a, -a]])
SEG = np.array([[i, (i + 1) % 12] for i in range(12)])


def make(path: str, h0: float, slope: float, hmax: float, n_refine: int) -> None:
    def hfun(x, y):                        # target edge length vs radius
        return np.clip(h0 + slope * np.hypot(x, y), h0, hmax)
    m = tr.triangulate({"vertices": CROSS, "segments": SEG}, "pq30a4.0")
    for _ in range(n_refine):              # size-function refinement
        V, T = m["vertices"], m["triangles"]
        cen = V[T].mean(1)
        area = 0.5 * np.abs(np.cross(V[T[:, 1]] - V[T[:, 0]],
                                     V[T[:, 2]] - V[T[:, 0]]))
        target = 0.43 * hfun(cen[:, 0], cen[:, 1]) ** 2
        mxa = np.where(area > target, target, -1.0)
        m = tr.triangulate(dict(vertices=V, triangles=T,
                                triangle_max_area=mxa), "rpa")
    V, T = m["vertices"], m["triangles"]
    ec: Counter = Counter()
    for t in T:
        for x, y in [(0, 1), (1, 2), (2, 0)]:
            ec[tuple(sorted((int(t[x]), int(t[y]))))] += 1
    be = [e for e, c in ec.items() if c == 1]
    bm = []
    for i, j in be:
        mx, my = 0.5 * (V[i] + V[j])
        if abs(my + R) < 0.5 and abs(mx) <= Wc + 0.5:
            bm.append("source")            # bottom face
        elif abs(mx + R) < 0.5 and abs(my) <= Wc + 0.5:
            bm.append("drain")             # left face
        else:
            bm.append("wall")
    save_mesh(path, V, T, np.array(be), bm)
    print(f"wrote {path}: {len(T)} triangles, {len(be)} boundary edges "
          f"(source={bm.count('source')}, drain={bm.count('drain')}, "
          f"wall={bm.count('wall')}); junction edge ~{h0}, arm edge ~{hmax}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mixer-tri.npz")
    ap.add_argument("--h0", type=float, default=0.20, help="junction edge length")
    ap.add_argument("--slope", type=float, default=0.10)
    ap.add_argument("--hmax", type=float, default=2.6, help="arm edge length")
    ap.add_argument("--refine", type=int, default=5)
    args = ap.parse_args()
    make(args.out, args.h0, args.slope, args.hmax, args.refine)
