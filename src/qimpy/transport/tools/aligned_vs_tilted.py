"""THE control experiment: identical physics, walls aligned vs tilted to the k-grid.

A rectangular channel -- contacts on two opposite faces, WALLS on the other two
-- is run twice: once axis-aligned, once with the whole geometry rotated.  The
rotation changes nothing physical (the material is isotropic and the mesh is the
same shape), but it changes whether the specular mirror k -> k - 2(k.n)n is a
grid symmetry:

    aligned  wall normals along +-x, +-y  ->  the mirror is an exact grid
             permutation, the reflector is exact to 1e-14
    tilted   wall normals at an odd angle ->  the mirror lands between grid
             points and must be interpolated

⛔ THIS IS THE EXPERIMENT THAT DECIDES WHETHER THE WALL IS THE DEVICE'S
DOMINANT SOURCE.  A single application of the reflector to a smooth analytic
drifted-FD is clean to 1e-30 once the stencil is cubic, yet the mixer still
leaves [0, 1] by ~1e-2 after 3000 steps.  Either the aligned channel is clean --
and the wall interpolation is confirmed as the cause, the residual being an
accumulation over bounces on a rough (non-analytic) input -- or it is NOT, and
the wall is merely necessary while something else supplies the amplitude.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import torch

from qimpy import rc
from qimpy.transport import Transport
from qimpy.transport.geometry import TensorList
from qimpy.transport.geometry._mesh import save_mesh

torch.set_default_dtype(torch.float64)

CFG = dict(kF=7.5e-3, vF=0.11194, M_theta=32, Nr=6, T=1.3301e-5, xi_max=6.0,
           tau_p=float("inf"), specularity=1.0, residual_damping=False,
           cartesian=dict(annulus_xi=0.0, te_fac_max=6.0, kD_max=1.2e-3,
                          dmu_max=1.2e-4, k_max=0.0132557160008, n_k=56))
DMU = 5.37e-5


def channel(nx: int, ny: int, L: float, W: float, deg: float, path: str) -> str:
    """Rectangular channel, contacts on the +-x ends, WALLS on the +-y sides,
    the whole thing rotated by `deg` about its centre."""
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, W, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    V = np.column_stack([X.ravel(), Y.ravel()])
    idx = lambda i, j: i * (ny + 1) + j
    tris, be, bm = [], [], []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [[a, b, c], [a, c, d]]
    for j in range(ny):                              # x = 0 and x = L: contacts
        be.append([idx(0, j), idx(0, j + 1)]); bm.append("source")
        be.append([idx(nx, j), idx(nx, j + 1)]); bm.append("drain")
    for i in range(nx):                              # y = 0 and y = W: walls
        be.append([idx(i, 0), idx(i + 1, 0)]); bm.append("wall")
        be.append([idx(i, ny), idx(i + 1, ny)]); bm.append("wall")
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    cen = V.mean(0)
    V = (V - cen) @ R.T + cen
    save_mesh(path, V, np.array(tris), np.array(be), bm)
    return path


def run(mesh: str, steps: int, n_k: int) -> dict:
    cfg = dict(CFG); cfg["cartesian"] = dict(CFG["cartesian"], n_k=n_k)
    t = Transport(fermi_surface=cfg,
                  spatial_transport=dict(mesh_file=mesh, compile=False,
                                         save_rho=True,
                                         contacts={"source": {"dmu": DMU,
                                                              "nonlinear": True},
                                                   "drain": {"dmu": -DMU,
                                                             "nonlinear": True}}),
                  time_evolution=dict(t_max=1e30, dt_save=1e30, n_collate=1))
    g = t.geometry
    f0 = t.material.representation._f0_lab[None, :]
    u = g.rho[0].clone()
    dt = float(g.dt_max)
    d = lambda w: g.rho_dot(TensorList([w]), 0.0)[0]
    lo, hi = 1.0, 0.0
    for s in range(steps):
        u = u + dt * d(u + 0.5 * dt * d(u))
        if not torch.isfinite(u).all():
            return dict(diverged_at=s + 1)
        if (s + 1) % 50 == 0 or s == steps - 1:
            f = f0 + u
            lo = min(lo, float(f.min())); hi = max(hi, float(f.max()))
    f = f0 + u
    return dict(min_f=lo, over=hi - 1.0, n_below=int((f < 0).sum()),
                n_above=int((f > 1).sum()), n_total=int(f.numel()),
                K=int(g.K))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--nk", type=int, nargs="+", default=[112])
    ap.add_argument("--angles", type=float, nargs="+", default=[0.0, 17.0])
    ap.add_argument("--nx", type=int, default=24)
    ap.add_argument("--ny", type=int, default=8)
    a = ap.parse_args()
    rc.init()
    tmp = tempfile.mkdtemp()
    print(f"  rectangular channel {a.nx}x{a.ny}, contacts on +-x, WALLS on +-y")
    print(f"  {a.steps} steps, dt = dt_max, interp = "
          f"{os.environ.get('QIMPY_REFL_INTERP', 'cubic')}")
    print(f"  {'n_k':>5} {'rotation':>9} {'min f':>13} {'max f - 1':>13}"
          f" {'#<0':>9} {'#>1':>8}")
    out = []
    for n_k in a.nk:
        for deg in a.angles:
            m = channel(a.nx, a.ny, 6.0, 2.0, deg,
                        os.path.join(tmp, f"ch{deg:g}.npz"))
            r = run(m, a.steps, n_k)
            r.update(n_k=n_k, deg=deg)
            out.append(r)
            if r.get("diverged_at"):
                print(f"  {n_k:>5} {deg:>8.1f}d   DIVERGED at {r['diverged_at']}",
                      flush=True)
                continue
            print(f"  {n_k:>5} {deg:>8.1f}d {r['min_f']:>13.3e}"
                  f" {r['over']:>13.3e} {r['n_below']:>9,} {r['n_above']:>8,}",
                  flush=True)
    json.dump(out, open("aligned_vs_tilted.json", "w"), indent=1)


if __name__ == "__main__":
    main()
