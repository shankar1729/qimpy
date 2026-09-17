"""Is the f-bound violation injected at the WALL, or by the interior scheme?

Same 2D mesh, three boundary variants, everything else identical:

    allcontact  every boundary edge is a contact -> the reflector NEVER runs
    cavity      every boundary edge is a wall    -> reflector only, no drive
    asis        the production mix

If `allcontact` stays inside [0, 1] and `cavity`/`asis` do not, the violation is
produced by the wall closure and there is nothing to fix in the reconstruction
or the time integrator.

WHY THE WALL IS THE SUSPECT.  The diffuse refill is alpha * f0 (1 - f0), so
Pauli requires alpha <= 1.  Measured alpha vs Te/T is 0.40 / 0.48 / 0.80 / 1.21
at Te/T = 1 / 2 / 4 / 5.6 -- it crosses 1 at Te/T ~ 4.7, and the production
mixer ran at Te/T = 5.69.  Separately, a uniform 1D mesh (source/drain only, NO
walls) is clean at CFL 0.9 with the non-SSP RK2 over 4000 steps.

⛔ >=1e3 STEPS.  A 40-step run showed nothing on a case that is badly unbounded
by 1e4 steps.  Early cleanliness proves nothing here.
⛔ REPORT (max f - 1), NOT max f.  The deep Fermi sea sits at exactly 1.0, so
max f prints as 1.0000000000 whether or not it is violated.
⛔ COMPARE AT EQUAL STEP COUNT from an identical start -- not at "converged".
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from qimpy import rc
from qimpy.transport import Transport
from qimpy.transport.geometry import TensorList

torch.set_default_dtype(torch.float64)

CFG_FS = dict(kF=7.5e-3, vF=0.11194, M_theta=32, Nr=6, T=1.3301e-5, xi_max=6.0,
              tau_p=float("inf"), specularity=1.0, residual_damping=False,
              cartesian=dict(annulus_xi=0.0, te_fac_max=6.0, kD_max=1.2e-3,
                             dmu_max=1.2e-4, k_max=0.0132557160008, n_k=56))
DMU = 5.37e-5


def make_mesh(base: str, kind: str, out: str) -> str:
    d = dict(np.load(base, allow_pickle=True))
    n = len(d["boundary_markers"])
    if kind == "asis":
        return base
    if kind == "allcontact":
        d["boundary_markers"] = np.array(
            ["source" if (i % 2 == 0) else "drain" for i in range(n)])
    elif kind == "cavity":
        d["boundary_markers"] = np.array(["wall"] * n)
    else:
        raise KeyError(kind)
    np.savez(out, **d)
    return out


def run(mesh: str, kind: str, steps: int, scheme: str, n_k: int = 56) -> dict:
    path = make_mesh(mesh, kind, f"/tmp/bb_{kind}.npz")
    names = set(str(x) for x in np.load(path, allow_pickle=True)["boundary_markers"])
    contacts = {}
    if "source" in names:
        contacts["source"] = {"dmu": DMU, "nonlinear": True}
    if "drain" in names:
        contacts["drain"] = {"dmu": -DMU, "nonlinear": True}
    cfg = dict(CFG_FS)
    cfg["cartesian"] = dict(CFG_FS["cartesian"], n_k=n_k)
    t = Transport(fermi_surface=cfg,
                  spatial_transport=dict(mesh_file=path, compile=False,
                                         save_rho=True, contacts=contacts),
                  time_evolution=dict(t_max=1e30, dt_save=1e30, n_collate=1))
    g = t.geometry
    f0 = t.material.representation._f0_lab[None, :]
    u = g.rho[0].clone()
    dt = float(g.dt_max)
    d = lambda w: g.rho_dot(TensorList([w]), 0.0)[0]
    lo, hi, lo_at, hi_at = 1.0, 0.0, -1, -1
    for s in range(steps):
        if scheme == "RK2":
            u = u + dt * d(u + 0.5 * dt * d(u))
        else:                                            # SSPRK3
            u1 = u + dt * d(u)
            u2 = 0.75 * u + 0.25 * (u1 + dt * d(u1))
            u = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * d(u2))
        if not torch.isfinite(u).all():
            return dict(kind=kind, scheme=scheme, n_k=n_k, diverged_at=s + 1)
        if (s + 1) % 50 == 0 or s == steps - 1:
            f = f0 + u
            mn, mx = float(f.min()), float(f.max())
            if mn < lo:
                lo, lo_at = mn, s + 1
            if mx > hi:
                hi, hi_at = mx, s + 1
    f = f0 + u
    bad = (f < 0.0).any(dim=1)                       # (K,) cells holding any f<0
    # ⛔ WHERE the violation sits is the whole point: a cell touching BOTH a wall
    # edge and a contact edge is a different diagnosis from a cell touching only
    # one, or from an interior cell.  Classify every cell by the boundary kinds
    # on its own faces.
    mesh = g.geom
    K = int(g.K)
    touches_wall = np.zeros(K, dtype=bool)
    touches_contact = np.zeros(K, dtype=bool)
    try:
        bm = np.array([str(x) for x in np.load(path, allow_pickle=True)
                       ["boundary_markers"]])
        be = np.asarray(np.load(path, allow_pickle=True)["boundary_edges"])
        tri = np.asarray(np.load(path, allow_pickle=True)["triangles"])
        vert_wall = set(be[bm == "wall"].ravel().tolist())
        vert_con = set(be[bm != "wall"].ravel().tolist())
        for c in range(min(K, len(tri))):
            vs = set(tri[c].tolist())
            touches_wall[c] = bool(vs & vert_wall)
            touches_contact[c] = bool(vs & vert_con)
    except Exception:
        pass
    badn = bad.cpu().numpy()[:K]
    both = int((badn & touches_wall & touches_contact).sum())
    wall_only = int((badn & touches_wall & ~touches_contact).sum())
    con_only = int((badn & ~touches_wall & touches_contact).sum())
    interior = int((badn & ~touches_wall & ~touches_contact).sum())
    return dict(kind=kind, scheme=scheme, n_k=n_k, steps=steps, diverged_at=None,
                min_f=lo, over=hi - 1.0, min_at=lo_at, over_at=hi_at,
                n_below_0=int((f < 0.0).sum()), n_above_1=int((f > 1.0).sum()),
                n_total=int(f.numel()), n_bad_cells=int(badn.sum()),
                bad_wall_and_contact=both, bad_wall_only=wall_only,
                bad_contact_only=con_only, bad_interior=interior,
                bounded=bool(lo >= -1e-14 and hi <= 1.0 + 1e-14))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--nk", type=int, nargs="+", default=[56])
    ap.add_argument("--kinds", nargs="+", default=["allcontact", "cavity", "asis"])
    ap.add_argument("--schemes", nargs="+", default=["RK2", "SSPRK3"])
    ap.add_argument("--out", default="/tmp/bound_by_boundary.json")
    a = ap.parse_args()
    rc.init()
    res = []
    print(f"  {a.steps} steps, production mixer material, dt = dt_max")
    print(f"  {'n_k':>5} {'boundary':>11} {'scheme':>7} {'min f':>12} "
          f"{'max f - 1':>12} {'#<0':>6} {'#>1':>6}  bounded")
    for n_k in a.nk:
      for kind in a.kinds:
        for scheme in a.schemes:
            r = run(a.mesh, kind, a.steps, scheme, n_k)
            res.append(r)
            if r.get("diverged_at"):
                print(f"  {n_k:>5} {kind:>11} {scheme:>7}   DIVERGED "
                      f"at {r['diverged_at']}",
                      flush=True)
                continue
            print(f"  {n_k:>5} {kind:>11} {scheme:>7} {r['min_f']:>12.3e} "
                  f"{r['over']:>12.3e} {r['n_below_0']:>6d} {r['n_above_1']:>6d}"
                  f"  {'yes' if r['bounded'] else 'NO'}", flush=True)
            if r["n_bad_cells"]:
                print(f"              cells f<0: {r['n_bad_cells']}"
                      f"  [wall+contact {r['bad_wall_and_contact']},"
                      f" wall-only {r['bad_wall_only']},"
                      f" contact-only {r['bad_contact_only']},"
                      f" interior {r['bad_interior']}]", flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
