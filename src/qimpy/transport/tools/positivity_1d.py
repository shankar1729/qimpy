"""Is the FV scheme bound-preserving on a UNIFORM 1D mesh?  (PR #48, point 2.)

Shankar's question: positivity was guaranteed in the previous (DG + Zhang-Shu
+ SSPRK3) formalism; is it guaranteed in the new FV scheme, at least for the
special case of a uniform 1D mesh?  It matters because Lindblad ab initio
scattering misbehaves when density-matrix eigenvalues leave [0, 1].

THE THEORY.  A second-order FV scheme is bound-preserving when three conditions
hold TOGETHER; drop any one and the guarantee is gone:

  (i)   the reconstruction never leaves the range of the cell's neighbourhood,
  (ii)  the time integrator is SSP -- a convex combination of forward-Euler
        steps, so it inherits FE's bound,
  (iii) the step satisfies the CFL that makes each of those FE steps a convex
        combination of neighbouring cell averages.

qimpy's FV meets (i): the Venkatakrishnan limiter is `.clamp(max=1.0)`, i.e.
bounded above by Barth-Jespersen, which is exactly the neighbourhood-range
condition.  (ii) is a USER CHOICE: `integrator: SSPRK3` is SSP, but the DEFAULT
`RK2` is the explicit midpoint rule, which is NOT.  (iii) is what this script
measures, because the admissible CFL is scheme- and mesh-dependent and we have
never determined it.

So on a uniform 1D mesh the expected answer is: YES with SSPRK3 below some CFL,
NO with the default RK2 at any CFL.  This script tests that rather than
asserting it.

⛔ MEASURE f, NOT THE STATE VECTOR.  In the modal representation the evolved
variable is a set of delta-f coefficients, and "coefficient < 0" says nothing
about occupancy.  This uses the CARTESIAN representation, where
f = f0_lab + delta-f is a genuine occupancy and [0, 1] is the physical bound --
the same quantity that reads [-1.66e-2, 1.026] on the production mixer.

⛔ 1e3 STEPS MINIMUM.  A 40-step run showed no violation on a case that is badly
unbounded by 1e4 steps; the growth is slow and early cleanliness proves nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from qimpy.transport.geometry import FiniteVolume
from qimpy.transport.geometry._mesh import save_mesh

torch.set_default_dtype(torch.float64)

# production mixer material, so the answer transfers to the runs we care about
CFG = dict(kF=7.5e-3, vF=0.11194, M_theta=32, Nr=6, T=1.3301e-5, xi_max=6.0,
           tau_p=float("inf"), specularity=1.0,
           cartesian=dict(annulus_xi=0.0, te_fac_max=6.0, kD_max=1.2e-3,
                          dmu_max=1.2e-4, k_max=0.0132557160008, n_k=56))
DMU = 5.37e-5


def uniform_line(nx: int, path: str, Lx: float = 1.0) -> str:
    """nx equal interval cells on [0, Lx] -- uniform by construction."""
    x = np.linspace(0.0, Lx, nx + 1)
    V = np.column_stack([x, np.zeros(nx + 1)])
    cells = np.column_stack([np.arange(nx), np.arange(1, nx + 1)])
    be = np.array([[0, 0], [nx, nx]], int)
    save_mesh(path, V, cells, be, ["source", "drain"])
    return path


def build(nx: int):
    pg = ProcessGrid("rk", (1, 1))
    mat = FermiSurface(process_grid=pg, **CFG)
    tmp = tempfile.mkdtemp()
    mesh = uniform_line(nx, os.path.join(tmp, "line.npz"))
    geom = FiniteVolume(
        material=mat, mesh_file=mesh, process_grid=pg, compile=False,
        contacts={"source": {"dmu": DMU, "nonlinear": True},
                  "drain": {"dmu": -DMU, "nonlinear": True}})
    return geom, mat


def advance(geom, u, dt, scheme):
    """One step.  RK2 = explicit midpoint (NOT SSP).  SSPRK3 = Shu-Osher."""
    d = lambda w: geom.rho_dot(type(geom.rho)([w]), 0.0)[0]
    if scheme == "RK2":
        return u + dt * d(u + 0.5 * dt * d(u))
    if scheme == "FE":
        return u + dt * d(u)
    if scheme == "SSPRK3":
        u1 = u + dt * d(u)
        u2 = 0.75 * u + 0.25 * (u1 + dt * d(u1))
        return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * d(u2))
    raise KeyError(scheme)


def run(nx: int, scheme: str, cfl: float, steps: int) -> dict:
    geom, mat = build(nx)
    f0 = mat.representation._f0_lab[None, :]
    u = geom.rho[0].clone()
    dt = cfl * float(geom.dt_max)
    lo, hi = 1.0, 0.0
    lo_step = hi_step = -1
    for s in range(steps):
        u = advance(geom, u, dt, scheme)
        if not torch.isfinite(u).all():
            return dict(nx=nx, scheme=scheme, cfl=cfl, steps=s, diverged=True,
                        min_f=float("nan"), max_f=float("nan"))
        if (s + 1) % 25 == 0 or s == steps - 1:
            f = f0 + u
            mn, mx = float(f.min()), float(f.max())
            if mn < lo:
                lo, lo_step = mn, s + 1
            if mx > hi:
                hi, hi_step = mx, s + 1
    # ⛔ REPORT THE EXCESS, NOT THE VALUE.  max f sits at exactly 1.0 because the
    # deep Fermi sea is occupied, so "max f = 1.0000000000" is the healthy
    # reading and a violation of 1e-6 is invisible at any sane print width.  The
    # violation is (max f - 1), and it must be printed in its own right.
    f = f0 + u
    n_lo = int((f < 0.0).sum())
    n_hi = int((f > 1.0).sum())
    return dict(nx=nx, scheme=scheme, cfl=cfl, steps=steps, diverged=False,
                min_f=lo, max_f=hi, over=hi - 1.0,
                n_below_0=n_lo, n_above_1=n_hi, n_total=int(f.numel()),
                first_min_step=lo_step, first_max_step=hi_step,
                bounded=bool(lo >= -1e-14 and hi <= 1 + 1e-14))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="/tmp/pos1d.json")
    a = ap.parse_args()
    rc.init()
    out = []
    print(f"  uniform 1D, nx={a.nx}, {a.steps} steps, production mixer material")
    print(f"  {'scheme':>8} {'CFL':>6} {'min f':>13} {'max f - 1':>13} "
          f"{'#<0':>7} {'#>1':>7}  bounded")
    for scheme in ("RK2", "SSPRK3"):
        for cfl in (0.9, 0.5, 0.25, 0.1, 0.05):
            r = run(a.nx, scheme, cfl, a.steps)
            out.append(r)
            if r["diverged"]:
                print(f"  {scheme:>8} {cfl:>6.2f}   DIVERGED at step {r['steps']}",
                      flush=True)
                continue
            tag = "yes" if r["bounded"] else "NO"
            print(f"  {scheme:>8} {cfl:>6.2f} {r['min_f']:>13.3e} "
                  f"{r['over']:>13.3e} {r['n_below_0']:>7d} {r['n_above_1']:>7d}"
                  f"  {tag}", flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
