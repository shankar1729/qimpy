"""Pointwise error of the Cartesian specular reflector, against the EXACT image.

HYPOTHESIS UNDER TEST.  The exact specular map k -> k* = k - 2(k.n)n preserves
|k| EXACTLY, so f0(k*) = f0(k) and the reflection does not move a state between
energy shells at all.  The implementation, however, interpolates delta-f
BILINEARLY on the (kx, ky) grid, and the four stencil corners around k* sit on
NEIGHBOURING |k| shells whose equilibrium occupancy differs by e^(+-dxi).
Linear interpolation across an exponential overshoots by cosh(dxi/2) - 1, which
is 12.6% at the production spacing dxi = 1.0.  Near the Fermi surface f0 ~ 0.5
and a 13% error on delta-f is harmless; where f0 has saturated to 0 or 1 there
is NO headroom and the same relative error takes f out of [0, 1].

THE CONTROLS THAT MAKE THIS FALSIFIABLE.
  * an AXIS-ALIGNED wall (phi = 0, 90 deg) mirrors grid point onto grid point,
    so the interpolation is exact and the error must be at roundoff.  If a
    tilted wall is bad and an aligned wall is clean, at identical physics, the
    interpolation is the cause and nothing else can be.
  * the error must fall as dxi^2 with n_k.
  * the error must be largest where f0 is saturated, not at the Fermi surface.

THE REFERENCE.  For an analytic input we can write the exact answer down: a
drifted Fermi-Dirac f(k) = FD(eps(k - kD)) reflects to f(k*) = FD(eps(k* - kD)),
evaluated at the exact real-valued k*, with NO grid involved.  Everything is
compared against that.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface

torch.set_default_dtype(torch.float64)


def fd(eps: torch.Tensor, mu: float, T: float) -> torch.Tensor:
    return torch.special.expit(-(eps - mu) / T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-k", type=int, nargs="+", default=[56, 112, 224])
    ap.add_argument("--k-max", type=float, default=0.0132557160008)
    ap.add_argument("--kF", type=float, default=7.5e-3)
    ap.add_argument("--vF", type=float, default=0.11194)
    ap.add_argument("--T", type=float, default=1.3301e-5)
    ap.add_argument("--drift", type=float, default=0.05, help="|kD| / kF")
    ap.add_argument("--angles", type=float, nargs="+",
                    default=[0.0, 15.0, 30.0, 45.0, 90.0])
    ap.add_argument("--te", type=float, nargs="+", default=[1.0],
                    help="input electron temperature Te/T (1 = drift only)")
    a = ap.parse_args()
    rc.init()

    print(f"  drifted-FD input, |kD| = {a.drift} kF; comparing the reflector to "
          f"the EXACT specular image\n")
    for n_k in a.n_k:
        fs = FermiSurface(
            kF=a.kF, vF=a.vF, M_theta=32, Nr=6, T=a.T, xi_max=6.0,
            tau_p=float("inf"), specularity=1.0,
            cartesian=dict(annulus_xi=0.0, te_fac_max=6.0, kD_max=1.2e-3,
                           dmu_max=1.2e-4, k_max=a.k_max, n_k=n_k),
            process_grid=ProcessGrid("rk", (1, 1)))
        rep = fs.representation
        k = rep.k
        m, mu, T = float(rep.m_star), float(rep.mu), float(rep.T_temp)
        dk = 2.0 * a.k_max / n_k
        dxi = (a.kF / m) * dk / T
        f0 = rep._f0_lab
        kD = torch.tensor([a.drift * a.kF, 0.0], dtype=k.dtype, device=k.device)
        xi = (((k ** 2).sum(-1)) / (2 * m) - mu) / T

        print(f"  n_k = {n_k}   dxi/point = {dxi:.2f}   W = "
              f"{os.environ.get('QIMPY_REFL_W','none')}"
              f"  interp={os.environ.get('QIMPY_REFL_INTERP','cubic')}"
              f"  radex={os.environ.get('QIMPY_REFL_RADEX','1')}")
        print(f"    {'Te/T':>5} {'wall':>6} {'rel err SHELL':>15} "
              f"{'rel err TAIL':>14} {'max|err|/max|df|':>18} {'min f after':>13}")
        # a drifted AND heated Fermi-Dirac; Te/T = 1 is pure drift
        for te in a.te:
            u_in = fd(((k - kD) ** 2).sum(-1) / (2 * m), mu, T * te) - f0
            for deg in a.angles:
                phi = np.deg2rad(deg)
                n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=k.dtype,
                                 device=k.device)
                refl = fs.get_reflector(n)
                out = refl(u_in[None, None, :])[0, 0]
                # exact image: reflect the COORDINATE, evaluate the analytic FD
                kn = (k * n[0]).sum(-1, keepdim=True)
                kstar = k - 2.0 * kn * n[0][None, :]
                exact = fd(((kstar - kD) ** 2).sum(-1) / (2 * m), mu, T * te) - f0
                err = out - exact
                vn = (k * n[0]).sum(-1)        # only the inflow half is written
                inflow = vn < 0
                sh = (xi.abs() < 6) & inflow
                tl = (xi > 20) & inflow

                def rel(sel):
                    if int(sel.sum()) == 0:
                        return float("nan")
                    d = exact[sel].abs().max().clamp(min=1e-300)
                    return float(err[sel].abs().max() / d)

                f_after = (f0 + out)[inflow]
                stg = getattr(refl, "_stage", None)
                extra = ""
                if stg:
                    # ⛔ WHICH OPERATION BREAKS THE BOUND?  Bilinear weights are
                    # a convex combination, so interpolating f cannot leave
                    # [0,1] -- but the rank-4 and rank-2 moment CLOSURES that
                    # run afterwards add correction vectors with no such
                    # property.  Report min f after each stage separately.
                    extra = "  minf[" + " ".join(
                        f"{nm}={float((f0 + v[0,0])[inflow].min()):+.2e}"
                        for nm, v in stg.items()
                        if not nm.startswith("alpha")) + "]"
                    al = [v for nm, v in stg.items() if nm.startswith("alpha")]
                    if al:
                        extra += "  alpha_max=%.3g" % max(al)
                print(f"    {te:>5.2f} {deg:>6.1f} {rel(sh):>15.3e}"
                      f" {rel(tl):>14.3e}"
                      f" {float(err.abs().max()/u_in.abs().max()):>18.3e}"
                      f" {float(f_after.min()):>13.3e}{extra}", flush=True)
        print()


if __name__ == "__main__":
    main()
