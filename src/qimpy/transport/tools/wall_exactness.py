"""What it takes to reflect off a wall at fp64, at ANY angle.

THE RESULT, in three measurements.  For a band-limited angular state on the
modal (delta-k) representation, with the wall normal at an arbitrary angle:

  1. THE REFLECTION ITSELF IS ALREADY EXACT.  Specular reflection is
     theta -> 2 alpha - theta, which on the angular harmonics is the per-mode
     rotation  (c_m, s_m) -> (c_m cos 2m alpha + s_m sin 2m alpha,
                              c_m sin 2m alpha - s_m cos 2m alpha),
     an orthogonal 2x2 block.  `_DeltaKReflector._specular_modal` implements it.
     Measured against the exact reflected function: 9.1e-15 at 17 deg, 1.3e-14
     at 30, 9.7e-15 at 63, 9.8e-15 at 11.37.  Every angle, machine precision.

  2. THE FLUX INTEGRAL IS WHAT IS NOT EXACT.  The wall's conservation laws are
     integrals of |v.n| f over a half-space, and |v.n| = vF |cos(theta - phi)|
     has a KINK at v.n = 0.  A kinked integrand on the uniform theta grid is a
     first-order quadrature no matter how smooth f is: residual 9.7e-4 at
     17 deg, 8.9e-3 at 30, 6.9e-2 at 63 -- and 1e-13 at 0 and 45, where the
     kink lands on the grid's own symmetry and the error cancels.  THAT is why
     axis-aligned walls always looked perfect and everything else did not.

  3. SPLITTING THE QUADRATURE AT THE KINK FIXES IT.  Integrate each half-space
     on its own panel, [phi - pi/2, phi + pi/2] and its complement, with Gauss
     nodes and f evaluated by exact trig interpolation.  |v.n| is smooth on each
     panel, so the rule converges spectrally instead of at first order:

        nodes/panel      17 deg       63 deg    11.37 deg
             34        8.10e-06     1.45e-05     8.88e-06
             48        2.71e-14     2.31e-14     2.65e-14
             64        2.59e-14     2.19e-14     2.23e-14
            128        1.82e-14     1.47e-14     1.68e-14

     Flat from 48 nodes on -- that is the fp64 floor, not a trend.

SO THE RECIPE FOR A MACHINE-PRECISION WALL IS: exact harmonic rotation for the
ghost, and split-panel quadrature for the flux moments.  With both, no closure
is needed at all -- and the bound comes free, because the ghost is an exact
evaluation of the same band-limited function and cannot leave its range.

⛔ EVERY EARLIER ATTEMPT TREATED A SYMPTOM.  The rank-4 numerics closure, the
rank-2 diffuse refill, a cubic stencil, radially exact weights, Sinkhorn
scaling -- all of them repair the consequences of a quadrature that cannot see
the kink.  None can reach fp64, because the kink is only C0 and no amount of
correction to a first-order rule makes it spectral.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface

torch.set_default_dtype(torch.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--angles", type=float, nargs="+",
                    default=[0.0, 11.37, 17.0, 30.0, 45.0, 63.0, 88.3])
    ap.add_argument("--panel-nodes", type=int, nargs="+", default=[34, 48, 128])
    a = ap.parse_args()
    rc.init()

    fs = FermiSurface(kF=1.0, vF=1.5, M_theta=a.M, Nr=1, T=1.0,
                      tau_p=np.inf, specularity=1.0,
                      process_grid=ProcessGrid("rk", (1, 1)))
    rep = fs.representation
    th = fs.angular.theta.clone()
    g = torch.Generator(device=th.device).manual_seed(1)
    c = torch.randn(a.M + 1, generator=g, device=th.device, dtype=torch.float64)
    s = torch.randn(a.M + 1, generator=g, device=th.device, dtype=torch.float64)

    def series(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for m in range(a.M + 1):
            out = out + c[m] * torch.cos(m * x) + s[m] * torch.sin(m * x)
        return out

    def series_np(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        cn, sn = c.cpu().numpy(), s.cpu().numpy()
        for m in range(a.M + 1):
            out += cn[m] * np.cos(m * x) + sn[m] * np.sin(m * x)
        return out

    u = series(th)
    N = int(fs.angular.N_theta)
    print(f"  band-limited state, M = {a.M}; uniform grid N_theta = {N}\n")
    print(f"  {'wall':>7} {'(1) rotation':>15} {'(2) uniform flux':>18} "
          + " ".join(f"(3) split n={n}" for n in a.panel_nodes))
    for deg in a.angles:
        phi = np.deg2rad(deg)
        n_ = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64,
                          device=th.device)
        R = fs.get_reflector(n_)
        rot = rep.from_modes(R._specular_modal(rep.to_modes(u[None, None, :])))[0, 0]
        exact = series(2 * (phi + np.pi / 2) - th)
        e_rot = float((rot - exact).abs().max() / u.abs().max())

        lo, hi = phi - np.pi / 2, phi + np.pi / 2
        gx, gw = np.polynomial.legendre.leggauss(400)          # reference
        X = 0.5 * (hi - lo) * gx + 0.5 * (lo + hi)
        W = 0.5 * (hi - lo) * gw
        ref = float(np.sum(W * np.cos(X - phi) * series_np(X)))

        thn = th.cpu().numpy()
        vdn = np.cos(thn - phi)
        uni = float((2 * np.pi / N) * np.sum(np.clip(vdn, 0, None)
                                             * series_np(thn)))
        cells = [f"{abs(uni - ref) / max(abs(ref), 1e-30):>18.3e}"]
        for npn in a.panel_nodes:
            gx2, gw2 = np.polynomial.legendre.leggauss(npn)
            X2 = 0.5 * (hi - lo) * gx2 + 0.5 * (lo + hi)
            W2 = 0.5 * (hi - lo) * gw2
            spl = float(np.sum(W2 * np.cos(X2 - phi) * series_np(X2)))
            cells.append(f"{abs(spl - ref) / max(abs(ref), 1e-30):>14.3e}")
        print(f"  {deg:>7.2f} {e_rot:>15.3e} " + " ".join(cells))


if __name__ == "__main__":
    main()
