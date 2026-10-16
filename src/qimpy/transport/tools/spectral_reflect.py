"""EXACT specular reflection ON THE CARTESIAN k-GRID, at any wall angle.

⛔ MY OWN OBJECTION WAS TOO NARROW.  I argued the Cartesian grid cannot do this
because an energy SHELL carries only ~8 samples (the D4 orbit), too few for the
32 harmonics in use.  That rules out a per-shell angular rotation and nothing
else.  The state is not 8-dimensional per shell: it lives in a TENSOR-PRODUCT
span, radial x angular, and 40320 grid points determine a few thousand
coefficients hugely overdeterminedly.  The angular information is there -- it is
just spread across shells rather than sitting inside one.

THE CONSTRUCTION.  Let B be the (Nk x Ndof) matrix of basis functions
    b_{p,m}(k) = g_p(xi(|k|)) * {cos m theta, sin m theta}
evaluated at the grid points, and B* the same basis evaluated at the MIRRORED
points k* = k - 2(k.n)n.  Then

    S = B* (B^T B)^-1 B^T

reproduces f(k*) exactly for every f in span(B), at any angle, with no
interpolation anywhere.  Reflection preserves |k|, so g_p is untouched and the
whole angular action is the exact rotation theta -> 2 alpha - theta.  S is a
FIXED linear operator -- no state dependence -- so the dense boundary cache in
_setup_boundary stays valid, and it is applied as two thin matmuls
(Nk -> Ndof -> Nk), never as a dense Nk x Nk matrix.

⛔ REFLECT delta-f, NOT f.  f0 is a sharp step of width ~1 in xi across a range
of ~100, so putting it in the span would need a huge radial order.  It does not
need to be there: |k*| = |k| exactly, so f0(k*) = f0(k) and the f0 part of the
reflection is the identity.  Only delta-f -- smooth, and confined to within a
few Te of the Fermi surface -- has to be spanned.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface

torch.set_default_dtype(torch.float64)


_KNOTS: dict = {}


def _bspline(xi: torch.Tensor, P: int) -> torch.Tensor:
    """Cubic B-spline design matrix, knots at quantiles of xi (cached)."""
    key = (P, int(xi.shape[0]), float(xi[0]))
    t = _KNOTS.get(key)
    if t is None:
        q = torch.linspace(0, 1, P - 2, device=xi.device, dtype=xi.dtype)
        inner = torch.quantile(xi.float(), q.float()).to(xi.dtype)
        inner = torch.unique(inner)
        lo = inner[0] - 1.0
        hi = inner[-1] + 1.0
        t = torch.cat([lo.repeat(4), inner[1:-1], hi.repeat(4)])
        _KNOTS[key] = t
    n = t.shape[0] - 4                                       # number of splines
    # Cox-de Boor, degree 0 -> 3
    x = xi.unsqueeze(-1)
    B = ((x >= t[:-1]) & (x < t[1:])).to(xi.dtype)
    B[..., -1] = torch.where(xi >= t[-2], torch.ones_like(xi), B[..., -1])
    for d in range(1, 4):
        num = t[d:-1] - t[:-(d + 1)]
        left = torch.where(num > 0, (x - t[:-(d + 1)]) / num.clamp(min=1e-300),
                           torch.zeros_like(x)) * B[..., :-1]
        num2 = t[d + 1:] - t[1:-d]
        right = torch.where(num2 > 0, (t[d + 1:] - x) / num2.clamp(min=1e-300),
                            torch.zeros_like(x)) * B[..., 1:]
        B = left + right
    return B[..., :n]


def basis(xi: torch.Tensor, th: torch.Tensor, P: int, M: int,
          s: float, kind: str = "tanh") -> torch.Tensor:
    """Chebyshev(u) x angular harmonics.

    ⛔ `linear` OVERFITS AND MUST NOT BE USED.  Spreading Chebyshev nodes over
    the raw xi range makes the radial functions oscillate wildly where the data
    is smooth, so the least-squares coefficients explode and the fit is garbage
    BETWEEN the sample points: measured 9.0e+04 at 17 deg and 3.7e+08 at P=192,
    while 0 deg still read 3.1e-04 because there the mirror lands on grid points
    and never leaves the samples.  A clean-looking aligned case with a diverging
    tilted one is the signature of overfitting, not of a reflection bug.

    ⛔ THE RADIAL MAP IS THE WHOLE GAME FOR THE HOT STATE.  With u = tanh(xi/2s)
    and s = 4 the cold (Te = T) case reaches 1.1e-11, but a Te = 5.69 T state
    extends to |xi| ~ 34 where tanh(34/8) = 0.9996 -- the entire outer structure
    is crushed into the last 0.04% of the map and Chebyshev cannot resolve it,
    which is why the hot error sat at 6e-5 for every s tried.  Raising the
    ANGULAR order does not help and actively hurts (M=64 drops the oversampling
    to 2.4x and the QR ill-conditions: hot 1.2e-2, cold 2.3e-5).  `linear`
    spreads Chebyshev nodes over the true xi range instead.
    """
    if kind == "bspline":
        # ⛔ CHEBYSHEV IS THE WRONG RADIAL FAMILY HERE, AND CONDITIONING IS WHY.
        # The grid's xi values are wildly non-uniform (most points sit at large
        # |xi|, since area grows with |k|), so global polynomials in a squashed
        # variable are near-dependent on this sample: raising P stops helping
        # and then hurts.  Cubic B-splines with knots at the QUANTILES of the
        # grid's own xi distribution are local, banded, uniformly resolved where
        # the data actually is, and stay well conditioned as P grows.
        rad = _bspline(xi, P)
    else:
        if kind == "linear":
            lo, hi = float(xi.min()), float(xi.max())
            u = (2.0 * xi - (hi + lo)) / (hi - lo)
        else:
            u = torch.tanh(xi / (2.0 * s))
        Tc = [torch.ones_like(u), u]
        for p in range(2, P):
            Tc.append(2.0 * u * Tc[-1] - Tc[-2])
        rad = torch.stack(Tc[:P], dim=-1)                   # (Nk, P)
        # ⛔ CHEBYSHEV IS SPARSEST EXACTLY WHERE delta-f IS SHARPEST.  Its nodes
        # cluster at |u| -> 1, i.e. |xi| -> inf, and thin out at u = 0 <-> xi = 0
        # -- which is where delta-f carries the Fermi step inherited from -f0.
        # Raising s to reach a hot tail flattens the map near xi = 0 and makes
        # that worse: the cold case gets ~5 modes across the step at s=4, the
        # hot case only ~3 at s=10, and the hot fit stalls at 2e-06 absolute
        # however many modes are added.  So put the sharp shapes IN the basis:
        # w_eq(xi; tau) xi^q spans the step and its low moments exactly, for a
        # spread of tau covering T .. te_fac_max * T.
        extra = []
        for tau in (1.0, 2.0, 4.0, 6.0):
            fw = torch.special.expit(-xi / tau)
            w = fw * (1.0 - fw)
            for q in range(4):
                extra.append(w * (xi / tau) ** q)
            extra.append(fw)
        rad = torch.cat([rad, torch.stack(extra, dim=-1)], dim=-1)
    ang = [torch.ones_like(th)]
    for m in range(1, M + 1):
        ang.append(torch.cos(m * th))
        ang.append(torch.sin(m * th))
    angt = torch.stack(ang, dim=-1)                         # (Nk, 2M+1)
    return (rad[:, :, None] * angt[:, None, :]).reshape(xi.shape[0], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-k", type=int, default=224)
    ap.add_argument("--k-max", type=float, default=0.0132557160008)
    ap.add_argument("--kF", type=float, default=7.5e-3)
    ap.add_argument("--vF", type=float, default=0.11194)
    ap.add_argument("--T", type=float, default=1.3301e-5)
    ap.add_argument("--P", type=int, default=128, help="radial order")
    ap.add_argument("--M", type=int, default=32, help="angular order")
    ap.add_argument("--s", type=float, default=4.0, help="xi mapping scale")
    ap.add_argument("--rcond", type=float, default=1e-10)
    ap.add_argument("--map", default="tanh", choices=["tanh", "linear", "bspline"])
    ap.add_argument("--drift", type=float, default=0.05)
    ap.add_argument("--te", type=float, nargs="+", default=[1.0, 5.69])
    ap.add_argument("--angles", type=float, nargs="+",
                    default=[0.0, 11.37, 17.0, 30.0, 45.0, 63.0, 88.3])
    a = ap.parse_args()
    rc.init()

    fs = FermiSurface(kF=a.kF, vF=a.vF, M_theta=32, Nr=6, T=a.T, xi_max=6.0,
                      tau_p=np.inf, specularity=1.0,
                      cartesian=dict(annulus_xi=0.0, te_fac_max=6.0,
                                     kD_max=1.2e-3, dmu_max=1.2e-4,
                                     k_max=a.k_max, n_k=a.n_k),
                      process_grid=ProcessGrid("rk", (1, 1)))
    rep = fs.representation
    k = rep.k
    m_s, mu, T = float(rep.m_star), float(rep.mu), float(rep.T_temp)
    f0 = rep._f0_lab
    xi = (k.square().sum(-1) / (2 * m_s) - mu) / T
    th = torch.atan2(k[:, 1], k[:, 0])
    Ndof = a.P * (2 * a.M + 1)
    print(f"  n_k={a.n_k}  Nk={k.shape[0]}  basis P={a.P} x (2M+1)={2*a.M+1}"
          f"  -> Ndof={Ndof}  (oversampling {k.shape[0]/Ndof:.1f}x)")

    B = basis(xi, th, a.P, a.M, a.s, a.map)                 # (Nk, Ndof)
    # ⛔ QR, NOT THE NORMAL EQUATIONS.  Cholesky on B^T B squares the condition
    # number, and at P=96 that is the difference between 1.9e-10 and a blown-up
    # 5.3e-01: the s=2, P=96 arm produced garbage purely from conditioning
    # while s=4, P=96 survived.  A reduced QR of B keeps kappa(B) itself.
    # ⛔ QR ALONE OVERFITS.  The fit reproduces the SAMPLE points to 1.3e-06
    # absolute but the REFLECTED evaluation is 3.2e-05 -- 25x worse -- because
    # the least-squares coefficients are slightly too large and only agree
    # between samples to that accuracy.  The region breakdown makes it
    # unambiguous: the residual is confined to xi in 20..40, where the state
    # amplitude is 5.6e-02, and neither more radial modes (P=256 made it worse)
    # nor more angular modes (M=64 -> 1.7e-03, M=80 -> 8.7e-02) help.  It is a
    # conditioning problem, so truncate the small singular values.
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    keep = S > a.rcond * float(S[0])
    print(f"  singular values: kept {int(keep.sum())} of {S.shape[0]}"
          f"  (cond kept = {float(S[0] / S[keep][-1]):.3e})")
    Ur, Sr, Vr = U[:, keep], S[keep], Vh[keep]
    kD = torch.tensor([a.drift * a.kF, 0.0], dtype=k.dtype, device=k.device)

    print(f"    {'Te/T':>5} {'wall':>7} {'rel err (inflow)':>18}"
          f" {'min f':>13} {'max f-1':>13} {'flux resid':>12}")
    for te in a.te:
        u_in = torch.special.expit(
            -(((k - kD) ** 2).sum(-1) / (2 * m_s) - mu) / (T * te)) - f0
        coef = Vr.T @ ((Ur.T @ u_in) / Sr)
        for deg in a.angles:
            phi = np.deg2rad(deg)
            n = torch.tensor([np.cos(phi), np.sin(phi)], dtype=k.dtype,
                             device=k.device)
            kn = (k * n).sum(-1, keepdim=True)
            kstar = k - 2.0 * kn * n[None, :]
            # |k*| = |k| exactly, so xi is UNCHANGED: only theta moves
            th_s = torch.atan2(kstar[:, 1], kstar[:, 0])
            Bs = basis(xi, th_s, a.P, a.M, a.s, a.map)
            out = Bs @ coef
            exact = torch.special.expit(
                -(((kstar - kD) ** 2).sum(-1) / (2 * m_s) - mu) / (T * te)) - f0
            vn = (k * n).sum(-1)
            inflow = vn < 0
            rel = float((out - exact)[inflow].abs().max()
                        / exact[inflow].abs().max().clamp(min=1e-300))
            f_out = (f0 + out)[inflow]
            vv = vn / m_s
            fo = float((vv.clamp(min=0) * u_in).sum())
            fi = float((vv.clamp(max=0) * out).sum())
            print(f"    {te:>5.2f} {deg:>7.2f} {rel:>18.3e}"
                  f" {float(f_out.min()):>13.3e} {float(f_out.max())-1:>13.3e}"
                  f" {abs(fo+fi)/max(abs(fo),1e-300):>12.3e}", flush=True)
        print()


if __name__ == "__main__":
    main()
