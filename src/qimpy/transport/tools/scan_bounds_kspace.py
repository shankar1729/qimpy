"""WHERE in k-space does f leave [0, 1]?  Is it k-space truncation?

Reads a checkpoint's `/geometry/rho`, rebuilds f = f0_lab + rho on the Cartesian
k-grid, and locates every violating entry in (|k|, xi, angle, cell).

THE HYPOTHESES IT SEPARATES:

  outer truncation   violations pile up at |k| -> k_max (the grid edge)
  band bottom        violations pile up at |k| -> 0     (eps -> 0)
  mask edge          violations pile up at |xi_local| = xi_max (the collision
                     operator's active-set cut)
  RESOLUTION         violations sit at the Fermi surface, |xi_lab| ~ few, far
                     from every edge -- the grid simply cannot resolve the step

⛔ THE GRID IS NOT TRUNCATED IN THE OBVIOUS SENSE.  For the production numbers
(kF 7.5e-3, vF 0.11194, k_max 0.01326) the box spans xi from -31.6 (k = 0, the
band bottom) to +67 at k_max, i.e. the WHOLE band.  And the collision mask
`xi.abs() < xi_max` uses xi = (eps' - mu_loc)/Te_loc -- normalised by the LOCAL
electron temperature -- so it keeps tanh(xi_max/2) = 99.5% of the local thermal
weight at ANY Te.  Neither is a truncation of occupied states.

★ WHAT IS MARGINAL IS RESOLUTION.  d(eps)/T per grid point at the Fermi surface
is 3.98 / 1.99 / 1.00 / 0.50 for n_k = 56 / 112 / 224 / 448, so the thermal
width Te = 5.69 T is spanned by only 1.4 / 2.9 / 5.7 / 11.4 points.

⛔ NO k-GRID METADATA IS STORED IN CHECKPOINTS (`/material/representation` holds
only `variant_name`).  k_max and n_k MUST be passed in from the run's cfg or the
grid loads silently wrong.
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface

torch.set_default_dtype(torch.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--k-max", type=float, required=True)
    ap.add_argument("--n-k", type=int, required=True)
    ap.add_argument("--kF", type=float, default=7.5e-3)
    ap.add_argument("--vF", type=float, default=0.11194)
    ap.add_argument("--T", type=float, default=1.3301e-5)
    ap.add_argument("--M-theta", type=int, default=32)
    ap.add_argument("--Nr", type=int, default=6)
    ap.add_argument("--xi-max", type=float, default=6.0)
    a = ap.parse_args()
    rc.init()

    fs = FermiSurface(
        kF=a.kF, vF=a.vF, M_theta=a.M_theta, Nr=a.Nr, T=a.T, xi_max=a.xi_max,
        tau_p=float("inf"), specularity=1.0, residual_damping=True,
        cartesian=dict(annulus_xi=0.0, te_fac_max=6.0, kD_max=1.2e-3,
                       dmu_max=1.2e-4, k_max=a.k_max, n_k=a.n_k),
        process_grid=ProcessGrid("rk", (1, 1)))
    rep = fs.representation

    with h5py.File(a.h5, "r") as fh:
        rho = np.asarray(fh["/geometry/rho"])
        cen = np.asarray(fh["/geometry/cell_centroid"])
        step = int(np.asarray(fh["/time_evolution"].attrs["i_step"])) \
            if "i_step" in fh["/time_evolution"].attrs else -1
    assert rho.shape[1] == int(rep.Nk), (
        f"⛔ Nk MISMATCH: checkpoint has {rho.shape[1]}, this k-grid has "
        f"{int(rep.Nk)} -- k_max/n_k are wrong, the load would be silent garbage")

    f0 = rep._f0_lab.detach().cpu().numpy()[None, :]
    f = f0 + rho
    k = rep.k.detach().cpu().numpy()
    kmag = np.linalg.norm(k, axis=-1)
    eps = kmag ** 2 / (2.0 * float(rep.m_star))
    xi_lab = (eps - float(rep.mu)) / float(rep.T_temp)

    lo, hi = f < 0.0, f > 1.0
    n_lo, n_hi = int(lo.sum()), int(hi.sum())
    print(f"  checkpoint step {step},  cells {f.shape[0]},  Nk {f.shape[1]}"
          f"  ({f.size:,} entries)")
    print(f"  min f = {f.min():.6e}   max f - 1 = {f.max() - 1.0:+.6e}")
    print(f"  f < 0 : {n_lo:,} ({100.0 * n_lo / f.size:.4f}%)"
          f"   f > 1 : {n_hi:,} ({100.0 * n_hi / f.size:.4f}%)")
    if n_lo == 0 and n_hi == 0:
        print("  no violations")
        return

    bad = lo | hi
    kb = kmag[np.where(bad)[1]]
    xb = xi_lab[np.where(bad)[1]]
    print(f"\n  {'quantity':<26}{'violating pts':>16}{'whole grid':>16}")
    for name, vb, va in (("|k| / kF", kb / a.kF, kmag / a.kF),
                         ("xi_lab = (eps-mu)/T", xb, xi_lab)):
        print(f"  {name:<26}"
              f"{f'{vb.min():.2f} .. {vb.max():.2f}':>16}"
              f"{f'{va.min():.2f} .. {va.max():.2f}':>16}")

    # --- edge tests -------------------------------------------------------
    dk = 2.0 * a.k_max / a.n_k
    k_out = kmag.max()
    print(f"\n  outermost active |k| = {k_out / a.kF:.4f} kF;  dk = {dk:.3e}")
    for label, sel in (("within 1 dk of the OUTER edge", kmag > k_out - dk),
                       ("within 1 dk of k = 0 (band bottom)", kmag < dk),
                       ("|xi_lab| < 12 (the Fermi shell)", np.abs(xi_lab) < 12.0)):
        share_grid = 100.0 * sel.mean()
        nb = int(bad[:, sel].sum())
        share_bad = 100.0 * nb / max(n_lo + n_hi, 1)
        print(f"  {label:<38} {share_bad:6.2f}% of violations "
              f"vs {share_grid:6.2f}% of the grid")

    # --- radial profile ---------------------------------------------------
    print(f"\n  violations by |k|/kF decile of the ACTIVE grid:")
    qs = np.quantile(kmag, np.linspace(0, 1, 11))
    for i in range(10):
        sel = (kmag >= qs[i]) & (kmag <= qs[i + 1])
        nb = int(bad[:, sel].sum())
        print(f"    {qs[i]/a.kF:5.2f}-{qs[i+1]/a.kF:5.2f} kF  "
              f"xi {(qs[i]**2/(2*float(rep.m_star)) - float(rep.mu))/float(rep.T_temp):+7.1f}"
              f" .. {(qs[i+1]**2/(2*float(rep.m_star)) - float(rep.mu))/float(rep.T_temp):+7.1f}"
              f"   {nb:>9,}  {100.0*nb/max(n_lo+n_hi,1):5.1f}%")

    # --- spatial ----------------------------------------------------------
    per_cell = bad.sum(axis=1)
    nz = np.where(per_cell > 0)[0]
    r = np.linalg.norm(cen[nz] - cen.mean(0), axis=1) if len(nz) else np.array([])
    print(f"\n  cells with any violation: {len(nz)} of {f.shape[0]}"
          f"   worst cell holds {per_cell.max():,} entries")
    if len(nz):
        rall = np.linalg.norm(cen - cen.mean(0), axis=1)
        print(f"  their distance from the device centre: "
              f"{r.min():.2f} .. {r.max():.2f}  (all cells: "
              f"{rall.min():.2f} .. {rall.max():.2f})")


if __name__ == "__main__":
    main()
