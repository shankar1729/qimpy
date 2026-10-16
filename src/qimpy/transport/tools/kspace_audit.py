"""Full k-space audit of a checkpoint: every grid point, every cell.

Answers "what does f actually look like in k, everywhere" rather than only
"where is it out of bounds".  Produces reduced arrays + figures.

WHAT IT MEASURES

  radial profile     f, delta-f and the violation rate binned by |k|
  RINGING            the grid-scale oscillation metric.  If the k-grid cannot
                     resolve the Fermi step, delta-f oscillates at the Nyquist
                     wavelength 2*dk, and the radial second difference
                     D2 = df(k+dk) - 2 df(k) + df(k-dk) approaches -4 df.
                     R = |D2| / (4 |df|) -> 1 is saturated grid-scale ringing,
                     R << 1 is a smooth, resolved field.  This is the direct
                     fingerprint of the under-resolution the n_k sweep implied.
  angular content    |a_m|(|k|): which harmonics live at which radius
  local frame        per-cell drift kD, electron temperature Te, chemical
                     potential mu recovered from f itself

⛔ NOT PLOTTED WITH qimpy.transport.plot, DELIBERATELY.  That module renders
DEVICE fields -- the RT0 reconstruction of per-edge fluxes on the mesh -- and
has no k-space path at all (`grep -n "def " plot.py` is all fv_*/rt0_* spatial
helpers).  The standing "always plot through plot.py" rule is about device
fields, and any spatial map here must still go through it.

⛔ CHECKPOINTS STORE NO k-GRID METADATA.  k_max and n_k must come from the run's
cfg (`/material/representation` holds only `variant_name`).  The Nk assert below
is the only thing standing between a typo and silently analysing a wrong grid.
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface

torch.set_default_dtype(torch.float64)


def build(a) -> FermiSurface:
    return FermiSurface(
        kF=a.kF, vF=a.vF, M_theta=a.M_theta, Nr=a.Nr, T=a.T, xi_max=a.xi_max,
        tau_p=float("inf"), specularity=1.0, residual_damping=True,
        cartesian=dict(annulus_xi=0.0, te_fac_max=6.0, kD_max=1.2e-3,
                       dmu_max=1.2e-4, k_max=a.k_max, n_k=a.n_k),
        process_grid=ProcessGrid("rk", (1, 1)))


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
    ap.add_argument("--out", default="kaudit")
    a = ap.parse_args()
    rc.init()

    fs = build(a)
    rep = fs.representation
    with h5py.File(a.h5, "r") as fh:
        rho = torch.as_tensor(np.asarray(fh["/geometry/rho"]), device=rc.device)
        cen = np.asarray(fh["/geometry/cell_centroid"])
    assert rho.shape[1] == int(rep.Nk), (
        f"⛔ Nk MISMATCH {rho.shape[1]} vs {int(rep.Nk)}: wrong k_max/n_k")

    f0 = rep._f0_lab
    f = f0[None, :] + rho
    k = rep.k
    kmag = k.norm(dim=-1)
    th = torch.atan2(k[:, 1], k[:, 0])
    eps = kmag.square() / (2.0 * float(rep.m_star))
    xi = (eps - float(rep.mu)) / float(rep.T_temp)
    K, Nk = f.shape
    dk = 2.0 * a.k_max / a.n_k
    print(f"  {K} cells x {Nk} k-points; dk = {dk:.4e}; "
          f"d(eps)/T at kF = {(a.kF/float(rep.m_star))*dk/a.T:.2f}")

    # ---- local frame per cell -------------------------------------------
    kD, Te, mu = rep._recover_frame(f)
    teT = (Te / float(rep.T_temp)).cpu().numpy()
    print(f"  local Te/T : {teT.min():.3f} .. {teT.max():.3f}  "
          f"(mean {teT.mean():.3f})")
    print(f"  |kD|/kF    : {(kD.norm(dim=-1)/a.kF).min():.3e} .. "
          f"{(kD.norm(dim=-1)/a.kF).max():.3e}")

    # ---- radial profile --------------------------------------------------
    nb = 80
    edges = torch.linspace(0.0, float(kmag.max()) * 1.0001, nb + 1,
                           device=rc.device)
    idx = torch.bucketize(kmag, edges) - 1
    idx = idx.clamp(0, nb - 1)
    prof = []
    for b in range(nb):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        fb = f[:, sel]
        db = rho[:, sel]
        prof.append(dict(
            kb=float(kmag[sel].mean() / a.kF), xib=float(xi[sel].mean()),
            npts=n,
            f_mean=float(fb.mean()), f_min=float(fb.min()), f_max=float(fb.max()),
            df_absmax=float(db.abs().max()), df_rms=float(db.square().mean().sqrt()),
            frac_below=float((fb < 0).float().mean()),
            frac_above=float((fb > 1).float().mean())))

    # ---- RINGING: radial second difference on the square grid ------------
    # scatter the active points back onto the n_k x n_k image, per cell
    ig = ((k[:, 0] + a.k_max) / dk - 0.5).round().long().clamp(0, a.n_k - 1)
    jg = ((k[:, 1] + a.k_max) / dk - 0.5).round().long().clamp(0, a.n_k - 1)
    img = torch.zeros(K, a.n_k, a.n_k, device=rc.device, dtype=rho.dtype)
    occ = torch.zeros(a.n_k, a.n_k, device=rc.device, dtype=torch.bool)
    img[:, ig, jg] = rho
    occ[ig, jg] = True
    # 1-D second difference along x, valid only where all three cells are active
    d2 = img[:, 2:, :] - 2.0 * img[:, 1:-1, :] + img[:, :-2, :]
    ok = occ[2:, :] & occ[1:-1, :] & occ[:-2, :]
    ctr = img[:, 1:-1, :]
    denom = 4.0 * ctr.abs().clamp(min=1e-300)
    R = (d2.abs() / denom)
    # per-radius ringing, restricted to the Fermi shell and to the tails
    kx = (torch.arange(a.n_k, device=rc.device) + 0.5) * dk - a.k_max
    KX, KY = torch.meshgrid(kx, kx, indexing="ij")
    KM = (KX.square() + KY.square()).sqrt()[1:-1, :]
    XI = (KM.square() / (2 * float(rep.m_star)) - float(rep.mu)) / float(rep.T_temp)
    ring = {}
    for name, sel in (("Fermi shell |xi|<12", XI.abs() < 12),
                      ("deep sea xi<-12", XI < -12),
                      ("empty tail xi>+12", XI > 12)):
        m = sel[None].expand_as(R) & ok[None].expand_as(R)
        if int(m.sum()) == 0:
            continue
        vals = R[m]
        # ⛔ torch.quantile refuses inputs above 2**24 elements ("input tensor is
        # too large") and this selection is ~1e8.  Compute the median and p90
        # from a fixed random subsample, and take the fractions on the FULL set
        # (a mean over a bool needs no sort and has no size limit).
        n_full = int(vals.numel())
        if n_full > 1 << 22:
            g = torch.Generator(device=vals.device).manual_seed(0)
            pick = torch.randint(n_full, (1 << 22,), device=vals.device,
                                 generator=g)
            sub = vals[pick]
        else:
            sub = vals
        ring[name] = dict(median=float(sub.median()),
                          p90=float(sub.quantile(0.9)),
                          frac_gt_0p5=float((vals > 0.5).float().mean()),
                          frac_gt_0p9=float((vals > 0.9).float().mean()),
                          n=n_full, n_sub=int(sub.numel()))
    print("\n  RINGING  R = |d2 df| / (4|df|)   (1 = saturated grid-scale "
          "oscillation, <<1 = resolved)")
    for nm, v in ring.items():
        print(f"    {nm:<22} median {v['median']:.3f}  p90 {v['p90']:.3f}"
              f"  frac(R>0.5) {100*v['frac_gt_0p5']:5.1f}%"
              f"  frac(R>0.9) {100*v['frac_gt_0p9']:5.1f}%  n={v['n']:,}")

    # ---- angular content vs radius --------------------------------------
    print("\n  angular harmonics |a_m| vs |k| (cell-averaged, normalised to m=0)")
    print(f"    {'|k|/kF':>8} {'xi':>8} " + " ".join(f"m={m}" for m in range(5)))
    dfm = rho.mean(0)
    for b in range(0, nb, max(1, nb // 12)):
        sel = idx == b
        if int(sel.sum()) < 8:
            continue
        t = th[sel]
        v = dfm[sel]
        amps = [float((v * torch.cos(m * t)).sum().abs()) for m in range(5)]
        a0 = max(amps[0], 1e-300)
        print(f"    {float(kmag[sel].mean()/a.kF):>8.3f} "
              f"{float(xi[sel].mean()):>8.1f} "
              + " ".join(f"{x/a0:7.3f}" for x in amps))

    np.savez(f"{a.out}.npz",
             prof_k=np.array([p["kb"] for p in prof]),
             prof_xi=np.array([p["xib"] for p in prof]),
             prof_fmin=np.array([p["f_min"] for p in prof]),
             prof_fmax=np.array([p["f_max"] for p in prof]),
             prof_fmean=np.array([p["f_mean"] for p in prof]),
             prof_dfabsmax=np.array([p["df_absmax"] for p in prof]),
             prof_dfrms=np.array([p["df_rms"] for p in prof]),
             prof_below=np.array([p["frac_below"] for p in prof]),
             prof_above=np.array([p["frac_above"] for p in prof]),
             teT=teT, cen=cen)
    json.dump(dict(ring=ring, prof=prof), open(f"{a.out}.json", "w"), indent=1)
    print(f"\n  wrote {a.out}.npz / {a.out}.json")

    # ---- worst-cell image for plotting -----------------------------------
    worst = int((f < 0).sum(dim=1).argmax())
    np.savez(f"{a.out}_cell.npz", img=img[worst].cpu().numpy(),
             occ=occ.cpu().numpy(), k_max=a.k_max, n_k=a.n_k, cell=worst,
             f_img=(f0[None, :] + rho)[worst].cpu().numpy(),
             kmag=kmag.cpu().numpy(), xi=xi.cpu().numpy(),
             f_cell=f[worst].cpu().numpy())
    print(f"  worst cell {worst}: {int((f[worst]<0).sum()):,} points f<0")


if __name__ == "__main__":
    main()
