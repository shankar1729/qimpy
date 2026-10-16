"""Per-CELL k-space audit: is the bound violation uniform in space, or not?

The radial audit aggregated over all 1792 cells at once and so could not say
whether the violation is spatially uniform, driven by the local electron
temperature, or concentrated at the boundary.  This resolves it per cell and
writes the per-cell scalars back into a checkpoint so the maps can be rendered
by qimpy's own `transport.plot` rather than by an ad-hoc plotter.

⛔ THE SPATIAL MAPS MUST GO THROUGH qimpy.transport.plot.  It flat-shades
cell-centred scalars as tripcolor -- the honest piecewise-constant FV picture --
and knows the mesh, the contacts and the RT0 face field.  Only the k-space
figures are matplotlib, because plot.py has no k-space path at all.

⛔ WRITE INTO A COPY.  Never modify the run's own checkpoint.
"""
from __future__ import annotations

import argparse
import shutil

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
    ap.add_argument("--mesh", default=None, help="npz, to classify boundary cells")
    ap.add_argument("--out", default="percell")
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
        rho = torch.as_tensor(np.asarray(fh["/geometry/rho"]), device=rc.device)
        cen = np.asarray(fh["/geometry/cell_centroid"])
        tris = np.asarray(fh["/geometry/mesh_triangles"])
    assert rho.shape[1] == int(rep.Nk), "⛔ Nk mismatch: wrong k_max/n_k"

    f0 = rep._f0_lab
    f = f0[None, :] + rho
    K = f.shape[0]
    k = rep.k
    kmag = k.norm(dim=-1)
    xi = (kmag.square() / (2.0 * float(rep.m_star)) - float(rep.mu)) \
        / float(rep.T_temp)
    shell = xi.abs() < 12.0

    kD, Te, mu = rep._recover_frame(f)
    teT = (Te / float(rep.T_temp)).cpu().numpy()
    kDn = (kD.norm(dim=-1) / a.kF).cpu().numpy()

    below = (f < 0.0)
    above = (f > 1.0)
    per = dict(
        n_below=below.sum(1).cpu().numpy().astype(float),
        n_above=above.sum(1).cpu().numpy().astype(float),
        min_f=f.min(1).values.cpu().numpy(),
        over_f=(f.max(1).values - 1.0).cpu().numpy(),
        min_f_shell=f[:, shell].min(1).values.cpu().numpy(),
        over_f_shell=(f[:, shell].max(1).values - 1.0).cpu().numpy(),
        df_absmax=rho.abs().max(1).values.cpu().numpy(),
        df_rms=rho.square().mean(1).sqrt().cpu().numpy(),
        teT=teT, kD=kDn)

    print(f"  {K} cells")
    print(f"  {'quantity':<16}{'min':>13}{'median':>13}{'max':>13}"
          f"{'cells at 0':>12}")
    for nm in ("n_below", "n_above", "min_f", "over_f", "min_f_shell",
               "over_f_shell", "df_absmax", "teT", "kD"):
        v = per[nm]
        z = int((v == 0).sum())
        print(f"  {nm:<16}{v.min():>13.4e}{np.median(v):>13.4e}"
              f"{v.max():>13.4e}{z:>12d}")

    # --- is the violation uniform, or does it track something? -----------
    print("\n  Spearman correlation of per-cell violation with:")
    def spear(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    rcen = np.linalg.norm(cen - cen.mean(0), axis=1)
    drivers = dict(local_Te=per["teT"], drift_kD=per["kD"],
                   df_absmax=per["df_absmax"], df_rms=per["df_rms"],
                   dist_from_centre=rcen)
    for nm, v in drivers.items():
        print(f"    {nm:<18} vs n_below {spear(per['n_below'], v):+.3f}"
              f"   vs n_above {spear(per['n_above'], v):+.3f}"
              f"   vs |min f| {spear(-per['min_f'], v):+.3f}")

    # --- boundary classification ------------------------------------------
    if a.mesh:
        d = np.load(a.mesh, allow_pickle=True)
        bm = np.array([str(x) for x in d["boundary_markers"]])
        be = np.asarray(d["boundary_edges"])
        vw = set(be[bm == "wall"].ravel().tolist())
        vc = set(be[bm != "wall"].ravel().tolist())
        tw = np.array([bool(set(t.tolist()) & vw) for t in tris])
        tc = np.array([bool(set(t.tolist()) & vc) for t in tris])
        print(f"\n  {'cell class':<22}{'n':>7}{'mean n_below':>14}"
              f"{'mean min f':>13}{'mean Te/T':>11}")
        for nm, sel in (("wall + contact", tw & tc), ("wall only", tw & ~tc),
                        ("contact only", ~tw & tc), ("interior", ~tw & ~tc)):
            if not sel.any():
                continue
            print(f"  {nm:<22}{int(sel.sum()):>7}"
                  f"{per['n_below'][sel].mean():>14.1f}"
                  f"{per['min_f'][sel].mean():>13.3e}"
                  f"{per['teT'][sel].mean():>11.3f}")

    np.savez(f"{a.out}.npz", cen=cen, **per)

    # --- write the scalars into a checkpoint copy for transport.plot ------
    dst = f"{a.out}_fields.h5"
    shutil.copyfile(a.h5, dst)
    names = ["min_f", "over_f", "n_below", "n_above", "teT", "df_absmax"]
    with h5py.File(dst, "r+") as fh:
        g = fh["/geometry"]
        obs = np.zeros((1, K, len(names)))
        for i, nm in enumerate(names):
            obs[0, :, i] = per[nm]
        del g["fv_observables"]
        g.create_dataset("fv_observables", data=obs)
        if "observable_names" in g:
            del g["observable_names"]
        g.create_dataset("observable_names",
                         data=np.bytes_("|".join(names)))
    print(f"\n  wrote {a.out}.npz and {dst} "
          f"(observables: {', '.join(names)}) -- render with transport.plot")


if __name__ == "__main__":
    main()
