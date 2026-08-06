"""WHY does the operator's in-span error grow with Nr?

Measured: coef = ||a_prod - proj(ref)|| / ||proj(ref)|| = 0.031, 0.032, 0.094,
0.176 at Nr = 2, 3, 4, 6, while the FIELD error keeps falling and prod stays
below the projection's own error at every rung.  A quantity that grows while
the thing it is supposed to measure shrinks is a red flag for the METRIC, not
the operator.

Three candidates, and this script separates them:

  (a) CONDITIONING.  A modal coefficient is obtained from a field by
      c = G^-1 <psi, .>, so a fixed field error is amplified into the
      coefficients by up to cond(G).  If cond(G) grows with Nr, the coefficient
      comparison inflates by construction and the field metric does not.
      Printed here for BOTH Grams: production's own 96-node Gauss rule from
      `_radial_galerkin`, and the harness's trapezoid rule.

  (b) OPERATOR QUADRATURE.  The auto rule is n_xi = ceil(max(16, 4 Nr) s), so
      n_xi = 16 all the way to Nr = 4 and only then starts moving.  The vertex
      energy integrals involve products of psi_l, which get more structured with
      l, so 16 Gauss nodes may simply stop resolving them.  Laddered here.

  (c) genuine loss of accuracy in the higher radial modes.

Discriminator: if (a), the SPECTRALLY RESTRICTED coefficient error -- computed
only on the directions where the Gram is well conditioned -- stays flat with Nr
while the full one grows.  If (b), the full one falls when n_xi is raised at
fixed Nr.  If neither, it is (c).

Env: QNRS QNXIS QM QNXP QTAG
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv2_common as qv

FEATS = [[0, 1, "cos", 1.0], [1, 2, "cos", 0.6], [0, 3, "sin", 0.8]]


def main():
    M = int(os.environ.get("QM", "6"))
    nrs = json.loads(os.environ.get("QNRS", "[2,3,4,6,8]"))
    nxis = json.loads(os.environ.get("QNXIS", "[0]"))    # 0 = auto
    n_xi_proj = int(os.environ.get("QNXP", "0"))
    tag = os.environ.get("QTAG", "cond")
    T = qv.T0
    xi_cut = 9.0

    torch.set_default_dtype(torch.float64)
    from qimpy import rc
    from qimpy.mpi import ProcessGrid
    from qimpy.transport.material.fermi_surface import FermiSurface

    feats = qv.scale_features([(int(p), int(m), str(k), float(a))
                               for p, m, k, a in FEATS], 0.30, xi_cut, T)
    out = []
    for Nr in nrs:
        for n_xi in nxis:
            kw = dict(epsilon_bg=qv.EPS_B, kappa=qv.KAPPA, nonlinear=False,
                      check_convergence=False, xi_cut=xi_cut)
            if n_xi:
                kw["n_xi"] = n_xi
            if n_xi_proj:
                kw["n_xi_proj"] = n_xi_proj
            fs = FermiSurface(kF=qv.KF, vF=qv.KF / qv.M_STAR, M_theta=M, Nr=Nr,
                              T=T, xi_max=6.0,
                              process_grid=ProcessGrid(rc.comm, "rk", (-1, 1)),
                              ee_scattering=kw)
            ee = fs.ee_scattering
            dim = fs.angular.dim
            psi_coeff, x_fine, P, Ginv, _ = ee._radial_galerkin(T)
            # production's own Gram (the one its projection actually inverts)
            G_prod = torch.linalg.inv(Ginv.cpu())
            cond_prod = float(torch.linalg.cond(G_prod))
            # the harness's trapezoid Gram, on the SAME grid the analyzer uses
            xg = torch.tensor(np.linspace(-xi_cut, xi_cut, 21),
                              dtype=torch.float64)
            psi_g = qv.psi_eval(xg, psi_coeff.cpu())
            G_harn = torch.einsum("xn,xm,x->nm", psi_g, psi_g,
                                  qv.w_occ(xg)) * float(xg[1] - xg[0])
            cond_harn = float(torch.linalg.cond(G_harn))
            # linear-block spectra: the near-null directions of L are where a
            # coefficient comparison is ill-posed independently of the Gram
            spec = {}
            for m in (2, 3, 4):
                c = 2 * m - 1
                if c < dim:
                    A = ee.L_coeff[c].cpu()
                    ev = torch.linalg.eigvalsh(0.5 * (A + A.T))
                    spec[m] = (float(ev.max()), float(ev.abs().min()),
                               float(ev.max() / ev.abs().min().clamp(min=1e-300)))
            a = qv.modes_from_features(feats, psi_coeff.cpu(), dim).to(rc.device)
            a4 = a.reshape(1, Nr, dim)
            L = (-torch.einsum("cij,bjc->bic", ee.L_coeff, a4)).reshape(-1).cpu()
            rec = dict(Nr=Nr, n_xi=ee.n_xi, n_phi=ee.n_phi,
                       n_xi_proj=ee.n_xi_proj, dim=dim,
                       cond_G_prod=cond_prod, cond_G_harness=cond_harn,
                       spec=spec)
            out.append(rec)
            np.savez(f"/tmp/v3lin_{tag}_Nr{Nr}_nxi{ee.n_xi}.npz",
                     meta=json.dumps(rec), psi_coeff=psi_coeff.cpu().numpy(),
                     L=L.numpy(), a=a.cpu().numpy())
            print(f"Nr={Nr:<2} n_xi={ee.n_xi:<3} n_xi_proj={ee.n_xi_proj:<4}"
                  f" cond(G_prod)={cond_prod:9.3e}"
                  f" cond(G_harness)={cond_harn:9.3e}"
                  + "".join(f"   L{m}: max/min={spec[m][2]:.2e}" for m in spec),
                  flush=True)
    json.dump(out, open(f"/tmp/v3cond_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
