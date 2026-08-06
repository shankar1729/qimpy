"""Two exact properties of eq (1) that no quadrature choice can excuse.

(a) DETAILED BALANCE.  The drifted, heated Fermi-Dirac

        f_le = sigma( -theta ( x - dmu - U sqrt(1 + t x) cos phi ) ),
        theta = T / T_e,  U = u kF / T

    is annihilated EXACTLY by the full nonlinear collision integral, for any
    (u, T_e, mu).  This is a nonlinear null state -- the LINEARIZED operator
    only annihilates the four collision invariants -- so it can only be tested
    against `a_dot`, never against `L_coeff`.

(b) H-THEOREM.  Entropy production
        Sdot = - Int fdot * ln( f / (1 - f) ) d^2k  >=  0
    for ANY f, with equality iff f is of the form above.

Both are checked on the reference evaluators (no truncation at all) and on the
production operator (where the residual is a truncation statement and must fall
with Nr, since Phi_le contains sqrt(1 + t x) which the feature basis represents
only approximately).

Env: QM QNR QTHETA QDMU QU QNX QNAZ QNXI QNPHI QTAG
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv2_common as qv


def main():
    M = int(os.environ.get("QM", "6"))
    Nr = int(os.environ.get("QNR", "3"))
    theta = float(os.environ.get("QTHETA", "0.769"))   # T / Te  (Te = 1.3 T)
    dmu = float(os.environ.get("QDMU", "0.5"))
    U = float(os.environ.get("QU", "0.30"))
    nx = int(os.environ.get("QNX", "21"))
    n_az = int(os.environ.get("QNAZ", "24"))
    n_xi = int(os.environ.get("QNXI", "48"))
    n_phi = int(os.environ.get("QNPHI", "1024"))
    xi_cut = 9.0
    tag = os.environ.get("QTAG", "eq")
    T = qv.T0
    t = T / qv.E_F

    torch.set_default_dtype(torch.float64)
    from qimpy import rc
    from qimpy.mpi import ProcessGrid
    from qimpy.transport.material.fermi_surface import FermiSurface
    from qimpy.transport.material.fermi_surface.scattering import _kernels

    f0 = lambda x: torch.sigmoid(-x)

    def f_le(x, p):
        return torch.sigmoid(
            -theta * (x - dmu - U * torch.sqrt(torch.clamp(1 + t * x, min=0))
                      * torch.cos(p)))

    df_le = lambda x, p: f_le(x, p) - f0(x)
    # a deliberately NON-equilibrium deviation of comparable size, to normalize
    df_neq = lambda x, p: 0.30 * qv.w_occ(x) / 0.25 * torch.cos(2 * p)

    xg, ph, Xf, Pf = qv.grids(nx, xi_cut, n_az)
    common = dict(kF=qv.KF, m_star=qv.M_STAR, T=T, epsilon_bg=qv.EPS_B,
                  kappa=qv.KAPPA, g_s=2.0)
    res = dict(tag=tag, theta=theta, dmu=dmu, U=U, M=M, Nr=Nr)
    print(f"peak |df_le| = {float(df_le(*torch.meshgrid(xg, ph, indexing='ij')).abs().max()):.4f}")

    # ---- (a) on the REFERENCE (no truncation, no basis) --------------------
    torch.set_default_device("cuda")
    Xc, Pc = Xf.cuda(), Pf.cuda()
    t0 = time.time()
    fdot_le = _kernels.exact_collision_reference(
        df_le, Xc, Pc, linearize=False, n_xi=n_xi, xi_cut=xi_cut,
        n_phi=n_phi, chunk=2, **common).cpu()
    fdot_neq = _kernels.exact_collision_reference(
        df_neq, Xc, Pc, linearize=False, n_xi=n_xi, xi_cut=xi_cut,
        n_phi=n_phi, chunk=2, **common).cpu()
    torch.set_default_device("cpu")
    ratio = float(fdot_le.abs().max() / fdot_neq.abs().max())
    res["ref_Cf_le_over_Cf_neq"] = ratio
    print(f"(a) reference  max|C[f_le]| / max|C[f_neq]| = {ratio:.3e}"
          f"   ({(time.time()-t0)/60:.1f} min)")

    # ---- (b) H-theorem on the reference -----------------------------------
    X, P = torch.meshgrid(xg, ph, indexing="ij")
    for name, dfun, fdot in (("f_le", df_le, fdot_le),
                             ("f_neq", df_neq, fdot_neq)):
        f = (f0(X) + dfun(X, P)).clamp(1e-14, 1 - 1e-14)
        s = -(fdot.reshape(nx, n_az) * torch.log(f / (1 - f)))
        Sdot = float(s.sum() * (float(xg[1] - xg[0])) * (2 * np.pi / n_az))
        res[f"Sdot_{name}"] = Sdot
        print(f"(b) reference  Sdot[{name}] = {Sdot:+.5e}"
              f"   {'OK (>=0)' if Sdot >= 0 else 'VIOLATION'}")

    # ---- (a) on PRODUCTION, vs truncation ---------------------------------
    fs = FermiSurface(kF=qv.KF, vF=qv.KF / qv.M_STAR, M_theta=M, Nr=Nr, T=T,
                      xi_max=6.0,
                      process_grid=ProcessGrid(rc.comm, "rk", (-1, 1)),
                      ee_scattering=dict(epsilon_bg=qv.EPS_B, kappa=qv.KAPPA,
                                         nonlinear=True,
                                         check_convergence=False))
    ee = fs.ee_scattering
    dim = fs.angular.dim
    psi_coeff = ee._radial_galerkin(T)[0].cpu()
    # project df_le / w_eq onto the basis (the operator's own input space)
    a_le = qv.project(df_le(X, P), xg, ph, psi_coeff, M, T)
    a_neq = qv.project(df_neq(X, P), xg, ph, psi_coeff, M, T)
    r_le = ee.a_dot(a_le.reshape(-1).to(rc.device)).cpu()
    r_neq = ee.a_dot(a_neq.reshape(-1).to(rc.device)).cpu()
    prod_ratio = float(r_le.abs().max() / r_neq.abs().max())
    res["prod_Cf_le_over_Cf_neq"] = prod_ratio
    res["repr_error"] = float(
        (qv.reconstruct(a_le, xg, ph, psi_coeff, M, T) - df_le(X, P)).abs().max()
        / df_le(X, P).abs().max())
    print(f"(a) production M={M} Nr={Nr}:  max|a_dot(a_le)| / max|a_dot(a_neq)|"
          f" = {prod_ratio:.3e}   (basis repr. error of f_le: "
          f"{res['repr_error']:.2e})")
    print(json.dumps(res), flush=True)
    json.dump(res, open(f"/tmp/v2equil_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
