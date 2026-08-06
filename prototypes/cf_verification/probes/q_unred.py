"""Ladder the REDUCTION-FREE reference's own systematics on Q.

The sigma-ladder held the number of x2 nodes per sigma FIXED (n_xi2 = 240, 320,
480, 640 at sigma = 0.40, 0.30, 0.20, 0.15 -- all exactly 0.188 sigma spacing).
Any error that lives on the OTHER axes is therefore sigma-INDEPENDENT by
construction, so the flat ratio does not discriminate what it was meant to.

The suspect axis is the ANGULAR one.  After the fine, uniform x2 quadrature
integrates the Gaussian across the shell, the surviving integrand in
(phi2, phi3, x3) carries a factor 1/|de/dx2| with

    de/dx2 = 1 - k4 cos(phi4 - phi2) / k2

which VANISHES on the near-collinear locus k2 || k4 -- the same van-Hove
caustic the reduced form shows as 1/|sin(phi4 - phi2)|.  The reduced kinematics
grade their azimuth grid for it (`_beta_grid`); this evaluator uses a PLAIN
UNIFORM grid of only n_phi = 112 points in BOTH phi2 and phi3.  Convergence
there is algebraic, not spectral, and independent of sigma.

Q from a cos(2 phi) input contains ONLY the harmonics {0, +-4} (the quadratic
vertex bins mo = ma + mb with ma, mb in {+-2}), so 5 azimuths give an EXACT
DFT (0 and 4 are distinct mod 5) -- half the cost of the previous run.

Env: QNPHI QNXI2 QNXI3 QXIC QSIG QNX QNAZ QCHUNK QDEV QTAG QMETA
"""
import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q_common as qc

M_OUT_Q = [0, 4]


def main():
    n_phi = int(os.environ.get("QNPHI", "112"))
    n_xi2 = int(os.environ.get("QNXI2", "320"))
    n_xi3 = int(os.environ.get("QNXI3", "20"))
    xi_cut = float(os.environ.get("QXIC", "9.0"))
    sigma = float(os.environ.get("QSIG", "0.30"))
    nx = int(os.environ.get("QNX", "13"))
    n_az = int(os.environ.get("QNAZ", "5"))
    chunk = int(os.environ.get("QCHUNK", "2"))
    dev = os.environ.get("QDEV", "cuda")
    tag = os.environ.get("QTAG", "u")
    meta = os.environ.get("QMETA", "/tmp/qprod_ref.npz")

    torch.set_default_dtype(torch.float64)
    from qimpy.transport.material.fermi_surface.scattering import _kernels

    z = np.load(meta)
    psi_coeff = torch.as_tensor(z["psi_coeff"]).to(torch.float64)
    null0 = torch.as_tensor(z["null0"]).to(torch.float64)
    amp = qc.amplitude(psi_coeff)

    xg, ph, Xf, Pf = qc.grids(nx, xi_cut, n_az)
    if dev == "cuda":
        torch.set_default_device("cuda")
        psi_coeff = psi_coeff.cuda()
        Xf, Pf = Xf.cuda(), Pf.cuda()
    df = qc.field(psi_coeff, amp)
    common = dict(kF=qc.KF, m_star=qc.M_STAR, T=qc.T0, epsilon_bg=qc.EPS_B,
                  kappa=qc.KAPPA, g_s=2.0)

    full = os.environ.get("QFULL", "0") == "1"   # all of L, Q, C (4 amplitudes)
    t0 = time.time()
    fd = {}
    for s in ((1.0, 2.0, -1.0, -2.0) if full else (1.0, -1.0)):
        fd[s] = _kernels.unreduced_collision_reference(
            (lambda s_: (lambda x, p: df(x, p) * s_))(s),
            Xf, Pf, linearize=False, sigma=sigma, n_xi2=n_xi2, n_xi3=n_xi3,
            n_phi=n_phi, xi_cut=xi_cut, x2chunk=chunk, **common).cpu()
        print(f"  s={s:+.0f} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    if dev == "cuda":
        torch.set_default_device("cpu")

    res = dict(tag=tag, n_phi=n_phi, n_xi2=n_xi2, n_xi3=n_xi3, xi_cut=xi_cut,
               sigma=sigma, nx=nx, n_az=n_az, full=full,
               min=(time.time() - t0) / 60)
    if full:
        raw = qc.orders_from_stencil(fd)
        m_out = qc.M_OUT
    else:
        raw = {"Q": 0.5 * (fd[1.0] + fd[-1.0])}
        m_out = M_OUT_Q
    save = {}
    for key, arr in raw.items():
        p = qc.project(arr, xg, ph, psi_coeff.cpu(), null0, m_out=m_out)
        save[key] = p.numpy()
        save["raw" + key] = arr.numpy()
        print(f"  {key}: " + "  ".join(
            f"({n},{m})={float(p[n, j]):+.5e}"
            for j, m in enumerate(m_out) for n in range(qc.NR)), flush=True)
    print(json.dumps(res), flush=True)
    np.savez(f"/tmp/qunred_{tag}.npz", meta=json.dumps(res), **save)


if __name__ == "__main__":
    main()
