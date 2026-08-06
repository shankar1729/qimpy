"""THIRD, independent evaluator: the exact REDUCED collision integral.

`exact_collision_reference` evaluates the raw bracket B - F pointwise on the
energy-shell-reduced kinematics.  It shares with production ONLY the kinematic
reduction (same two azimuthal roots, same `_beta_grid`); it shares NO vertex
algebra, NO modal assembly, NO radial Galerkin build, and it has its own,
independently controllable quadrature.

Three-way triage of Q(n=1, m=4):

    exact  ==  unreduced   ->  the reduction is sound; the defect is in the
                              production vertex assembly or its quadrature
    exact  ==  production  ->  the defect is in the reduction itself (or in a
                              quadrature the two of them share)

Env knobs:  QNXI (n_xi), QNPHI (n_phi), QXIC (xi_cut), QNX (radial grid points),
            QXG (radial grid half-width), QCHUNK, QDEV, QTAG, QMETA (npz path)
"""
import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q_common as qc


def main():
    n_xi = int(os.environ.get("QNXI", "48"))
    n_phi = int(os.environ.get("QNPHI", "1024"))
    xi_cut = float(os.environ.get("QXIC", "12.0"))
    nx = int(os.environ.get("QNX", "25"))
    xi_grid = float(os.environ.get("QXG", "9.0"))
    chunk = int(os.environ.get("QCHUNK", "4"))
    dev = os.environ.get("QDEV", "cuda")
    tag = os.environ.get("QTAG", "exact")
    meta = os.environ.get("QMETA", "/tmp/qprod_ref.npz")

    torch.set_default_dtype(torch.float64)
    from qimpy.transport.material.fermi_surface.scattering import _kernels

    z = np.load(meta)
    psi_coeff = torch.as_tensor(z["psi_coeff"]).to(torch.float64)
    null0 = torch.as_tensor(z["null0"]).to(torch.float64)
    amp = qc.amplitude(psi_coeff)
    dim = json.loads(str(z["meta"]))["dim"]

    xg, ph, Xf, Pf = qc.grids(nx, xi_grid)
    if dev == "cuda":
        torch.set_default_device("cuda")
        psi_coeff = psi_coeff.cuda()
        Xf, Pf = Xf.cuda(), Pf.cuda()
    df = qc.field(psi_coeff, amp)
    common = dict(kF=qc.KF, m_star=qc.M_STAR, T=qc.T0, epsilon_bg=qc.EPS_B,
                  kappa=qc.KAPPA, g_s=2.0)

    t0 = time.time()
    fd = {}
    for s in (1.0, 2.0, -1.0, -2.0):
        fd[s] = _kernels.exact_collision_reference(
            (lambda s_: (lambda x, p: df(x, p) * s_))(s),
            Xf, Pf, linearize=False, n_xi=n_xi, xi_cut=xi_cut, n_phi=n_phi,
            chunk=chunk, **common).cpu()
        print(f"  s={s:+.0f} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    if dev == "cuda":
        torch.set_default_device("cpu")

    raw = qc.orders_from_stencil(fd)
    res = dict(tag=tag, n_xi=n_xi, n_phi=n_phi, xi_cut=xi_cut, nx=nx,
               xi_grid=xi_grid, amp=amp, min=(time.time() - t0) / 60)
    print(json.dumps(res))
    save = {}
    for key in ("L", "Q", "C"):
        p = qc.project(raw[key], xg, ph, psi_coeff.cpu(),
                       null0 if key != "L" else null0)
        save[key] = p.numpy()
        print(f"  {key}: " + "  ".join(
            f"({n},{m})={float(p[n, j]):+.5e}"
            for j, m in enumerate(qc.M_OUT) for n in range(qc.NR)
            if abs(float(p[n, j])) > 1e-16))
    np.savez(f"/tmp/qexact_{tag}.npz", meta=json.dumps(res), **save,
             **{f"raw{k}": v.numpy() for k, v in raw.items()})


if __name__ == "__main__":
    main()
