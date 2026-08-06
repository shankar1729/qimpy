"""Production-side L, Q, C at CONTROLLED quadrature.

The even amplitude stencil removes the cubic exactly, so for a Q-only ladder the
cubic build can be skipped (QZC=1) -- that makes n_xi / n_phi / xi_cut cheap to
ladder.  QZC=0 keeps the cubic so L and C come out too.

Env knobs:
    QB    backend: dense | matrix_free | auto     (default dense)
    QNXI  n_xi     (0 = auto)
    QNPHI n_phi    (0 = auto)
    QXIC  xi_cut   (0 = auto = 10.0)
    QNXP  n_xi_proj (0 = auto = 96)
    QZC   1 = zero the cubic vertex (default 1)
    QTAG  output tag
"""
import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q_common as qc


def main():
    backend = os.environ.get("QB", "dense")
    n_xi = int(os.environ.get("QNXI", "0"))
    n_phi = int(os.environ.get("QNPHI", "0"))
    xi_cut = float(os.environ.get("QXIC", "0"))
    n_xi_proj = int(os.environ.get("QNXP", "0"))
    zero_cubic = os.environ.get("QZC", "1") == "1"
    tag = os.environ.get("QTAG", "prod")

    torch.set_default_dtype(torch.float64)
    from qimpy import rc
    from qimpy.mpi import ProcessGrid
    from qimpy.transport.material.fermi_surface import FermiSurface
    from qimpy.transport.material.fermi_surface.scattering import _kernels

    if zero_cubic:
        def _zero_cubic(*, ti, work_dtype=torch.complex128, work_device=None,
                        **kw):
            return torch.zeros(len(ti), dtype=work_dtype, device=work_device)
        _kernels.cubic_packed_node = _zero_cubic

    kw = dict(epsilon_bg=qc.EPS_B, kappa=qc.KAPPA, nonlinear=True,
              check_convergence=False, backend=backend)
    if n_xi:
        kw["n_xi"] = n_xi
    if n_phi:
        kw["n_phi"] = n_phi
    if xi_cut:
        kw["xi_cut"] = xi_cut
    if n_xi_proj:
        kw["n_xi_proj"] = n_xi_proj

    t0 = time.time()
    fs = FermiSurface(kF=qc.KF, vF=qc.KF / qc.M_STAR, M_theta=qc.M_THETA,
                      Nr=qc.NR, T=qc.T0, xi_max=6.0,
                      process_grid=ProcessGrid(rc.comm, "rk", (-1, 1)),
                      ee_scattering=kw)
    ee = fs.ee_scattering
    dim = fs.angular.dim
    build_min = (time.time() - t0) / 60
    psi_coeff = ee._radial_galerkin(qc.T0)[0].cpu()
    amp = qc.amplitude(psi_coeff)

    a = torch.zeros(qc.NR * dim, device=rc.device)
    a[qc.N_IN * dim + (2 * qc.M_IN - 1)] = amp
    a4 = a.reshape(1, qc.NR, dim)
    lin = (-torch.einsum("cij,bjc->bic", ee.L_coeff, a4)).reshape(-1).cpu()
    ad_p, ad_m = ee.a_dot(a).cpu(), ee.a_dot(-a).cpu()
    quad = 0.5 * (ad_p + ad_m)
    cub = 0.5 * (ad_p - ad_m) - lin
    nulls = ee._null_covectors(qc.T0)
    null0 = torch.stack([v.cpu() for v in nulls[0]], 1)

    out = dict(tag=tag, backend=ee.backend, n_xi=ee.n_xi, n_phi=ee.n_phi,
               xi_cut=ee.xi_cut, n_xi_proj=ee.n_xi_proj, build_min=build_min,
               amp=amp, dim=dim, zero_cubic=zero_cubic)
    np.savez(f"/tmp/qprod_{tag}.npz",
             psi_coeff=psi_coeff.numpy(), null0=null0.numpy(),
             L=lin.numpy(), Q=quad.numpy(), C=cub.numpy(),
             meta=json.dumps(out))
    ci = 2 * qc.M_IN - 1          # cos(2 phi) channel
    c4 = 2 * (2 * qc.M_IN) - 1    # cos(4 phi) channel
    print(json.dumps(out))
    print(f"  Q(0,4) = {float(quad[0*dim + c4]):+.6e}"
          f"   Q(1,4) = {float(quad[1*dim + c4]):+.6e}"
          f"   Q(2,4) = {float(quad[2*dim + c4]):+.6e}")
    print(f"  L(0,2) = {float(lin[0*dim + ci]):+.6e}"
          f"   C(0,2) = {float(cub[0*dim + ci]):+.6e}")


if __name__ == "__main__":
    main()
