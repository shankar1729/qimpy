"""Production side, v2: arbitrary multi-mode input, any (M, Nr), any quadrature.

Dumps the order-resolved modal output AND enough metadata to reconstruct the
fdot field, so the same run serves the packing test, the truncation ladder and
the conservation check.

Env:  QM QNR QNXI QNPHI QXIC QNXP QB(backend) QTAG QFEATS(json) QPEAK QT
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv2_common as qv

# default input: three modes, mixed radial features AND a sin harmonic, chosen
# so the nonlinear packing sees off-diagonal pairs (m1 != m2) and triples with
# multiplicity 3 and 6 -- the branches a single-mode input never reaches.
DEFAULT_FEATS = [[0, 1, "cos", 1.0],      # constant in xi, cos(phi)
                 [1, 2, "cos", 0.6],      # linear in xi,   cos(2 phi)
                 [0, 3, "sin", 0.8]]      # constant in xi, sin(3 phi)


def main():
    M = int(os.environ.get("QM", "6"))
    Nr = int(os.environ.get("QNR", "3"))
    n_xi = int(os.environ.get("QNXI", "0"))
    n_phi = int(os.environ.get("QNPHI", "0"))
    xi_cut = float(os.environ.get("QXIC", "9.0"))
    n_xi_proj = int(os.environ.get("QNXP", "0"))
    backend = os.environ.get("QB", "auto")
    tag = os.environ.get("QTAG", "v2")
    peak = float(os.environ.get("QPEAK", "0.30"))
    T = float(os.environ.get("QT", str(qv.T0)))
    feats = json.loads(os.environ.get("QFEATS", json.dumps(DEFAULT_FEATS)))
    feats = [(int(p), int(m), str(k), float(a)) for p, m, k, a in feats]

    torch.set_default_dtype(torch.float64)
    from qimpy import rc
    from qimpy.mpi import ProcessGrid
    from qimpy.transport.material.fermi_surface import FermiSurface

    kw = dict(epsilon_bg=qv.EPS_B, kappa=qv.KAPPA, nonlinear=True,
              check_convergence=False, backend=backend)
    for k, v in (("n_xi", n_xi), ("n_phi", n_phi), ("n_xi_proj", n_xi_proj)):
        if v:
            kw[k] = v
    if xi_cut:
        kw["xi_cut"] = xi_cut

    t0 = time.time()
    fs = FermiSurface(kF=qv.KF, vF=qv.KF / qv.M_STAR, M_theta=M, Nr=Nr, T=T,
                      xi_max=6.0,
                      process_grid=ProcessGrid(rc.comm, "rk", (-1, 1)),
                      ee_scattering=kw)
    ee = fs.ee_scattering
    dim = fs.angular.dim
    build_min = (time.time() - t0) / 60
    psi_coeff = ee._radial_galerkin(T)[0].cpu()

    # QFEATSETS lets ONE (expensive) operator build serve many input fields --
    # the operator does not depend on the input, so the whole input-mode sweep
    # costs one build plus a handful of applies.
    sets = os.environ.get("QFEATSETS")
    featsets = ([[(int(p), int(m), str(k), float(a)) for p, m, k, a in fs_]
                 for fs_ in json.loads(sets)] if sets else [feats])
    tags = json.loads(os.environ.get("QSUBTAGS",
                                     json.dumps([f"{tag}_{i}" for i in
                                                 range(len(featsets))])))

    for sub, fset in zip(tags, featsets):
        fset = qv.scale_features(fset, peak, xi_cut, T)
        a = qv.modes_from_features(fset, psi_coeff, dim).to(rc.device)
        a4 = a.reshape(1, Nr, dim)
        lin = (-torch.einsum("cij,bjc->bic", ee.L_coeff, a4)).reshape(-1).cpu()
        ad_p, ad_m = ee.a_dot(a).cpu(), ee.a_dot(-a).cpu()
        ad_2p, ad_2m = ee.a_dot(2 * a).cpu(), ee.a_dot(-2 * a).cpu()
        # same signed stencil as the reference, applied to the OPERATOR:
        o1 = 0.5 * (ad_p - ad_m)
        o2 = 0.5 * (ad_2p - ad_2m)
        out = dict(L=lin, Q=0.5 * (ad_p + ad_m), C=(o2 - 2.0 * o1) / 6.0,
                   Lstencil=(8.0 * o1 - o2) / 6.0, full=ad_p)
        meta = dict(tag=sub, M=M, Nr=Nr, dim=dim, backend=ee.backend,
                    n_xi=ee.n_xi, n_phi=ee.n_phi, xi_cut=ee.xi_cut,
                    n_xi_proj=ee.n_xi_proj, build_min=build_min, T=T,
                    feats=[list(f) for f in fset], peak=peak)
        np.savez(f"/tmp/v2prod_{sub}.npz", meta=json.dumps(meta),
                 psi_coeff=psi_coeff.numpy(), a=a.cpu().numpy(),
                 **{k: v.numpy() for k, v in out.items()})
        print(json.dumps(meta), flush=True)
        # L_coeff and the amplitude stencil must agree: an internal consistency
        # check of a_dot's own order separation (catches any leak of the
        # quadratic/cubic into what the solver integrates as "linear")
        rel = float((out["L"] - out["Lstencil"]).abs().max()
                    / out["L"].abs().max().clamp(min=1e-300))
        print(f"  L_coeff vs amplitude-stencil L: {rel:.2e}")
        for key in ("L", "Q", "C"):
            v = out[key].reshape(Nr, dim)
            nz = sum(1 for n in range(Nr) for c in range(dim)
                     if abs(float(v[n, c])) > 1e-12 * float(v.abs().max()))
            print(f"  {key}: |max| = {float(v.abs().max()):.5e}, "
                  f"{nz} nonzero of {Nr * dim}")


if __name__ == "__main__":
    main()
