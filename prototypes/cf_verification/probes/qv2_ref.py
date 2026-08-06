"""Reference side, v2: the same multi-mode field through EITHER reference.

QEVAL=unred -> `unreduced_collision_reference` (shares nothing with production)
QEVAL=exact -> `exact_collision_reference`     (shares the reduction only)

Dumps the RAW order-resolved fdot fields on the (x1, phi1) grid.  Everything
downstream -- modal projection at any (M, Nr), field reconstruction, conserved
moments -- is post-processing on these arrays, so the expensive part is paid
once and reused.

Env: QEVAL QNPHI QNXI QNXI2 QNXI3 QSIG QXIC QNX QNAZ QCHUNK QTAG QFEATS QPEAK QT
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv2_common as qv
from qv2_prod import DEFAULT_FEATS


def main():
    evaluator = os.environ.get("QEVAL", "unred")
    n_phi = int(os.environ.get("QNPHI", "0"))
    n_xi = int(os.environ.get("QNXI", "48"))          # exact only
    n_xi2 = int(os.environ.get("QNXI2", "320"))       # unred only
    n_xi3 = int(os.environ.get("QNXI3", "20"))
    sigma = float(os.environ.get("QSIG", "0.30"))
    xi_cut = float(os.environ.get("QXIC", "9.0"))
    nx = int(os.environ.get("QNX", "21"))
    n_az = int(os.environ.get("QNAZ", "26"))
    chunk = int(os.environ.get("QCHUNK", "2"))
    tag = os.environ.get("QTAG", "v2ref")
    peak = float(os.environ.get("QPEAK", "0.30"))
    T = float(os.environ.get("QT", str(qv.T0)))
    feats = json.loads(os.environ.get("QFEATS", json.dumps(DEFAULT_FEATS)))
    feats = [(int(p), int(m), str(k), float(a)) for p, m, k, a in feats]
    feats = qv.scale_features(feats, peak, xi_cut, T)

    torch.set_default_dtype(torch.float64)
    from qimpy.transport.material.fermi_surface.scattering import _kernels

    xg, ph, Xf, Pf = qv.grids(nx, xi_cut, n_az)
    torch.set_default_device("cuda")
    Xf, Pf = Xf.cuda(), Pf.cuda()
    df = qv.field_from_features(feats, T)
    common = dict(kF=qv.KF, m_star=qv.M_STAR, T=T, epsilon_bg=qv.EPS_B,
                  kappa=qv.KAPPA, g_s=2.0)

    t0 = time.time()
    fd = {}
    for s in (1.0, 2.0, -1.0, -2.0):
        fn = (lambda s_: (lambda x, p: df(x, p) * s_))(s)
        if evaluator == "unred":
            v = _kernels.unreduced_collision_reference(
                fn, Xf, Pf, linearize=False, sigma=sigma, n_xi2=n_xi2,
                n_xi3=n_xi3, n_phi=n_phi, xi_cut=xi_cut, x2chunk=chunk,
                **common)
        else:
            v = _kernels.exact_collision_reference(
                fn, Xf, Pf, linearize=False, n_xi=n_xi, xi_cut=xi_cut,
                n_phi=(n_phi or 1024), chunk=chunk, **common)
        fd[s] = v.cpu()
        print(f"  s={s:+.0f} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.set_default_device("cpu")

    raw = qv.orders_from_stencil(fd)
    meta = dict(tag=tag, evaluator=evaluator, n_phi=n_phi, n_xi=n_xi,
                n_xi2=n_xi2, n_xi3=n_xi3, sigma=sigma, xi_cut=xi_cut, nx=nx,
                n_az=n_az, T=T, min=(time.time() - t0) / 60,
                feats=[list(f) for f in feats], peak=peak)
    np.savez(f"/tmp/v2ref_{tag}.npz", meta=json.dumps(meta),
             xg=xg.numpy(), ph=ph.numpy(),
             **{f"raw{k}": v.numpy() for k, v in raw.items()},
             **{f"amp{s:+.0f}": v.numpy() for s, v in fd.items()})
    print(json.dumps(meta), flush=True)
    for k, v in raw.items():
        print(f"  {k}: |max fdot| = {float(v.abs().max()):.5e}")


if __name__ == "__main__":
    main()
