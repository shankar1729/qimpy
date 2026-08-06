"""WHERE in the modal vector does the growing-with-Nr error live?

`coef` is a single ratio over the whole output vector.  That is exactly the kind
of aggregate that hides its own cause.  Break it down by RADIAL INDEX n:

    ||a_prod[n]||, ||a_ref[n]||, ||a_prod[n] - a_ref[n]||

If the growth sits in the high-n rows -- which carry little of the field, since
the reconstruction weights them by psi_n and the field norm is dominated by the
low modes -- then `coef` is measuring the accuracy of components that barely
affect the answer, and the field metric is the honest one.  If instead the error
grows in the LOW-n rows, the operator really is losing accuracy and the field
metric is hiding it behind the truncation error.

Usage: python3 qv3_where.py <ref.npz> <prod.npz> [<prod.npz> ...]
"""
import sys, json
import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import qv2_common as qv

torch.set_default_dtype(torch.float64)


def main():
    ref = np.load(sys.argv[1])
    rmeta = json.loads(str(ref["meta"]))
    xg, ph = torch.as_tensor(ref["xg"]), torch.as_tensor(ref["ph"])
    T = rmeta.get("T", qv.T0)
    print(f"reference: nx={rmeta['nx']} n_az={rmeta['n_az']} "
          f"evaluator={rmeta['evaluator']}\n")
    raw = {k: torch.as_tensor(ref[f"raw{k}"]).reshape(len(xg), len(ph))
           for k in ("L", "Q", "C")}

    for path in sys.argv[2:]:
        p = np.load(path)
        pm = json.loads(str(p["meta"]))
        M, Nr, dim = pm["M"], pm["Nr"], pm["dim"]
        psi_coeff = torch.as_tensor(p["psi_coeff"])
        free = torch.ones(dim, dtype=torch.bool); free[:3] = False
        print(f"--- {pm['tag']}  M={M} Nr={Nr} n_xi={pm['n_xi']} "
              f"n_phi={pm['n_phi']}")
        for key in ("L", "Q", "C"):
            a = torch.as_tensor(p[key]).reshape(Nr, dim)
            r = qv.project(raw[key], xg, ph, psi_coeff, M, T)
            tot_e = float((a[:, free] - r[:, free]).norm())
            tot_r = float(r[:, free].norm())
            # how much of the FIELD each radial row actually carries: the
            # reconstruction weight ||psi_n||_w, so a row's coefficient error is
            # only as important as psi_n is big
            psi_g = qv.psi_eval(xg, psi_coeff)
            wn = [float((psi_g[:, n] ** 2 * qv.w_occ(xg)).sum().sqrt())
                  for n in range(Nr)]
            rows = []
            for n in range(Nr):
                e = float((a[n, free] - r[n, free]).norm())
                rows.append(f"n={n}: |ref| {float(r[n, free].norm()):.2e}"
                            f" err {e:.2e} ({e/tot_e:5.1%} of total err,"
                            f" psi_w {wn[n]:.2f})")
            print(f"  {key}: coef = {tot_e/tot_r:.4f}")
            for s in rows:
                print("      " + s)


if __name__ == "__main__":
    main()
