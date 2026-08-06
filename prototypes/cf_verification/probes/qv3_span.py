"""Does the reference's projection DOMAIN explain the top-radial-mode error?"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/home/ubuntu")
import qv2_common as qv
torch.set_default_dtype(torch.float64)
ref = np.load(sys.argv[1]); rm = json.loads(str(ref["meta"]))
xg, ph = torch.as_tensor(ref["xg"]), torch.as_tensor(ref["ph"])
T = rm.get("T", qv.T0)
p = np.load(sys.argv[2]); pm = json.loads(str(p["meta"]))
M, Nr, dim = pm["M"], pm["Nr"], pm["dim"]
psi = torch.as_tensor(p["psi_coeff"])
free = torch.ones(dim, dtype=torch.bool); free[:3] = False
print(f"reference grid |x| <= {float(xg.max()):.0f} with {len(xg)} points; "
      f"production Galerkin domain is FIXED at |x| <= 8")
for key in ("L", "Q", "C"):
    arr = torch.as_tensor(ref[f"raw{key}"]).reshape(len(xg), len(ph))
    a = torch.as_tensor(p[key]).reshape(Nr, dim)
    print(f"  --- {key}")
    for lim in (8.0, 9.0, 12.0, 14.0):
        sel = xg.abs() <= lim + 1e-9
        if sel.sum() < 5 or float(xg.max()) < lim - 1e-9:
            continue
        r = qv.project(arr[sel], xg[sel], ph, psi, M, T)
        e = float((a[:, free] - r[:, free]).norm() / r[:, free].norm())
        rows = "  ".join(f"n{n} {float((a[n,free]-r[n,free]).norm()):.2e}"
                         for n in range(Nr))
        print(f"      project over |x|<={lim:<4}  coef {e:.4f}   {rows}")
