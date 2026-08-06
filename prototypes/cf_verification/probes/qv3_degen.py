"""Are the odd radial modes near-degenerate, so that only their SUM is determined?

Features {1, xi, v^2, v, v^4, v^3} with v = tanh(xi/2).  Near the Fermi surface
v ~ xi/2, so psi_1 (from xi) and psi_3 (from v) are nearly parallel there and
differ only in the tails.  If that is the story, the per-row coefficient errors
should CANCEL when reconstructed together: the operator gets the physical field
right while the split between near-parallel modes is ill-determined.
"""
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
sel = xg.abs() <= 14.0
a = torch.as_tensor(p["L"]).reshape(Nr, dim)
r = qv.project(torch.as_tensor(ref["rawL"]).reshape(len(xg), len(ph))[sel],
               xg[sel], ph, psi, M, T)
d = a - r
psi_g = qv.psi_eval(xg, psi)
# overlap of the basis functions in the w_eq measure: how parallel are they?
W = qv.w_occ(xg) * float(xg[1] - xg[0])
G = torch.einsum("xn,xm,x->nm", psi_g, psi_g, W)
nrm = G.diagonal().sqrt()
print(f"Nr={Nr}  basis overlaps <psi_i,psi_j>/(|psi_i||psi_j|):")
for i in range(Nr):
    print("   " + "  ".join(f"{float(G[i,j]/(nrm[i]*nrm[j])):+.3f}"
                            for j in range(Nr)))
# do the row errors cancel in the reconstructed field?
tot = qv.reconstruct(d, xg, ph, psi, M, T).norm()
indiv = sum(qv.reconstruct(torch.where(
    torch.arange(Nr)[:, None] == n, d, torch.zeros_like(d)),
    xg, ph, psi, M, T).norm() for n in range(Nr))
print(f"\n  ||sum of row errors as a FIELD||   = {float(tot):.4e}")
print(f"  sum of ||each row error as field|| = {float(indiv):.4e}")
print(f"  cancellation factor                = {float(indiv/tot):.2f}x")
