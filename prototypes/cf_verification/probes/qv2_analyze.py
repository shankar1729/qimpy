"""Post-process: channel table, truncation convergence, conserved moments.

Usage:  python3 qv2_analyze.py <ref.npz> <prod.npz> [<prod.npz> ...]

The reference field is truncation-independent, so ONE reference run serves the
whole (M, Nr) ladder.  For each production run we
  * project the reference onto that run's (M, Nr) basis and compare channels;
  * reconstruct the production field and report ||prod - ref|| / ||ref||, which
    is the only comparison meaningful ACROSS truncations;
  * report conserved moments of both -- computed from the fields, not by
    contracting against the projector's own null covectors.
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
    xg = torch.as_tensor(ref["xg"])
    ph = torch.as_tensor(ref["ph"])
    print(f"reference: {json.dumps(rmeta)}\n")
    T = rmeta.get("T", qv.T0)
    raw = {k: torch.as_tensor(ref[f"raw{k}"]).reshape(len(xg), len(ph))
           for k in ("L", "Q", "C")}
    full = torch.as_tensor(ref["amp+1"]).reshape(len(xg), len(ph))

    print("CONSERVED MOMENTS of the reference field (must vanish; normalized "
          "by the field's own L1 norm):")
    for k, v in list(raw.items()) + [("full", full)]:
        _, rel = qv.moments(v, xg, ph)
        print(f"  {k:5s}  N {rel[0]:+.2e}   Px {rel[1]:+.2e}   "
              f"Py {rel[2]:+.2e}   E {rel[3]:+.2e}")
    print()

    # Three numbers, and the distinction between them is the whole point:
    #   basis  = || rec(proj(ref)) - ref || / ||ref||   how much of C[f]'s output
    #            simply does not FIT in the retained (M, Nr) span -- a property
    #            of the truncation alone, with production nowhere in sight;
    #   prod   = || rec(a_prod)     - ref || / ||ref||   the total error;
    #   coef   = || a_prod - proj(ref) || / ||proj(ref)||  the OPERATOR's own
    #            error inside the span.
    # prod ~= basis with coef << 1 means production is as accurate as the
    # truncation permits, and the residual is a basis-completeness statement,
    # not an operator defect.
    hdr = (f"{'run':>12} {'M':>3} {'Nr':>3} {'basis':>8} {'prod':>8} "
           f"{'coef':>8}")
    print(hdr + "     per-order  basis / prod / coef")
    print("-" * 100)
    for path in sys.argv[2:]:
        p = np.load(path)
        pm = json.loads(str(p["meta"]))
        M, Nr, dim = pm["M"], pm["Nr"], pm["dim"]
        psi_coeff = torch.as_tensor(p["psi_coeff"])
        # mo = 0 and 1 carry the conserved (null) directions, which production
        # PROJECTS OUT and the reference does not.  Comparing them would need
        # the projector on both sides -- which is precisely the tautology this
        # whole exercise exists to avoid -- so the coefficient error is reported
        # over mo >= 2, and conservation is verified independently through the
        # field moments above.
        free = torch.ones(dim, dtype=torch.bool)
        free[:3] = False                      # m0, cos1, sin1
        e = {}
        for key in ("L", "Q", "C", "full"):
            a = torch.as_tensor(p[key]).reshape(Nr, dim)
            tgt = full if key == "full" else raw[key]
            pr = qv.project(tgt, xg, ph, psi_coeff, M, T)
            e[key] = (
                float((qv.reconstruct(pr, xg, ph, psi_coeff, M, T) - tgt).norm()
                      / tgt.norm()),
                float((qv.reconstruct(a, xg, ph, psi_coeff, M, T) - tgt).norm()
                      / tgt.norm()),
                float((a[:, free] - pr[:, free]).norm()
                      / pr[:, free].norm().clamp(min=1e-300)))
        print(f"{pm['tag']:>12} {M:>3} {Nr:>3} {e['full'][0]:>8.4f} "
              f"{e['full'][1]:>8.4f} {e['full'][2]:>8.4f}     "
              + "  ".join(f"{k} {e[k][0]:.3f}/{e[k][1]:.3f}/{e[k][2]:.3f}"
                          for k in ("L", "Q", "C")))

    # channel-by-channel at the FIRST production run's truncation
    p = np.load(sys.argv[2])
    pm = json.loads(str(p["meta"]))
    M, Nr, dim = pm["M"], pm["Nr"], pm["dim"]
    psi_coeff = torch.as_tensor(p["psi_coeff"])
    print(f"\nCHANNEL TABLE at M={M}, Nr={Nr}  (tag {pm['tag']});"
          " residual as % of that order's peak")
    names = ["m0"] + [f"{'cs'[i%2]}{(i+2)//2}" for i in range(dim - 1)]
    for key in ("L", "Q", "C"):
        pr = torch.as_tensor(p[key]).reshape(Nr, dim)
        rf = qv.project(raw[key], xg, ph, psi_coeff, M, T)
        peak = float(rf.abs().max())
        print(f"  --- {key}  (reference peak {peak:.4e})")
        for n in range(Nr):
            for c in range(dim):
                a, b = float(pr[n, c]), float(rf[n, c])
                if max(abs(a), abs(b)) < 5e-3 * peak:
                    continue
                note = "  [null-projected: not comparable]" if c < 3 else ""
                print(f"      n={n} {names[c]:>4}  prod {a:+.5e}  ref {b:+.5e}"
                      f"   {abs(a-b)/peak:7.2%} of peak{note}")


if __name__ == "__main__":
    main()
