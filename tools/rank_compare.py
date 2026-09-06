"""Rank-invariance verdict for the torch.distributed port.

Domain decomposition must not change physics.  For each (mesh, boundary kind)
the 1-rank run is the reference and 2/3/4 ranks must reproduce it.

⛔ THE BAR IS NOT BIT-IDENTITY.  Splitting the cells reorders every reduction
and changes the order of flux accumulation, so agreement is to a tolerance.
But the tolerance has to be tight enough to catch a real defect: a halo that
drops a face, or an off-by-one in the send/recv lists, changes the solution at
the 1e-3 level within a few steps, not the 1e-14 level.  1e-9 relative
separates "floating-point reordering" from "wrong".

⛔ dt_max is checked for EXACT equality: it comes from a min-reduction over
cell inradii, and min is order-independent in floating point.  Any difference
there is a broken ReduceOp.MIN, not reordering.

⛔ Reporting only the worst case over all quantities would hide which one
broke, so every quantity is reported per configuration.
"""
from __future__ import annotations
import glob
import json
import os
import sys
from collections import defaultdict

TOL = float(os.environ.get("TOL", "1e-9"))


def rel(a: float, b: float) -> float:
    d = max(abs(a), abs(b))
    return 0.0 if d == 0.0 else abs(a - b) / d


def main(pattern: str) -> int:
    runs = defaultdict(dict)
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)[3:-5]          # strip "sw_" and ".json"
        key, _, rk = base.rpartition("_r")
        try:
            runs[key][int(rk)] = json.load(open(path))
        except Exception as e:
            print(f"  UNREADABLE {path}: {e}")
    if not runs:
        print("  no results yet")
        return 1

    bad = 0
    print(f"  tolerance {TOL:.0e} relative;  dt_max requires EXACT equality\n")
    hdr = f"  {'configuration':<40} {'ranks':>5} {'dt_max':>8} {'min f':>10} " \
          f"{'max f':>10} {'n_ch':>10} {'I_src':>10} {'I_drn':>10}"
    print(hdr)
    for key in sorted(runs):
        by_rank = runs[key]
        if 1 not in by_rank:
            print(f"  {key:<40}   NO 1-RANK REFERENCE")
            bad += 1
            continue
        ref = by_rank[1]
        for n in sorted(by_rank):
            if n == 1:
                continue
            r = by_rank[n]
            dt_ok = (r["dt_max"] == ref["dt_max"])
            cells = [
                "EXACT" if dt_ok else "DIFFER",
                f"{rel(r['min_f'], ref['min_f']):.2e}",
                f"{rel(r['max_f'], ref['max_f']):.2e}",
                f"{rel(r['n_ch_sum'], ref['n_ch_sum']):.2e}",
            ]
            for c in ("source", "drain"):
                a, b = r.get("currents", {}), ref.get("currents", {})
                cells.append(f"{rel(a[c], b[c]):.2e}" if (c in a and c in b)
                             else "n/a")
            worst = max([float(x) for x in cells[1:] if x != "n/a"])
            flag = "" if (dt_ok and worst < TOL) else "   <-- FAIL"
            if flag:
                bad += 1
            print(f"  {key:<40} {n:>5} " + " ".join(f"{c:>10}" for c in cells)
                  + flag)
    print()
    if bad:
        print(f"  {bad} configuration(s) FAILED rank-invariance")
    else:
        print("  ALL configurations rank-invariant")
    # a defect would also show as min f leaving [0,1] only at some rank counts
    print("\n  absolute occupancy bounds (all runs):")
    for key in sorted(runs):
        for n, r in sorted(runs[key].items()):
            if r["min_f"] < -1e-12 or r["max_f"] > 1 + 1e-12:
                print(f"    {key} r{n}: f in [{r['min_f']:+.3e}, {r['max_f']:.10f}]"
                      "   OUT OF [0,1]")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sw_*.json"))
