"""Run the exact ballistic solver on a qimpy transport input file.

    python -m qimpy.transport.ballistic -i input.yaml [--n-ang 1024]

Reads the SAME yaml as ``python -m qimpy.transport``: the mesh comes from
``spatial_transport.mesh_file``, the band from ``fermi_surface`` and the drive
from ``spatial_transport.contacts``.  Anything the FV solver can run, this can
run -- which is the point: it is a cross-check, not a separate model.

⛔ n_ang IS NOT CONVERGED AT SMALL VALUES, and it is not monotone, so a
single run proves nothing -- ladder it.  Measured on mixer-refined-gaas at
dmu = 5.37e-5, n_edge = 48, max_bounce = 3200:

    n_ang    512     1024     2048     4096
    I (uA)  9.6908  9.5130   9.7649   9.7863

2048 -> 4096 moves 0.22%, so ~9.79 uA is the converged answer; 1024 is 2.8%
low and would have looked perfectly plausible on its own.

⛔ It solves the STEADY BALLISTIC limit only.  If the input specifies
``ee_scattering`` or a finite ``tau_p`` the answer is not the same physics, so
that is reported rather than silently ignored.
"""
from __future__ import annotations

import argparse
import sys

import yaml

from ._solver import Ballistic


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m qimpy.transport.ballistic")
    ap.add_argument("-i", "--input-file", required=True)
    ap.add_argument("--n-ang", type=int, default=2048,
                    help="angular samples per boundary point; LADDER IT")
    ap.add_argument("--n-edge", type=int, default=48,
                    help="quadrature points per contact segment")
    ap.add_argument("--max-bounce", type=int, default=12800)
    a = ap.parse_args(argv)

    with open(a.input_file) as fh:
        cfg = yaml.safe_load(fh)
    fs = cfg.get("fermi_surface")
    st = cfg.get("spatial_transport")
    if fs is None or st is None:
        print("input needs both 'fermi_surface' and 'spatial_transport'")
        return 2
    for bad, why in (("ee_scattering", "collisions"),
                     ("tau_p", "momentum relaxation")):
        v = fs.get(bad)
        if v is not None and v not in (float("inf"), ".inf", "inf"):
            print(f"  NOTE: input specifies {bad} ({why}); the ballistic "
                  "solver ignores it and reports the collisionless limit.")

    contacts = {k: float(v.get("dmu", 0.0))
                for k, v in (st.get("contacts") or {}).items()}
    b = Ballistic(st["mesh_file"], kF=float(fs["kF"]), vF=float(fs["vF"]),
                  T=float(fs["T"]), contacts=contacts)
    print(f"  mesh      {st['mesh_file']}")
    print(f"  contacts  {b.poly.contact_names}   driven {contacts}")
    print(f"  device    {b.device}")
    tot = 0.0
    for nm in b.poly.contact_names:
        I, trap = b.contact_current(nm, n_ang=a.n_ang, n_edge=a.n_edge,
                                    max_bounce=a.max_bounce)
        tot += I
        print(f"  I[{nm:<10}] = {I:+.8e} a.u. = {I / 1.5097e-4:+10.4f} uA"
              f"   unresolved {trap:.4f}")
    print(f"  sum(I)      = {tot:+.3e} a.u.   (should vanish: charge "
          "conservation across all contacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
