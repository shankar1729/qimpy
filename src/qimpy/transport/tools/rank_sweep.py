"""Rank x geometry sweep for the torch.distributed port (merge ad7ab150).

WHY THIS EXISTS.  `SpatialDecomp.exchange` returns early when recv/send are
empty, which is always true at world_size = 1 -- so the ported halo exchange
(dist.isend / dist.irecv / broadcast_object_list / ReduceOp.MIN) has never
executed a single instruction, in any test, ever.  136 passing tests say
nothing about it.  Only a multi-rank run does.

THE INVARIANT.  Domain decomposition is a pure implementation detail: the same
mesh, drive and k-grid must give the SAME physics at any rank count.  So run
each geometry at 1, 2, 3, 4 ranks from an identical initial state and require

  * contact currents          agree with the 1-rank reference
  * min f / max f             agree
  * dt_max                    agrees   (this is the ReduceOp.MIN path)
  * per-channel particle number agrees (this is the halo + flux assembly)

⛔ Rank counts that do NOT divide the cell count are the interesting ones: 3
ranks on 1792 cells gives uneven blocks and exercises the ragged edge of the
partition, which is exactly where an off-by-one in send/recv lists hides.

⛔ Compared at EQUAL STEP COUNT from an identical start, never at "converged" --
two runs that both converged would agree even if the halo were silently
dropping data, because the steady state is set by the boundary conditions.
A short run diverges immediately if the exchange is wrong.

⛔ Bit-identity is NOT the bar.  Different rank counts reorder floating-point
reductions, so agreement is to a tolerance (default 1e-10 relative); anything
larger than that is a real defect, not reordering.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc
from qimpy.transport import Transport
from qimpy.transport.geometry import TensorList

torch.set_default_dtype(torch.float64)

CFG_FS = dict(kF=7.5e-3, vF=0.11194, M_theta=32, Nr=6, T=1.3301e-5, xi_max=6.0,
              tau_p=float("inf"), specularity=1.0, residual_damping=False,
              cartesian=dict(annulus_xi=0.0, te_fac_max=6.0, kD_max=1.2e-3,
                             dmu_max=1.2e-4, k_max=0.0132557160008, n_k=56))
DMU = 5.37e-5


def make_mesh(base: str, kind: str, out: str) -> str:
    """Variants of a base mesh that change WHICH boundary operators are live."""
    d = dict(np.load(base, allow_pickle=True))
    mk = d["boundary_markers"]
    if kind == "asis":
        return base
    if kind == "allcontact":       # no reflector anywhere
        d["boundary_markers"] = np.array(
            ["source" if (i % 2 == 0) else "drain" for i in range(len(mk))])
    elif kind == "cavity":         # every boundary a wall: closed system
        d["boundary_markers"] = np.array(["wall"] * len(mk))
    else:
        raise KeyError(kind)
    np.savez(out, **d)
    return out


def _stage(msg: str) -> None:
    # ⛔ Every rank prints, unconditionally and flushed: a hang is diagnosed by
    # seeing WHICH rank stopped printing and where, and rank-0-only logging
    # hides exactly the asymmetric-deadlock case we are hunting.
    if os.environ.get("STAGE"):
        import socket
        print(f"  [stage] {socket.gethostname()} r{os.environ.get('OMPI_COMM_WORLD_RANK','?')} {msg}",
              flush=True)


def run(mesh: str, steps: int, n_k: int) -> dict:
    fs = dict(CFG_FS)
    fs["cartesian"] = dict(CFG_FS["cartesian"])
    fs["cartesian"]["n_k"] = n_k
    d = dict(np.load(mesh, allow_pickle=True))
    names = set(str(x) for x in d["boundary_markers"])
    contacts = {}
    if "source" in names:
        contacts["source"] = {"dmu": DMU, "nonlinear": True}
    if "drain" in names:
        contacts["drain"] = {"dmu": -DMU, "nonlinear": True}
    _stage('before Transport')
    t = Transport(
        fermi_surface=fs,
        spatial_transport=dict(mesh_file=mesh, compile=False, save_rho=True,
                               contacts=contacts),
        time_evolution=dict(t_max=1e30, dt_save=1e30, n_collate=1))
    _stage('Transport built')
    g = t.geometry
    assert float(t.material.rho_dot(
        torch.randn(4, g.Nk, device=rc.device), 0.0, 0).abs().max()) == 0.0, \
        "material is not ballistic"
    f0 = t.material.representation._f0_lab[None, :]
    u = g.rho[0].clone()
    dt = float(g.dt_max)
    _stage(f'stepping {steps} (K={g.K} own={g._own_stop-g._own_start})')
    for _ in range(steps):
        uh = u + (0.5 * dt) * g.rho_dot(TensorList([u]), 0.0)[0]
        u = u + dt * g.rho_dot(TensorList([uh]), 0.5 * dt)[0]
    # ⛔ geometry.rho is the FULL (K, Nk) array INCLUDING the halo copies of
    # other ranks' cells -- not the owned slice.  Writing it back through a
    # [_own_start:_own_stop] slice fails (896 vs 1792); copy the whole thing.
    g._u.copy_(u)

    _stage('stepped')
    f = f0 + u
    # ⛔ but REDUCE over owned cells only: every rank holds halo duplicates of
    # its neighbours' cells, so summing the full array double-counts them and
    # the "conservation" check would pass on a broken exchange.
    own = slice(g._own_start, g._own_stop)
    area = g.geom.area[own, None]
    n_ch = (area * f[own]).sum(0)
    loc = torch.stack([f[own].min().reshape(()), (-f[own].max()).reshape(())])
    # ⛔ Drive BOTH trees from one harness: pre-merge exposes an mpi4py
    # `comm` (comm.size, comm.Allreduce), merged exposes a torch.distributed
    # `group` (group.size(), dist.all_reduce).  Without this the pre-merge arm
    # of an equivalence test dies on AttributeError and silently compares
    # nothing.
    grp = getattr(g, "group", None)
    if grp is not None:                                   # merged tree
        n_ranks = grp.size()
        if g._mpi:
            dist.all_reduce(n_ch, group=grp)
            dist.all_reduce(loc, op=dist.ReduceOp.MIN, group=grp)
    else:                                                 # pre-merge tree
        from qimpy import MPI
        from qimpy.mpi import BufferView
        comm = g.comm
        n_ranks = comm.size
        if g._mpi:
            comm.Allreduce(MPI.IN_PLACE, BufferView(n_ch))
            comm.Allreduce(MPI.IN_PLACE, BufferView(loc), op=MPI.MIN)
    out = dict(dt_max=dt, min_f=float(loc[0]), max_f=float(-loc[1]),
               n_ch_sum=float(n_ch.sum()), n_ch_absmax=float(n_ch.abs().max()),
               K=int(g.K), Nk=int(g.Nk), ranks=n_ranks)
    _stage('reduced')
    try:
        out["currents"] = {k: float(v) for k, v in g.contact_currents(0.0).items()}
    except Exception as e:
        out["currents"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--kind", default="asis")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--nk", type=int, default=56)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rc.init()
    mesh = make_mesh(a.mesh, a.kind, f"/tmp/mesh_{a.kind}.npz")
    res = run(mesh, a.steps, a.nk)
    if rc.i_proc == 0:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(json.dumps(res))


if __name__ == "__main__":
    main()
