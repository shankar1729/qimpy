"""Spatial-MPI parallelization of the DG advection.

qimpy's process grid has two dimensions, real-space ("r") and momentum ("k").
The material divides channels over "k"; this module populates the so-far-unused
"r" dimension with a spatial domain decomposition of the triangle mesh, so a
run with process grid (Pr, Pk) parallelizes over BOTH space and momentum.

Design: partition the mesh by elements (METIS min-cut). Each rank runs the
serial operators (DG2D/AdvectTorch) on its local submesh = owned elements + a
one-element ghost ring. Before every RHS evaluation the ghost element state is
exchanged over the "r" communicator, so faces that straddle a rank boundary
read the correct neighbor trace. Both ranks sharing a cut face then compute the
identical upwind flux, so mass telescopes across the cut and the distributed
result is bit-for-bit the serial one.

The local solver (`DistributedAdvect`) is communicator-agnostic; the halo
transport is pluggable -- `mpi_halo_exchange` (real mpi4py) for production,
`emulate_halo` (in-process array routing) for testing without launching MPI.
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch

from qimpy import MPI
from qimpy.mpi import BufferView
from ._dg2d import DG2D
from ._dg_torch import AdvectTorch


def _global_adjacency(EToV):
    e2el = defaultdict(list)
    for k, t in enumerate(EToV):
        for x, y in [(0, 1), (1, 2), (2, 0)]:
            e2el[tuple(sorted((int(t[x]), int(t[y]))))].append(k)
    nbr = defaultdict(set)
    for els in e2el.values():
        if len(els) == 2:
            a, b = els; nbr[a].add(b); nbr[b].add(a)
    return nbr


def compute_partition(mesh, nparts: int) -> np.ndarray:
    """METIS min-cut element partition of the global mesh -> part[K]."""
    K = len(mesh.EToV)
    if nparts == 1:
        return np.zeros(K, np.int32)
    try:
        import pymetis
    except ImportError as e:                       # optional dependency
        raise ImportError("spatial MPI decomposition needs pymetis "
                          "(pip install pymetis)") from e
    nbr = _global_adjacency(mesh.EToV)
    _, part = pymetis.part_graph(nparts, adjacency=[sorted(nbr[k]) for k in range(K)])
    return np.asarray(part, np.int32)


class SpatialPartition:
    """Local submesh (owned + ghost ring) for one rank, plus halo plans.
    Communicator-free: built from the shared global `part` array so it can be
    constructed identically under real MPI or in-process emulation."""

    def __init__(self, mesh, order: int, part: np.ndarray, rank: int):
        VX, VY, EToV = mesh.VX, mesh.VY, mesh.EToV
        nbr = _global_adjacency(EToV)
        self.part, self.rank = part, rank
        owned = np.sort(np.where(part == rank)[0])
        ghost = np.array(sorted({n for o in owned for n in nbr[o]
                                 if part[n] != rank}), int)
        self.n_owned = len(owned)
        local = (np.concatenate([owned, ghost]).astype(int)
                 if len(ghost) else owned.astype(int))
        self.local2global_elem = local
        g2l = {int(g): i for i, g in enumerate(local)}

        used = np.unique(EToV[local].ravel())
        g2lv = {int(g): i for i, g in enumerate(used)}
        self.local2global_vert = used
        EToVl = np.array([[g2lv[int(v)] for v in EToV[g]] for g in local], int)
        self.dg = DG2D(order, VX[used].copy(), VY[used].copy(), EToVl)
        self.edge_marker, self.marker_names = mesh.edge_marker, mesh.marker_names

        send, recv = defaultdict(set), defaultdict(set)
        for o in owned:
            for n in nbr[o]:
                if part[n] != rank:
                    send[part[n]].add(int(o))
        for gh in ghost:
            recv[part[gh]].add(int(gh))
        # sorting by global id aligns r.send_plan[q] with q.recv_plan[r]
        self.send_plan = {q: np.array([g2l[g] for g in sorted(s)], int)
                          for q, s in send.items()}
        self.recv_plan = {q: np.array([g2l[g] for g in sorted(s)], int)
                          for q, s in recv.items()}


class DistributedAdvect:
    """Local DG advection on a SpatialPartition. Ghost rows of `w` must be
    refreshed by the chosen halo transport before `local_rhs` is called."""

    def __init__(self, part: SpatialPartition, material, contacts, comm=None):
        self.p = part
        self.comm_sp = comm                       # spatial comm (face reductions)
        v_t = material.transport_velocity
        self.Nk = v_t.shape[0]
        self.vx = v_t[:, 0].contiguous(); self.vy = v_t[:, 1].contiguous()
        self.coupling = getattr(material, "coupling", None)  # modal flux operator
        self.adv = AdvectTorch(part.dg, device=v_t.device, dtype=v_t.dtype)
        # number-density projection (wk * observable_0) for contact-current readout
        self.mass_proj = (material.wk * material.get_observables(0.0)[0]).to(v_t.dtype)
        self._floating = []                       # (name, pos, FloatingContactor)
        self._contact_sel = {}                    # name -> sel (indices into mapB)
        self._setup_boundaries(material, contacts)

    def _setup_boundaries(self, material, contacts):
        dg = self.p.dg
        if dg.mapB.size == 0:
            self.adv.set_boundary(reflect_sel=np.array([], int), reflector=None,
                                  contacts=[], specularity=1.0); return
        l2gv = self.p.local2global_vert
        vn = np.array([[0, 1], [1, 2], [2, 0]]); Nfp, Nfaces = dg.Nfp, dg.Nfaces
        fm = {}
        for k in range(dg.K):
            for f in range(dg.Nfaces):
                if dg.EToE[k, f] != k:
                    continue
                ga = int(l2gv[dg.EToV[k, vn[f, 0]]]); gb = int(l2gv[dg.EToV[k, vn[f, 1]]])
                m = self.p.edge_marker.get(tuple(sorted((ga, gb))), 0)
                fm[(k, f)] = (self.p.marker_names[m]
                              if m < len(self.p.marker_names) else "wall")
        node_name = np.empty(dg.mapB.size, object)
        for i, pp in enumerate(dg.mapB):
            t = int(pp) // Nfp
            node_name[i] = fm.get((t // Nfaces, t % Nfaces), "wall")
        n_all = self.adv.boundary_normals()
        cops = []; assigned = np.zeros(dg.mapB.size, bool)
        for name, params in (contacts or {}).items():
            sel = np.where(node_name == name)[0]
            if not sel.size:
                continue
            assigned[sel] = True
            ten = torch.as_tensor(sel, dtype=torch.long)
            self._contact_sel[name] = ten
            p = params or {}
            if p.get("floating", False):
                # voltage probe: isotropic ghost whose level floats to zero current
                fc = FloatingContactor(dim=self._coupling_dim(),
                                       device=self.adv.device, dtype=self.adv.dtype)
                cops.append((sel, fc))
                self._floating.append((name, self.adv.mapB[ten], fc))
            else:
                cops.append((sel, material.get_contactor(n_all[ten], **p)))
        rsel = np.where(~assigned)[0]
        refl = (material.get_reflector(n_all[torch.as_tensor(rsel, dtype=torch.long)])
                if rsel.size else None)
        self.adv.set_boundary(reflect_sel=rsel, reflector=refl,
                              contacts=cops, specularity=1.0)

    def _coupling_dim(self):
        c = getattr(self, "coupling", None)
        return None if c is None else c.dim

    def _update_floating(self, u):
        """Set each voltage-probe level mu_c = (face number-flux out)/(face inflow
        capacity), so the net current through the contact is zero. With an isotropic
        ghost uP = mu_c e_0, the boundary number-flux is (A_n^+ uM)_0 + mu_c (A_n^- e_0)_0,
        so mu_c = -sum (A_n^+ uM)_0 / sum (A_n^- e_0)_0 (flux-weighted over the face).
        Scalar channels are the diagonal special case A_n = diag(v.n)."""
        cpl = self.coupling
        uf = u.permute(1, 0, 2).reshape(self.adv.Np * self.adv.K, -1)
        uM = uf[self.adv.vmapM]
        for name, pos, fc in self._floating:
            nx = self.adv.nxf[pos]; ny = self.adv.nyf[pos]
            uMb = uM[pos]; wq = self.adv.face_quad_w[pos]
            if cpl is None:
                adn = nx[:, None] * self.vx[None, :] + ny[:, None] * self.vy[None, :]
                num = (adn.clamp(min=0.0) * uMb).sum(-1)   # (v.n)^+ . uM  per node
                den = (-adn.clamp(max=0.0)).sum(-1)        # (v.n)^- capacity per node
            else:
                phi = torch.atan2(ny, nx)
                e0 = torch.zeros_like(uMb); e0[:, 0] = 1.0
                An_uM = nx[:, None] * (uMb @ cpl.Ax.T) + ny[:, None] * (uMb @ cpl.Ay.T)
                An_e0 = nx[:, None] * (e0 @ cpl.Ax.T) + ny[:, None] * (e0 @ cpl.Ay.T)
                num = 0.5 * (An_uM + cpl.abs_flux(uMb, phi))[:, 0]   # (A_n^+ uM)_0
                den = -0.5 * (An_e0 - cpl.abs_flux(e0, phi))[:, 0]   # -(A_n^- e_0)_0
            acc = torch.stack([(wq * num).sum(), (wq * den).sum()])
            if self.comm_sp is not None and self.comm_sp.size > 1:
                self.comm_sp.Allreduce(MPI.IN_PLACE, BufferView(acc))
            fc.level = acc[0] / acc[1].clamp(min=1e-300)

    def local_rhs(self, w, t: float = 0.0):
        if self._floating:
            self._update_floating(self.adv.apply_mass_inv(w))
        return self.adv.rhs_w(w, self.vx, self.vy, t, coupling=self.coupling)

    def contact_currents(self, w, t: float = 0.0) -> dict:
        """Net outward current through each contact, I_c = integral_Gamma_c
        (number-flux of the numerical flux) dl. Positive = out of the device.
        For a floating probe this is ~0 by construction; its `level` is the
        floating potential. Conservative: sum over all boundaries = -d/dt of mass."""
        u = self.adv.apply_mass_inv(w)
        if self._floating:
            self._update_floating(u)
        _, fstar_b = self.adv._residual(u, self.vx, self.vy, t,
                                        coupling=self.coupling, return_fstar=True)
        massflux = fstar_b @ self.mass_proj            # (Nb,) per boundary node
        wb = self.adv.face_quad_w[self.adv.mapB]
        out = {}
        for name, sel in self._contact_sel.items():
            I = (wb[sel] * massflux[sel]).sum()
            if self.comm_sp is not None and self.comm_sp.size > 1:
                buf = I.clone()[None]
                self.comm_sp.Allreduce(MPI.IN_PLACE, BufferView(buf)); I = buf[0]
            out[name] = float(I)
        return out

    def contact_potentials(self) -> dict:
        """Floating (voltage-probe) potentials read at the last RHS evaluation."""
        return {name: float(fc.level) for name, _, fc in self._floating}


def build_distributed(mesh, order, material, contacts, comm):
    """Real-MPI entry: rank 0 partitions, broadcasts, each rank builds its local
    solver. Returns (SpatialPartition, DistributedAdvect) for this rank."""
    K = len(mesh.EToV)
    part = np.empty(K, np.int32)
    if comm.rank == 0:
        part[:] = compute_partition(mesh, comm.size)
    comm.Bcast(part, root=0)
    sp = SpatialPartition(mesh, order, part, comm.rank)
    return sp, DistributedAdvect(sp, material, contacts, comm)


class FloatingContactor:
    """Voltage-probe contact. Holds an isotropic ghost level (the floating
    electrochemical potential mu_c), updated each step to zero the contact
    current. Returns a per-direction constant (scalar representation) or a
    level * e_0 modal vector (`dim` set)."""

    def __init__(self, dim=None, device=None, dtype=None):
        self.dim = dim
        self.level = torch.zeros((), device=device, dtype=dtype)

    def __call__(self, t):
        if self.dim is None:
            return self.level                       # scalar: broadcast to channels
        v = torch.zeros(self.dim, device=self.level.device, dtype=self.level.dtype)
        v[0] = self.level                           # modal: isotropic m=0
        return v


def mpi_halo_exchange(comm, part: SpatialPartition, w):
    """Production transport: non-blocking exchange of ghost element state."""
    from mpi4py import MPI
    if not part.send_plan and not part.recv_plan:
        return
    Np, _, Nk = w.shape; wn = w.detach().cpu().numpy()
    reqs = []; sbufs = []; rbufs = {}
    for q, idx in part.recv_plan.items():
        buf = np.empty((Np, len(idx), Nk)); rbufs[q] = (buf, idx)
        reqs.append(comm.Irecv(buf, source=q, tag=7))
    for q, idx in part.send_plan.items():
        sb = np.ascontiguousarray(wn[:, idx, :]); sbufs.append(sb)
        reqs.append(comm.Isend(sb, dest=q, tag=7))
    MPI.Request.Waitall(reqs)
    for q, (buf, idx) in rbufs.items():
        w[:, idx, :] = torch.from_numpy(buf).to(w)


def emulate_halo(parts, ws):
    """Test transport: route owned->ghost in-process by the exact array slices
    Isend/Irecv would move (same buffers, same ordering). Lets the decomposition
    be validated against serial without launching an MPI job."""
    snap = [w.detach().clone() for w in ws]
    for r, P in enumerate(parts):
        for q, ridx in P.recv_plan.items():
            sidx = parts[q].send_plan[r]
            assert len(sidx) == len(ridx)
            ws[r][:, ridx, :] = snap[q][:, sidx, :]
