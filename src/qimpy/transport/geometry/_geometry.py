"""Cell-centered 2nd-order finite-volume transport on an unstructured triangle mesh.

One cell average per triangle per momentum channel, ``u: (K, Nk)``. MUSCL
reconstruction (least-squares cell gradient, Venkatakrishnan limited) feeds a
scalar per-channel upwind flux ``F_c = (v_c.n) u_upwind`` -- each delta-k channel
streams with its own Fermi velocity, so the upwind side of each edge is fixed by
sign(v_c.n). Walls/contacts supply the exterior trace via the (reused)
FermiSurface reflector/contactor; collisions come from the material. Time
stepping is plain RK2 (see _time_evolution).

Boundary conditions: reflective walls, fixed-voltage/drift contacts, floating
(zero-current) probes and current sources (per-step scalar level solve), and
periodic faces paired through the mesh lattice. A METIS spatial decomposition
splits cells across the ``r`` comm with a thin halo exchange (FermiSurface
couples k-channels, so k is never split); see :class:`SpatialDecomp` below.

All velocity-independent per-edge weights are precomputed once, so a step is just
one limited-reconstruction pass, two gathers, and three scatters.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

from scipy.sparse import csr_array
import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc, log, TreeNode
from qimpy.io import CheckpointPath, InvalidInputException, CheckpointContext
from qimpy.mpi import ProcessGrid, all_reduce_scalars
from qimpy.math import ceildiv
from qimpy.profiler import StopWatch
from ..material import Material
from . import Mesh


class SpatialDecomp:
    """Owned/ghost bookkeeping and halo exchange over a renumbered cell mesh.

    Construct after the cells have been renumbered by :func:`partition` and the
    geometry built, passing the vertex-neighbour table ``nbr`` (global, in the
    renumbered order) and the per-rank ``bounds``. Owned cells of rank ``r`` are
    the contiguous slice ``[bounds[r], bounds[r+1])``.
    """

    def __init__(
        self, mesh: Mesh, bounds: np.ndarray, group: dist.ProcessGroup
    ) -> None:
        self.group = group
        self.size = group.size()
        self.rank = dist.get_rank(group)
        self.K = len(mesh.cells)
        self.offset = np.asarray(bounds, int)
        self.start = int(self.offset[self.rank])
        self.stop = int(self.offset[self.rank + 1])
        self.owned = np.arange(self.start, self.stop)
        adj_vert = mesh.cell_adjacency_vertex
        adj_edge = mesh.cell_adjacency_edge

        def ring(cells: np.ndarray, adj: csr_array) -> np.ndarray:
            """1-ring vertex closure of a set of cells (cells + their neighbours)."""
            v = np.zeros(self.K)
            v[cells] = 1
            v += adj @ v
            return np.where(v)[0]

        ring1 = ring(self.owned, adj_edge)  # cells to reconstruct (+edge-neighbors)
        ring2 = ring(ring1, adj_vert)  # cells accessed overall (+vertex-neighbors)
        self.recon_rows = ring1
        ghosts = ring2[np.where((ring2 < self.start) | (ring2 >= self.stop))]

        # Plan communications:
        recv: list[np.ndarray] = []
        send: list[np.ndarray] = []
        for q in range(self.size):
            if q == self.rank:
                recv.append(np.empty(0, dtype=int))
                send.append(np.empty(0, dtype=int))
            else:
                # Receive each ghost from its owning rank:
                recv.append(
                    ghosts[(ghosts >= self.offset[q]) & (ghosts < self.offset[q + 1])]
                )
                # Send the owned cells that other ranks need:
                owned_q = np.arange(self.offset[q], self.offset[q + 1])
                ring2_q = ring(ring(owned_q, adj_edge), adj_vert)
                need_q = ring2_q[
                    np.where(
                        (ring2_q < self.offset[q]) | (ring2_q >= self.offset[q + 1])
                    )
                ]
                send.append(self.owned[np.isin(self.owned, need_q)])
        self.recv_index = torch.from_numpy(np.concatenate(recv)).to(rc.device)
        self.send_index = torch.from_numpy(np.concatenate(send)).to(rc.device)
        self.recv_counts = [len(idx) for idx in recv]
        self.send_counts = [len(idx) for idx in send]

    def exchange(self, u: torch.Tensor) -> None:
        """Fill this rank's ghost rows of ``u`` (K, Nk) with their owners' values."""
        send_buf = u[self.send_index].contiguous()
        recv_buf = torch.empty(
            (len(self.recv_index), u.shape[1]), dtype=u.dtype, device=u.device
        )
        dist.all_to_all_single(
            recv_buf, send_buf, self.recv_counts, self.send_counts, group=self.group
        )
        u[self.recv_index] = recv_buf


@dataclass
class _Contact:
    """One boundary contact. ``fixed`` holds a prescribed ghost; a feedback
    contact (``floating`` probe or ``current`` source) solves a scalar level
    each evaluation so its net current hits ``target`` (0 for floating)."""

    name: str
    idx: torch.Tensor  # boundary-edge indices of this contact
    cur: torch.Tensor  # (Nsel, Nk) outward number-flux operator
    kind: str = "fixed"
    ghost: Optional[torch.Tensor] = None  # fixed: prescribed exterior trace
    unit: Optional[torch.Tensor] = None  # feedback: ghost per unit level
    drift: Optional[torch.Tensor] = None  # feedback: prescribed drift part
    cur_out: Optional[torch.Tensor] = None  # feedback: outflow-only flux op
    den: float = 1.0  # feedback: inflow capacity / level
    base: float = 0.0  # feedback: drift inflow current
    target: float = 0.0  # feedback: desired net current
    level: float = 0.0  # feedback: last solved level
    cur_in: Optional[torch.Tensor] = None  # feedback(nonlinear): inflow-only flux op
    bn: Optional[torch.Tensor] = None  # feedback(nonlinear): boundary normals
    vD: float = 0.0  # feedback: contact drift velocity
    nonlinear: bool = False  # feedback: full-FD Newton (vs affine)
    hmu: float = 1e-3  # feedback(nonlinear): Newton FD step in dmu


class Geometry(TreeNode):
    """Cell-centered finite-volume geometry on an external triangle mesh."""

    def __init__(
        self,
        *,
        material: Material,
        mesh: Mesh | dict | None = None,
        contacts: dict[str, Optional[dict]],
        cfl: float = 0.4,
        epsilon_sq: float = 0.0,
        compile: bool = False,
        save_rho: bool = False,
        save_terms: bool = False,
        probe_interval: int = 0,
        probe_file: str = "",
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Parameters
        ----------
        mesh
            :yaml:`Specification of 2D or 1D mesh geometry for transport.`
        contacts
            :yaml:`Dictionary of contact names to parameters (match mesh markers).`
            Each value selects the contact kind: ``{dmu, vD}`` a fixed
            voltage/drift source, ``{floating: true}`` a zero-current probe, and
            ``{I_set: <current>}`` a current source (with optional ``vD``).
        cfl
            :yaml:`CFL number for the explicit step (dt = cfl * inradius / vmax).`
        epsilon_sq
            :yaml:`Flux-squared threshold on flux limiter.`
            Default of 0 yields a pure smooth limiter. Set to ~(mesh-scale variation)^2
            to not limit smooth variations and recover strict linearity preservation.
        compile
            :yaml:`torch.compile the limited reconstruction (fuses the per-step
            limiter kernels).` ~3x faster steps on GPU at the cost of a one-time
            compile; leave off for short runs and the test suite.
        """
        TreeNode.__init__(self)
        self.material = material
        self.add_child("mesh", Mesh, mesh, checkpoint_in)
        self.group = process_grid.get_group("r")
        self.contacts = contacts
        self.save_rho = save_rho
        self.save_terms = save_terms
        self.epsilon_sq = float(epsilon_sq)

        self._mpi = self.group.size() > 1
        bounds = self.mesh.partition_cells(self.group)
        geom = self.mesh.get_fv_geom(dtype=material.transport_velocity.dtype)
        self.geom = geom
        v = material.transport_velocity  # (Nk, 2)
        self.Nk = v.shape[0]
        self.K = int(geom.area.shape[0])
        self._nf = int(geom.recon.shape[1])  # faces/cell: 3 (tri) or 2 (1D)

        # Spatial decomposition: owned cell block, reconstruction rows (owned +
        # 1-ring), owned-incident edges and the halo exchange (see SpatialDecomp).
        if self._mpi:
            self._decomp = SpatialDecomp(self.mesh, bounds, self.group)
            self._own_start, self._own_stop = self._decomp.start, self._decomp.stop
            self._R = torch.as_tensor(
                self._decomp.recon_rows, device=rc.device, dtype=torch.long
            )
            self._owned_mask = torch.zeros(
                self.K, 1, dtype=torch.bool, device=rc.device
            )
            self._owned_mask[self._own_start : self._own_stop] = True
        else:
            self._decomp = None
            self._own_start, self._own_stop = 0, self.K
            self._R = None
            self._owned_mask = None

        n_cells_mine = self._own_stop - self._own_start
        self._faces_chunk_size = ceildiv(n_cells_mine * self._nf, geom.recon.shape[2])

        # Precompute velocity-weighted, area-scaled edge operators (constant):
        #   into eL: -a*elen/area_L,  into eR: +a*elen/area_R,  a = v_c.n
        a_int = geom.en @ v.t()  # (Ne, Nk)
        inv_area = 1 / geom.area
        self._maskL = a_int > 0
        self._wL = -a_int * (geom.elen * inv_area[geom.eL])[:, None]
        self._wR = a_int * (geom.elen * inv_area[geom.eR])[:, None]
        self._a_bnd = geom.bn @ v.t()  # (Nb, Nk)
        self._maskB = self._a_bnd > 0
        self._wB = -self._a_bnd * (geom.blen * inv_area[geom.bcell])[:, None]
        # Per-channel density weight; outward number-flux operator per boundary edge.
        self._ncoef = material.get_observables(0.0)[0]  # (Nk,)
        self._cur_b = (
            geom.blen[:, None] * self._ncoef[None, :]
        ) * self._a_bnd  # (Nb,Nk)

        # Face-flux output: fluxes (currents, heat) are emitted on edges as the
        # face-normal flux F = elen * sum_k u_face_k (v_k.n) g_k -- the SAME upwind
        # trace the scheme streams with, so F is the conserved flux (exactly 0
        # through specular walls).  Scalars stay cell-centred.  (Serial only.)
        self._flux_names = list(material.get_flux_names())
        self._flux_g = material.get_flux_weights()  # (Nflux, Nk) or None
        self._do_flux = (self._flux_g is not None) and (self._decomp is None)
        if self._do_flux:
            self._a_int = a_int  # (Ne, Nk) interior v.n
            eLc = geom.eL.detach().cpu().numpy()
            eRc = geom.eR.detach().cpu().numpy()
            bc = geom.bcell.detach().cpu().numpy()
            self._edge_cells = np.concatenate(
                [  # (n_edge, 2), eR=-1 on bnd
                    np.stack([eLc, eRc], 1),
                    np.stack([bc, -np.ones_like(bc)], 1),
                ]
            ).astype(np.int64)
            self._edge_normal = np.concatenate(
                [  # (n_edge, 2) out of first cell
                    geom.en.detach().cpu().numpy(),
                    geom.bn.detach().cpu().numpy(),
                ]
            )
            self._edge_len = np.concatenate(
                [  # (n_edge,)
                    geom.elen.detach().cpu().numpy(),
                    geom.blen.detach().cpu().numpy(),
                ]
            )
            self._stash_flux = []

        # Restrict per-step work to cells/edges this rank owns (all of them serially).
        lo, hi = self._own_start, self._own_stop
        if self._mpi:
            own_e = ((geom.eL >= lo) & (geom.eL < hi)) | (
                (geom.eR >= lo) & (geom.eR < hi)
            )
            self._eloc = torch.where(own_e)[0]
            self._bloc = torch.where((geom.bcell >= lo) & (geom.bcell < hi))[0]
        else:
            self._eloc = None
            self._bloc = None
        self._setup_boundary(material)

        # Following should be made consistent with the existing observables mechanism
        # ---- dense in-run probe (every probe_interval steps) -------------
        # dt_save is sized by checkpoint cost (rho is ~0.4 GB/frame), so the
        # only V(t) / I(t) record a CLI run leaves behind is one point per
        # checkpoint.  That is far too sparse to watch a transient relax.  This
        # writes a small CSV every probe_interval steps instead: per-contact
        # currents plus the mean chemical potential over each named region.
        # Cost is bounded by evaluating get_cell_scalars on the PROBE CELLS
        # ONLY (the per-cell frame Newton is the expensive part), so a typical
        # 4-region mixer probe every 10 steps is ~1% of a collision step.
        self.probe_interval = int(probe_interval)
        self.probe_file = str(probe_file)
        self._probe_regions = {}
        self._probe_idx = None
        self._probe_header = False
        if self.probe_interval > 0:
            names = self.mesh.cell_regions
            if names is None:
                raise InvalidInputException(
                    f"probe_interval > 0 but {self.mesh_file} defines no"
                    " cell_regions; label the probe cells in the mesh"
                    " generator (regions are geometry, not run config)"
                )
            lo, hi = self._own_start, self._own_stop
            local = np.asarray(names)[lo:hi]
            uniq = sorted({n for n in local.tolist() if n})
            sel_all = np.zeros(hi - lo, dtype=bool)
            for n in uniq:
                m = local == n
                self._probe_regions[n] = m
                sel_all |= m
            idx = np.flatnonzero(sel_all)
            # self._u does not exist yet at this point in __init__; realise the
            # index tensor lazily on the first probe instead.
            self._probe_idx = idx
            pos = {int(g): i for i, g in enumerate(idx)}
            self._probe_regions = {
                n: np.array([pos[int(g)] for g in np.flatnonzero(m)], dtype=int)
                for n, m in self._probe_regions.items()
            }
            log.info(
                f"Probing regions {uniq} ({len(idx)} cells) every "
                f"{self.probe_interval} steps -> {self.probe_file}"
            )

        # Operator-splitting support: TimeEvolution suppresses the collision
        # during the streaming stages when collision_interval > 1, and applies
        # it separately over the longer sub-step via collision_dot below.
        self.collision_enabled = True
        rm = getattr(material, "rates_modal", None)
        ks = getattr(material, "k_speed", 0.0)
        has_ee = hasattr(
            material, "ee_scattering"
        )  # microscopic e-e: NOT in rates_modal
        self._skip_collision = bool(
            rm is not None
            and float(rm.abs().sum()) == 0.0
            and float(ks) == 0.0
            and not has_ee
        )

        # Compile the face limiter which involves several elemntwise operations:
        if compile:
            mode = os.environ.get("QIMPY_COMPILE_MODE", "max-autotune")
            if mode == "jit-script":
                self._face_limiter = torch.jit.script(_face_limiter)
            else:
                self._face_limiter = torch.compile(_face_limiter, mode=mode)
        else:
            self._face_limiter = _face_limiter  # eager mode

        vmax = float(v.norm(dim=1).max())
        dt_local = float(cfl) * float(geom.inradius.min()) / max(vmax, 1e-300)
        self.dt_max = all_reduce_scalars(dt_local, dist.ReduceOp.MIN, self.group)

        rho0 = getattr(material, "rho0", None)
        if (self._decomp is None) and checkpoint_in and checkpoint_in.member("rho"):
            cp, path = checkpoint_in  # warm start (serial)
            self._u = torch.as_tensor(
                np.array(cp[f"{path}/rho"]), device=rc.device, dtype=v.dtype
            )
            if tuple(self._u.shape) != (self.K, self.Nk):
                raise InvalidInputException(
                    f"checkpoint rho shape {tuple(self._u.shape)} != (K, Nk) ="
                    f" ({self.K}, {self.Nk}): k-representation config (n_k /"
                    f" circular / annulus_xi) must match the checkpoint"
                )
        elif rho0 is not None:
            self._u = rho0.flatten().to(rc.device, v.dtype)[None, :].repeat(self.K, 1)
        else:
            self._u = torch.zeros(self.K, self.Nk, device=rc.device, dtype=v.dtype)
        self._stash_t, self._stash_i, self._stash_obs = [], [], []
        self._stash_terms = []  # per-frame (4, K_own, Nr*dim): [a, lin, quad, cub]
        self.material.initialize_fields(self._u, {}, id(self))

    def _setup_boundary(self, material: Material) -> None:
        """Group boundary edges by marker into a wall (reflector) set and one
        contact object per parametrized marker; unparametrized markers reflect.

        Under decomposition each rank keeps only the boundary edges on its owned
        cells; a feedback contact's capacity (``den``/``base``) is summed across
        ranks so every rank solves the same global level."""
        g = self.geom
        names = self.mesh.boundary_names
        name_of = [
            names[m] if 0 <= m < len(names) else "wall" for m in g.bmark.tolist()
        ]
        lo, hi = self._own_start, self._own_stop
        bcell = g.bcell.tolist()
        owns = [lo <= bcell[i] < hi for i in range(len(name_of))]  # this rank's edges
        is_c = lambda nm: (nm in self.contacts) and (self.contacts[nm] is not None)
        wall = np.array(
            [i for i, nm in enumerate(name_of) if owns[i] and not is_c(nm)], int
        )
        self._wall = torch.as_tensor(wall, device=rc.device, dtype=torch.long)
        self._reflector = (
            material.get_reflector(g.bn[self._wall]) if wall.size else None
        )

        def allreduce(x: float) -> float:
            return (
                all_reduce_scalars(x, dist.ReduceOp.SUM, self.group) if self._mpi else x
            )

        self._contacts: list[_Contact] = []
        for nm, params in self.contacts.items():
            if params is None:
                continue
            sel = np.array(
                [i for i, x in enumerate(name_of) if x == nm and owns[i]], int
            )
            ci = torch.as_tensor(sel, device=rc.device, dtype=torch.long)
            cur = self._cur_b[ci]  # (Nsel, Nk) (maybe empty)
            params = dict(params)
            floating = bool(params.pop("floating", False))
            i_set = params.pop("I_set", None)
            # nonlinear: for feedback (I_set/floating) contacts it selects the
            # Newton-on-exact-FD level solve; for fixed contacts it passes
            # through to the representation's contactor (exact reservoir vs its
            # linearization).  None = each contactor's own default.
            nl_key = params.pop("nonlinear", None)
            nonlinear = bool(nl_key)
            vD = float(params.get("vD", 0.0))
            if floating or (i_set is not None):
                # Feedback ghost is affine in dmu: g(dmu) = dmu*unit + drift, so the
                # unit is d(ghost)/d(dmu).  Take it as the LINEAR slope from a small
                # dmu: exact for the (linear-in-dmu) delta-k contactor, and the correct
                # df0/dmu for the Cartesian full-f Fermi-Dirac -- where dmu=1 would
                # saturate the FD and inject current into the inert empty high-energy
                # tail instead of the Fermi surface (no drive at all).
                bn_ci = g.bn[ci]
                eps = 0.1 * float(getattr(material, "T_temp", 1.0))
                unit = material.get_contactor(bn_ci, dmu=eps)(0.0) / eps
                drift = (
                    material.get_contactor(bn_ci, vD=vD)(0.0)
                    if vD
                    else torch.zeros_like(unit)
                )
                cur_in = torch.where(self._a_bnd[ci] < 0, cur, torch.zeros_like(cur))
                cur_out = torch.where(self._a_bnd[ci] > 0, cur, torch.zeros_like(cur))
                den = allreduce(float(-(cur_in * unit).sum()))  # global inflow capacity
                base = allreduce(float((cur_in * drift).sum()))  # global drift inflow
                self._contacts.append(
                    _Contact(
                        name=nm,
                        idx=ci,
                        cur=cur,
                        kind="current",
                        unit=unit,
                        drift=drift,
                        cur_out=cur_out,
                        den=(den if abs(den) > 1e-300 else 1e-300),
                        base=base,
                        target=(0.0 if floating else float(i_set)),
                        cur_in=cur_in,
                        bn=bn_ci,
                        vD=vD,
                        nonlinear=nonlinear,
                        hmu=0.01 * float(getattr(material, "T_temp", 1.0)),
                    )
                )
            else:
                if nl_key is not None:
                    params["nonlinear"] = bool(nl_key)
                ghost = material.get_contactor(g.bn[ci], **params)(0.0)
                self._contacts.append(
                    _Contact(name=nm, idx=ci, cur=cur, kind="fixed", ghost=ghost)
                )

    def _faces(self, u: torch.Tensor) -> torch.Tensor:
        """Reconstructed face values, (K, n_face, Nk). Serial reconstructs every
        cell; under decomposition only the rows this rank needs (owned + 1-ring)
        are filled, the rest left zero (their faces are never read).
        Chunked over cells with `_faces_chunk_size`."""
        g = self.geom
        rows = self._R  # None (serial) or owned+1ring
        idx = torch.arange(self.K, device=u.device) if rows is None else rows
        uf = u.new_zeros(self.K, self._nf, self.Nk)
        chunk = self._faces_chunk_size
        for lo in range(0, len(idx), chunk):
            ci = idx[lo : lo + chunk]
            uf[ci] = self._face_limiter(
                u[ci], u, g.nbr[ci], g.recon[ci], self.epsilon_sq
            )
        return uf

    def _exterior(self, uMb: torch.Tensor, t: float) -> torch.Tensor:
        """Exterior ghost at boundary edges: reflector on walls, prescribed or
        feedback-solved contactor on contacts. Feedback contacts solve a scalar
        level so the net current equals their target; only inflow channels are
        consumed downstream by the upwind flux."""
        uP = uMb.clone()
        if self._reflector is not None:
            uP[self._wall] = self._reflector(uMb[self._wall][None])[0]
        for c in self._contacts:
            if c.kind == "fixed":
                uP[c.idx] = c.ghost.to(uP)
            else:
                # Feedback: solve the contact level so the net current hits target.
                # I_net = num_out + inflow(ghost); num_out is the interior outflow.
                num = float((c.cur_out * uMb[c.idx]).sum())
                if self._mpi:
                    num = all_reduce_scalars(num, dist.ReduceOp.SUM, self.group)
                if c.nonlinear:
                    # Full physical contact: ghost is the exact Pauli-bounded Fermi-
                    # Dirac deviation FD(mu0+dmu)-f0, NONLINEAR in dmu.  Newton-solve
                    # dmu (warm-started from last step) so I_net(dmu)=target -- captures
                    # the contact nonlinearity the affine linear-response feedback drops.
                    dmu = c.level
                    for _ in range(8):
                        inflow = self.material.get_contactor(c.bn, dmu=dmu, vD=c.vD)(
                            0.0
                        )
                        i0 = float((c.cur_in * inflow).sum())
                        gh = self.material.get_contactor(
                            c.bn, dmu=dmu + c.hmu, vD=c.vD
                        )(0.0)
                        ih = float((c.cur_in * gh).sum())
                        if self._mpi:
                            i0 = all_reduce_scalars(i0, dist.ReduceOp.SUM, self.group)
                            ih = all_reduce_scalars(ih, dist.ReduceOp.SUM, self.group)
                        dIdmu = (ih - i0) / c.hmu
                        if abs(dIdmu) < 1e-300:
                            break
                        d = (num + i0 - c.target) / dIdmu
                        dmu = dmu - d
                        if abs(d) <= 1e-6 * (abs(dmu) + c.hmu):
                            break
                    c.level = dmu
                    uP[c.idx] = self.material.get_contactor(c.bn, dmu=dmu, vD=c.vD)(
                        0.0
                    ).to(uP)
                else:
                    c.level = (num + c.base - c.target) / c.den
                    uP[c.idx] = (c.level * c.unit + c.drift).to(uP)
        return uP

    def _spatial_rhs(self, u: torch.Tensor, t: float) -> torch.Tensor:
        g = self.geom
        self.material.constrain(u)
        uf = self._faces(u).reshape(-1, self.Nk)  # (K*3, Nk)
        dudt = torch.zeros_like(u)
        e = self._eloc  # owned-incident edges
        eL = g.eL if e is None else g.eL[e]
        eR = g.eR if e is None else g.eR[e]
        eLF = g.eLF if e is None else g.eLF[e]
        eRF = g.eRF if e is None else g.eRF[e]
        maskL = self._maskL if e is None else self._maskL[e]
        wL = self._wL if e is None else self._wL[e]
        wR = self._wR if e is None else self._wR[e]
        uup = torch.where(maskL, uf[eLF], uf[eRF])  # interior upwind trace
        dudt.index_add_(0, eL, wL * uup)
        dudt.index_add_(0, eR, wR * uup)
        if g.bcell.numel():
            uMb = uf[g.bF]
            uup_b = torch.where(self._maskB, uMb, self._exterior(uMb, t))
            wbu = self._wB * uup_b
            if self._bloc is None:
                dudt.index_add_(0, g.bcell, wbu)
            else:
                b = self._bloc
                dudt.index_add_(0, g.bcell[b], wbu[b])
        return dudt

    def rho_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        watch = StopWatch("Geometry.rho_dot")
        if self._decomp is not None:
            self._decomp.exchange(rho)  # fill halo ghost rows
        out = self._spatial_rhs(rho, t)
        watch.stop()
        if self.collision_enabled and not self._skip_collision:
            lo, hi = self._own_start, self._own_stop
            out[lo:hi] = out[lo:hi] + self.material.rho_dot(rho[lo:hi], t, id(self))
        if self._owned_mask is not None:
            out = out * self._owned_mask
        return out

    def collision_dot(self, rho: torch.Tensor) -> torch.Tensor:
        """The material collision term ALONE, with no spatial RHS.

        Used by operator splitting (``collision_interval > 1``).  The collision
        is local to a cell, so unlike :meth:`rho_dot` this needs no halo exchange.
        """
        out = torch.zeros_like(rho)
        if not self._skip_collision:
            lo, hi = self._own_start, self._own_stop
            out[lo:hi] = self.material.rho_dot(rho[lo:hi], 0.0, id(self))
        if self._owned_mask is not None:
            out = out * self._owned_mask
        return out

    @property
    def rho(self) -> torch.Tensor:
        return self._u

    @rho.setter
    def rho(self, rho_new: torch.Tensor) -> None:
        self._u = rho_new
        self.material.constrain(self._u)

    @property
    def density(self) -> torch.Tensor:
        return self._u

    # ---- contact diagnostics ----
    def contact_currents(self, t: float = 0.0) -> dict[str, float]:
        """Net outward number-current through each contact (positive = out of the
        device). Floating probes read ~0; current sources read their I_set; fixed
        contacts read a response. Sum over all boundaries = -d/dt of mass."""
        if self._decomp is not None:
            self._decomp.exchange(self._u)
        uf = self._faces(self._u).reshape(-1, self.Nk)
        uMb = uf[self.geom.bF]
        uup_b = torch.where(self._maskB, uMb, self._exterior(uMb, t))
        out = {}
        for c in self._contacts:
            I = float((c.cur * uup_b[c.idx]).sum())  # this rank's edges
            out[c.name] = (
                all_reduce_scalars(I, dist.ReduceOp.SUM, self.group) if self._mpi else I
            )
        return out

    def maybe_probe(self, i_step: int, t: float) -> None:
        """Append one probe row every ``probe_interval`` steps (else a no-op)."""
        if self.probe_interval <= 0 or self._probe_idx is None:
            return
        if i_step % self.probe_interval:
            return
        if not torch.is_tensor(self._probe_idx):
            self._probe_idx = torch.as_tensor(
                np.asarray(self._probe_idx), dtype=torch.long, device=self._u.device
            )
        with torch.no_grad():
            cur = self.contact_currents(t)
            names = self.material.get_cell_scalar_names()
            i_mu = names.index("chemical_potential")
            sc = (
                self.material.get_cell_scalars(self._u[self._probe_idx], t)[:, i_mu]
                .detach()
                .cpu()
                .numpy()
            )
        row = {"i_step": i_step, "t": t}
        row.update({f"I_{k}": v for k, v in cur.items()})
        for name, sel in self._probe_regions.items():
            if self._mpi:
                tot = all_reduce_scalars(
                    float(sc[sel].sum()), dist.ReduceOp.SUM, self.group
                )
                cnt = all_reduce_scalars(int(sel.size), dist.ReduceOp.SUM, self.group)
            else:
                tot, cnt = float(sc[sel].sum()), int(sel.size)
            row[f"V_{name}"] = tot / max(cnt, 1)
        if self._mpi and dist.get_rank(self.group):
            return
        new = not self._probe_header
        with open(self.probe_file, "a") as fh:
            if new:
                fh.write(",".join(row.keys()) + "\n")
                self._probe_header = True
            fh.write(
                ",".join(
                    f"{v!r}" if isinstance(v, int) else f"{v:.10e}"
                    for v in row.values()
                )
                + "\n"
            )

    def contact_potentials(self) -> dict[str, float]:
        """Self-adjusting level of each feedback contact, from the last evaluation."""
        return {c.name: c.level for c in self._contacts if c.kind != "fixed"}

    def update_stash(self, i_step: int, t: float) -> None:
        ce = getattr(
            getattr(self.material, "representation", None), "check_envelope", None
        )
        if ce is not None:
            ce(self._u[self._own_start : self._own_stop])
        # Stash cell-centred scalar fields for this rank's owned cells.
        u_own = self._u[self._own_start : self._own_stop]
        obs = self.material.get_cell_scalars(u_own, t)
        self._stash_i.append(i_step)
        self._stash_t.append(t)
        self._stash_obs.append(obs.detach().cpu().numpy())
        if self._do_flux:
            # Face-normal fluxes on every edge, from the upwind face trace.
            uf = self._faces(self._u).reshape(-1, self.Nk)
            g = self.geom
            uup_i = torch.where(self._maskL, uf[g.eLF], uf[g.eRF])  # (Ne, Nk)
            uMb = uf[g.bF]
            uup_b = torch.where(self._maskB, uMb, self._exterior(uMb, t))  # (Nb, Nk)
            Fi = g.elen[None, :] * (
                self._flux_g @ (uup_i * self._a_int).t()
            )  # (Nflux, Ne)
            Fb = g.blen[None, :] * (
                self._flux_g @ (uup_b * self._a_bnd).t()
            )  # (Nflux, Nb)
            F = torch.cat([Fi, Fb], dim=1).t()  # (n_edge, Nflux)
            self._stash_flux.append(F.detach().cpu().numpy())
        if self.save_terms and hasattr(self.material, "ee_scattering"):
            # Per-(m,l) collision breakdown, computed WARM in the evolution loop
            # (the same context as the in-run apply) to dodge the cold
            # standalone-apply slow path. a is the modal distribution; lin/quad/
            # cub are the linear, quadratic-in-deltaf and cubic-in-deltaf parts
            # of the e-e operator. All are modal, flattenable to (Nr, dim).
            with torch.no_grad():
                a = self.material.to_modes(u_own)  # (K_own, Nr*dim)
                lin, quad, cub = self.material.ee_scattering.a_dot_breakdown(a)
                self._stash_terms.append(
                    np.stack(
                        [
                            a.detach().cpu().numpy(),
                            lin.detach().cpu().numpy(),
                            quad.detach().cpu().numpy(),
                            cub.detach().cpu().numpy(),
                        ],
                        axis=0,
                    )
                )  # (4, K_own, Nr*dim)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        g = self.geom
        names = self.material.get_cell_scalar_names()
        # NOTE: checkpoint attrs are fed back as constructor kwargs on restart
        # (qimpy convention: attrs == constructor params), so only real
        # constructor arguments may be written here.
        saved = [
            cp_path.write("cell_centroid", torch.from_numpy(g.centroid_np)),
            "fv_observables",
        ]
        cp_path.write_str("contact_names", ",".join(self.contacts.keys()))
        cp_path.write_str("observable_names", ",".join(names))
        cp_path["t"] = np.array(self._stash_t)
        cp_path["i_step"] = np.array(self._stash_i)
        checkpoint, path = cp_path
        n_stash = len(self._stash_t)
        # Collective: every rank creates the global dataset, then writes the
        # slice of cells it owns ([own_start, own_stop)). Serially this is the
        # whole array; under MPI it is each rank's contiguous block (needs an
        # mpio-enabled h5py to run multi-rank).
        CheckpointPath(checkpoint, path).create_dataset(
            "fv_observables", (n_stash, self.K, len(names)), np.float64
        )
        if checkpoint is not None and n_stash:
            checkpoint.write_slice(
                checkpoint[f"{path}/fv_observables"],
                (0, self._own_start, 0),
                torch.from_numpy(np.stack(self._stash_obs)),
            )
        if self._do_flux and n_stash:
            # Flux observables live on edges: geometry + face-normal flux per frame.
            saved += [
                cp_path.write("edge_cells", torch.from_numpy(self._edge_cells)),
                cp_path.write("edge_normal", torch.from_numpy(self._edge_normal)),
                cp_path.write("edge_len", torch.from_numpy(self._edge_len)),
                "fv_edge_flux",
            ]
            cp_path.write_str("flux_names", ",".join(self._flux_names))
            n_edge = self._edge_len.shape[0]
            CheckpointPath(checkpoint, path).create_dataset(
                "fv_edge_flux", (n_stash, n_edge, len(self._flux_names)), np.float64
            )
            if checkpoint is not None:
                checkpoint.write_slice(
                    checkpoint[f"{path}/fv_edge_flux"],
                    (0, 0, 0),
                    torch.from_numpy(np.stack(self._stash_flux)),
                )
        if self.save_rho:
            # Raw per-cell state (n_cells, n_channels), for exact restart /
            # steady-state warm start. Each rank writes its owned cell block.
            u_own = self._u[self._own_start : self._own_stop].detach().cpu().numpy()
            CheckpointPath(checkpoint, path).create_dataset(
                "rho", (self.K, self.Nk), u_own.dtype
            )
            if checkpoint is not None:
                checkpoint.write_slice(
                    checkpoint[f"{path}/rho"],
                    (self._own_start, 0),
                    torch.from_numpy(u_own),
                )
            saved.append("rho")
        if self.save_terms and len(self._stash_terms):
            # Per-(m,l) collision breakdown stacked over saved frames:
            # (n_stash, 4, K, Nr*dim) with channel order [a, lin, quad, cub].
            # Reshape the last axis to (Nr, dim) offline for the radial/angular
            # (l, m) contributions. dim = 2*M_theta+1 (m = -M..M).
            fs = self.material
            Nr_m = int(fs.Nr)
            dim_m = int(fs.angular.dim)
            n_modal = Nr_m * dim_m
            # (Nr, dim) as a dataset, NOT attrs -- attrs round-trip into
            # constructor kwargs on restart and these are not constructor args.
            cp_path.write("terms_shape", torch.tensor([Nr_m, dim_m], dtype=torch.int64))
            cp_path.write_str("terms_channels", "a,lin,quad,cub")
            terms_own = np.stack(self._stash_terms)  # (n_stash,4,K_own,n_modal)
            CheckpointPath(checkpoint, path).create_dataset(
                "fv_terms", (n_stash, 4, self.K, n_modal), terms_own.dtype
            )
            if checkpoint is not None:
                checkpoint.write_slice(
                    checkpoint[f"{path}/fv_terms"],
                    (0, 0, self._own_start, 0),
                    torch.from_numpy(terms_own),
                )
            saved.append("fv_terms")
        self._stash_t, self._stash_i, self._stash_obs = [], [], []
        self._stash_terms = []
        if self._do_flux:
            self._stash_flux = []
        return saved


def _face_limiter(
    uc: torch.Tensor,
    u: torch.Tensor,
    nbr: torch.Tensor,
    recon: torch.Tensor,
    epsilon_sq: float,
) -> torch.Tensor:
    """Venkatakrishnan-limited face values for a subset of cells with values `uc`,
    where `u` is the full data, `nbr` is the neighbor list and `recon` is the
    fused gradient->face operator. Limiter function applied per face is:
        phi = (D1^2 + 2 D1 d + epsilon^2) / (D1^2 + D1 d + 2 d^2 + epsilon^2),
    where d = u_face - u, and D1 is the same-sign headroom to the u differences."""
    du = u[nbr] - uc[:, None]
    d = recon @ du
    # Compute limiter (in [0, 1]) for the gradients on each face:
    lo, hi = du.aminmax(dim=1, keepdim=True)
    lo.clip_(max=0.0)
    hi.clip_(min=0.0)
    D1 = torch.where(d >= 0, hi, lo)
    common = D1.square() + epsilon_sq
    dD1 = d * D1
    phi = (common + 2 * dD1) / (common + dD1 + 2 * d.square())
    phi.nan_to_num_(1.0)  # Handle possible NANs from d underflow
    phi.clamp_(max=1.0)
    limiter = phi.amin(dim=1, keepdim=True)
    return uc[:, None] + limiter * d
