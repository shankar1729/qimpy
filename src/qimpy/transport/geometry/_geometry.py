from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc, TreeNode
from qimpy.io import CheckpointPath, InvalidInputException, CheckpointContext
from qimpy.mpi import ProcessGrid, all_reduce_scalars, get_block_slices
from qimpy.profiler import StopWatch, stopwatch
from ..material import Material
from . import Mesh


class Geometry(TreeNode):
    """Cell-centered finite-volume geometry on a 1D/2D `Mesh`."""

    def __init__(
        self,
        *,
        material: Material,
        mesh: Mesh | dict | None = None,
        contacts: dict[str, dict],
        cfl: float = 0.4,
        epsilon_sq: float = 0.0,
        faces_block_size: int = 0,
        compile: bool = False,
        save_rho: bool = False,
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
        faces_block_size
            :yaml:`Number of faces to process per block in the face flux limiter.`
            By default, process all the faces together, which is fastest on large-enough
            GPUs, but consumes more memory. Set to a smaller chunk size to reduce memory
            footprint, or to speed up on CPUs, where that may help with memory locality.
        compile
            :yaml:`Whether to compile the face flux limiter kernel.`
            Speeds up sequential elemntise operations by kernel fusion when possible.
        """
        TreeNode.__init__(self)
        self.material = material
        self.add_child("mesh", Mesh, mesh, checkpoint_in)
        self.group = process_grid.get_group("r")
        self.save_rho = save_rho
        self.epsilon_sq = float(epsilon_sq)
        self.faces_block_size = faces_block_size

        self._mpi = self.group.size() > 1
        self.cell_division = self.mesh.partition_cells(self.group)
        v = material.transport_velocity  # (Nk, 2)
        geom = self.mesh.get_fv_geom(self.cell_division, dtype=v.dtype)
        self.geom = geom
        self.Nk = v.shape[0]
        self.K = int(geom.area.shape[0])  # local cells including padding
        self._nf = int(geom.recon.shape[1])  # faces/cell: 3 (tri) or 2 (1D)

        # Precompute velocity-weighted, area-scaled edge operators (constant):
        #   into eL: -a*elen/area_L,  into eR: +a*elen/area_R,  a = v_c.n
        a_int = geom.en @ v.t()  # (Ne, Nk)
        inv_area = 1 / geom.area
        maskL = a_int > 0
        wL = -a_int * (geom.elen * inv_area[geom.eL])[:, None]
        wR = a_int * (geom.elen * inv_area[geom.eR])[:, None]
        ik = torch.arange(self.Nk, device=rc.device)
        idx_in = self.Nk * torch.where(maskL, geom.eLF[:, None], geom.eRF[:, None]) + ik
        idx_out = self.Nk * torch.cat((geom.eL, geom.eR))[:, None] + ik
        n_recon = self.cell_division.n_recon
        self.interior_advect_mat = torch.sparse_coo_tensor(
            indices=torch.stack((idx_out.flatten(), idx_in.flatten().tile(2))),
            values=torch.cat((wL.flatten(), wR.flatten())),
            size=(len(geom.area) * self.Nk, n_recon * self._nf * self.Nk),
        ).to_sparse_csr()

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
        self._flux_indices = material.get_flux_indices()
        self._do_flux = self._flux_indices and (self.cell_division is None)
        if self._do_flux:
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

        self._setup_boundary(material, contacts)
        self.collision_enabled = True

        # Set up cell region (interior) probes:
        n_mine = self.cell_division.n_mine
        w_cells = geom.area[:n_mine]
        i_region_all = self.mesh.cells[:, -1]
        assert i_region_all.min() >= -1
        n_regions = int(i_region_all.max()) + 1
        i_region = torch.from_numpy(
            i_region_all[self.cell_division.global_index[:n_mine]]
        ).to(rc.device)
        probe_cells = torch.where(i_region >= 0)[0]
        i_region = i_region[probe_cells]
        w_cells = w_cells[probe_cells]
        w_tot = torch.zeros(n_regions, dtype=w_cells.dtype, device=rc.device)
        w_tot.index_add_(0, i_region, w_cells)
        dist.all_reduce(w_tot, group=self.group)
        n_probe = len(probe_cells)
        self.probe_cells = probe_cells
        self.probe_mat_cell = torch.sparse_coo_tensor(
            indices=torch.stack((i_region, torch.arange(n_probe, device=rc.device))),
            values=w_cells / w_tot[i_region],
            size=(n_regions, n_probe),
        ).to_sparse_csr()  # multiply by per-cell observables to get region averages

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
        if (
            (self.cell_division is None)
            and checkpoint_in
            and checkpoint_in.member("rho")
        ):
            cp, path = checkpoint_in  # warm start (serial)
            self._u = torch.as_tensor(
                np.array(cp[f"{path}/rho"]), device=rc.device, dtype=v.dtype
            )
            if tuple(self._u.shape) != (self.K, self.Nk):
                raise InvalidInputException(
                    f"checkpoint rho shape {tuple(self._u.shape)} does not"
                    f" match({self.K}, {self.Nk}): k-representation config (n_k /"
                    f" circular / annulus_xi) must match the checkpoint"
                )
        elif rho0 is not None:
            self._u = rho0.flatten().to(rc.device, v.dtype)[None, :].repeat(self.K, 1)
        else:
            self._u = torch.zeros(self.K, self.Nk, device=rc.device, dtype=v.dtype)
        self.stash = Geometry.Stash()
        self.stash_probe = Geometry.Stash()  # coarse-grained, more frequent version
        self.material.initialize_fields(self._u, {}, id(self))

    @dataclass
    class Contact:
        """Contact boundary conditions, along with the edges they apply to."""

        indices: torch.Tensor  #: boundary-edge indices of this contact
        flux_operator: torch.Tensor  #: outward number-flux operator (len(indices), Nk)
        contactor: Callable[[float], torch.Tensor]  #: material-specific contact

    def _setup_boundary(self, material: Material, contacts: dict[str, dict]) -> None:
        """Group boundary edges by marker into a wall (reflector) set and one
        contact object per parametrized marker; unparametrized markers reflect."""
        g = self.geom
        names = self.mesh.boundary_names
        edge_used = torch.full(
            (len(g.bmark),), False, device=rc.device, dtype=torch.bool
        )
        self.contacts: dict[str, Geometry.Contact] = {}
        for name, params in contacts.items():
            if params is None:
                continue
            if name not in names:
                continue
            indices = torch.where(g.bmark == names.index(name))[0]
            n = g.bn[indices]
            flux_operator = self._cur_b[indices]
            self.contacts[name] = Geometry.Contact(
                indices, flux_operator, material.get_contactor(n, **params)
            )
            edge_used[indices] = True

        # Mark all boundary edges not in a contact as reflecting:
        self._wall = torch.where(~edge_used)[0]
        if len(self._wall):
            self._reflector = material.get_reflector(g.bn[self._wall])
        else:
            self._reflector = None

    def _faces(self, u: torch.Tensor) -> torch.Tensor:
        """Reconstruct face values, (K, n_face, Nk)."""
        g = self.geom
        n_recon = self.cell_division.n_recon
        uf = u.new_zeros(n_recon, self._nf, self.Nk)
        block_size = self.faces_block_size if self.faces_block_size else n_recon
        for sel in get_block_slices(n_recon, block_size):
            uf[sel] = self._face_limiter(
                u[sel], u, g.nbr[sel], g.recon[sel], self.epsilon_sq
            )
        return uf

    def _spatial_rhs(self, u: torch.Tensor, t: float) -> torch.Tensor:
        g = self.geom
        self.material.constrain(u)
        uf = self._faces(u).reshape(-1, self.Nk)  # (K*3, Nk)
        dudt = (self.interior_advect_mat @ uf.view(-1)).view(-1, self.Nk)
        if g.bcell.numel():
            ub = uf[g.bF]
            # Apply reflections and contacts:
            ub_in = ub.detach().clone()
            if self._reflector is not None:
                ub_in[self._wall] = self._reflector(ub[self._wall])
            for contact in self.contacts.values():
                ub_in[contact.indices] = contact.contactor(t)
            # Propagate incoming modes from boundary to relevant cells:
            dudt.index_add_(0, g.bcell, self._wB * torch.where(self._maskB, ub, ub_in))
        return dudt

    def rho_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        watch = StopWatch("Geometry.rho_dot")
        if self.cell_division is not None:
            self.cell_division.exchange(rho)  # update ghost regions
        out = self._spatial_rhs(rho, t)
        watch.stop()
        n_mine = self.cell_division.n_mine
        if self.collision_enabled:
            out[:n_mine] += self.material.rho_dot(rho[:n_mine], t, id(self))
        out[n_mine:].zero_()  # mask out ghost zhones
        return out

    def collision_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        """The material collision term ALONE, with no spatial RHS.
        Used by operator splitting (``collision_interval > 1``)."""
        out = torch.zeros_like(rho)
        n_mine = self.cell_division.n_mine
        out[:n_mine] += self.material.rho_dot(rho[:n_mine], t, id(self))
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
        if self.cell_division is not None:
            self.cell_division.exchange(self._u)
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

    @dataclass
    class Stash:
        """Stash for probes and observables that are collated into checkpoints."""

        i_step: list[int] = field(default_factory=list)
        t: list[float] = field(default_factory=list)
        observables: list[torch.Tensor] = field(default_factory=list)
        flux: list[torch.Tensor] = field(default_factory=list)

    @stopwatch
    def update_probes(self, i_step: int, t: float) -> None:
        """Stash coarse-grained probes on defined cell regions and contacts."""
        self.stash_probe.i_step.append(i_step)
        self.stash_probe.t.append(t)
        # Interior cell regions:
        obserables = self.material.measure_observables(self._u[self.probe_cells], t)
        region_averages = (self.probe_mat_cell @ obserables).contiguous()
        dist.all_reduce(region_averages, group=self.group)
        self.stash_probe.observables.append(region_averages.detach().cpu())
        # TO DO: boundary region fluxes

    @stopwatch
    def update_observables(self, i_step: int, t: float) -> None:
        """Stash observables (and related) for output in next checkpoint."""
        u_own = self._u[: self.cell_division.n_mine]
        obserables = self.material.measure_observables(u_own, t)
        self.stash.i_step.append(i_step)
        self.stash.t.append(t)
        self.stash.observables.append(obserables.detach().cpu())
        if self._do_flux:
            # Face-normal fluxes on every edge, from the upwind face trace.
            uf = self._faces(self._u).reshape(-1, self.Nk)
            g = self.geom
            uup_i = torch.where(self._maskL, uf[g.eLF], uf[g.eRF])  # (Ne, Nk)
            uMb = uf[g.bF]
            uup_b = torch.where(self._maskB, uMb, self._exterior(uMb, t))  # (Nb, Nk)
            flux_g = self.material.get_observables(t)[self._flux_indices]
            Fi = g.elen[None, :] * (flux_g @ (uup_i * self._a_int).t())  # (Nflux, Ne)
            Fb = g.blen[None, :] * (flux_g @ (uup_b * self._a_bnd).t())  # (Nflux, Nb)
            F = torch.cat([Fi, Fb], dim=1).t()  # (n_edge, Nflux)
            self.stash.flux.append(F.detach().cpu())
        self.material.update_stash(u_own, self.cell_division)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        checkpoint, path = cp_path
        assert checkpoint is not None
        names = self.material.get_observable_names()
        saved = [
            cp_path.write("flux_indices", np.array(self._flux_indices)),
            cp_path.write_str("contact_names", ",".join(self.contacts.keys())),
            cp_path.write_str("observable_names", ",".join(names)),
            cp_path.write("t", np.array(self.stash.t)),
            cp_path.write("i_step", np.array(self.stash.i_step)),
            "observables",
            cp_path.write("t_probe", np.array(self.stash_probe.t)),
            cp_path.write("i_step_probe", np.array(self.stash_probe.i_step)),
            "probes",
        ]

        # Observables:
        n_stash = len(self.stash.t)
        div = self.cell_division
        dset = cp_path.create_dataset(
            "observables", (n_stash, div.n_tot, len(names)), np.float64
        )
        checkpoint.write_slice(
            dset, (0, div.i_start, 0), torch.stack(self.stash.observables)
        )
        self.stash = Geometry.Stash()  # reset now-written cached observables

        # Probes:
        cp_path.write("probes", torch.stack(self.stash_probe.observables))
        self.stash_probe = Geometry.Stash()  # reset now-written cached probes

        # TO DO: revamp this:
        if self._do_flux and n_stash:
            # Flux observables live on edges: geometry + face-normal flux per frame.
            saved += [
                cp_path.write("edge_cells", self._edge_cells),
                cp_path.write("edge_normal", self._edge_normal),
                cp_path.write("edge_len", self._edge_len),
                "edge_flux",
            ]
            n_edge = self._edge_len.shape[0]
            dset = cp_path.create_dataset(
                "edge_flux", (n_stash, n_edge, len(self._flux_indices)), np.float64
            )
            checkpoint.write_slice(dset, (0, 0, 0), torch.stack(self.stash.flux))

        if self.save_rho:
            # Optional full state output (large) for continuing calculations:
            u_own = self._u[: div.n_mine].detach()
            dset = cp_path.create_dataset("rho", (div.n_tot, self.Nk), u_own.dtype)
            checkpoint.write_slice(dset, (div.i_start, 0), u_own)
            saved.append("rho")

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
