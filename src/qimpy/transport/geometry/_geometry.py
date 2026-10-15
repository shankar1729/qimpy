from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc, log, TreeNode
from qimpy.io import CheckpointPath, InvalidInputException, CheckpointContext
from qimpy.mpi import ProcessGrid, all_reduce_scalars, get_block_slices
from qimpy.profiler import StopWatch
from ..material import Material
from . import Mesh


class Geometry(TreeNode):
    """Cell-centered finite-volume geometry on a 1D/2D `Mesh`."""

    def __init__(
        self,
        *,
        material: Material,
        mesh: Mesh | dict | None = None,
        contacts: dict[str, Optional[dict]],
        cfl: float = 0.4,
        epsilon_sq: float = 0.0,
        faces_block_size: int = 0,
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
        self.save_terms = save_terms
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
        self._flux_indices = material.get_flux_indices()
        self._do_flux = self._flux_indices and (self.cell_division is None)
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

        self._setup_boundary(material, contacts)
        self.collision_enabled = True

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
        self._stash_t, self._stash_i, self._stash_obs = [], [], []
        self._stash_terms = []  # per-frame (4, K_own, Nr*dim): [a, lin, quad, cub]
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
        dudt = torch.zeros_like(u)
        uup = torch.where(self._maskL, uf[g.eLF], uf[g.eRF])  # interior upwind trace
        dudt.index_add_(0, g.eL, self._wL * uup)
        dudt.index_add_(0, g.eR, self._wR * uup)
        if g.bcell.numel():
            ub = uf[g.bF]
            # Apply reflections and contacts:
            ub_in = ub.detach().clone()
            if self._reflector is not None:
                ub_in[self._wall] = self._reflector(ub[self._wall][None])[0]
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
        # Stash cell-centred scalar fields for this rank's owned cells.
        u_own = self._u[: self.cell_division.n_mine]
        obs = self.material.measure_observables(u_own, t)
        self._stash_i.append(i_step)
        self._stash_t.append(t)
        self._stash_obs.append(obs.detach().cpu())
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
            self._stash_flux.append(F.detach().cpu())
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
                    torch.stack(
                        [
                            a.detach().cpu(),
                            lin.detach().cpu(),
                            quad.detach().cpu(),
                            cub.detach().cpu(),
                        ],
                        dim=0,
                    )
                )  # (4, K_own, Nr*dim)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        checkpoint, path = cp_path
        assert checkpoint is not None
        names = self.material.get_observable_names()
        saved = [
            "observables",
            cp_path.write("flux_indices", np.array(self._flux_indices)),
            cp_path.write_str("contact_names", ",".join(self.contacts.keys())),
            cp_path.write_str("observable_names", ",".join(names)),
            cp_path.write("t", np.array(self._stash_t)),
            cp_path.write("i_step", np.array(self._stash_i)),
        ]
        n_stash = len(self._stash_t)
        dset = cp_path.create_dataset(
            "observables",
            (n_stash, self.cell_division.n_tot, len(names)),
            np.float64,
        )
        checkpoint.write_slice(
            dset, (0, self.cell_division.i_start, 0), torch.stack(self._stash_obs)
        )
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
            checkpoint.write_slice(dset, (0, 0, 0), torch.stack(self._stash_flux))
        if self.save_rho:
            # Raw per-cell state (n_cells, n_channels), for exact restart /
            # steady-state warm start. Each rank writes its owned cell block.
            u_own = self._u[: self.cell_division.n_mine].detach()
            dset = cp_path.create_dataset(
                "rho", (self.cell_division.n_tot, self.Nk), u_own.dtype
            )
            checkpoint.write_slice(dset, (self.cell_division.i_start, 0), u_own)
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
            terms_own = torch.stack(self._stash_terms)  # (n_stash,4,K_own,n_modal)
            dset = cp_path.create_dataset(
                "fv_terms",
                (n_stash, 4, self.cell_division.n_tot, n_modal),
                terms_own.dtype,
            )
            checkpoint.write_slice(
                dset, (0, 0, self.cell_division.i_start, 0), terms_own
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
