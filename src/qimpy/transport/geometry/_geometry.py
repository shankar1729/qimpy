from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc, log, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid, get_block_slices, globalreduce
from qimpy.profiler import StopWatch, stopwatch
from ..material import Material
from ..advect import limiter
from . import Mesh


class Geometry(TreeNode):
    """Cell-centered finite-volume geometry on a 1D/2D `Mesh`."""

    def __init__(
        self,
        *,
        material: Material,
        mesh: Mesh | dict | None = None,
        contacts: dict[str, dict],
        courant_number: float = 1.0,
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
        courant_number
            :yaml:`Courant number for selecting the maximum stable time step.`
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
        self.group = process_grid.get_group("r")
        self.material = material
        self.add_child("mesh", Mesh, mesh, checkpoint_in, group=self.group)
        self.save_rho = save_rho
        self.epsilon_sq = float(epsilon_sq)
        self.faces_block_size = faces_block_size

        # Initialize cell to face reconstruction operator:
        neighbors, reconstruct = self.mesh.get_face_reconstruction_op()
        self.neighbors = torch.from_numpy(neighbors).to(rc.device)
        self.reconstruct = torch.from_numpy(reconstruct).to(rc.device)
        if compile:
            mode = os.environ.get("QIMPY_COMPILE_MODE", "max-autotune")
            if mode == "jit-script":
                self._face_limiter = torch.jit.script(_face_limiter)
            else:
                self._face_limiter = torch.compile(_face_limiter, mode=mode)
        else:
            self._face_limiter = _face_limiter  # eager mode

        # Get properties of local faces and their mapping to cells:
        face_weights, face_map = self.mesh.get_face_weights()
        face_division = self.mesh.face_division
        boundary_weights = face_weights[: face_division.n_boundary_mine]
        interior_weights = face_weights[face_division.n_boundary_mine :]
        boundary_region = face_map[: face_division.n_boundary_mine, 1, 1]
        boundary_map = face_map[: face_division.n_boundary_mine, 0]
        interior_map = face_map[face_division.n_boundary_mine :].swapaxes(0, 1)
        # --- flattened indices by cell and face-within-cell:
        n_faces = self.mesh.faces.shape[1]
        boundary_cf = boundary_map[:, 0] * n_faces + boundary_map[:, 1]
        interior_cf = interior_map[..., 0] * n_faces + interior_map[..., 1]
        # --- further flattened indices including kbb dimension as well:
        v = material.transport_velocity.cpu().numpy()
        n_kbb_mine = v.shape[0]
        i_kbb = np.arange(n_kbb_mine)
        boundary_cfk = (i_kbb + n_kbb_mine * boundary_cf[:, None]).flatten()
        interior_cfk = (i_kbb + n_kbb_mine * interior_cf[..., None]).reshape(2, -1)
        self.n_faces = n_faces
        self.n_kbb_mine = n_kbb_mine

        # Initialize face to cell -v.del operator for advection:
        # Mask which cell's flux contributes to each interior face based on flux (per k)
        cell_weights = self.mesh.get_cell_weights()
        interior_flux = interior_weights @ v.T  # outward flux per face and kbb
        boundary_flux = boundary_weights @ v.T
        interior_cfk_sel = np.where(interior_flux.flatten() > 0, *interior_cfk)
        index_cfk = np.concatenate((interior_cfk_sel, interior_cfk_sel, boundary_cfk))
        index_c = np.concatenate((*interior_map[..., 0], boundary_map[:, 0]))
        index_ck = (i_kbb + n_kbb_mine * index_c[:, None]).flatten()
        values = (
            np.concatenate((-interior_flux, interior_flux, -boundary_flux))
            / cell_weights[index_c, None]
        ).flatten()
        cell_division = self.mesh.cell_division
        n_reconstruct = cell_division.n_reconstruct
        sel_mine = np.where(index_ck < cell_division.n_mine * n_kbb_mine)[0]
        self.minus_v_dot_del = torch.sparse_coo_tensor(
            indices=rc.from_np(np.stack((index_ck[sel_mine], index_cfk[sel_mine]))),
            values=rc.from_np(values[sel_mine]),
            size=(len(cell_weights) * n_kbb_mine, n_reconstruct * n_faces * n_kbb_mine),
        ).to_sparse_csr()
        rates = self.minus_v_dot_del.values().abs()
        self.dt_max = courant_number / globalreduce.max(rates, dist.group.WORLD).item()
        log.info(f"Set dt_max = {self.dt_max:.4g} for Courant number {courant_number}")

        # Set up boundary conditions:
        boundary_args = (
            rc.from_np(boundary_map),
            rc.from_np(boundary_region),
            rc.from_np(boundary_weights),
        )
        self.boundary = Boundary(self, contacts, *boundary_args)

        # Initialize flux collection on faces:
        self.flux_indices = material.get_flux_indices()
        if self.flux_indices:
            i_face_global = face_division.global_index[: face_division.n_mine]
            i_face_mine = i_face_global - face_division.i_start  # mine in global order

            # Construct flux matrix (selects appropriate direction and multiplies v):
            n_interior_fk = face_division.n_interior_mine * n_kbb_mine
            index_cfk = np.concatenate((boundary_cfk, interior_cfk_sel[:n_interior_fk]))
            index_fk = (i_kbb + n_kbb_mine * i_face_mine[:, None]).flatten()
            values = np.concatenate(
                (boundary_flux.flatten(), interior_flux.flatten()[:n_interior_fk])
            )
            self.v_multiply_select = torch.sparse_coo_tensor(
                indices=rc.from_np(np.stack((index_fk, index_cfk))),
                values=rc.from_np(values),
                size=(len(values), n_reconstruct * n_faces * n_kbb_mine),
            ).to_sparse_csr()

            # Set up boundary flux probes:
            i_boundary_all = self.mesh.boundary_faces[:, -1]
            assert i_boundary_all.min() >= -1
            n_boundaries = int(i_boundary_all.max()) + 1  # marked boundary regions
            probe_f = np.where(boundary_region >= 0)[0]  # face indices within probes
            probe_region = boundary_region[probe_f]
            index_fk = np.arange(len(probe_f) * n_kbb_mine)
            index_pk = (i_kbb + n_kbb_mine * probe_region[:, None]).flatten()
            values = (self.material.wk * rc.from_np(boundary_flux[probe_f])).flatten()
            self.boundary_flux_op = torch.sparse_coo_tensor(
                indices=rc.from_np(np.stack((index_pk, index_fk))),
                values=values,
                size=(n_boundaries * n_kbb_mine, len(probe_f) * n_kbb_mine),
            ).to_sparse_csr()
            self.boundary_probe = Boundary(
                self, contacts, *boundary_args, rc.from_np(probe_f)
            )

            # Set up boundary scalar probes:
            index_f = torch.arange(len(probe_f), device=rc.device)
            index_p = rc.from_np(probe_region)
            w_faces = rc.from_np(np.linalg.norm(face_weights[probe_f], axis=1))
            w_tot = torch.zeros(n_boundaries, dtype=w_faces.dtype, device=rc.device)
            w_tot.index_add_(0, index_p, w_faces)
            dist.all_reduce(w_tot, group=self.group)
            self.boundary_scalar_op = torch.sparse_coo_tensor(
                indices=torch.stack((index_p, index_f)),
                values=w_faces / w_tot[index_p],
                size=(n_boundaries, len(probe_f)),
            ).to_sparse_csr()

        # Set up cell region (interior) probes:
        n_mine = cell_division.n_mine
        i_region_all = self.mesh.cells[:, -1]
        assert i_region_all.min() >= -1
        n_regions = int(i_region_all.max()) + 1
        i_region = rc.from_np(i_region_all[cell_division.global_index[:n_mine]])
        probe_cells = torch.where(i_region >= 0)[0]  # cell indices within probes
        i_region = i_region[probe_cells]
        w_cells = rc.from_np(cell_weights)[probe_cells]
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

        # Initialize state (rho):
        n_cells_padded = len(cell_division.global_index)
        self.rho = material.rho0.flatten()[None, :].repeat(n_cells_padded, 1)
        if checkpoint_in and checkpoint_in.member("rho"):
            cp, path = checkpoint_in
            n_bands_sq = material.n_bands**2
            self.rho[: cell_division.n_mine] = cp.read_slice(
                cp[f"{path}/rho"],
                (cell_division.i_start, material.k_division.i_start * n_bands_sq),
                (cell_division.n_mine, material.k_division.n_mine * n_bands_sq),
            )
        self.stash = Stash()
        self.stash_probe = Stash()  # coarse-grained, more frequent version
        self.material.initialize_fields(self.rho, {}, id(self))

    def to_faces(
        self, rho: torch.Tensor, t: float, boundary_only: bool = False
    ) -> torch.Tensor:
        """Reconstruct face values from cell-centred values and boundary conditions.
        Result is flattened over the cell and face dimensions.
        If `boundary_only`, result is computed only for marked boundary faces."""
        # Update ghost regions if needed:
        self.mesh.cell_division.communicate_ghosts(rho)

        # Reconstruct rho at faces from rho at cell centroids:
        neighbors = self.neighbors
        reconstruct = self.reconstruct
        if boundary_only:
            sel = self.boundary_probe.boundary_c
            rho_faces = self._face_limiter(
                rho[sel], rho, neighbors[sel], reconstruct[sel], self.epsilon_sq
            )
            # Apply boundary conditions and return only boundary faces:
            return self.boundary_probe.apply(rho_faces.flatten(0, 1), t, update=False)
        else:
            n_reconstruct = len(reconstruct)
            if self.faces_block_size:
                rho_faces = rho.new_zeros(n_reconstruct, self.n_faces, self.n_kbb_mine)
                for sel in get_block_slices(n_reconstruct, self.faces_block_size):
                    rho_faces[sel] = self._face_limiter(
                        rho[sel], rho, neighbors[sel], reconstruct[sel], self.epsilon_sq
                    )
            else:
                rho_faces = self._face_limiter(
                    rho[:n_reconstruct], rho, neighbors, reconstruct, self.epsilon_sq
                )
            # Apply boundary conditions and return all faces:
            rho_faces = rho_faces.flatten(0, 1)
            self.boundary.apply(rho_faces, t, update=True)
            return rho_faces

    def get_flux(
        self, t: float, boundary_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute flux on all faces, or only boundary regions if `boundary_only`.
        Also return `rho_faces`, if needed for calculating observables on faces."""
        rho_faces = self.to_faces(self.rho, t, boundary_only)
        v_op = self.boundary_flux_op if boundary_only else self.v_multiply_select
        v_rho = v_op @ rho_faces.flatten()
        flux_op = self.material.get_observables(t)[self.flux_indices].T
        flux = (v_rho.view(-1, self.n_kbb_mine) @ flux_op).contiguous()
        dist.all_reduce(flux, group=(None if boundary_only else self.material.group))
        return flux, rho_faces

    def rho_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        watch = StopWatch("Geometry.rho_dot")
        out = (self.minus_v_dot_del @ self.to_faces(rho, t).flatten()).view(rho.shape)
        watch.stop()
        n_mine = self.mesh.cell_division.n_mine
        out[:n_mine] += self.material.rho_dot(rho[:n_mine], t, id(self))
        return out

    @stopwatch
    def update_probes(self, i_step: int, t: float) -> None:
        """Stash coarse-grained probes on defined cell regions and contacts."""
        self.stash_probe.i_step.append(i_step)
        self.stash_probe.t.append(t)
        # Interior cell regions:
        obserables = self.material.measure_observables(self.rho[self.probe_cells], t)
        region_averages = (self.probe_mat_cell @ obserables).contiguous()
        dist.all_reduce(region_averages, group=self.group)
        if self.flux_indices:
            flux, rho_faces = self.get_flux(t, boundary_only=True)
            observables = self.material.measure_observables(rho_faces, t)
            surface_averages = (self.boundary_scalar_op @ observables).contiguous()
            dist.all_reduce(surface_averages, group=self.group)
            region_averages = torch.cat((region_averages, surface_averages))
            self.stash_probe.flux.append(flux.detach().cpu())
        self.stash_probe.observables.append(region_averages.detach().cpu())

    @stopwatch
    def update_observables(self, i_step: int, t: float) -> None:
        """Stash observables (and related) for output in next checkpoint."""
        rho_mine = self.rho[: self.mesh.cell_division.n_mine]
        obserables = self.material.measure_observables(rho_mine, t)
        self.stash.i_step.append(i_step)
        self.stash.t.append(t)
        self.stash.observables.append(obserables.detach().cpu())
        if self.flux_indices:
            self.stash.flux.append(self.get_flux(t)[0].detach().cpu())
        self.material.update_stash(rho_mine, self.mesh.cell_division)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        checkpoint, path = cp_path
        assert checkpoint is not None
        names = self.material.get_observable_names()
        saved = [
            cp_path.write("flux_indices", np.array(self.flux_indices)),
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
        div = self.mesh.cell_division
        dset = cp_path.create_dataset(
            "observables", (n_stash, div.n_tot, len(names)), np.float64
        )
        checkpoint.write_slice(
            dset, (0, div.i_start, 0), torch.stack(self.stash.observables)
        )
        if self.flux_indices:
            # Flux observables live on edges: geometry + face-normal flux per frame.
            saved += ["face_flux"]
            face_div = self.mesh.face_division
            flux = torch.stack(self.stash.flux)
            dset = cp_path.create_dataset(
                "face_flux", (n_stash, face_div.n_tot, len(flux)), np.float64
            )
            checkpoint.write_slice(dset, (0, face_div.i_start, 0), flux)
        self.stash = Stash()  # reset now-written cached observables

        # Probes:
        cp_path.write("probes", torch.stack(self.stash_probe.observables))
        cp_path.write("probe_flux", torch.stack(self.stash_probe.flux))
        self.stash_probe = Stash()  # reset now-written cached probes

        if self.save_rho:
            # Optional full state output (large) for continuing calculations:
            rho_mine = self.rho[: div.n_mine].detach()
            dset = cp_path.create_dataset(
                "rho", (div.n_tot, self.n_kbb_mine), rho_mine.dtype
            )
            checkpoint.write_slice(dset, (div.i_start, 0), rho_mine)
            saved.append("rho")

        return saved


def _face_limiter(
    rho_sel: torch.Tensor,
    rho: torch.Tensor,
    neighbors: torch.Tensor,
    recon: torch.Tensor,
    epsilon_sq: float,
) -> torch.Tensor:
    """Compute limited face values for a subset of cells with values `rho_sel`,
    where `rho` is the full data, `neighbors` lists the neighboring cell indices for
    the selected cells, and `reconstruct` is the fused gradient->face operator."""
    drho = rho[neighbors] - rho_sel[:, None]
    d = recon @ drho
    lo, hi = drho.aminmax(dim=1, keepdim=True)
    return rho_sel[:, None] + limiter(d, lo, hi, 1, epsilon_sq) * d


@dataclass
class Stash:
    """Stash for probes and observables that are collated into checkpoints."""

    i_step: list[int] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    observables: list[torch.Tensor] = field(default_factory=list)
    flux: list[torch.Tensor] = field(default_factory=list)


class Boundary:
    """Boundary operators and indices for the entire boundary, or a selection of it."""

    contacts: list[tuple[torch.Tensor, None | Callable[[float], torch.Tensor]]]
    wall: tuple[torch.Tensor, None | Callable[[torch.Tensor], torch.Tensor]]
    is_outward: torch.Tensor  #: whether each face, k velocity is outwards
    boundary_c: torch.Tensor | None  #: subset of cells at which rho_faces is needed
    boundary_cf: torch.Tensor  #: flattened index of boundary faces into rho_faces

    def __init__(
        self,
        geometry: Geometry,
        contacts: dict[str, dict],
        boundary_map: torch.Tensor,
        boundary_region: torch.Tensor,
        boundary_weights: torch.Tensor,
        selection: torch.Tensor | None = None,
    ) -> None:
        names = geometry.mesh.boundary_names
        material = geometry.material
        normals = boundary_weights / boundary_weights.norm(dim=-1, keepdim=True)
        n_faces = geometry.n_faces
        if selection is None:
            self.boundary_c = None
            self.boundary_cf = boundary_map[:, 0] * n_faces + boundary_map[:, 1]
        else:
            boundary_region = boundary_region[selection]
            normals = normals[selection]
            # Find the cells for which rho_faces is needed for `selection`:
            cells, i_face = boundary_map[selection].T
            self.boundary_c, i_cell = torch.unique(cells, return_inverse=True)
            self.boundary_cf = i_cell * n_faces + i_face

        face_used = torch.full(
            (len(boundary_region),), False, device=rc.device, dtype=torch.bool
        )
        self.contacts = []
        for name, params in contacts.items():
            indices = torch.where(boundary_region == names.index(name))[0]
            if len(indices):
                face_used[indices] = True
                contactor = material.get_contactor(normals[indices], **params)
                self.contacts.append((indices, contactor))

        # Mark all boundary faces not in a contact as reflecting:
        indices = torch.where(~face_used)[0]
        reflector = material.get_reflector(normals[indices]) if len(indices) else None
        self.wall = (indices, reflector)
        self.is_outward = (normals @ material.transport_velocity.T) > 0

    def apply(self, rho_faces: torch.Tensor, t: float, update: bool) -> torch.Tensor:
        """Apply boundary conditions to `rho_faces`, modifying it in place
        if `update`. Otherwise, return the updated component alone."""
        rho_outward = rho_faces[self.boundary_cf]  # boundary faces, valid outwards
        # Apply reflections and contacts:
        rho_inward = rho_outward.detach().clone()  # prepare inward valid version
        indices, reflector = self.wall
        if reflector is not None:
            rho_inward[indices] = reflector(rho_outward[indices])
        for indices, contactor in self.contacts:
            rho_inward[indices] = contactor(t)
        # Set back to faces / return:
        rho_faces_sel = torch.where(self.is_outward, rho_outward, rho_inward)
        if update:
            rho_faces[self.boundary_cf] = rho_faces_sel
        return rho_faces_sel
