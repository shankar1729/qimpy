from __future__ import annotations
from typing import Optional

import torch
import numpy as np

from qimpy import MPI, rc
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid, TaskDivision, BufferView
from qimpy.profiler import stopwatch
from qimpy.transport.material import Material
from . import (
    Geometry,
    Advect,
    BicubicPatch,
    parse_svg,
    QuadSet,
    SubQuadSet,
    subdivide,
    select_division,
)


class PatchSet(Geometry):
    """PatchSet specification."""

    grid_spacing: float  #: Grid spacing used for discretization
    contact_names: list[str]  #: Names of contacts used in SVG specification and plots
    quad_set: QuadSet  #: Original geometry specification from SVG
    sub_quad_set: SubQuadSet  #: Division into smaller quads for tuning parallelization
    patches: list[Advect]  #: Advection for each quad patch local to this process
    patch_division: TaskDivision  #: Division of patches over `comm`
    stash: ResultStash  #: Saved results for collating into fewer checkpoints

    # v_F and N_theta should eventually be material paramteters
    def __init__(
        self,
        *,
        material: Material,
        svg_file: str,
        svg_unit: float = 1.0,
        grid_spacing: float,
        contacts: dict[str, dict],
        grid_size_max: int = 0,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize PatchSet parameters.

        Parameters
        ----------
        svg_file
            :yaml:`Path to an SVG file containing the input geometry.
        svg_unit
            :yaml:`Real length corresponding to one unit of distance in SVG.`
        grid_spacing
            :yaml:`Maximum spacing between grid points anywhere in the geometry.`
            This is used to select the number of grid points in each domain.
        grid_size_max
            :yaml:`Maximum grid points per dimension after quad subdvision.`
            If 0, will be determined automatically from number of processes.
            Note that this only affects parallelization and performance by
            changing how data is divided into patches, and does not affect
            the accuracy of format of the output.
        """
        super().__init__(
            material=material,
            process_grid=process_grid,
        )
        self.grid_spacing = grid_spacing
        self.contact_names = list(contacts.keys())
        self.quad_set = parse_svg(svg_file, svg_unit, grid_spacing, self.contact_names)
        aperture_circles = torch.from_numpy(self.quad_set.apertures).to(rc.device)
        contact_circles = torch.from_numpy(self.quad_set.contacts).to(rc.device)
        contact_params = list(contacts.values())

        # Subdivide:
        if not grid_size_max:
            grid_size_max = select_division(self.quad_set, self.comm.size)
        self.sub_quad_set = subdivide(self.quad_set, grid_size_max)
        self.patch_division = TaskDivision(
            n_tot=len(self.sub_quad_set.quad_index),
            n_procs=self.comm.size,
            i_proc=self.comm.rank,
        )

        # Build an advect object for each sub-quad local to this process:
        self.patches = []
        mine = slice(self.patch_division.i_start, self.patch_division.i_stop)
        for i_quad, grid_start, grid_stop, adjacency, has_apertures in zip(
            self.sub_quad_set.quad_index[mine],
            self.sub_quad_set.grid_start[mine],
            self.sub_quad_set.grid_stop[mine],
            self.sub_quad_set.adjacency[mine],
            self.sub_quad_set.has_apertures[mine],
        ):
            boundary = torch.from_numpy(self.quad_set.get_boundary(i_quad))
            transformation = BicubicPatch(boundary=boundary.to(rc.device))
            self.patches.append(
                Advect(
                    transformation=transformation,
                    grid_size_tot=tuple(self.quad_set.grid_size[i_quad]),
                    grid_start=grid_start,
                    grid_stop=grid_stop,
                    material=material,
                    is_reflective=(adjacency[:, 0] == -1),
                    has_apertures=has_apertures,
                    aperture_circles=aperture_circles,
                    contact_circles=contact_circles,
                    contact_params=contact_params,
                )
            )
        self.dt_max = self.comm.allreduce(
            min((patch.dt_max for patch in self.patches), default=np.inf), op=MPI.MIN
        )
        self.stash = ResultStash(len(self.patches))

    @property
    def rho_list(self) -> list[torch.Tensor]:
        return [patch.rho for patch in self.patches]

    @rho_list.setter
    def rho_list(self, rho_list_new) -> None:
        for patch, rho_new in zip(self.patches, rho_list_new):
            patch.rho = rho_new

    @stopwatch
    def apply_boundaries(self, rho_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply all boundary conditions to `rho` and produce ghost-padded version.
        The list contains the data for each patch."""
        # Create padded version for all patches:
        out_list = []
        for patch, rho in zip(self.patches, rho_list):
            out = torch.zeros(patch.rho_padded_shape, device=rc.device)
            out[Advect.NON_GHOST, Advect.NON_GHOST] = rho
            out_list.append(out)

        # Populate ghost zones across patches where needed:
        requests = []
        pending_reads = []  # keep reference to data so that it doesn't deallocate
        pending_writes = []  # keep plans for writes till transfers complete
        for i_patch, adjacency in enumerate(self.sub_quad_set.adjacency):
            for i_edge, (other_patch, other_edge) in enumerate(adjacency):
                # Reflections (always local):
                if self.patch_division.is_mine(i_patch):
                    i_patch_mine = i_patch - self.patch_division.i_start
                    patch = self.patches[i_patch_mine]
                    reflector = patch.reflectors[i_edge]
                    if reflector is not None:
                        # Fetch the data in appropriate orientation:
                        ghost_data = rho_list[i_patch_mine][IN_SLICES[i_edge]]
                        if i_edge % 2 == 0:
                            ghost_data = ghost_data.swapaxes(0, 1)  # short axis first
                        # Reflect:
                        ghost_data = reflector(ghost_data)  # reciprocal space changes
                        ghost_data = ghost_data.flip(dims=(0,))  # flip short axis
                        # Apply contacts, if any:
                        for contact_slice, contact_rho in patch.contacts[i_edge]:
                            ghost_data[:, contact_slice] = contact_rho[None]
                        # Store back:
                        if i_edge % 2 == 0:
                            ghost_data = ghost_data.swapaxes(0, 1)  # restore axis order
                        out_list[i_patch_mine][OUT_SLICES[i_edge]] = ghost_data

                # Pass-through boundaries (may involve MPI communication):
                if other_patch >= 0:
                    read_mine = self.patch_division.is_mine(other_patch)
                    write_mine = self.patch_division.is_mine(i_patch)
                    tag = 4 * i_patch + i_edge  # unique for each message
                    if read_mine:
                        rho = rho_list[other_patch - self.patch_division.i_start]
                        ghost_data = rho[IN_SLICES[other_edge]]
                        delta_edge = other_edge - i_edge
                        if delta_edge % 2:
                            ghost_data = ghost_data.swapaxes(0, 1)
                        if flip_dims := FLIP_DIMS[delta_edge]:
                            ghost_data = ghost_data.flip(dims=flip_dims)
                        if not write_mine:
                            write_whose = self.patch_division.whose(i_patch)
                            ghost_data = ghost_data.contiguous()
                            pending_reads.append(ghost_data)  # hold till transfers done
                            requests.append(
                                self.comm.Isend(
                                    BufferView(ghost_data), write_whose, tag
                                )
                            )
                    if write_mine:
                        i_patch_mine = i_patch - self.patch_division.i_start
                        mask = self.patches[i_patch_mine].aperture_selections[i_edge]
                        if read_mine:
                            set_ghost_zone(
                                out_list[i_patch_mine], i_edge, ghost_data, mask
                            )
                        else:
                            read_whose = self.patch_division.whose(other_patch)
                            ghost_data = torch.empty_like(
                                out_list[i_patch_mine][OUT_SLICES[i_edge]]
                            )
                            requests.append(
                                self.comm.Irecv(BufferView(ghost_data), read_whose, tag)
                            )
                            pending_writes.append(
                                [i_patch_mine, i_edge, ghost_data, mask]
                            )

        # Finish pending data transfers and writes:
        if requests:
            MPI.Request.Waitall(requests)
            for i_patch_mine, i_edge, ghost_data, mask in pending_writes:
                set_ghost_zone(out_list[i_patch_mine], i_edge, ghost_data, mask)
        return out_list

    def rho_dot(
        self,
        rho_list_eval: list[torch.Tensor],
        t: float,
    ) -> list[torch.Tensor]:
        """Compute f(rho_eval), ingredient of time step"""
        material = self.material
        rho_list_padded = self.apply_boundaries(rho_list_eval)
        return [
            (patch.rho_dot(rho_padded) + material.rho_dot(rho_eval, t))
            for rho_padded, rho_eval, patch in zip(
                rho_list_padded, rho_list_eval, self.patches
            )
        ]

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        saved_list = [
            cp_path.write("vertices", torch.from_numpy(self.quad_set.vertices)),
            cp_path.write("quads", torch.from_numpy(self.quad_set.quads)),
            cp_path.write(
                "displacements", torch.from_numpy(self.quad_set.displacements)
            ),
            cp_path.write("adjacency", torch.from_numpy(self.quad_set.adjacency)),
            cp_path.write("grid_size", torch.from_numpy(self.quad_set.grid_size)),
            cp_path.write("contacts", torch.from_numpy(self.quad_set.contacts)),
            cp_path.write("apertures", torch.from_numpy(self.quad_set.apertures)),
            "q",
            "g",
            "density",
            "flux",
        ]
        cp_path.attrs["grid_spacing"] = self.grid_spacing
        cp_path.attrs["contact_names"] = ",".join(self.contact_names)
        cp_path.attrs["aperture_names"] = ",".join(self.quad_set.aperture_names)
        cp_path.attrs["observable_names"] = ",".join(
            self.material.get_observable_names()
        )
        stash = self.stash
        cp_path.attrs["t"] = np.array(stash.t)
        cp_path.attrs["i_step"] = np.array(stash.i_step)
        # Collect MPI-split data to be written from head (avoids slow h5-mpio):
        checkpoint, path = cp_path
        for i_quad, grid_size_np in enumerate(self.quad_set.grid_size):
            cp_quad = CheckpointPath(checkpoint, f"{path}/quad{i_quad}")
            n_stash = len(stash.t)
            grid_size = tuple(grid_size_np)
            n_obs = len(self.material.get_observable_names())
            stashed_size = (n_stash,) + grid_size + (n_obs,)
            q = torch.empty(grid_size + (2,))
            g = torch.empty(grid_size)
            density = torch.empty(stashed_size)
            flux = torch.empty(stashed_size + (2,))
            for i_patch in np.where(self.sub_quad_set.quad_index == i_quad)[0]:
                tag = 3 * i_patch
                i_proc = self.comm.rank
                whose = self.patch_division.whose(i_patch)
                local = i_proc == whose
                if local:
                    i_patch_mine = i_patch - self.patch_division.i_start
                    patch = self.patches[i_patch_mine]
                    q_cur = patch.q
                    g_cur = patch.g[..., 0]
                    density_cur = torch.stack(stash.density[i_patch_mine], dim=0)
                    flux_cur = torch.stack(stash.flux[i_patch_mine], dim=0)
                    if i_proc:
                        # Send to head for write:
                        self.comm.Send(BufferView(q_cur), 0, tag=tag)
                        self.comm.Send(BufferView(g_cur), 0, tag=tag + 1)
                        self.comm.Send(BufferView(density_cur), 0, tag=tag + 2)
                        self.comm.Send(BufferView(flux_cur), 0, tag=tag + 3)
                if not i_proc:
                    # Receive and write from head:
                    grid_start = self.sub_quad_set.grid_start[i_patch]
                    grid_stop = self.sub_quad_set.grid_stop[i_patch]
                    patch_size = tuple(grid_stop - grid_start)
                    slice0 = slice(grid_start[0], grid_stop[0])
                    slice1 = slice(grid_start[1], grid_stop[1])
                    if not local:
                        q_cur = torch.empty(patch_size + (2,))
                        g_cur = torch.empty(patch_size)
                        density_cur = torch.empty((n_stash,) + patch_size + (n_obs,))
                        flux_cur = torch.empty(
                            (n_stash,) + patch_size + (n_obs,) + (2,)
                        )
                        self.comm.Recv(BufferView(q_cur), whose, tag=tag)
                        self.comm.Recv(BufferView(g_cur), whose, tag=tag + 1)
                        self.comm.Recv(BufferView(density_cur), whose, tag=tag + 2)
                        self.comm.Recv(BufferView(flux_cur), whose, tag=tag + 3)
                    q[slice0, slice1] = q_cur
                    g[slice0, slice1] = g_cur
                    density[:, slice0, slice1] = density_cur
                    flux[:, slice0, slice1] = flux_cur
            cp_quad.write("q", q)
            cp_quad.write("g", g)
            cp_quad.write("density", density)
            cp_quad.write("flux", flux)
        self.stash = ResultStash(len(self.patches))  # Clear stashed history
        return saved_list

    def update_stash(self, i_step: int, t: float) -> None:
        """Stash results for current step for a future save_checkpoint call."""
        stash = self.stash
        stash.i_step.append(i_step)
        stash.t.append(t)
        for i_patch_mine, patch in enumerate(self.patches):
            density, flux = self.material.measure_observables(patch.rho, t)
            stash.density[i_patch_mine].append(density)
            stash.flux[i_patch_mine].append(flux)


# Constants for edge data transfer:
IN_SLICES = [
    (slice(None), Advect.GHOST_L),
    (Advect.GHOST_R, slice(None)),
    (slice(None), Advect.GHOST_R),
    (Advect.GHOST_L, slice(None)),
]  #: input slice for each edge orientation during edge communication

OUT_SLICES = [
    (Advect.NON_GHOST, Advect.GHOST_L),
    (Advect.GHOST_R, Advect.NON_GHOST),
    (Advect.NON_GHOST, Advect.GHOST_R),
    (Advect.GHOST_L, Advect.NON_GHOST),
]  #: output slice for each edge orientation during edge communication

FLIP_DIMS = [(0, 1), (0,), None, (1,)]  #: which dims to flip during edge transfer


def set_ghost_zone(
    data: torch.Tensor,
    i_edge: int,
    ghost_data: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> None:
    """Set ghost-zone data, accounting for an aperture mask if any."""
    if mask is None:
        data[OUT_SLICES[i_edge]] = ghost_data
    else:
        mask_sel = (slice(None), mask) if (i_edge % 2) else (mask, slice(None))
        data[OUT_SLICES[i_edge]][mask_sel] = ghost_data[mask_sel]


class ResultStash:
    """Stashed results for collating I/O into fewer checkpoints."""

    i_step: list[int]
    t: list[float]
    density: list[list[torch.Tensor]]
    flux: list[list[torch.Tensor]]

    def __init__(self, n_patches_mine: int):
        self.i_step = []
        self.t = []
        self.density = [[] for _ in range(n_patches_mine)]
        self.flux = [[] for _ in range(n_patches_mine)]
