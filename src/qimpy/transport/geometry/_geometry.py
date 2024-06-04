from __future__ import annotations
from abc import abstractmethod

import torch
import numpy as np

from qimpy import log, rc, TreeNode, MPI
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid, TaskDivision, BufferView
from ..material import Material
from . import (
    TensorList,
    QuadSet,
    SubQuadSet,
    subdivide,
    Patch,
    BicubicPatch,
    select_division,
)


class Geometry(TreeNode):
    """Geometry specification."""

    comm: MPI.Comm  #: Communicator for real-space split over patches
    material: Material  #: Corresponding material
    grid_spacing: float  #: Grid spacing used for discretization
    contact_names: list[str]  #: Names of contacts used in SVG specification and plots
    quad_set: QuadSet  #: Original geometry specification from SVG
    sub_quad_set: SubQuadSet  #: Division into smaller quads for tuning parallelization
    patches: list[Patch]  #: Advection for each quad patch local to this process
    patch_division: TaskDivision  #: Division of patches over `comm`
    stash: ResultStash  #: Saved results for collating into fewer checkpoints
    dt_max: float  #: Maximum stable time step
    save_rho: bool  #: whether to write rho to checkpoint file

    def __init__(
        self,
        *,
        material: Material,
        quad_set: QuadSet,
        grid_spacing: float,
        contacts: dict[str, dict],
        grid_size_max: int,
        save_rho: bool = False,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize geometry parameters, typically used from a derived class.

        Parameters
        ----------
        quad_set
            Geometry specification from derived class.
        grid_spacing
            :yaml:`Maximum spacing between grid points anywhere in the geometry.`
            This is used to select the number of grid points in each domain.
        contacts
            :yaml:`Dictionary of contact names to parameters.`
            The available contact parameters depend on the contact models
            implemented in the corresponding material.
        grid_size_max
            :yaml:`Maximum grid points per dimension after quad subdvision.`
            If 0, will be determined automatically from number of processes.
            Note that this only affects parallelization and performance by
            changing how data is divided into patches, and does not affect
            the accuracy of format of the output.
        save_rho
            :yaml:`Whether to write the full density matrices to the checkpoint file.`
            If not (default), only observables are written to the checkpoint file.
        """
        super().__init__()
        self.comm = process_grid.get_comm("r")
        self.material = material
        self.quad_set = quad_set
        self.grid_spacing = grid_spacing
        self.contact_names = list(contacts.keys())
        aperture_circles = torch.from_numpy(quad_set.apertures).to(rc.device)
        contact_circles = torch.from_numpy(quad_set.contacts).to(rc.device)
        contact_params = list(contacts.values())

        # Subdivide:
        if grid_size_max:
            log.info(f"Using specified {grid_size_max = }")
        else:
            grid_size_max = select_division(quad_set, self.comm.size)
        self.sub_quad_set = subdivide(quad_set, grid_size_max)
        log.info(
            f"Subdivided {len(quad_set.quads)} quads to "
            f"{len(self.sub_quad_set.quad_index)} for split "
            f"over {self.comm.size} processes."
        )
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
            boundary = torch.from_numpy(quad_set.get_boundary(i_quad))
            transformation = BicubicPatch(boundary=boundary.to(rc.device))
            self.patches.append(
                Patch(
                    transformation=transformation,
                    grid_size_tot=tuple(quad_set.grid_size[i_quad]),
                    grid_start=grid_start,
                    grid_stop=grid_stop,
                    material=material,
                    is_reflective=(adjacency[:, 0] == -1),
                    has_apertures=has_apertures,
                    aperture_circles=aperture_circles,
                    contact_circles=contact_circles,
                    contact_params=contact_params,
                    checkpoint_in=checkpoint_in.relative(f"quad{i_quad}"),
                )
            )
        self.dt_max = self.comm.allreduce(
            min((patch.dt_max for patch in self.patches), default=np.inf), op=MPI.MIN
        )
        self.stash = ResultStash(len(self.patches))
        self.save_rho = save_rho

    @abstractmethod
    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        """Return list of drho/dt corresponding to each rho at time `t`."""

    @property
    def rho(self) -> TensorList:
        """Get current values of density matrices."""
        return TensorList(patch.rho for patch in self.patches)

    @rho.setter
    def rho(self, rho_new: TensorList) -> None:
        """Set current values of density matrices."""
        for patch, rho_new_i in zip(self.patches, rho_new):
            patch.rho = rho_new_i

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
            "rho",
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
        nkpts, nbands = self.material.E.shape
        Nkbb = nkpts * nbands**2
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
            if self.save_rho:
                rho = torch.empty(grid_size + (Nkbb,))
            for i_patch in np.where(self.sub_quad_set.quad_index == i_quad)[0]:
                tag = 5 * i_patch
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
                    if self.save_rho:
                        rho_cur = patch.rho
                    if i_proc:
                        # Send to head for write:
                        self.comm.Send(BufferView(q_cur), 0, tag=tag)
                        self.comm.Send(BufferView(g_cur), 0, tag=tag + 1)
                        self.comm.Send(BufferView(density_cur), 0, tag=tag + 2)
                        self.comm.Send(BufferView(flux_cur), 0, tag=tag + 3)
                        if self.save_rho:
                            self.comm.Send(BufferView(rho_cur), 0, tag=tag + 4)
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
                        if self.save_rho:
                            rho_cur = torch.empty(patch_size + (Nkbb,))
                            self.comm.Recv(BufferView(rho_cur), whose, tag=tag + 4)
                    q[slice0, slice1] = q_cur
                    g[slice0, slice1] = g_cur
                    density[:, slice0, slice1] = density_cur
                    flux[:, slice0, slice1] = flux_cur
                    if self.save_rho:
                        rho[slice0, slice1] = rho_cur
            cp_quad.write("q", q)
            cp_quad.write("g", g)
            cp_quad.write("density", density)
            cp_quad.write("flux", flux)
            if self.save_rho:
                cp_quad.write("rho", rho)
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
