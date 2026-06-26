from __future__ import annotations
from typing import Optional
from abc import abstractmethod

import torch
import numpy as np

from qimpy import log, rc, TreeNode, MPI
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid, TaskDivision
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
    contacts: dict[str, Optional[dict]]  #: SVG contact names to material parameters
    quad_set: QuadSet  #: Original geometry specification from SVG
    sub_quad_set: SubQuadSet  #: Division into smaller quads for tuning parallelization
    patches: list[Patch]  #: Advection for each quad patch local to this process
    patch_division: TaskDivision  #: Division of patches over `comm`
    stash: ResultStash  #: Saved results for collating into fewer checkpoints
    dt_max: float  #: Maximum stable time step
    save_rho: bool  #: whether to write rho to checkpoint file
    cent_diff_deriv: bool  # using simple central difference derivative

    def __init__(
        self,
        *,
        material: Material,
        quad_set: QuadSet,
        grid_spacing: float,
        contacts: dict[str, Optional[dict]],
        grid_size_max: int,
        save_rho: bool = False,
        cent_diff_deriv: bool = False,
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
            implemented in the corresponding material. If None, the contact
            is treated like a regular boundary, but stored in the checkpoint,
            enabling consistent plotting of 'floating' contacts.
        grid_size_max
            :yaml:`Maximum grid points per dimension after quad subdvision.`
            If 0, will be determined automatically from number of processes.
            Note that this only affects parallelization and performance by
            changing how data is divided into patches, and does not affect
            the accuracy of format of the output.
        save_rho
            :yaml:`Whether to write the full density matrices to the checkpoint file.`
            If not (default), only observables are written to the checkpoint file.
        cent_diff_deriv
            :yaml:`Whether to use the simple central-difference derivative operator.`
            The default is choosing from the backward, central or forward derivative.
        """
        super().__init__()
        self.comm = process_grid.get_comm("r")
        self.material = material
        self.quad_set = quad_set
        self.grid_spacing = grid_spacing
        self.cent_diff_deriv = cent_diff_deriv
        self.contacts = contacts
        aperture_circles = torch.from_numpy(quad_set.apertures).to(rc.device)
        contact_params: list[dict] = []
        contact_sel: list[int] = []
        for i_contact, params in enumerate(contacts.values()):
            if params is not None:
                contact_params.append(params)
                contact_sel.append(i_contact)
        contact_circles = torch.from_numpy(quad_set.contacts[contact_sel]).to(rc.device)

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
                    cent_diff_deriv=self.cent_diff_deriv,
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
            "observables",
        ]
        if self.save_rho:
            saved_list.append("rho")
        cp_path.write_str("contact_names", ",".join(self.contacts.keys()))
        cp_path.write_str("aperture_names", ",".join(self.quad_set.aperture_names))
        cp_path.write_str(
            "observable_names", ",".join(self.material.get_observable_names())
        )
        stash = self.stash
        cp_path["t"] = np.array(stash.t)
        cp_path["i_step"] = np.array(stash.i_step)

        n_observables = len(self.material.get_observable_names())
        Nkbb = self.material.k_division.n_tot * self.material.n_bands**2
        checkpoint, path = cp_path
        for i_quad, grid_size_np in enumerate(self.quad_set.grid_size):
            # Create group and datasets together from all processes:
            cp_quad = CheckpointPath(checkpoint, f"{path}/quad{i_quad}")
            n_stash = len(stash.t)
            grid_size = tuple(grid_size_np)
            cp_quad.create_dataset("q", grid_size + (2,), np.float64)
            cp_quad.create_dataset("g", grid_size, np.float64)
            cp_quad.create_dataset(
                "observables", (n_stash,) + grid_size + (n_observables,), np.float64
            )
            if self.save_rho:
                cp_quad.create_dataset("rho", grid_size + (Nkbb,), np.float64)

            # Write independently from each patch on this quad:
            my_patches = slice(self.patch_division.i_start, self.patch_division.i_stop)
            for i_patch_mine in np.where(
                self.sub_quad_set.quad_index[my_patches] == i_quad
            )[0]:
                patch = self.patches[i_patch_mine]
                observables = torch.stack(stash.observables[i_patch_mine], dim=0)
                patch.save_checkpoint(cp_quad, observables, self.save_rho)

        self.stash = ResultStash(len(self.patches))  # Clear stashed history
        return saved_list

    def update_stash(self, i_step: int, t: float) -> None:
        """Stash results for current step for a future save_checkpoint call."""
        stash = self.stash
        stash.i_step.append(i_step)
        stash.t.append(t)
        for i_patch_mine, patch in enumerate(self.patches):
            stash.observables[i_patch_mine].append(
                self.material.measure_observables(patch.rho, t)
            )


class ResultStash:
    """Stashed results for collating I/O into fewer checkpoints."""

    i_step: list[int]
    t: list[float]
    observables: list[list[torch.Tensor]]

    def __init__(self, n_patches_mine: int):
        self.i_step = []
        self.t = []
        self.observables = [[] for _ in range(n_patches_mine)]
