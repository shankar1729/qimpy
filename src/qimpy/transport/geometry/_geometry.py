from __future__ import annotations

import torch

from qimpy import TreeNode, rc
from qimpy.io import CheckpointPath
from qimpy.profiler import stopwatch
from qimpy.transport.material import Material
from . import (
    Advect,
    BicubicPatch,
    parse_svg,
    QuadSet,
    SubQuadSet,
    subdivide,
    select_division,
)


class Geometry(TreeNode):
    """Geometry specification."""

    quad_set: QuadSet  #: Original geometry specification from SVG
    sub_quad_set: SubQuadSet  #: Division into smaller quads for tuning parallelization
    patches: list[Advect]  #: Advection for each quad patch

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

    # v_F and N_theta should eventually be material paramteters
    def __init__(
        self,
        *,
        material: Material,
        svg_file: str,
        grid_spacing: float,
        grid_size_max: int = 0,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize geometry parameters.

        Parameters
        ----------
        svg_file
            :yaml:`Path to an SVG file containing the input geometry.
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
        super().__init__()
        self.quad_set = parse_svg(svg_file, grid_spacing)

        # Subdivide:
        if not grid_size_max:
            # TODO: select appropriate dimension of process grid, once implemented
            grid_size_max = select_division(self.quad_set, rc.n_procs)
        self.sub_quad_set = subdivide(self.quad_set, grid_size_max)

        # Build an advect object for each quad
        self.patches = []
        v = material.transport_velocity
        for i_quad, grid_start, grid_stop in zip(
            self.sub_quad_set.quad_index,
            self.sub_quad_set.grid_start,
            self.sub_quad_set.grid_stop,
        ):
            boundary = torch.from_numpy(self.quad_set.get_boundary(i_quad))
            transformation = BicubicPatch(boundary=boundary.to(rc.device))
            self.patches.append(
                Advect(
                    transformation=transformation,
                    grid_size_tot=tuple(self.quad_set.grid_size[i_quad]),
                    grid_start=grid_start,
                    grid_stop=grid_stop,
                    v=v,
                )
            )
        self.dt = min(patch.dt_max for patch in self.patches)

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
        for i_patch, (out, adjacency) in enumerate(
            zip(out_list, self.sub_quad_set.adjacency)
        ):
            for i_edge, (other_patch, other_edge) in enumerate(adjacency):
                if other_patch < 0:
                    # TODO: handle reflecting boundaries
                    # For now they will be sinks (hence pass)
                    pass
                else:
                    # Pass-through boundary:
                    ghost_data = rho_list[other_patch][Geometry.IN_SLICES[other_edge]]
                    delta_edge = other_edge - i_edge
                    if delta_edge % 2:
                        ghost_data = ghost_data.swapaxes(0, 1)
                    if flip_dims := Geometry.FLIP_DIMS[delta_edge]:
                        ghost_data = ghost_data.flip(dims=flip_dims)
                    out[Geometry.OUT_SLICES[i_edge]] = ghost_data
        return out_list

    def next_rho_list(
        self,
        dt: float,
        rho_list_initial: list[torch.Tensor],
        rho_list_eval: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Ingredient of time step: compute rho_initial + dt * f(rho_eval)."""
        return [
            (rho_initial + patch.drho(dt, rho_eval))
            for rho_initial, rho_eval, patch in zip(
                rho_list_initial, self.apply_boundaries(rho_list_eval), self.patches
            )
        ]

    def time_step(self) -> None:
        """Second-order correct time step."""
        rho_list_init = self.rho_list
        rho_list_half = self.next_rho_list(0.5 * self.dt, rho_list_init, rho_list_init)
        self.rho_list = self.next_rho_list(self.dt, rho_list_init, rho_list_half)
