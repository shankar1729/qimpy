from __future__ import annotations
from typing import Optional

import torch

from qimpy import MPI
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid, BufferView
from qimpy.profiler import stopwatch
from qimpy.transport.material import Material
from qimpy.transport.advect import N_GHOST, NON_GHOST
from . import TensorList, Geometry, parse_svg


class PatchSet(Geometry):
    """Spatial transport starting from a SVG cubic spline specification."""

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
        save_rho: bool = False,
        cent_diff_deriv: bool = False,
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
        cent_diff_deriv
            :yaml:`Whether to use the simple central-difference derivative operator.`
            The default is choosing from the backward, central or forward derivative.
        """
        self.svg_file = svg_file
        self.svg_unit = svg_unit
        self.grid_spacing_max = grid_size_max
        super().__init__(
            material=material,
            process_grid=process_grid,
            grid_spacing=grid_spacing,
            contacts=contacts,
            grid_size_max=grid_size_max,
            quad_set=parse_svg(svg_file, svg_unit, grid_spacing, list(contacts.keys())),
            save_rho=save_rho,
            cent_diff_deriv=cent_diff_deriv,
            checkpoint_in=checkpoint_in,
        )

        # Initialize spatially-dependent fields, if any:
        field_params: dict[str, torch.Tensor] = {}  # TODO: fields varying in space
        for patch in self.patches:
            material.initialize_fields(patch.rho, field_params, id(patch))

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["svg_file"] = self.svg_file
        attrs["svg_unit"] = self.svg_unit
        attrs["grid_spacing"] = self.grid_spacing
        # attrs["contacts"] = self.contacts  # TODO: serialize contacts
        attrs["grid_size_max"] = self.grid_spacing_max
        attrs["save_rho"] = self.save_rho
        attrs["cent_diff_deriv"] = self.cent_diff_deriv
        return list(attrs.keys()) + super()._save_checkpoint(cp_path, context)

    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        # Geometry contributions:
        grho_padded = self.boundaries_pre(rho, t)
        grho_dot = [
            patch.rho_dot(grho_padded_i)
            for patch, grho_padded_i in zip(self.patches, grho_padded)
        ]
        self.boundaries_post(grho_dot)
        # Add material contributions:
        return TensorList(
            grho_dot_i / patch.g + self.material.rho_dot(rho_i, t, id(patch))
            for (grho_dot_i, _), rho_i, patch in zip(grho_dot, rho, self.patches)
        )

    @stopwatch
    def boundaries_pre(self, rho_list: TensorList, t: float) -> list[torch.Tensor]:
        """Apply all boundary conditions to `rho` at time `t` and produce
        ghost-padded, g=sqrt(metric)-multipled version suitable for advection.
        The list contains the data for each patch."""
        # Create padded g-multiplied version for all patches:
        out_list = [
            torch.nn.functional.pad(rho * patch.g, (0, 0) + (N_GHOST,) * 4)
            for rho, patch in zip(rho_list, self.patches)
        ]

        # Populate ghost zones across patches where needed:
        requests = []
        pending_reads = []  # keep reference to data so that it doesn't deallocate
        pending_writes = list[tuple[int, int, torch.Tensor, Optional[torch.Tensor]]]()
        for i_patch, adjacency in enumerate(self.sub_quad_set.adjacency):
            for i_edge, (other_patch, other_edge) in enumerate(adjacency):
                # Reflections (always local):
                if self.patch_division.is_mine(i_patch):
                    i_patch_mine = i_patch - self.patch_division.i_start
                    patch = self.patches[i_patch_mine]
                    reflector = patch.reflectors[i_edge]
                    if reflector is not None:
                        # Fetch the data in appropriate orientation:
                        g_edge = patch.g[EDGES[i_edge]]
                        ghost_data = g_edge * rho_list[i_patch_mine][EDGES[i_edge]]
                        # Reflect:
                        ghost_data = reflector(ghost_data[None])[
                            0
                        ]  # reciprocal space changes
                        # Apply contacts, if any:
                        for contact_slice, contactor in patch.contacts[i_edge]:
                            g = patch.g[EDGES[i_edge]][contact_slice]
                            ghost_data[contact_slice] = g * contactor(t)[0]
                        # Store back:
                        out_list[i_patch_mine][GHOSTS[i_edge]] = ghost_data

                # Pass-through boundaries (may involve MPI communication):
                if other_patch >= 0:
                    read_mine = self.patch_division.is_mine(other_patch)
                    write_mine = self.patch_division.is_mine(i_patch)
                    tag = 4 * i_patch + i_edge  # unique for each message
                    if read_mine:
                        other_patch_mine = other_patch - self.patch_division.i_start
                        rho = rho_list[other_patch_mine]
                        g_edge = self.patches[other_patch_mine].g[EDGES[other_edge]]
                        ghost_data = g_edge * rho[EDGES[other_edge]]
                        if (other_edge < 2) ^ (i_edge >= 2):
                            ghost_data = ghost_data.flip(dims=(0,))
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
                                out_list[i_patch_mine][GHOSTS[i_edge]]
                            )
                            requests.append(
                                self.comm.Irecv(BufferView(ghost_data), read_whose, tag)
                            )
                            pending_writes.append(
                                (i_patch_mine, i_edge, ghost_data, mask)
                            )

        # Finish pending data transfers and writes:
        if requests:
            MPI.Request.Waitall(requests)
            for i_patch_mine, i_edge, ghost_data, mask in pending_writes:
                set_ghost_zone(out_list[i_patch_mine], i_edge, ghost_data, mask)
        return out_list

    @stopwatch
    def boundaries_post(
        self, grho_dot_list: list[tuple[torch.Tensor, list[torch.Tensor]]]
    ) -> None:
        """Accumulate edge contributions of `grho_dot` into appropriate domain points.
        This is necessary for exact norm conservation in reflection and pass-throughs,
        when velocities don't map exactly across the boundary."""
        requests = []
        pending_reads = []  # keep reference to data so that it doesn't deallocate
        pending_writes = list[tuple[int, int, torch.Tensor, Optional[torch.Tensor]]]()
        for i_patch, adjacency in enumerate(self.sub_quad_set.adjacency):
            for i_edge, (other_patch, other_edge) in enumerate(adjacency):
                # Reflections (always local):
                if self.patch_division.is_mine(i_patch):
                    i_patch_mine = i_patch - self.patch_division.i_start
                    grho_dot, grho_dot_edges = grho_dot_list[i_patch_mine]
                    patch = self.patches[i_patch_mine]
                    reflector = patch.reflectors[i_edge]
                    if reflector is not None:
                        # Fetch and reflect the edge data as 1 x N x Nkbb:
                        edge_data = reflector(grho_dot_edges[i_edge][None])[0]
                        # Mask out contacts, if any:
                        for contact_slice, _ in patch.contacts[i_edge]:
                            edge_data[contact_slice] = 0.0
                        # Mask out apertures, if any:
                        if (mask := patch.aperture_selections[i_edge]) is not None:
                            edge_data[mask] = 0.0
                        # Accumulate contribution:
                        grho_dot[EDGES[i_edge]] += edge_data

                # Pass-through boundaries (may involve MPI communication):
                if other_patch >= 0:
                    read_mine = self.patch_division.is_mine(other_patch)
                    write_mine = self.patch_division.is_mine(i_patch)
                    tag = 4 * i_patch + i_edge  # unique for each message
                    if read_mine:
                        other_patch_mine = other_patch - self.patch_division.i_start
                        edge_data = grho_dot_list[other_patch_mine][1][other_edge]
                        if (other_edge < 2) ^ (i_edge >= 2):
                            edge_data = edge_data.flip(dims=(0,))
                        if not write_mine:
                            write_whose = self.patch_division.whose(i_patch)
                            edge_data = edge_data.contiguous()
                            pending_reads.append(edge_data)  # hold till transfers done
                            requests.append(
                                self.comm.Isend(BufferView(edge_data), write_whose, tag)
                            )
                    if write_mine:
                        i_patch_mine = i_patch - self.patch_division.i_start
                        mask = self.patches[i_patch_mine].aperture_selections[i_edge]
                        if read_mine:
                            accumulate_edge(
                                grho_dot_list[i_patch_mine][0], i_edge, edge_data, mask
                            )
                        else:
                            read_whose = self.patch_division.whose(other_patch)
                            edge_data = torch.empty_like(
                                grho_dot_list[i_patch_mine][0][EDGES[i_edge]]
                            )
                            requests.append(
                                self.comm.Irecv(BufferView(edge_data), read_whose, tag)
                            )
                            pending_writes.append(
                                (i_patch_mine, i_edge, edge_data, mask)
                            )

        # Finish pending data transfers and writes:
        if requests:
            MPI.Request.Waitall(requests)
            for i_patch_mine, i_edge, edge_data, mask in pending_writes:
                accumulate_edge(grho_dot_list[i_patch_mine][0], i_edge, edge_data, mask)


# Constants for edge data transfer:
GHOSTS = [
    (NON_GHOST, 0),
    (-1, NON_GHOST),
    (NON_GHOST, -1),
    (0, NON_GHOST),
]  #: slices for the edges of the ghost zone

EDGES = [
    (slice(None), 0),
    (-1, slice(None)),
    (slice(None), -1),
    (0, slice(None)),
]  #: slices for the edges of the domain


def set_ghost_zone(
    data: torch.Tensor,
    i_edge: int,
    ghost_data: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> None:
    """Set ghost-zone data, accounting for an aperture mask if any."""
    if mask is None:
        data[GHOSTS[i_edge]] = ghost_data
    else:
        data[GHOSTS[i_edge]][mask] = ghost_data[mask]


def accumulate_edge(
    data: torch.Tensor,
    i_edge: int,
    edge_data: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> None:
    """Accumulate edge contributions, accounting for an aperture mask if any."""
    if mask is None:
        data[EDGES[i_edge]] += edge_data
    else:
        data[EDGES[i_edge]][mask] += edge_data[mask]
