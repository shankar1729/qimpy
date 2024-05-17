from __future__ import annotations
from typing import Optional

import torch

from qimpy import MPI, rc
from qimpy.io import CheckpointPath
from qimpy.mpi import ProcessGrid, BufferView
from qimpy.profiler import stopwatch
from qimpy.transport.material import Material
from . import TensorList, Geometry, Patch, parse_svg


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
        """
        super().__init__(
            material=material,
            process_grid=process_grid,
            grid_spacing=grid_spacing,
            contacts=contacts,
            grid_size_max=grid_size_max,
            quad_set=parse_svg(svg_file, svg_unit, grid_spacing, list(contacts.keys())),
            checkpoint_in=checkpoint_in,
        )

        # Initialize spatially-dependent fields, if any:
        field_params = {}  # TODO: mechanism for input of spatially-varying fields
        for patch in self.patches:
            material.initialize_fields(patch.rho, field_params, id(patch))

    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        material = self.material
        rho_padded = self.apply_boundaries(rho, t)
        return TensorList(
            (patch.rho_dot(rho_padded_i) + material.rho_dot(rho_i, t, id(patch)))
            for rho_padded_i, rho_i, patch in zip(rho_padded, rho, self.patches)
        )

    @stopwatch
    def apply_boundaries(self, rho_list: TensorList, t: float) -> TensorList:
        """Apply all boundary conditions to `rho` at time `t` and produce
        ghost-padded version. The list contains the data for each patch."""
        # Create padded version for all patches:
        out_list = TensorList()
        for patch, rho in zip(self.patches, rho_list):
            out = torch.zeros(patch.rho_padded_shape, device=rc.device)
            out[Patch.NON_GHOST, Patch.NON_GHOST] = rho
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
                        for contact_slice, contactor in patch.contacts[i_edge]:
                            ghost_data[:, contact_slice] = contactor(t)
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


# Constants for edge data transfer:
IN_SLICES = [
    (slice(None), Patch.GHOST_L),
    (Patch.GHOST_R, slice(None)),
    (slice(None), Patch.GHOST_R),
    (Patch.GHOST_L, slice(None)),
]  #: input slice for each edge orientation during edge communication

OUT_SLICES = [
    (Patch.NON_GHOST, Patch.GHOST_L),
    (Patch.GHOST_R, Patch.NON_GHOST),
    (Patch.NON_GHOST, Patch.GHOST_R),
    (Patch.GHOST_L, Patch.NON_GHOST),
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
