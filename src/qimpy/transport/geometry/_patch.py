from __future__ import annotations
from typing import Callable, Optional, NamedTuple

import numpy as np
import torch

from qimpy import rc
from qimpy.io import CheckpointPath
from qimpy.mpi import globalreduce
from qimpy.profiler import stopwatch
from qimpy.transport.material import Material
from qimpy.transport.advect import Advect, N_GHOST, NON_GHOST, Mask
from . import within_circles


class Contact(NamedTuple):
    """Definition of a contact."""

    selection: slice  #: Slice of edge's data that are within the contact
    contactor: Callable[[float], torch.Tensor]  #: Corresponding distribution calculator


class Patch:
    """Quad-patch with real-space advection on an arbitrary transformation."""

    q: torch.Tensor  #: Nx x Ny x 2 Cartesian coordinates
    g: torch.Tensor  #: Nx x Ny x 1 sqrt(metric), with extra dimensipm for broadcasting
    V0: torch.Tensor  #: Nx_padded x Ny x Nkbb mesh coordinate velocities in first axis
    V1: torch.Tensor  #: Nx x Ny_padded x Nkbb mesh coordinate velocities in second axis
    dt_max: float  #: Maximum stable time step
    wk: float  #: Integration weight for the flattened density matrix dimensions
    rho_offset: tuple[int, ...]  #: Offset of density matrix data within that of quad
    rho_shape: tuple[int, ...]  #: Shape of density matrix on patch
    rho: torch.Tensor  #: current density matrix on this patch
    advect: torch.jit.ScriptModule  #: Underlying advection logic
    cent_diff_deriv: bool  # using simple central difference operator

    material: Material
    aperture_selections: list[Optional[torch.Tensor]]  #: Aperture indices for each edge
    reflectors: list[
        Optional[Callable[[torch.Tensor], torch.Tensor]]
    ]  #: Material-dependent reflector for each edge that needs one
    contacts: list[list[Contact]]  #: Contact calculators (multiple possibly) by edge
    edge_masks0: list[Mask]  #: Masks for zeroing edge flux in first axis
    edge_masks1: list[Mask]  #: Masks for zeroing edge flux in second axis

    def __init__(
        self,
        *,
        transformation: Callable[[torch.Tensor], torch.Tensor],
        grid_size_tot: tuple[int, ...],
        grid_start: tuple[int, ...],
        grid_stop: tuple[int, ...],
        is_reflective: np.ndarray,
        has_apertures: np.ndarray,
        aperture_circles: torch.Tensor,
        contact_circles: torch.Tensor,
        contact_params: list[dict],
        material: Material,
        cent_diff_deriv: bool,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        # Initialize mesh:
        grids1d = [
            (torch.arange(grid_start_i, grid_stop_i, device=rc.device) + 0.5)
            for grid_start_i, grid_stop_i in zip(grid_start, grid_stop)
        ]
        Q = torch.stack(torch.meshgrid(*grids1d, indexing="ij"), dim=-1)

        # Initialize transformed coordinates and jacobian using auto-grad
        N = tuple(
            (grid_stop_i - grid_start_i)
            for grid_start_i, grid_stop_i in zip(grid_start, grid_stop)
        )
        N_tot = torch.tensor(grid_size_tot, device=rc.device)
        grad_q = torch.tile(
            torch.eye(2, device=rc.device)[:, None, None], (1,) + N + (1,)
        )
        Q.requires_grad = True
        Qfrac = Q / N_tot
        q = transformation(Qfrac)
        jacobian = torch.autograd.grad(
            q, Q, grad_outputs=grad_q, is_grads_batched=True, retain_graph=False
        )[0]
        jacobian = torch.permute(jacobian, (1, 2, 0, 3)).detach()
        Q.requires_grad = False
        self.q = q.detach()

        # Initialize metric:
        metric = torch.einsum("...aB, ...aC -> ...BC", jacobian, jacobian)
        self.g = torch.linalg.det(metric).sqrt()[:, :, None]

        # Initialize velocities:
        v = material.transport_velocity
        V = torch.einsum("ka, ...Ba -> ...kB", v, torch.linalg.inv(jacobian))
        self.dt_max = 0.5 / globalreduce.max(V.abs(), material.comm)
        self.wk = material.wk

        # Initialize distribution function:
        Nkbb = v.shape[0]  # flattened density-matrix count (Nkbb_mine of material)
        nk_prev = material.k_division.n_prev[material.comm.rank]
        Nkbb_offset = nk_prev * (material.n_bands**2)
        self.rho_offset = tuple(grid_start) + (Nkbb_offset,)
        self.rho_shape = (N[0], N[1], Nkbb)
        if checkpoint_in:
            checkpoint, path = checkpoint_in.relative("rho")
            assert checkpoint is not None
            self.rho = checkpoint.read_slice(
                checkpoint[path], self.rho_offset, self.rho_shape
            )
        else:
            self.rho = torch.tile(material.rho0.flatten(), (N[0], N[1], 1))

        # Store mesh velocities with padding needed for advection:
        self.V0 = torch.nn.functional.pad(V[..., 0], (0, 0, 0, 0, N_GHOST, N_GHOST))
        self.V0[0] = V[0, ..., 0]
        self.V0[-1] = V[-1, ..., 0]
        self.V1 = torch.nn.functional.pad(V[..., 1], (0, 0, N_GHOST, N_GHOST))
        self.V1[:, 0] = V[:, 0, :, 1]
        self.V1[:, -1] = V[:, -1, :, 1]

        # Initialize v*drho/dx calculator:
        self.advect = torch.jit.script(
            Advect(cent_diff_deriv=cent_diff_deriv, pad=True)
        )

        # Initialize reflectors if needed:
        self.material = material
        self.aperture_selections = [None] * 4
        self.reflectors = [None] * 4
        self.contacts = [[] for _ in range(4)]  # Note: [[]]*N makes N refs to one []!
        edge_masks: list[Mask] = [True] * 4
        for i_edge, (is_reflective_i, has_apertures_i) in enumerate(
            zip(is_reflective, has_apertures)
        ):
            if not (is_reflective_i or has_apertures_i):
                continue  # Entirely pass-through (neither reflective nor has apertures)

            i_dim = i_edge % 2  # long direction of edge
            j_dim = 1 - i_dim  # other direction

            # Compute coordinates along edge:
            Q_edge = torch.empty((N[i_dim], 2), device=rc.device)
            Q_edge[:, i_dim] = grids1d[i_dim]
            Q_edge[:, j_dim] = (grid_start if (i_edge in {0, 3}) else grid_stop)[j_dim]
            Q_edge_frac = Q_edge / N_tot
            Q_edge_frac.requires_grad = True
            q_edge = transformation(Q_edge_frac)

            # Compute tangent direction:
            grad_q_edge = torch.tile(
                torch.eye(2, device=rc.device)[:, None], (1, N[i_dim], 1)
            )
            jacobian_edge = torch.autograd.grad(
                q_edge,
                Q_edge_frac,
                grad_outputs=grad_q_edge,
                is_grads_batched=True,
                retain_graph=False,
            )[0]
            jacobian_edge = torch.permute(jacobian_edge, (1, 0, 2)).detach()
            tangent = jacobian_edge[..., i_dim]  # derivative along edge
            if i_edge >= 2:
                tangent *= -1  # so that it follows the counter-clockwise diretcion
            normal = torch.stack((tangent[..., 1], -tangent[..., 0]), dim=-1)
            normal *= (1.0 / normal.norm(dim=-1))[..., None]  # unit outward normal
            self.reflectors[i_edge] = material.get_reflector(normal)

            # Initialize pass-through indices for edges with apertures:
            if has_apertures_i:
                within = within_circles(aperture_circles, q_edge.detach())
                self.aperture_selections[i_edge] = torch.where(within.any(dim=0))[0]

            # Check for any contacts:
            if not is_reflective_i:
                continue  # ignore contacts in periodic/pass-through
            within = within_circles(contact_circles, q_edge.detach())
            for i_contact, contact_params_i in enumerate(contact_params):
                if len(selection := torch.argwhere(within[i_contact])):
                    selection_start = selection.min().item()
                    selection_stop = selection.max().item() + 1
                    assert (selection_stop - selection_start) == len(selection)
                    # Assume each edge intersection with contact is contiguous
                    # Using this reduce selection to a slice (more convenient):
                    contact_slice = slice(selection_start, selection_stop)
                    contactor = material.get_contactor(
                        normal[contact_slice], **contact_params_i
                    )
                    self.contacts[i_edge].append(Contact(contact_slice, contactor))
            edge_mask = torch.where(torch.logical_not(within.any(dim=0)))[0]
            if len(edge_mask) == 0:
                edge_masks[i_edge] = False  # Disable mask for full-edge contact
            elif len(edge_mask) < within.shape[1]:
                edge_masks[i_edge] = edge_mask  # Partial mask outside contact only

        self.edge_masks0 = [edge_masks[3], edge_masks[1]]
        self.edge_masks1 = [edge_masks[0], edge_masks[2]]

    def save_checkpoint(
        self, cp_path: CheckpointPath, observables: torch.Tensor, save_rho: bool
    ) -> None:
        """Save observables, and optionally density matrix, to checkpoint."""
        cp, path = cp_path
        assert cp is not None
        grid_offset = self.rho_offset[:-1]
        if self.material.comm.rank == 0:
            # Write quantities not divided over material:
            cp.write_slice(cp[path + "/q"], grid_offset + (0,), self.q)
            cp.write_slice(cp[path + "/g"], grid_offset, self.g[:, :, 0])
            cp.write_slice(
                cp[path + "/observables"], (0,) + grid_offset + (0,), observables
            )
        if save_rho:
            cp.write_slice(cp[path + "/rho"], self.rho_offset, self.rho)

    @stopwatch
    def rho_dot(self, grho: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute g*rho_dot, given current g*rho.
        The input is ghost-padded, while the output contains the contributions within
        the domain and the edge contributions that must be reflected/passed-through."""
        out0, outL, outR = self.advect(grho[:, NON_GHOST], self.V0, 0, self.edge_masks0)
        out1, outB, outT = self.advect(grho[NON_GHOST, :], self.V1, 1, self.edge_masks1)
        return out0 + out1, [outB, outR, outT, outL]


def to_numpy(f: torch.Tensor) -> np.ndarray:
    """Move torch.Tensor to numpy array, regardless of input device etc."""
    return f.detach().cpu().numpy()
