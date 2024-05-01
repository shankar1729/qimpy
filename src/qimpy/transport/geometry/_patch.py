from __future__ import annotations
from typing import Callable, Optional, NamedTuple

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import globalreduce
from qimpy.profiler import stopwatch
from qimpy.transport.material import Material
from . import within_circles


class Contact(NamedTuple):
    """Definition of a contact."""

    selection: slice  #: Slice of edge's data that are within the contact
    contactor: Callable[[float], torch.Tensor]  #: Corresponding distribution calculator


class Patch:
    """Quad-patch with real-space advection on an arbitrary transformation."""

    q: torch.Tensor  #: Nx x Ny x 2 Cartesian coordinates
    g: torch.Tensor  #: Nx x Ny x 1 sqrt(metric), with extra dimensipm for broadcasting
    v: torch.tensor  #: Nkbb x 2 Cartesian velocities (where Nkbb is flattened k, b, b')
    V: torch.Tensor  #: Nx x Ny x Nkbb x 2 mesh coordinate velocities
    dt_max: float  #: Maximum stable time step
    wk: float  #: Integration weight for the flattened density matrix dimensions
    rho_shape: tuple[int, ...]  #: Shape of density matrix on patch
    rho_padded_shape: tuple[int, ...]  #: Shape of density matrix with ghost padding
    rho: torch.Tensor  #: current density matrix on this patch
    v_prime: torch.jit.ScriptModule  #: Underlying advection logic

    material: Material
    aperture_selections: list[Optional[torch.Tensor]]  #: Aperture indices for each edge
    reflectors: list[
        Optional[Callable[[torch.Tensor], torch.Tensor]]
    ]  #: Material-dependent reflector for each edge that needs one
    contactors: list[list[Contact]]  #: Contact calculators (multiple possibly) by edge

    N_GHOST: int = 2  #: currently a constant, but could depend on slope method later

    # Initialize slices for accessing ghost regions in padded version:
    # These are also the slices for the boundary region in non-padded version
    NON_GHOST = slice(N_GHOST, -N_GHOST)
    GHOST_L = slice(0, N_GHOST)  #: ghost indices on left/bottom
    GHOST_R = slice(-N_GHOST, None)  #: ghost indices on right/top side

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
        self.v = material.transport_velocity
        self.V = torch.einsum("ka, ...Ba -> ...kB", self.v, torch.linalg.inv(jacobian))
        self.dt_max = 0.5 / globalreduce.max(self.V.abs(), material.comm)
        self.wk = material.wk

        # Initialize distribution function:
        Nkbb = self.v.shape[0]  # flattened density-matrix count (Nkbb_mine of material)
        padding = 2 * Patch.N_GHOST
        self.rho_shape = (N[0], N[1], Nkbb)
        self.rho_padded_shape = (N[0] + padding, N[1] + padding, Nkbb)
        self.rho = torch.tile(material.rho0.flatten(), (N[0], N[1], 1))

        # Initialize v*drho/dx calculator:
        self.v_prime = torch.jit.script(Vprime())

        # Initialize reflectors if needed:
        self.material = material
        self.aperture_selections = [None] * 4
        self.reflectors = [None] * 4
        self.contacts = [[] for _ in range(4)]  # Note: [[]]*N makes N refs to one []!
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

    @stopwatch
    def rho_dot(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute rho_dot, given current rho."""
        return -1.0 * (
            self.v_prime(rho[:, Patch.NON_GHOST], self.V[..., 0], axis=0)
            + self.v_prime(rho[Patch.NON_GHOST, :], self.V[..., 1], axis=1)
        )


def to_numpy(f: torch.Tensor) -> np.ndarray:
    """Move torch.Tensor to numpy array, regardless of input device etc."""
    return f.detach().cpu().numpy()


def minmod(f: torch.Tensor, axis: int) -> torch.Tensor:
    """Return min|`f`| along `axis` when all same sign, and 0 otherwise."""
    fmin, fmax = torch.aminmax(f, dim=axis)
    return torch.where(
        fmin < 0.0,
        torch.clamp(fmax, max=0.0),  # fmin < 0, so fmax if also < 0, else 0.
        fmin,  # fmin >= 0, so this is the min mod
    )


class Vprime(torch.nn.Module):
    def __init__(self, slope_lim_theta: float = 2.0):
        # Initialize convolution that computes slopes using 3 difference formulae.
        # Here, `slope_lim_theta` controls the scaling of the forward/backward
        # difference formulae relative to the central difference one.
        # This convolution takes input Nbatch x 1 x N and produces output with
        # dimensions Nbatch x 3 x (N-2), containing backward, central and forward
        # difference computations of the slope.
        super().__init__()
        self.slope_conv = torch.nn.Conv1d(1, 3, 3, bias=False)
        self.slope_conv.weight.data = torch.tensor(
            [
                [-slope_lim_theta, slope_lim_theta, 0.0],
                [-0.5, 0.0, 0.5],
                [0.0, -slope_lim_theta, slope_lim_theta],
            ],
            device=rc.device,
        ).view(
            3, 1, 3
        )  # add singleton in_channels dim
        self.slope_conv.weight.requires_grad = False

    def slope_minmod(self, f: torch.Tensor) -> torch.Tensor:
        """Compute slope of `f` along its last axis with a minmod limiter."""
        # Flatten all but last axis into a single batch dimension:
        batch_shape = f.shape[:-1]
        f = f.flatten(0, -2)[:, None]  # n_batch x 1 x n_axis
        # Compute slopes by convolution and apply minmod filter:
        slope = minmod(self.slope_conv(f), axis=1)  # n_batch x n_axis
        return slope.unflatten(0, batch_shape)  # restore dimensions

    def forward(self, rho: torch.Tensor, v: torch.Tensor, axis: int) -> torch.Tensor:
        """Compute v * d`rho`/dx, with velocity `v` along `axis`."""
        # Bring active axis to end
        rho = rho.swapaxes(axis, -1)
        v = v.swapaxes(axis, -1)

        # Reconstruction
        half_slope = 0.5 * self.slope_minmod(rho)

        # Central difference from half points & Riemann selection based on velocity:
        rho_diff = rho[..., 1:-1].diff(dim=-1)
        half_slope_diff = half_slope.diff(dim=-1)
        result_minus = (rho_diff - half_slope_diff)[..., 1:]
        result_plus = (rho_diff + half_slope_diff)[..., :-1]
        delta_rho = torch.where(v < 0.0, result_minus, result_plus)
        return (v * delta_rho).swapaxes(axis, -1)  # original axis order
