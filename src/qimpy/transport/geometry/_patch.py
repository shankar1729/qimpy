from __future__ import annotations
from typing import Callable, Optional

import numpy as np
import torch

from qimpy import rc
from qimpy.io import CheckpointPath
from qimpy.transport.material import Material


class Patch:
    """Structured quad grid holding coordinates, metric and density-matrix state.

    This is a passive grid/state container (used by `ParameterGrid` for batched
    dynamics). The spatial finite-volume advection that previously lived here
    has been removed; spatial transport is now handled by the DG solver
    (`TriSet`). Momentum-space advection lives in the material.
    """

    q: torch.Tensor  #: Nx x Ny x 2 Cartesian coordinates
    g: torch.Tensor  #: Nx x Ny x 1 sqrt(metric)
    dt_max: float  #: Maximum stable time step (unused here; set by Geometry)
    wk: float  #: Integration weight for the flattened density-matrix dimensions
    rho_offset: tuple[int, ...]  #: Offset of density matrix within that of the quad
    rho_shape: tuple[int, ...]  #: Shape of the density matrix on this patch
    rho: torch.Tensor  #: current density matrix on this patch
    material: Material

    def __init__(
        self,
        *,
        transformation: Callable[[torch.Tensor], torch.Tensor],
        grid_size_tot: tuple[int, ...],
        grid_start: tuple[int, ...],
        grid_stop: tuple[int, ...],
        is_reflective: np.ndarray = None,
        has_apertures: np.ndarray = None,
        aperture_circles: Optional[torch.Tensor] = None,
        contact_circles: Optional[torch.Tensor] = None,
        contact_params: Optional[list[dict]] = None,
        material: Material,
        cent_diff_deriv: bool = False,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        # Initialize mesh:
        grids1d = [
            (torch.arange(grid_start_i, grid_stop_i, device=rc.device) + 0.5)
            for grid_start_i, grid_stop_i in zip(grid_start, grid_stop)
        ]
        Q = torch.stack(torch.meshgrid(*grids1d, indexing="ij"), dim=-1)

        # Transformed coordinates and Jacobian via auto-grad:
        N = tuple(
            (grid_stop_i - grid_start_i)
            for grid_start_i, grid_stop_i in zip(grid_start, grid_stop)
        )
        N_tot = torch.tensor(grid_size_tot, device=rc.device)
        grad_q = torch.tile(
            torch.eye(2, device=rc.device)[:, None, None], (1,) + N + (1,)
        )
        Q.requires_grad = True
        q = transformation(Q / N_tot)
        jacobian = torch.autograd.grad(
            q, Q, grad_outputs=grad_q, is_grads_batched=True, retain_graph=False
        )[0]
        jacobian = torch.permute(jacobian, (1, 2, 0, 3)).detach()
        Q.requires_grad = False
        self.q = q.detach()

        # Metric:
        metric = torch.einsum("...aB, ...aC -> ...BC", jacobian, jacobian)
        self.g = torch.linalg.det(metric).sqrt()[:, :, None]

        self.material = material
        self.wk = material.wk
        self.dt_max = np.inf  # no spatial advection here; set by the geometry

        # Distribution function:
        v = material.transport_velocity
        Nkbb = v.shape[0]
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

    def save_checkpoint(
        self, cp_path: CheckpointPath, observables: torch.Tensor, save_rho: bool
    ) -> None:
        """Save observables, and optionally density matrix, to checkpoint."""
        cp, path = cp_path
        assert cp is not None
        grid_offset = self.rho_offset[:-1]
        if self.material.comm.rank == 0:
            cp.write_slice(cp[path + "/q"], grid_offset + (0,), self.q)
            cp.write_slice(cp[path + "/g"], grid_offset, self.g[:, :, 0])
            cp.write_slice(
                cp[path + "/observables"], (0,) + grid_offset + (0,), observables
            )
        if save_rho:
            cp.write_slice(cp[path + "/rho"], self.rho_offset, self.rho)


def to_numpy(f: torch.Tensor) -> np.ndarray:
    """Move torch.Tensor to numpy array, regardless of input device etc."""
    return f.detach().cpu().numpy()
