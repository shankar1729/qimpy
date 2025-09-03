from __future__ import annotations

import torch

from qimpy import rc, TreeNode
from qimpy.io import (
    CheckpointPath,
    CheckpointContext,
    TensorCompatible,
    cast_tensor,
)
from qimpy.profiler import stopwatch
from qimpy.transport import material
from qimpy.transport.advect import Advect
from qimpy.lattice import Lattice
from qimpy.io import Unit

class EMField(TreeNode):
    """Magnetic field pulse to rotate spins by specific angles."""

    ab_initio: material.ab_initio.AbInitio

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters
    grad_phi: dict[int, torch.Tensor]  #: Electric field strength

    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        grad_phi: TensorCompatible = (0.0, 0.0, 0.0),
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize EM field interaction.

        Parameters
        ----------
        grad_phi
            :yaml:`Scalar potential.`
        """
        super().__init__()
        self.ab_initio = ab_initio
        self.constant_params = dict(
            grad_phi=cast_tensor(grad_phi),
        )

        # Initialize v*drho/dx calculator:
        self.advect = torch.jit.script(Advect(cent_diff_deriv=True))
        assert self.ab_initio.k_adj is not None

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["grad_phi"] = self.constant_params["grad_phi"].to(rc.cpu)
        return list(attrs.keys())

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        self._initialize_fields(patch_id, **params)

    def _initialize_fields(self, patch_id: int, *, grad_phi: torch.Tensor) -> None:
        # Spatial gradient of scalar potential
        dk = torch.diag(torch.tensor([1/nk for nk in self.ab_initio.nk_grid], device=rc.device))
        Gbasis = 2 * torch.pi * torch.linalg.inv(self.ab_initio.R.T)
        invJ = torch.linalg.inv(Gbasis@dk)
        # Factor of 1/2 added by Hermitian conjugate
        self.grad_phi = torch.einsum("...i,ji->j...", grad_phi, invJ)/2

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        # Rho dot shape: x1, x2, k, b1, b2
        result = torch.zeros_like(rho)
        F = self.grad_phi
        for comp, k_adj in enumerate(self.ab_initio.k_adj.swapaxes(0, 1)):
            rho_intermediate = rho[:, :, k_adj].real.swapaxes(3, -1)
            result += self.advect(rho_intermediate, F[comp][..., None, None, None, None], axis=-1).squeeze(dim=5)
        return result
