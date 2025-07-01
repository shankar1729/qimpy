from __future__ import annotations

import numpy as np
import torch

from qimpy import rc, TreeNode
from qimpy.io import (
    CheckpointPath,
    Unit,
    CheckpointContext,
    TensorCompatible,
    cast_tensor,
)
from qimpy.profiler import stopwatch
from qimpy.transport import material
from qimpy.transport.advect import Advect


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
        self.advect = torch.jit.script(Advect(cent_diff_deriv=False))
        assert self.ab_initio.k_adj is not None

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["grad_phi"] = self.constant_params["grad_phi"].to(rc.cpu)
        return list(attrs.keys())

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        self._initialize_fields(patch_id, **params)

    def _initialize_fields(
        self,
        patch_id: int,
        *,
        grad_phi: torch.Tensor
    ) -> None:

        # Spatial gradient of scalar potential
        self.grad_phi = grad_phi

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        # Rho dot shape: x1, x2, k, b1, b2
        # Expand rho to include dimensions for adjacent k-points along each component
        rho = rho[..., None, None].expand(-1, -1, -1, -1, -1, 5, 3).to(torch.float64)
        F = self.grad_phi
        nk = self.ab_initio.k.shape[0]
        # For each component, copy data for adjacent k-points (along axis 5)
        for ik in range(nk):
            for comp in range(3):
                for ind, jk in enumerate(self.ab_initio.k_adj[ik, comp]):
                    # Center k-point (index 2) at a k-point index is identity
                    rho[:, :, ik, :, :, ind, comp] = rho[:, :, jk, :, :, 2, comp] 
        # Sum contributions along each component
        result = (self.advect(rho[...,0], F[0], axis=-1) + \
               self.advect(rho[...,1], F[1], axis=-1) + \
               self.advect(rho[...,2], F[2], axis=-1))
        return result[..., 0]
        
