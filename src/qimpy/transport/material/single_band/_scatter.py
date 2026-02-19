from __future__ import annotations

import torch

from qimpy import TreeNode
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.transport import material


class Scatter(TreeNode):
    """Explicit k-dependent scattering kernel.
    Currently, e-e scattering only, but can be extended to e-ph.
    """

    single_band: material.single_band.SingleBand
    dE: float
    epsilon_bg: float
    lambda_D: float

    def __init__(
        self,
        *,
        single_band: material.single_band.SingleBand,
        dE: float,
        epsilon_bg: float,
        lambda_D: float,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize ab initio Lindbladian scattering.

        Parameters
        ----------
        dE
            :yaml:`Energy conservation width.`
            Determines resolution of internal frequency grid.
        epsilon_bg
            :yaml:`Background dielectric constant.`
        lambda_D
            :yaml:`Debye screening length of electrons.`
        """
        super().__init__()
        self.single_band = single_band
        self.dE = dE
        self.epsilon_bg = epsilon_bg
        self.lambda_D = lambda_D

        # Initialize scattering operator
        raise NotImplementedError

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["dE"] = self.dE
        attrs["epsilon_bg"] = self.epsilon_bg
        attrs["lambda_D"] = self.lambda_D
        return list(attrs.keys())

    def rho_dot(self, rho: torch.Tensor) -> torch.Tensor:
        return NotImplemented
