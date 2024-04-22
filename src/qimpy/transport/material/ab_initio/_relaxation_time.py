from __future__ import annotations

import numpy as np
import torch

from qimpy import TreeNode
from qimpy.io import CheckpointPath
from qimpy.transport import material


class RelaxationTime(TreeNode):
    ab_initio: material.ab_initio.AbInitio
    tau_p: float  #: momentum relaxation time
    tau_s: float  #: spin relaxation time

    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        tau_p: float = np.inf,
        tau_s: float = np.inf,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize relaxation time approximation for scattering.

        Parameters
        ----------
        tau_p
            :yaml:`Momentum relaxation time.`
        tau_s
            :yaml:`Spin relaxation time.`
        """
        super().__init__()
        self.ab_initio = ab_initio
        self.tau_p = tau_p
        self.tau_s = tau_s

    def __bool__(self) -> bool:
        """Whether there is non-zero scattering."""
        return bool(np.isfinite(self.tau_p) or np.isfinite(self.tau_s))

    def rho_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        if not self:
            return torch.zeros_like(rho)
        return NotImplemented
