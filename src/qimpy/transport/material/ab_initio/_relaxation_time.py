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

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters

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

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        pass  # TODO

    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        raise NotImplementedError
