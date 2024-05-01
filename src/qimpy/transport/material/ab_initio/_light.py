from __future__ import annotations
from typing import Optional

import torch
import numpy as np

from qimpy import rc, TreeNode
from qimpy.io import CheckpointPath, InvalidInputException
from qimpy.transport import material


class Light(TreeNode):
    ab_initio: material.ab_initio.AbInitio
    coherent: bool  #: Whether term is Coherent or Lindbladian
    gauge: str  #: Gauge: one of velocity or length
    A0: torch.Tensor  #: Vector potential amplitude
    E0: torch.Tensor  #: Electric field amplitude
    omega: float  #: light frequency
    t0: float  #: center of Gaussian pulse, if sigma is non-zero
    sigma: float  #: width of Gaussian pulse in time, if non-zero
    A0_dot_P: torch.Tensor  #: Precomputed A0 . P matrix elements
    E0_dot_R: torch.Tensor  #: Precomputed E0 . R matrix elements

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters

    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        coherent: bool = True,
        gauge: str = "velocity",
        A0: Optional[list[complex]] = None,
        E0: Optional[list[complex]] = None,
        omega: float,
        t0: float = 0.0,
        sigma: float = 0.0,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize coherent light interaction.

        Parameters
        ----------
        coherent
            :yaml:`Switch between coherent and Lindbladian implementations.`
        gauge
            :yaml:`Switch between 'velocity' or 'length' gauge.`
        A0
            :yaml:`Vector potential amplitude.`
            TODO: specify details about differences in CW vs pulse mode.
            Exactly one of A0 or E0 must be specified.
        E0:
            :yaml:`Electric-field amplitude.`
            Exactly one of A0 or E0 must be specified.
        omega
            :yaml:`Angular frequency / photon energy of the light.`
        t0
            :yaml:`Center of Gaussian pulse, used only if sigma is non-zero.`
        sigma
            :yaml:`Time width of Gaussian pulse, if non-zero.`
        """
        super().__init__()
        self.coherent = coherent
        self.ab_initio = ab_initio
        self.gauge = gauge

        # Get amplitude from A0 or E0:
        if (A0 is None) == (E0 is None):
            raise InvalidInputException("Exactly one of A0 and E0 must be specified")
        if A0 is not None:
            self.A0 = torch.tensor(A0, device=rc.device)
        elif E0 is not None:
            self.A0 = torch.tensor(E0, device=rc.device) / omega
        if self.A0.shape[-1] == 2:  # handle complex tensors
            self.A0 = torch.view_as_complex(self.A0)
        else:
            self.A0 = self.A0.to(torch.complex128)
        self.E0 = self.A0 * omega

        if self.gauge == "velocity":
            self.A0_dot_P = torch.einsum("i, kiab -> kab", self.A0, ab_initio.P)
        elif self.gauge == "length":
            self.E0_dot_R = torch.einsum("i, kiab -> kab", self.E0, ab_initio.R)
        else:
            raise InvalidInputException(
                "Parameter gauge should only be velocity or length"
            )

        self.omega = omega
        self.t0 = t0
        self.sigma = sigma
        self.coherent = coherent

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        pass  # TODO

    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        prefac = -0.5j * np.exp(1j * self.omega * t)  # with Louiville, symmetrization
        if self.sigma:
            prefac *= np.exp(-((t - self.t0) ** 2) / (2 * self.sigma**2)) / np.sqrt(
                np.sqrt(np.pi) * self.sigma
            )

        if self.gauge == "velocity":
            light_interaction = prefac * self.A0_dot_P
        else:  # self.gauge == "length"
            light_interaction = prefac * self.E0_dot_R

        return (light_interaction - light_interaction.swapaxes(-2, -1).conj()) @ rho
