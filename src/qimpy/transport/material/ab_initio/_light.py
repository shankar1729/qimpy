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
    omega: float  #: light frequency
    t0: float  #: center of Gaussian pulse, if sigma is non-zero
    sigma: float  #: width of Gaussian pulse in time, if non-zero

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
        if self.gauge == "velocity":
            if A0 is not None:
                self.A0 = torch.tensor(A0, device=rc.device)
            elif E0 is not None:
                self.A0 = torch.tensor(E0, device=rc.device) / omega
            if self.A0.shape[-1] == 2:  # handle complex tensors
                self.A0 = torch.view_as_complex(self.A0)
            else:
                self.A0 = self.A0 * (1.0 + 0.0j)
            self.A0real_dot_P = torch.einsum("i, kiab -> kab", (1.0 + 0.0j) * self.A0.real, ab_initio.P)
            self.A0imag_dot_P = torch.einsum("i, kiab -> kab", (1.0 + 0.0j) * self.A0.imag, ab_initio.P) 
        elif self.gauge == "length":
            if E0 is not None:
                self.E0 = torch.tensor(E0, device=rc.device)
            elif A0 is not None:
                self.E0 = torch.tensor(A0, device=rc.device) * omega
            if self.E0.shape[-1] == 2:  # handle complex tensors
                self.E0 = torch.view_as_complex(self.E0)
            else:
                self.E0 = self.E0 * (1.0 + 0.0j)
            self.E0real_dot_R = torch.einsum("i, kiab -> kab", (1.0 + 0.0j) * self.E0.real, ab_initio.R)
            self.E0imag_dot_R = torch.einsum("i, kiab -> kab", (1.0 + 0.0j) * self.E0.imag, ab_initio.R)
        else:
            raise InvalidInputException("Parameter gauge should only be velocity or length")

        self.omega = omega
        self.t0 = t0
        self.sigma = sigma
        self.coherent = coherent

    def rho_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        prefac = np.exp( -(t-t0)**2/(2*self.sigma**2) )/np.sqrt(np.sqrt(np.pi)*self.sigma) if self.sigma > 0 else 1
        if self.gauge == "velocity":
            light_interaction = prefac * ((-1.0j * np.cos(self.omega * t)) * self.A0real_dot_P +
                               (-1.0j * np.sin(self.omega * t)) * self.A0imag_dot_P)
        elif self.gauge == "length":
            light_interaction = prefac * ((-1.0j * np.cos(self.omega * t)) * self.E0real_dot_R +
                               (-1.0j * np.sin(self.omega * t)) * self.E0imag_dot_R)
        return light_interaction @ rho
        
