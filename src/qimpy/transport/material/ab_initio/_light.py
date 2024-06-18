from __future__ import annotations
from typing import Optional

import torch
import numpy as np

from qimpy import rc, TreeNode
from qimpy.io import CheckpointPath, InvalidInputException
from qimpy.profiler import stopwatch
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
    smearing: float  #: Width of Gaussian
    amp_mat: torch.Tensor  #: Amplitude matrix, precomputed A0 . P or E0 . R

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters

    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        coherent: bool = True,
        gauge: str = "velocity",
        A0: Optional[list[complex]] = None,
        E0: Optional[list[complex]] = None,
        omega: float = 0.0,
        t0: float = 0.0,
        sigma: float = 0.0,
        smearing: float = 0.001,
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
        smearing
            :yaml:`Width of Gaussian function to represent delta function.`
        """
        super().__init__()
        self.coherent = coherent
        self.ab_initio = ab_initio
        self.gauge = gauge

        # Get amplitude from A0 or E0:
        if (A0 is None) == (E0 is None):
            raise InvalidInputException("Exactly one of A0 and E0 must be specified")
        if A0 is not None:
            A0 = torch.tensor(A0, device=rc.device)
        elif E0 is not None:
            A0 = torch.tensor(E0, device=rc.device) / omega
        if A0.shape[-1] == 2:  # handle complex tensors
            A0 = torch.view_as_complex(A0)
        else:
            A0 = A0.to(torch.complex128)

        self.constant_params = dict(
            A0=A0,
            omega=torch.tensor(omega, device=rc.device),
            t0=torch.tensor(t0, device=rc.device),
            sigma=torch.tensor(sigma, device=rc.device),
            smearing=torch.tensor(smearing, device=rc.device),
        )
        self.t0 = {}
        self.sigma = {}
        if self.coherent:
            self.amp_mat = {}
            self.omega = {}
        else:
            self.plus = {}
            self.plus_deg = {}
            self.minus = {}
            self.minus_deg = {}

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        self._initialize_fields(patch_id, **params)

    def _initialize_fields(
        self,
        patch_id: int,
        *,
        A0: torch.Tensor,
        omega: torch.Tensor,
        t0: torch.Tensor,
        sigma: torch.Tensor,
        smearing: torch.Tensor,
    ) -> None:
        ab_initio = self.ab_initio
        if self.gauge == "velocity":
            amp_mat = torch.einsum("i, kiab -> kab", A0, ab_initio.P)
        elif self.gauge == "length":
            amp_mat = torch.einsum("i, kiab -> kab", A0 * omega, ab_initio.R)
        else:
            raise InvalidInputException(
                "Parameter gauge should only be velocity or length"
            )

        omega = omega * torch.ones([1,1]).to(rc.device) # easy for broadcasting
        self.t0[patch_id] = t0
        self.sigma[patch_id] = sigma
        if self.coherent:
            self.amp_mat[patch_id] = amp_mat
            self.omega[patch_id] = omega
        else:  #: lindblad version
            prefac = torch.sqrt(torch.sqrt(torch.pi / (8 * smearing**2)))
            exp_factor = -1.0 / (smearing**2)
            Nk, Nb = ab_initio.E.shape
            dE = ab_initio.E.reshape([1, 1, Nk, Nb, 1]) - ab_initio.E.reshape([1, 1, Nk, 1, Nb])
            plus = prefac * amp_mat * torch.exp(
                exp_factor * ((dE + omega_[:, :, None, None, None]) ** 2)
            )
            minus = prefac * amp_mat * torch.exp(
                exp_factor * ((dE - omega_[:, :, None, None, None]) ** 2)
            )
            plus_deg = plus.swapaxes(-2, -1).conj()
            minus_deg = minus.swapaxes(-2, -1).conj()
            self.identity_mat = torch.tile(torch.eye(Nb), (1, Nk, 1, 1)).to(rc.device)
            self.plus[patch_id] = plus
            self.plus_deg[patch_id] = plus_deg
            self.minus[patch_id] = minus
            self.minus_deg[patch_id] = minus_deg

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        t0 = self.t0[patch_id]
        sigma = self.sigma[patch_id]
        if sigma > 0:
            prefac = torch.exp(-((t - t0) ** 2) / (2 * sigma**2)) / torch.sqrt(
                torch.sqrt(np.pi * sigma**2)
            )
        else:
            prefac = 1.0

        if self.coherent:
            omega = self.omega[patch_id]
            prefac = -0.5j * torch.exp(-1j * omega * t) * prefac  # Louiville, symmetrization
            interaction = prefac[:, :, None, None, None] * self.amp_mat[patch_id][None, None, ...]
            return (interaction - interaction.swapaxes(-2, -1).conj()) @ rho
        else:
            prefac = 0.5 * prefac**2
            I_minus_rho = self.identity_mat - rho
            plus = self.plus[patch_id]
            minus = self.minus[patch_id]
            plus_deg = self.plus_deg[patch_id]
            minus_deg = self.minus_deg[patch_id]
            return prefac * (
                commutator(I_minus_rho @ plus @ rho, plus_deg)
                + commutator(I_minus_rho @ minus @ rho, minus_deg)
            )

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["coherent"] = self.coherent
        attrs["gauge"] = self.gauge
        attrs.update(
            {param: self.constant_params[param].cpu() for param in self.constant_params}
        )
        return ["coherent", "gauge", "A0", "omega", "t0", "sigma", "smearing"]


def commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Commutator of two tensors (along final two dimensions)."""
    return A @ B - B @ A
