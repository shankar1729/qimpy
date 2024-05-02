from __future__ import annotations
from typing import Sequence

import numpy as np
import torch

from qimpy import rc, TreeNode
from qimpy.io import CheckpointPath, Unit
from qimpy.transport import material


class PulseB(TreeNode):
    """Magnetic field pulse to rotate spins by specific angles."""

    ab_initio: material.ab_initio.AbInitio

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters
    H0: dict[int, torch.Tensor]  #: Perturbing Hamiltonian amplitude
    omega: dict[int, torch.Tensor]  #: Larmor precession frequencies
    t_start: dict[int, torch.Tensor]  #: Pulse start times
    t_stop: dict[int, torch.Tensor]  #: Pulse stop times

    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        B0: Sequence[float] = (0.0, 0.0, 0.0),
        g: float = 0.0,
        t_start: float = 0.0,
        angle: float = np.pi,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize relaxation time approximation for scattering.

        Parameters
        ----------
        B0
            :yaml:`Magnetic field amplitude vector.`
        g
            :yaml:`Effective gyromagnetic ratio to calculate mean Larmor frequency.`
        t_start
            :yaml:`Start time of oscillating magnetic field pulse.`
        angle
            :yaml:`Target spin rotation angle (in radians).`
        """
        super().__init__()
        self.ab_initio = ab_initio
        self.constant_params = dict(
            B0=torch.tensor(B0, device=rc.device),
            g=torch.tensor(g, device=rc.device),
            t_start=torch.tensor(t_start, device=rc.device),
            angle=torch.tensor(angle, device=rc.device),
        )
        self.H0 = {}
        self.omega = {}
        self.t_start = {}
        self.t_stop = {}

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        self._initialize_fields(patch_id, **params)

    def _initialize_fields(
        self,
        patch_id: int,
        *,
        B0: torch.Tensor,
        g: torch.Tensor,
        t_start: torch.Tensor,
        angle: torch.Tensor,
    ) -> None:
        B = self.ab_initio.B  # constant magnetic field
        mu_B = Unit.MAP["mu_B"]  # Bohr magneton
        assert B is not None
        omega = abs(g * mu_B * B.norm())  # Larmor frequency
        duration = angle / abs(g * mu_B * 0.5 * B0.norm(dim=-1))
        self.H0[patch_id] = self.ab_initio.zeemanH(B0)
        self.omega[patch_id] = omega
        self.t_start[patch_id] = t_start
        self.t_stop[patch_id] = t_start + duration

    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        t_start = self.t_start[patch_id]
        t_stop = self.t_stop[patch_id]
        omega = self.omega[patch_id]
        in_range = torch.where(torch.logical_and(t >= t_start, t <= t_stop), 1.0, 0.0)
        prefactor = 1j * (torch.sin(omega * (t - t_start)) * in_range)
        return prefactor[..., None, None, None] * (rho @ self.H0[patch_id])
