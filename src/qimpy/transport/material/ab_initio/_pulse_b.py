from __future__ import annotations
from typing import Sequence

import numpy as np
import torch

from qimpy import rc, TreeNode
from qimpy.io import CheckpointPath, Unit
from qimpy.profiler import stopwatch
from qimpy.transport import material


class PulseB(TreeNode):
    """Magnetic field pulse to rotate spins by specific angles."""

    ab_initio: material.ab_initio.AbInitio

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters
    H0: dict[int, torch.Tensor]  #: Perturbing Hamiltonian amplitude
    omega: dict[int, torch.Tensor]  #: Larmor precession frequencies
    t_starts: dict[int, torch.Tensor]  #: Pulse start times
    t_stops: dict[int, torch.Tensor]  #: Pulse stop times

    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        B0: Sequence[float] = (0.0, 0.0, 0.0),
        g: float = 0.0,
        t_starts: Sequence[float] = (0.0,),
        angles: Sequence[float] = (np.pi,),
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
        t_starts
            :yaml:`Start times of oscillating magnetic field pulse.`
        angles
            :yaml:`Corresponding target spin rotation angles (in radians).`
        """
        super().__init__()
        self.ab_initio = ab_initio
        self.constant_params = dict(
            B0=torch.tensor(B0, device=rc.device),
            g=torch.tensor(g, device=rc.device),
            t_starts=torch.tensor(t_starts, device=rc.device),
            angles=torch.tensor(angles, device=rc.device),
        )
        self.H0 = {}
        self.omega = {}
        self.t_starts = {}
        self.t_stops = {}

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        self._initialize_fields(patch_id, **params)

    def _initialize_fields(
        self,
        patch_id: int,
        *,
        B0: torch.Tensor,
        g: torch.Tensor,
        t_starts: torch.Tensor,
        angles: torch.Tensor,
    ) -> None:
        B = self.ab_initio.B  # constant magnetic field
        mu_B = Unit.MAP["mu_B"]  # Bohr magneton
        assert B is not None
        omega = abs(g * mu_B * B.norm())  # Larmor frequency
        durations = angles / abs(g * mu_B * 0.5 * B0.norm(dim=-1))[..., None]
        self.H0[patch_id] = self.ab_initio.zeemanH(B0)
        self.omega[patch_id] = omega
        self.t_starts[patch_id] = t_starts
        self.t_stops[patch_id] = t_starts + durations

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        t_starts = self.t_starts[patch_id]
        t_stops = self.t_stops[patch_id]
        omega = self.omega[patch_id]
        in_range = torch.where(torch.logical_and(t >= t_starts, t <= t_stops), 1.0, 0.0)
        prefactor = 1j * (torch.cos(omega * t) * in_range.sum(dim=-1))
        return prefactor[..., None, None, None] * (rho @ self.H0[patch_id])
