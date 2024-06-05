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
        g_flip: float = 0.0,
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
            :yaml:`Gyromagnetic ratio magnitude to calculate mean Larmor frequency.`
        g_flip
            :yaml:`Gyromagnetic ratio magnitude to calculate flip time.`
            May be different from g for anisotropic system.
            If unspecified or set to zero, assumed to be same as `g`.
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
            g_flip=torch.tensor(g_flip, device=rc.device),
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
        g_flip: torch.Tensor,
        t_starts: torch.Tensor,
        angles: torch.Tensor,
    ) -> None:
        # Construct orthogonal frame for rotating field
        B = self.ab_initio.B  # constant magnetic field
        assert B is not None
        Bhat = B / B.norm()
        B0r = B0 - (B0 @ Bhat)[..., None] * Bhat  # remove parallel component
        B0i = torch.linalg.cross(Bhat, B0r)  # direction after 90 degree rotation

        # Get rotation frequencies and durations
        mu_B = Unit.MAP["mu_B"]  # Bohr magneton
        omega = g * mu_B * B.norm()  # Larmor frequency
        g_flip = torch.where(g_flip == 0.0, g, g_flip)
        durations = angles / abs(g_flip * mu_B * B0r.norm(dim=-1))[..., None]

        self.H0[patch_id] = self.ab_initio.zeemanH(B0r + 1j * B0i)
        self.omega[patch_id] = omega
        self.t_starts[patch_id] = t_starts
        self.t_stops[patch_id] = t_starts + durations

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        t_starts = self.t_starts[patch_id]
        t_stops = self.t_stops[patch_id]
        omega = self.omega[patch_id]
        in_range = torch.where(torch.logical_and(t >= t_starts, t <= t_stops), 1.0, 0.0)
        prefactor = -0.5j * torch.exp(-1j * omega * t) * in_range.sum(dim=-1)
        iH = prefactor[..., None, None, None] * self.H0[patch_id]
        return (iH - iH.swapaxes(-1, -2).conj()) @ rho

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs.update(
            {param: self.constant_params[param].cpu() for param in self.constant_params}
        )
        return ["B0", "g", "g_flip", "t_starts", "angles"]