from __future__ import annotations
from typing import Callable

import numpy as np
import torch

from qimpy import rc, MPI
from qimpy.mpi import ProcessGrid
from qimpy.io import CheckpointPath
from . import Material


class FermiCircle(Material):
    """Fermi-circle representation suitable for graphene and 2DEGs."""

    kF: float  #: Fermi wave-vector
    vF: float  #: Fermi velocity

    def __init__(
        self,
        *,
        kF: float,
        vF: float,
        N_theta: int,
        theta0: float = 0.0,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize ab initio material.

        Parameters
        ----------
        kF
            :yaml:`Fermi wave vector in atomic units.`
        vF
            :yaml:`Fermi velocity in atomic units.`
        N_theta
            :yaml:`Number of k along Fermi circle.`
        theta0
            :yaml:`Angle of first k-point.`
        """
        self.kF = kF
        self.vF = vF

        # Create theta grid on Fermi circle:
        dtheta = 2 * np.pi / N_theta
        theta = theta0 + torch.arange(N_theta, device=rc.device) * dtheta
        k_hat = torch.stack([theta.cos(), theta.sin()], dim=-1)
        k = k_hat * kF
        v = k_hat[:, None] * vF
        E = torch.zeros((N_theta, 1))
        wk = 1.0 / N_theta
        super().__init__(
            k=k, wk=wk, E=E, v=v, checkpoint_in=checkpoint_in, process_grid=process_grid
        )

    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        return SpecularReflector(n, self.v, self.comm)


class SpecularReflector:
    """Reflect velocities specularly i.e. with reflection angle = incidence angle."""

    def __init__(self, n: torch.Tensor, v: torch.Tensor, comm: MPI.Comm) -> None:
        assert comm.size == 1  # TODO: implement k-point split
        assert v.shape[1] == 1  # only for single-band case
        v_normal = torch.einsum(
            "ri, rk -> rki", n, torch.einsum("ri, ki -> rk", n, v[:, 0])
        )
        v_reflected = v[None, :, 0] - 2 * v_normal
        v_diff = (v_reflected[:, :, None] - v[None, None, :, 0]).norm(dim=-1)
        self.i_reflected = v_diff.argmin(dim=2)

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(rho)
        for ik, index in enumerate(self.i_reflected):
            out[ik] = rho[ik, :, index]
        return out
