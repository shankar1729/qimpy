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
    tau_p: float  #: Momentum relaxation time
    tau_ee: float  #: Electron internal scattering time (momentum-conserving)

    def __init__(
        self,
        *,
        kF: float,
        vF: float,
        N_theta: int,
        tau_p: float,
        tau_ee: float,
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
        self.tau_p = tau_p
        self.tau_ee = tau_ee

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

    def get_contact_distribution(
        self, n: torch.Tensor, *, dmu: float = 0.0, vD: float = 0.0
    ) -> torch.Tensor:
        """Return contact distribution function for specified chemical potential
        shift and drift velocity. Note that positive vD corresponds to current
        flowing into the device (along -n), while negative vD flows out (along +n)."""
        v_hat = self.transport_velocity / self.vF
        return dmu - (n @ v_hat.T) * (vD / self.vF)  # TODO: check

    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        return SpecularReflector(n, self.v, self.comm)

    def rho_dot_scatter(self, rho: torch.Tensor) -> torch.Tensor:
        # Compute stationary and moving equlibrium carrier densities:
        v = self.transport_velocity
        rho_0 = rho.mean(dim=-1)[..., None]
        rho_v = (
            torch.einsum(
                "...i, ki -> ...k", torch.einsum("...k, ki -> ...i", rho, v), v
            )
            / v.square().sum()
        )
        return -(rho - rho_0) / self.tau_p - (rho - rho_0 - rho_v) / self.tau_ee


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
        i_reflected = v_diff.argmin(dim=2)

        # Compute flattened index on Nr x Nk for efficient indexing
        self.i_reflected_flat = (
            i_reflected + torch.arange(len(n), device=rc.device)[:, None] * len(v)
        ).flatten()

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        rho_flat = rho.flatten(1, 2)  # flatten r and k indices
        out_flat = rho_flat[:, self.i_reflected_flat]
        return out_flat.unflatten(1, rho.shape[1:])  # restore r and k
