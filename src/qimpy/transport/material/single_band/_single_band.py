from __future__ import annotations
from typing import Union, Sequence, Callable
from dataclasses import dataclass

import numpy as np
import torch

from qimpy import rc, log
from qimpy.mpi import ProcessGrid
from qimpy.io import CheckpointPath
from qimpy.lattice import Lattice
from .. import Material, fermi


class SingleBand(Material):
    """Single-band model for charge transport with energy resolution."""

    def __init__(
        self,
        *,
        lattice: Union[Lattice, dict],
        kmesh: Union[Sequence[int], np.ndarray],
        dispersion: Union[dict, Callable[[torch.Tensor], torch.Tensor]],
        mu: float,
        T: float,
        nT_below: float = 5,
        nT_above: float = 5,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """Initialize a single-band model material.

        Parameters
        ----------
        lattice
            :yaml:`Bulk lattice vectors / unit cell definition.`
        kmesh
            :yaml:`Uniform k-point mesh from which to select relevant k-points.`
        dispersion
            :yaml:`Dispersion relation for the single band model.`
            Supported options include:
            * `v = value` for a linear dispersion relation with velocity `m`
            * `m = value` for a quadratic dispersion relation with effective mass `m`
            * In code, custom function that returns energy given Cartesian k
              (in atomic units) and the lattice. For convenience, the Cartesian k
              will be pre-wrapped aroung k = 0. In principle the energy function
              should be periodic on the reciprocal lattice, but this will not matter
              if the selected k are far from the Brillouin zone boundaries.
        mu
            :yaml:`Backround/reference chemical potential of initial state`
        T
            :yaml:`Backround/reference temperature of initial state`
        nT_below
            :yaml:`Include states with energies starting at this many T below mu.`
        nT_above
            :yaml:`Include states with energies up to this many T above mu.`
        """
        super().__init__()
        self.add_child("lattice", Lattice, lattice, checkpoint_in)
        self.kmesh = kmesh
        self.mu = mu
        self.T = T
        self.nT_below = nT_below
        self.nT_above = nT_above

        # Prepare dispersion relation:
        if isinstance(dispersion, dict):
            if "v" in dispersion:
                self.dispersion = LinearDispersion(**dispersion)
            elif "m" in dispersion:
                self.dispersion = QuadraticDispersion(**dispersion)
            else:
                raise KeyError("Unrecognized dispersion relation")
        else:
            assert callable(dispersion)
            self.dispersion = dispersion

        # Select k-points:
        k0, k1, k2 = [
            torch.cat(
                (
                    torch.arange((Nk + 1) // 2, device=rc.device),
                    torch.arange((Nk + 1) // 2 - Nk, 0, device=rc.device),
                )
            )
            / Nk
            for Nk in kmesh
        ]
        k12 = torch.stack(
            torch.meshgrid(k0[:1], k1, k2, indexing="ij"), dim=-1
        ).flatten(
            0, 2
        )  # k1-k2 slice with k0 = 0
        Emin = mu - nT_below * T
        Emax = mu + nT_above * T
        k_sel = []
        for k0_i in k0:
            k12[:, 0] = k0_i
            k_cart = k12 @ self.lattice.Gbasis.T
            E = self.dispersion(k_cart, self.lattice)
            sel = torch.where(torch.logical_and(E >= Emin, E <= Emax))[0]
            if len(sel):
                k_sel.append(k12[sel].clone())
        k_all = torch.cat(k_sel, dim=0)
        nk = len(k_all)
        wk = 2 / np.prod(kmesh)
        log.info(f"Selected {nk} k-points from k-mesh with dimensions {kmesh}")
        self.initialize(nk=nk, wk=wk, n_bands=1, n_dim=3, process_grid=process_grid)

        # Initialize further properties for selected k-points:
        self.k[:] = k_all[self.k_mine]
        k_cart = self.k @ self.lattice.Gbasis.T
        k_cart.requires_grad_(True)
        E = self.dispersion(k_cart, self.lattice)
        E.sum().backward()  # compute v = dE/dk in k_cart.grad
        self.E[:, 0] = E.detach()
        self.v[:, 0, :] = k_cart.grad.detach()
        self.rho0[:, 0, 0] = fermi(self.E[:, 0], mu, T)

        from matplotlib import pyplot as plt

        kx, ky, _ = k_cart.detach().T
        f = self.rho0[:, 0, 0]
        plt.scatter(kx, ky, c=f * (1 - f) * self.v[:, 0, 1], cmap="viridis")
        plt.gca().set_aspect("equal")
        plt.colorbar()
        plt.show()
        exit()


@dataclass
class LinearDispersion:
    v: float  #: velocity

    def __call__(self, k_cart: torch.Tensor, lattice: Lattice) -> torch.Tensor:
        return self.v * k_cart.norm(dim=-1)


@dataclass
class QuadraticDispersion:
    m: float  #: effective mass

    def __call__(self, k_cart: torch.Tensor, lattice: Lattice) -> torch.Tensor:
        return (0.5 / self.m) * (k_cart**2).sum(dim=-1)
