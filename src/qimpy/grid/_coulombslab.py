from __future__ import annotations

import numpy as np
import torch

from . import Grid, FieldH


class Coulomb_Slab:
    def __init__(self, grid: Grid, n_ions: int, iDir: int) -> None:
        self.iDir = iDir
        self.grid = grid
        self.update_lattice_dependent(n_ions)

    def update_lattice_dependent(self, n_ions: int) -> None:
        grid = self.grid
        iDir = self.iDir

        Rsq = (self.grid.lattice.Rbasis).square().sum(dim=0)
        hlfL = torch.sqrt(Rsq[self.iDir])
        iG = grid.get_mesh("H").to(torch.double)
        Gi = iG @ grid.lattice.Gbasis.T
        Gsqi = Gi.square()
        Gsq = Gsqi.sum(dim=-1)
        Gplane = torch.sqrt(Gsq - Gsqi[..., iDir])
        Gperp = Gi[..., iDir]
        self._kernel = torch.where(
            Gsq == 0.0,
            -0.5 * hlfL**2,
            (4 * np.pi)
            * (1 - torch.exp(-Gplane * hlfL) * torch.cos(np.pi * Gperp))
            / Gsq,
        )

    def __call__(self, rho: FieldH, correct_G0_width: bool = False) -> FieldH:
        """Apply coulomb operator on charge density `rho`.
        If correct_G0_width = True, rho is a point charge distribution
        widened by `ion_width` and needs a corresponding G=0 correction.
        """
        assert self.grid is rho.grid
        result = FieldH(self.grid, data=(self._kernel * rho.data))
        return result

    def stress(self, rho1: FieldH, rho2: FieldH) -> torch.Tensor:
        grid = self.grid
        Rsq = (grid.lattice.Rbasis).square().sum(dim=0)
        hlfL = torch.sqrt(Rsq[self.iDir]) / 2

        G = grid.get_mesh("H").to(torch.double) @ grid.lattice.Gbasis.T
        G = G.permute(3, 0, 1, 2)  # Bringing cartesian coordinate to first axis
        Gplane = G
        Gplane[self.iDir, ...] = 0  # Set perpendicular direction elements to zero
        Gplaneabs = (Gplane.square().sum(dim=0)).sqrt()
        Gperp = G[self.iDir, ...]

        Gsq = G.square().sum(dim=0)

        stress_kernel_plane_1 = torch.where(
            Gsq == 0.0,
            0.0,
            (8 * np.pi)
            * (1 - torch.exp(-Gplaneabs * hlfL) * torch.cos(Gperp * hlfL))
            / (Gsq * Gsq)
            * Gplane[None]
            * Gplane[:, None],
        )
        stress_kernel_plane_2 = torch.where(
            Gsq == 0.0,
            0.0,
            (4 * np.pi)
            * (-torch.exp(-Gplaneabs * hlfL) * torch.cos(Gperp * hlfL))
            * hlfL
            / (Gsq * Gplaneabs)
            * Gplane[None]
            * Gplane[:, None],
        )
        stress_kernel = stress_kernel_plane_1 + stress_kernel_plane_2
        stress_kernel[self.iDir, self.iDir, ...] = (
            8
            * torch.pi
            / (Gsq * Gsq)
            * (1 - torch.exp(-Gplaneabs * hlfL))
            * Gperp
            * Gperp
            + 4
            * torch.pi
            / Gsq
            * torch.exp(-Gplaneabs * hlfL)
            * torch.cos(Gperp * hlfL)
            * Gplaneabs
            * hlfL
        )

        stress_rho2 = FieldH(self.grid, data=(stress_kernel * rho2.data))

        return rho1 ^ stress_rho2

    def ewald(self, positions: torch.Tensor, Z: torch.Tensor) -> float:
        sigma = self.sigma
        # sigmaSq = sigma**2
        eta = np.sqrt(0.5) / sigma
        # lattice = self.grid.lattice

        # Position independent terms:
        # Ztot = Z.sum()
        ZsqTot = (Z**2).sum()
        # First calculate the self-energy correction:

        E = -ZsqTot * eta * (1 / np.sqrt(np.pi))

        return E
