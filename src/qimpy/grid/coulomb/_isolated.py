from __future__ import annotations

import numpy as np
import torch

from qimpy import log
from qimpy.lattice import Lattice
from qimpy.grid import Grid, FieldH, coulomb


class KernelIsolated:
    """Coulomb interactions between fields with no periodicity: Wigner-Seitz version."""

    grid: Grid
    _kernel: torch.Tensor  # Coulomb kernel

    def __init__(self, coul: coulomb.Coulomb) -> None:
        self.grid = coul.grid
        raise NotImplementedError

    def __call__(self, rho: FieldH, correct_G0_width: bool = False) -> FieldH:
        assert self.grid is rho.grid
        raise NotImplementedError

    def stress(self, rho1: FieldH, rho2: FieldH) -> torch.Tensor:
        raise NotImplementedError


class KernelSpherical:
    """Coulomb interactions between fields with no periodicity: analytical version."""

    grid: Grid
    radius: float
    _kernel: torch.Tensor  # Coulomb kernel

    def __init__(self, coul: coulomb.Coulomb) -> None:
        """Initialize truncated coulomb calculation"""
        self.grid = grid = coul.grid
        if coul.radius:
            self.radius = coul.radius
        else:
            raise NotImplementedError  # TODO: determine in-radius
        iG = grid.get_mesh("H").to(torch.double)
        Gsq = (iG @ grid.lattice.Gbasis.T).square().sum(dim=-1)
        self._kernel = torch.where(
            Gsq == 0.0,
            2 * np.pi * (self.radius**2),
            (4 * np.pi) * (1 - torch.cos(self.radius * torch.sqrt(Gsq))) / Gsq,
        )

    def __call__(self, rho: FieldH, correct_G0_width: bool = False) -> FieldH:
        assert self.grid is rho.grid
        result = FieldH(self.grid, data=(self._kernel * rho.data))
        return result

    def stress(self, rho1: FieldH, rho2: FieldH) -> torch.Tensor:
        grid = self.grid
        lattice = grid.lattice
        iG = grid.get_mesh("H").to(torch.double)
        Gsq = (iG @ lattice.Gbasis.T).square().sum(dim=-1)
        Gradius = torch.sqrt(Gsq) * self.radius
        stress_prefac = torch.where(
            Gsq == 0,
            0,
            (4 * np.pi)
            * (2 * (1 - torch.cos(Gradius)) - Gradius * torch.sin(Gradius))
            / (Gsq * Gsq),
        )
        stress_kernel = torch.einsum(
            "ijk, ijka, ijkb -> abijk",
            stress_prefac,
            iG @ lattice.Gbasis.T,
            iG @ lattice.Gbasis.T,
        )
        stress_rho2 = FieldH(self.grid, data=(stress_kernel * rho2.data))
        return rho1 ^ stress_rho2


class EwaldIsolated:
    """Coulomb interactions between point charges with no periodicity."""

    lattice: Lattice

    def __init__(self, lattice: Lattice) -> None:
        self.lattice = lattice

    def __call__(self, positions: torch.Tensor, Z: torch.Tensor) -> float:
        lattice = self.lattice
        Zprod = Z.view(-1, 1) * Z.view(1, -1)
        x = positions.view(-1, 1, 3) - positions.view(1, -1, 3)
        log.info(f"Positions in Coulomb_Isolated.ewald: {positions}")
        rVec = x @ lattice.Rbasis.T  # Cartesian separations for all pairs
        log.info(f"Z (atomic charges in Coulomb_Isolated.ewald): {Z}")
        r = rVec.norm(dim=-1)
        rinv = torch.where(r == 0, 0, 1 / r)
        E = 0.5 * torch.einsum("ij, ij -> ", Zprod, rinv)
        log.info(f"r @ lattice.Rbasis shape: {(rVec @ lattice.Rbasis).shape}")
        minus_dE_r_by_r = torch.einsum(
            "ij, ij, ijk -> ik", Zprod, rinv**3, (rVec @ lattice.Rbasis)
        )
        if positions.requires_grad:
            positions.grad += -minus_dE_r_by_r
            torch.set_printoptions(19)
            log.info(f"Forces in Coulomb_Isolated.ewald: {minus_dE_r_by_r}")

        if lattice.requires_grad:
            real_sum_stress = 0.5 * torch.einsum(
                "ij, ij, ija, ijb -> ab", Zprod, rinv**3, rVec, rVec
            )
            log.info(f"Stresses in Coulomb_Isolted.ewald: {real_sum_stress}")
            lattice.grad -= real_sum_stress
        return E.item()
