from __future__ import annotations

import numpy as np
import torch
from qimpy import log, rc

from . import Grid, FieldH


class Coulomb_Isolated:
    """Coulomb interactions between fields and point charges in a truncated Slab geometry"""

    grid: Grid
    ion_width: float
    _kernel: torch.Tensor  # Coulomb kernel

    def __init__(self, grid: Grid, n_ions: int) -> None:
        """Initialize truncated coulomb calculation"""
        self.grid = grid
        h_max = (
            (grid.lattice.Rbasis.norm(dim=0).to(rc.cpu) / torch.tensor(grid.shape))
            .max()
            .item()
        )
        self.ion_width = 2.2 * h_max  # Best balance from test_nyquist()
        self.update_lattice_dependent(n_ions)

    def update_lattice_dependent(self, n_ions: int) -> None:
        grid = self.grid
        lattice = grid.lattice
        iG = grid.get_mesh("H").to(torch.double)
        Gsq = (iG @ lattice.Gbasis.T).square().sum(dim=-1)
        Rc = 8
        self._kernel = torch.where(
            Gsq == 0.0,
            2 * np.pi * (Rc**2),
            (4 * np.pi) * (1 - torch.cos(Rc * torch.sqrt(Gsq))) / Gsq,
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
        lattice = grid.lattice
        iG = grid.get_mesh("H").to(torch.double)
        Gsq = (iG @ lattice.Gbasis.T).square().sum(dim=-1)
        Rc = 8
        GRc = torch.sqrt(Gsq) * Rc
        stress_prefac = torch.where(
            Gsq == 0,
            0,
            (4 * np.pi)
            * (2 * (1 - torch.cos(GRc)) - GRc * torch.sin(GRc))
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

    def ewald(self, positions: torch.Tensor, Z: torch.Tensor) -> float:
        lattice = self.grid.lattice
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
        return E
