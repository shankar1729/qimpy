from __future__ import annotations

import numpy as np
import torch

from qimpy import log, rc, grid
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

    def ewald() -> None:
        pass

    def stress() -> None:
        pass
