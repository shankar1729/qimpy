from __future__ import annotations

import numpy as np
import torch
from qimpy import log, grid, rc

from . import Grid, FieldH


class Coulomb_Slab:
    """Coulomb interactions between fields and point charges in a truncated Slab geometry"""

    grid: Grid
    ion_width: float
    sigma: float
    iDir: int  # Truncated direction (zero-based indexing)
    iR: torch.Tensor  # Ewald real-space mesh points
    iG: torch.Tensor  # Ewald reciprocal-space mesh points
    _kernel: torch.Tensor  # Coulomb kernel

    def __init__(self, grid: Grid, n_ions: int, iDir: int) -> None:
        """Initialize truncated coulomb calculation"""
        self.iDir = iDir
        self.grid = grid
        self.update_lattice_dependent(n_ions)

    def update_lattice_dependent(self, n_ions: int) -> None:
        grid = self.grid
        iDir = self.iDir
        lattice = grid.lattice

        Rsq = (self.grid.lattice.Rbasis).square().sum(dim=0)
        hlfL = torch.sqrt(Rsq[self.iDir]) / 2
        self.hlfL = hlfL
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

        # Calculate the Gaussian width for the Ewald sums:
        self.sigma = (
            5
            * lattice.volume**2
            / ((2 * np.pi) ** 2 * (2 * hlfL) ** 2 * max(1, n_ions))
        ) ** (1.0 / 4)

        self.iR = get_mesh(
            self.sigma,
            lattice.Rbasis,
            lattice.Gbasis,
            include_margin=True,
            exclude_zero=False,
        )
        self.iG = get_mesh(
            1.0 / self.sigma,  # flip sigma <-> 1/sigma for reciprocal
            lattice.Gbasis,  # flip R <-> G for reciprocal
            lattice.Rbasis,  # flip R <-> G for reciprocal
            include_margin=False,
            exclude_zero=True,
        )

        log.info(
            f"Ewald:  sigma: {self.sigma:f}"
            f"  nR: {self.iR.shape[0]}  nG: {self.iG.shape[0]}"
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
        sigmaSq = sigma**2
        eta = np.sqrt(0.5) / sigma
        lattice = self.grid.lattice

        # Position independent terms:
        # Ztot = Z.sum()
        ZsqTot = (Z**2).sum()
        Zprod = Z.view(-1, 1, 1) * Z.view(1, -1, 1)

        # First calculate the self-energy correction:

        E = -ZsqTot * eta * (1 / np.sqrt(np.pi))

        # Next calculate the real space sum:

        rCut = 1e-6  # cutoff to detect self-term
        pos = positions - torch.floor(0.5 + positions)  # in [-0.5,0.5)
        x = self.iR.view(1, 1, -1, 3) + (pos.view(-1, 1, 1, 3) - pos.view(1, -1, 1, 3))
        rVec = x @ lattice.Rbasis.T  # Cartesian separations for all pairs
        r = rVec.norm(dim=-1)
        r[torch.where(r < rCut)] = grid.N_SIGMAS_PER_WIDTH * sigma
        Eterm = (0.5 * Zprod * torch.erfc(eta * r) / r).sum()
        E += Eterm
        minus_E_r_by_r = (
            Eterm + (2.0 * eta / np.sqrt(np.pi)) * Zprod * torch.exp(-((eta * r) ** 2))
        ) / (r**2)

        if positions.requires_grad:
            positions.grad -= ((rVec @ lattice.Rbasis) * minus_E_r_by_r[..., None]).sum(
                dim=(1, 2)
            )
        if lattice.requires_grad:
            lattice.grad -= 0.5 * torch.einsum(
                "rij, rija, rijb -> ab", minus_E_r_by_r, rVec, rVec
            )

        # Next calculate reciprocal space sum
        restrict_sum = torch.ones(Z.size(dim=0), Z.size(dim=0)).triu()
        restrict_sum = restrict_sum.view(Z.size(dim=0), Z.size(dim=0), 1)

        delta_sum = 2 - torch.ones(Z.size(dim=0)).diag()
        prefac = (
            np.pi * 2 * self.hlfL * Zprod * restrict_sum * delta_sum / lattice.volume()
        )
        r12 = pos.view(-1, 1, 1, 3) - pos.view(1, -1, 1, 3)
        z12 = r12[..., self.iDir]

        c = 1  # Replace with cosine term
        G = ((self.iG @ lattice.Gbasis.T).square().sum(dim=-1)).sqrt()
        expPlus = torch.exp(G * z12)
        expMinus = 1 / expPlus
        erfcPlus = torch.erfc(eta * (sigmaSq * G + z12))
        erfcMinus = torch.erfc(eta * (sigmaSq * G - z12))

        zTerm = (0.5 / G) * (expPlus * erfcPlus + expMinus * erfcMinus)
        E12 = prefac * c * zTerm
        E += E12
        # E12_r12 += (prefac * -s * zTerm * (2*M_PI)) * iG;
        # E12_r12[iDir] += prefac * c * zTermPrime * L;

        if lattice.requires_grad:
            lattice.grad -= E12 * torch.eye(3)

        return E


def get_mesh(
    sigma: float,
    Rbasis: torch.Tensor,
    Gbasis: torch.Tensor,
    iDir: int,
    include_margin: bool,
    exclude_zero: bool,
) -> torch.Tensor:
    """Create mesh to cover non-negligible terms of Gaussian with width `sigma`.
    The mesh is integral in the coordinates specified by `Rbasis`,
    with `Gbasis` being the corresponding reciprocal basis.
    Negligible is defined at double precision using `grid.N_SIGMAS_PER_WIDTH`.
    Optionally, include a margin of 1 in grid units if `include_margin`.
    Optionally, exclude the zero (origin) point if `exclude_zero`."""
    Rcut = grid.N_SIGMAS_PER_WIDTH * sigma

    # Create parallelopiped mesh:
    RlengthInv = Gbasis.norm(dim=0).to(rc.cpu) / (2 * np.pi)
    Ncut = 1 + torch.ceil(Rcut * RlengthInv).to(torch.int)
    Ncut[iDir] = 0
    iRgrids = tuple(
        torch.arange(-N, N + 1, device=rc.device, dtype=torch.double) for N in Ncut
    )
    iR = torch.stack(torch.meshgrid(*iRgrids, indexing="ij")).flatten(1).T
    if exclude_zero:
        iR = iR[torch.where(iR.abs().sum(dim=-1))[0]]

    # Add margin mesh:
    marginGrid = torch.tensor([-1, 0, +1] if include_margin else [0], device=rc.device)
    iMargin = torch.stack(torch.meshgrid([marginGrid] * 3, indexing="ij")).flatten(1).T
    iRmargin = iR[None, ...] + iMargin[:, None, :]

    # Select points within Rcut (including margin):
    RmagMin = (iRmargin @ Rbasis.T).norm(dim=-1).min(dim=0)[0]
    return iR[torch.where(RmagMin <= Rcut)[0]]
