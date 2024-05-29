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
        self.iDir = iDir  # Figure out where to initialize thi
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
        iDir = self.iDir
        lattice = grid.lattice

        Rsq = (self.grid.lattice.Rbasis).square().sum(dim=0)

        hlfL = torch.sqrt(Rsq[iDir]) / 2
        self.hlfL = hlfL
        log.info(f"hlfL in update_lattice_dependent: {hlfL}")

        iG = grid.get_mesh("H").to(torch.double)
        Gi = iG @ grid.lattice.Gbasis.T
        Gsqi = Gi.square()
        Gsq = Gsqi.sum(dim=-1)
        Gplane = torch.sqrt(Gsq - Gsqi[..., iDir])
        self._kernel = torch.where(
            Gsq == 0.0,
            -2 * np.pi * hlfL**2,
            (4 * np.pi)
            * (1 - torch.exp(-Gplane * hlfL) * torch.cos(np.pi * iG[..., iDir]))
            / Gsq,
        )
        # Calculate the Gaussian width for the Ewald sums:
        self.sigma = (
            5 * lattice.volume**2 / ((2 * np.pi) ** 2 * (2 * hlfL) ** 2)
        ) ** (1.0 / 4)

        self.iR = get_mesh(
            self.sigma,
            lattice.Rbasis,
            lattice.Gbasis,
            iDir,
            include_margin=True,
            exclude_zero=False,
        )
        self.iG = get_mesh(
            1.0 / self.sigma,  # flip sigma <-> 1/sigma for reciprocal
            lattice.Gbasis,  # flip R <-> G for reciprocal
            lattice.Rbasis,  # flip R <-> G for reciprocal
            iDir,
            include_margin=False,
            exclude_zero=True,
        )

        log.info(
            f"Ewald:  sigma: {self.sigma:f}"
            f"  nR: {self.iR.shape[0]}  nG: {self.iG.shape[0]}"
        )
        log.info(f"Ewald iR shape: {self.iR.shape}\n Ewald iG shape: {self.iG.shape}")

    def __call__(self, rho: FieldH, correct_G0_width: bool = False) -> FieldH:
        """Apply coulomb operator on charge density `rho`.
        If correct_G0_width = True, rho is a point charge distribution
        widened by `ion_width` and needs a corresponding G=0 correction.
        """
        assert self.grid is rho.grid
        result = FieldH(self.grid, data=(self._kernel * rho.data))
        return result

    def stress(self, rho1: FieldH, rho2: FieldH) -> torch.Tensor:
        hlfL = self.hlfL
        iG = self.grid.get_mesh("H").to(torch.double)
        G = self.grid.get_mesh("H").to(torch.double) @ self.grid.lattice.Gbasis.T
        Gplane = G.clone()
        Gplane[..., self.iDir] = 0  # Set perpendicular direction elements to zero

        iGplane = iG.clone()
        iGplane[..., self.iDir] = 0  # Set perpendicular direction elements to zero

        iGperp = iG[..., self.iDir]
        Gplaneabs = (Gplane.square().sum(dim=-1)).sqrt()
        Gplaneabsinv = torch.where(Gplaneabs == 0, 0, 1 / Gplaneabs)
        Gsq = G.square().sum(dim=-1)

        expCosTerm = torch.exp(-Gplaneabs * hlfL) * torch.cos(np.pi * iGperp)
        Gsqinv = torch.where(Gsq == 0, 0, 1 / Gsq)
        prefac1 = 2 * (1 - expCosTerm) * Gsqinv
        stress_kernel = torch.einsum(
            "ijk, ijkl, ijkm->lmijk",
            4 * np.pi * Gsqinv * (prefac1 - expCosTerm * hlfL * Gplaneabsinv),
            Gplane,
            Gplane,
        )
        GGT_idir_idir = self.grid.lattice.Gbasis[self.iDir, self.iDir] ** 2

        stress_kernel[self.iDir, self.iDir, ...] = (
            4
            * np.pi
            * torch.where(
                Gsq == 0,
                -(hlfL**2),
                Gsqinv
                * (
                    prefac1 * GGT_idir_idir * iG[..., self.iDir] ** 2
                    + expCosTerm * hlfL * Gplaneabs
                ),
            )
        )

        stress_rho2 = FieldH(self.grid, data=(stress_kernel * rho2.data))
        log.info(f"Stress from CoulombSlab.stress {rho1 ^ stress_rho2}")
        return rho1 ^ stress_rho2

    def ewald(self, positions: torch.Tensor, Z: torch.Tensor) -> float:
        sigma = self.sigma
        sigmaSq = sigma**2
        eta = np.sqrt(0.5) / sigma
        etaSqrtPiInv = 1 / (eta * np.sqrt(np.pi))
        etaSq = eta**2

        lattice = self.grid.lattice
        log.info(f"Positions of ions: {positions} with tensor size {positions.size()}")
        log.info(f"Charges of ions: {Z} with tensor size {Z.size()}")
        # Position independent terms:
        ZsqTot = (Z**2).sum()
        Zprod = Z.view(-1, 1, 1) * Z.view(1, -1, 1)
        # Self-energy correction:

        E = -ZsqTot * eta * (1 / np.sqrt(np.pi))
        log.info(f"Self energy term in ewald sum: {E}")

        # Next calculate the real space sum:

        rCut = 1e-6  # cutoff to detect self-term
        pos0 = torch.tensor((0, 0, 0))
        pos0[self.iDir] = positions[
            0, self.iDir
        ]  # The coordinate in the truncated direction for the first atom in the system.
        pos = positions - torch.floor(
            0.5 + positions - pos0
        )  # positions in [-0.5, 0.5)
        x = self.iR.view(1, 1, -1, 3) + (pos.view(-1, 1, 1, 3) - pos.view(1, -1, 1, 3))
        rVec = x @ lattice.Rbasis.T  # Cartesian separations for all pairs
        r = rVec.norm(dim=-1)
        r[torch.where(r < rCut)] = grid.N_SIGMAS_PER_WIDTH * sigma
        Eterm = (0.5 * Zprod * torch.erfc(eta * r) / r).sum()
        E += Eterm
        minus_E_r_by_r = (
            Zprod * torch.erfc(eta * r) / r
            + (2.0 * eta / np.sqrt(np.pi)) * Zprod * torch.exp(-((eta * r) ** 2))
        ) / (r**2)
        # Next calculate reciprocal space sum
        volPrefac = np.pi * 2 * self.hlfL / lattice.volume
        prefac = (
            volPrefac * Zprod
        )  # Factor of two above is to convert halfL to full length in the truncated direction
        r12 = pos.view(-1, 1, 1, 3) - pos.view(1, -1, 1, 3)
        z12 = r12[..., self.iDir]
        z12 *= 2 * self.hlfL
        G = ((self.iG @ lattice.Gbasis.T).square().sum(dim=-1)).sqrt()

        expPlus = torch.exp(G * z12)
        expMinus = 1 / expPlus
        erfcPlus = torch.erfc(eta * (sigmaSq * G + z12))
        erfcMinus = torch.erfc(eta * (sigmaSq * G - z12))

        c = torch.cos(2 * np.pi * torch.einsum("ijkl, gl->ijkg", r12, self.iG))
        s = torch.sin(2 * np.pi * torch.einsum("ijkl,gl->ijkg", r12, self.iG))

        zTerm = (1 / G) * (expPlus * erfcPlus)
        zTermG0 = -z12 * torch.erf(z12 * eta) - etaSqrtPiInv * torch.exp(
            -etaSq * z12 * z12
        )
        zTerm_force = (1 / G) * (expPlus * erfcPlus + expMinus * erfcMinus)
        zTerm_prime = expPlus * erfcPlus - expMinus * erfcMinus

        E12 = torch.einsum("ijk, ijkl, ijl-> ", prefac, c, zTerm)
        E12 += (prefac * zTermG0).sum()
        E += E12
        E12_r12 = (
            2
            * np.pi
            * torch.einsum(
                "ijk, ijkl, ijl, lm -> im ", prefac, -s, zTerm_force, self.iG
            )
        )
        E12_r12[:, self.iDir] += (
            2 * self.hlfL * torch.einsum("ijk, ijkl, ijl -> i", prefac, c, zTerm_prime)
        )
        E12_r12[:, self.iDir] += (
            2
            * self.hlfL
            * torch.einsum("ijk, ijk -> i", prefac, -2 * torch.erf(z12 * eta))
        )

        if positions.requires_grad:
            real_sum_forces = -(
                (rVec @ lattice.Rbasis) * minus_E_r_by_r[..., None]
            ).sum(dim=(1, 2))
            reciprocal_sum_forces = E12_r12
            positions.grad += real_sum_forces + reciprocal_sum_forces

        if lattice.requires_grad:
            real_sum_stress = -0.5 * torch.einsum(
                "rij, rija, rijb -> ab",
                minus_E_r_by_r,
                rVec,
                rVec,
            )

            lattice.grad += real_sum_stress

            zHat = torch.zeros((3, 3))
            zHat[self.iDir, self.iDir] = 1
            E_RRTzz = (
                torch.einsum(
                    "ijk, ijkl, ijl, ijk -> ", prefac, c, 1 / 2 * zTerm_prime, z12
                )
                * zHat
            )
            E_RRTzz += (
                torch.einsum("ijk, ijk, ijk -> ", prefac, -torch.erf(z12 * eta), z12)
                * zHat
            )
            E_RRTzz += E12 * zHat
            minus_zTerm_G_by_G_1 = torch.einsum("ijl, l -> ijl", zTerm, 1 / G**2)
            minus_zTerm_G_by_G_2 = -torch.einsum(
                "ijl, ijl, l -> ijl", z12, zTerm_prime / 2, 1 / G**2
            )
            minus_zTerm_G_by_G_3 = torch.einsum(
                "l, ijl -> ijl",
                sigma
                * np.sqrt(2 / np.pi)
                * torch.exp(-0.5 * sigmaSq * G**2)
                / G**2,
                torch.exp(-etaSq * z12**2),
            )
            minus_zTerm_G_by_G = (
                minus_zTerm_G_by_G_1 + minus_zTerm_G_by_G_2 + minus_zTerm_G_by_G_3
            )
            reciprocal_sum_stress_nonvol = torch.einsum(
                "ijk, ijkl, ijl, la, lb -> ab",
                prefac,
                c,
                minus_zTerm_G_by_G,
                self.iG @ lattice.Gbasis.T,
                self.iG @ lattice.Gbasis.T,
            )
            reciprocal_sum_stress_nonvol += E_RRTzz

            reciprocal_sum_stress_vol = -E12 * torch.eye(3)
            lattice.grad += reciprocal_sum_stress_vol
            lattice.grad += reciprocal_sum_stress_nonvol
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
    log.info(f"Ncut found in get_mesh: {Ncut}")
    Ncut[iDir] = 0  # Do not sum over truncated direction
    iRgrids = tuple(
        torch.arange(-N, N + 1, device=rc.device, dtype=torch.double) for N in Ncut
    )
    iR = torch.stack(torch.meshgrid(*iRgrids, indexing="ij")).flatten(1).T
    if exclude_zero:
        iR = iR[torch.where(iR.abs().sum(dim=-1))[0]]
    # log.info(f"iR: {iR} with shape {iR.shape}")

    # Add margin mesh:
    marginGrid = torch.tensor([-1, 0, +1] if include_margin else [0], device=rc.device)
    iMargin = torch.stack(torch.meshgrid([marginGrid] * 3, indexing="ij")).flatten(1).T
    iRmargin = iR[None, ...] + iMargin[:, None, :]

    # Select points within Rcut (including margin):
    RmagMin = (iRmargin @ Rbasis.T).norm(dim=-1).min(dim=0)[0]
    return iR[torch.where(RmagMin <= Rcut)[0]]
