from __future__ import annotations

import numpy as np
import torch

from qimpy import log, rc
from qimpy.lattice import Lattice
from qimpy.grid import Grid, FieldH, coulomb


class KernelSlab:
    """Coulomb interactions between fields in a truncated Slab geometry."""

    grid: Grid
    i_dir: int  # Truncated direction (zero-based indexing)
    radius: float  # Range of truncation
    _kernel: torch.Tensor  # Coulomb kernel

    def __init__(self, coul: coulomb.Coulomb, i_dir: int) -> None:
        """Initialize truncated coulomb calculation"""
        self.grid = grid = coul.grid
        self.i_dir = i_dir
        if coul.radius:
            self.radius = coul.radius
        else:
            self.radius = grid.lattice.Rbasis[:, i_dir].norm() * 0.5
        iG = grid.get_mesh("H").to(torch.double)
        Gi = iG @ grid.lattice.Gbasis.T
        Gsqi = Gi.square()
        Gsq = Gsqi.sum(dim=-1)
        Gplane = torch.sqrt(Gsq - Gsqi[..., i_dir])
        self._kernel = torch.where(
            Gsq == 0.0,
            -2 * np.pi * self.radius**2,
            (4 * np.pi)
            * (1 - torch.exp(-Gplane * self.radius) * torch.cos(np.pi * iG[..., i_dir]))
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
        radius = self.radius
        iG = self.grid.get_mesh("H").to(torch.double)
        G = self.grid.get_mesh("H").to(torch.double) @ self.grid.lattice.Gbasis.T
        Gplane = G.clone()
        Gplane[..., self.i_dir] = 0  # Set perpendicular direction elements to zero

        iGplane = iG.clone()
        iGplane[..., self.i_dir] = 0  # Set perpendicular direction elements to zero

        iGperp = iG[..., self.i_dir]
        Gplaneabs = (Gplane.square().sum(dim=-1)).sqrt()
        Gplaneabsinv = torch.where(Gplaneabs == 0, 0, 1 / Gplaneabs)
        Gsq = G.square().sum(dim=-1)

        expCosTerm = torch.exp(-Gplaneabs * radius) * torch.cos(np.pi * iGperp)
        Gsqinv = torch.where(Gsq == 0, 0, 1 / Gsq)
        prefac1 = 2 * (1 - expCosTerm) * Gsqinv
        stress_kernel = torch.einsum(
            "ijk, ijkl, ijkm->lmijk",
            4 * np.pi * Gsqinv * (prefac1 - expCosTerm * radius * Gplaneabsinv),
            Gplane,
            Gplane,
        )
        GGT_idir_idir = self.grid.lattice.Gbasis[self.i_dir, self.i_dir] ** 2

        stress_kernel[self.i_dir, self.i_dir, ...] = (
            4
            * np.pi
            * torch.where(
                Gsq == 0,
                -(radius**2),
                Gsqinv
                * (
                    prefac1 * GGT_idir_idir * iG[..., self.i_dir] ** 2
                    + expCosTerm * radius * Gplaneabs
                ),
            )
        )

        stress_rho2 = FieldH(self.grid, data=(stress_kernel * rho2.data))
        return rho1 ^ stress_rho2


class EwaldSlab:
    """Coulomb interactions between point charges in a truncated Slab geometry."""

    lattice: Lattice
    i_dir: int  # Truncated direction (zero-based indexing)
    area: float  #: Area of periodic dimensions
    sigma: float  #: Ewald range-separation parameter
    iR: torch.Tensor  #: Ewald real-space mesh points
    iG: torch.Tensor  #: Ewald reciprocal-space mesh points

    def __init__(self, lattice: Lattice, i_dir: int) -> None:
        self.lattice = lattice
        self.i_dir = i_dir
        self.area = lattice.volume / lattice.Rbasis[:, i_dir].norm().item()

        # Calculate the Gaussian width for the Ewald sums:
        self.sigma = (5 * (self.area / (2 * np.pi)) ** 2) ** (1.0 / 4)

        self.iR = get_mesh(
            self.sigma,
            lattice.Rbasis,
            lattice.Gbasis,
            i_dir,
            include_margin=True,
            exclude_zero=False,
        )
        self.iG = get_mesh(
            1.0 / self.sigma,  # flip sigma <-> 1/sigma for reciprocal
            lattice.Gbasis,  # flip R <-> G for reciprocal
            lattice.Rbasis,  # flip R <-> G for reciprocal
            i_dir,
            include_margin=False,
            exclude_zero=True,
        )

        log.info(
            f"Ewald:  sigma: {self.sigma:f}"
            f"  nR: {self.iR.shape[0]}  nG: {self.iG.shape[0]}"
        )
        log.info(f"Ewald iR shape: {self.iR.shape}\n Ewald iG shape: {self.iG.shape}")

    def __call__(self, positions: torch.Tensor, Z: torch.Tensor) -> float:
        lattice = self.lattice
        Lz = lattice.volume / self.area
        sigma = self.sigma
        sigmaSq = sigma**2
        eta = np.sqrt(0.5) / sigma
        etaSqrtPiInv = 1 / (eta * np.sqrt(np.pi))
        etaSq = eta**2

        # Position independent terms:
        ZsqTot = (Z**2).sum()
        Zprod = Z.view(-1, 1) * Z.view(1, -1)
        # Self-energy correction:

        E = -ZsqTot * eta * (1 / np.sqrt(np.pi))

        # Next calculate the real space sum:

        rCut = 1e-6  # cutoff to detect self-term
        pos0 = torch.tensor((0, 0, 0))
        pos0[self.i_dir] = positions[
            0, self.i_dir
        ]  # The coordinate in the truncated direction for the first atom in the system.
        pos = positions - torch.floor(
            0.5 + positions - pos0
        )  # positions in [-0.5, 0.5)
        x = self.iR.view(1, 1, -1, 3) + (pos.view(-1, 1, 1, 3) - pos.view(1, -1, 1, 3))
        rVec = x @ lattice.Rbasis.T  # Cartesian separations for all pairs
        r = rVec.norm(dim=-1)
        r[torch.where(r < rCut)] = coulomb.N_SIGMAS_PER_WIDTH * sigma
        Eterm = 0.5 * torch.einsum("ij, ijk -> ", Zprod, torch.erfc(eta * r) / r)
        E += Eterm

        minus_E_r_by_r = torch.einsum(
            "ij, ijk -> ijk",
            Zprod,
            torch.erfc(eta * r) / r**3
            + (2.0 * eta / np.sqrt(np.pi)) * torch.exp(-((eta * r) ** 2)) / (r**2),
        )
        # Next calculate reciprocal space sum
        prefac = Zprod * np.pi / self.area
        r12 = pos.view(-1, 1, 3) - pos.view(1, -1, 3)
        z12 = r12[..., self.i_dir]
        z12 *= Lz
        G = ((self.iG @ lattice.Gbasis.T).square().sum(dim=-1)).sqrt()

        expPlus = torch.exp(torch.einsum("i, jk -> jki", G, z12))
        expMinus = 1 / expPlus
        erfcPlus = torch.erfc(
            eta * (sigmaSq * G.view(1, 1, -1) + z12.view(*z12.size(), 1))
        )
        erfcMinus = torch.erfc(
            eta * (sigmaSq * G.view(1, 1, -1) - z12.view(*z12.size(), 1))
        )

        c = torch.cos(2 * np.pi * torch.einsum("ijl, gl -> ijg", r12, self.iG))
        s = torch.sin(2 * np.pi * torch.einsum("ijl, gl -> ijg", r12, self.iG))

        zTerm = torch.einsum("i, jki -> jki", 1 / G, expPlus * erfcPlus)
        zTermG0 = -z12 * torch.erf(z12 * eta) - etaSqrtPiInv * torch.exp(
            -etaSq * z12 * z12
        )
        zTerm_force = torch.einsum(
            "i, jki -> jki", 1 / G, expPlus * erfcPlus + expMinus * erfcMinus
        )
        zTerm_prime = expPlus * erfcPlus - expMinus * erfcMinus

        E12 = torch.einsum("ij, ijl, ijl-> ", prefac, c, zTerm)
        E12 += (prefac * zTermG0).sum()
        E += E12
        E12_r12 = (
            2
            * np.pi
            * torch.einsum("ij, ijl, ijl, lm -> im ", prefac, -s, zTerm_force, self.iG)
        )
        E12_r12[:, self.i_dir] += Lz * torch.einsum(
            "ij, ijl, ijl -> i", prefac, c, zTerm_prime
        )
        E12_r12[:, self.i_dir] += Lz * torch.einsum(
            "ij, ij -> i", prefac, -2 * torch.erf(z12 * eta)
        )

        if positions.requires_grad:
            real_sum_forces = -(
                (rVec @ lattice.Rbasis) * minus_E_r_by_r[..., None]
            ).sum(dim=(1, 2))
            reciprocal_sum_forces = E12_r12
            positions.grad += real_sum_forces + reciprocal_sum_forces
            torch.set_printoptions(19)

        if lattice.requires_grad:
            real_sum_stress = -0.5 * torch.einsum(
                "rij, rija, rijb -> ab",
                minus_E_r_by_r,
                rVec,
                rVec,
            )

            lattice.grad += real_sum_stress

            zHat = torch.zeros((3, 3))
            zHat[self.i_dir, self.i_dir] = 1
            E_RRTzz = (
                torch.einsum(
                    "ij, ijl, ijl, ij -> ", prefac, c, 1 / 2 * zTerm_prime, z12
                )
                * zHat
            )
            E_RRTzz += (
                torch.einsum("ij, ij, ij -> ", prefac, -torch.erf(z12 * eta), z12)
                * zHat
            )
            E_RRTzz += E12 * zHat
            minus_zTerm_G_by_G_1 = torch.einsum("ijl, l -> ijl", zTerm, 1 / G**2)
            minus_zTerm_G_by_G_2 = -torch.einsum(
                "ij, ijl, l -> ijl", z12, zTerm_prime / 2, 1 / G**2
            )
            minus_zTerm_G_by_G_3 = torch.einsum(
                "l, ij -> ijl",
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
                "ij, ijl, ijl, la, lb -> ab",
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
    i_dir: int,
    include_margin: bool,
    exclude_zero: bool,
) -> torch.Tensor:
    """Create mesh to cover non-negligible terms of Gaussian with width `sigma`.
    The mesh is integral in the coordinates specified by `Rbasis`,
    with `Gbasis` being the corresponding reciprocal basis.
    Negligible is defined at double precision using `coulomb.N_SIGMAS_PER_WIDTH`.
    Optionally, include a margin of 1 in grid units if `include_margin`.
    Optionally, exclude the zero (origin) point if `exclude_zero`."""
    Rcut = coulomb.N_SIGMAS_PER_WIDTH * sigma

    # Create parallelopiped mesh:
    RlengthInv = Gbasis.norm(dim=0).to(rc.cpu) / (2 * np.pi)
    Ncut = 1 + torch.ceil(Rcut * RlengthInv).to(torch.int)
    Ncut[i_dir] = 0  # Do not sum over truncated direction
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
