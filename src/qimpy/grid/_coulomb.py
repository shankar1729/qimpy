from __future__ import annotations
import qimpy as qp
import numpy as np
import torch


class Coulomb:
    """Coulomb interactions between fields and point charges.
    TODO: support non-periodic geometries (truncation)."""

    grid: qp.grid.Grid  #: Grid associated with fields for coulomb interaction
    ion_width: float  #: Ion-charge gaussian width for embedding and solvation
    sigma: float  #: Ewald range-separation parameter
    iR: torch.Tensor  #: Ewald real-space mesh points
    iG: torch.Tensor  #: Ewald reciprocal-space mesh points
    _kernel: torch.Tensor  #: Cached coulomb kernel

    def __init__(self, grid: qp.grid.Grid, n_ions: int) -> None:
        """Initialize coulomb interactions.

        Parameters
        ----------
        grid
            Fields for coulomb interaction will be on this grid.
        n_ions
            Number of point charges to optimize Ewald sums for.
        """
        self.grid = grid
        rc = qp.rc
        lattice = grid.lattice

        # Determine ionic width from maximum grid spacing
        h_max = (
            (lattice.Rbasis.norm(dim=0).to(rc.cpu) / torch.tensor(grid.shape))
            .max()
            .item()
        )
        self.ion_width = 2.2 * h_max  # Best balance from test_nyquist()
        qp.log.info(f"Ionic width for embedding / fluids: {self.ion_width:f}")

        # Determine optimum gaussian width for Ewald sums:
        # Uses fact that number of reciprocal cells is proportional to lattice
        # volume and number of real space cells inversely proportional to it.
        # Also accounts for real space cost ~ Natoms^2/cell,
        # while reciprocal space cost ~ Natoms/cell.
        self.sigma = (lattice.volume ** 2 / ((2 * np.pi) ** 3 * max(1, n_ions))) ** (
            1.0 / 6
        )

        # Compute Ewald real and reciprocal meshes:
        def get_mesh(
            Rbasis: torch.Tensor,
            Gbasis: torch.Tensor,
            sigma: float,
            include_margin: bool,
            exclude_zero: bool,
        ) -> torch.Tensor:
            """Create mesh that includes all non-zero terms based on sigma,
            optionally including a margin of 1 in grid units,
            and optionaly excluding the zero (origin) point"""
            Rcut = qp.grid.N_SIGMAS_PER_WIDTH * sigma
            # Create parallelopiped mesh:
            RlengthInv = Gbasis.norm(dim=0).to(rc.cpu) / (2 * np.pi)
            Ncut = 1 + torch.ceil(Rcut * RlengthInv).to(torch.int)
            iRgrids = [
                torch.arange(-N, N + 1, device=rc.device, dtype=torch.double)
                for N in Ncut
            ]
            iR = (
                torch.stack(torch.meshgrid(*tuple(iRgrids), indexing="ij")).flatten(1).T
            )
            if exclude_zero:
                iR = iR[torch.where(iR.abs().sum(dim=-1))[0]]
            # Add margin mesh:
            marginGrid = torch.tensor(
                [-1, +1] if include_margin else [0], device=rc.device
            )
            iMargin = (
                torch.stack(torch.meshgrid([marginGrid] * 3, indexing="ij"))
                .flatten(1)
                .T
            )
            iRmargin = iR[None, ...] + iMargin[:, None, :]
            # Select points within Rcut (including margin):
            RmagMin = (iRmargin @ Rbasis.T).norm(dim=-1).min(dim=0)[0]
            return iR[torch.where(RmagMin <= Rcut)[0]]

        # --- real-space mesh
        self.iR = get_mesh(
            lattice.Rbasis,
            lattice.Gbasis,
            self.sigma,
            include_margin=True,
            exclude_zero=False,
        )
        # --- reciprocal space mesh (flip R <-> G, sigma <-> 1/sigma)
        self.iG = get_mesh(
            lattice.Gbasis,
            lattice.Rbasis,
            1.0 / self.sigma,
            include_margin=False,
            exclude_zero=True,
        )
        qp.log.info(
            f"Ewald:  sigma: {self.sigma:f}"
            f"  nR: {self.iR.shape[0]}  nG: {self.iG.shape[0]}"
        )

        # Set up kernel:
        iG = grid.get_mesh("H").to(torch.double)  # half-space
        Gsq = (iG @ grid.lattice.Gbasis.T).square().sum(dim=-1)
        self._kernel = torch.where(Gsq == 0.0, 0.0, (4 * np.pi) / Gsq)

    def __call__(
        self, rho: qp.grid.FieldH, correct_G0_width: bool = False
    ) -> qp.grid.FieldH:
        """Apply coulomb operator on charge density `rho`.
        If correct_G0_width = True, rho is a point charge distribution
        widened by `ion_width` and needs a corresponding G=0 correction.
        """
        assert self.grid is rho.grid
        result = qp.grid.FieldH(self.grid, data=(self._kernel * rho.data))
        if correct_G0_width:
            result.o += rho.o * (4 * np.pi * (-0.5 * (self.ion_width ** 2)))
        return result

    def stress(self, rho1: qp.grid.FieldH, rho2: qp.grid.FieldH) -> torch.Tensor:
        """Return stress due to Coulomb interaction between `rho1` and `rho2`.
        The result has dimensions of energy, appropriate for adding to `lattice.grad`.
        """
        G = self.grid.get_mesh("H").to(torch.double) @ self.grid.lattice.Gbasis.T
        Gsq = G.square().sum(dim=-1)
        G = G.permute(3, 0, 1, 2)  # bring gradient direction to front
        stress_kernel = (
            torch.where(Gsq == 0.0, 0.0, (8 * np.pi) / (Gsq * Gsq))
            * G[None]
            * G[:, None]
        )
        stress_rho2 = qp.grid.FieldH(self.grid, data=(stress_kernel * rho2.data))
        return rho1 ^ stress_rho2

    def ewald(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
    ) -> float:
        """Compute Ewald energy, and optionally accumulate gradients.
        Each gradient contribution is accumulated to a `grad` attribute,
        only if the corresponding `requires_grad` is enabled.
        Force contributions are collected in `positions.grad`.
        Stress contributions are collected in `self.grid.lattice.grad`.

        Parameters
        ----------
        positions
            Positions (fractional coordinates) of point charges
        Z
            Charges of each point charge
        """
        sigma = self.sigma
        sigmaSq = sigma ** 2
        eta = np.sqrt(0.5) / sigma
        lattice = self.grid.lattice

        # Position independent terms:
        Ztot = Z.sum()
        ZsqTot = (Z ** 2).sum()
        E_Gzero = 0.5 * ((4 * np.pi) * (-0.5 * sigmaSq * (Ztot ** 2)) / lattice.volume)
        E_self = 0.5 * (-ZsqTot * eta * (2.0 / np.sqrt(np.pi)))
        E = E_Gzero + E_self
        if lattice.requires_grad:
            lattice.grad -= E_Gzero * torch.eye(3, device=Z.device)

        # Real-space sum:
        rCut = 1e-6  # cutoff to detect self-term
        pos = positions - torch.floor(0.5 + positions)  # in [-0.5,0.5)
        x = self.iR.view(1, 1, -1, 3) + (pos.view(-1, 1, 1, 3) - pos.view(1, -1, 1, 3))
        rVec = x @ lattice.Rbasis.T  # Cartesian separations for all pairs
        r = rVec.norm(dim=-1)
        r[torch.where(r < rCut)] = qp.grid.N_SIGMAS_PER_WIDTH * sigma
        Zprod = Z.view(-1, 1, 1) * Z.view(1, -1, 1)
        Eterm = Zprod * torch.erfc(eta * r) / r
        minus_E_r_by_r = (
            Eterm + (2.0 * eta / np.sqrt(np.pi)) * Zprod * torch.exp(-((eta * r) ** 2))
        ) / (
            r ** 2
        )  # -(dE/dr)/r
        E += 0.5 * Eterm.sum()
        if positions.requires_grad:
            positions.grad -= ((rVec @ lattice.Rbasis) * minus_E_r_by_r[..., None]).sum(
                dim=(1, 2)
            )
        if lattice.requires_grad:
            lattice.grad -= 0.5 * torch.einsum(
                "rij, rija, rijb -> ab", minus_E_r_by_r, rVec, rVec
            )

        # Reciprocal space sum:
        Sf = torch.exp((-2j * np.pi) * (self.iG @ pos.T))  # Translation phases
        SfZ = (Sf * Z[None, :]).sum(dim=-1)  # Charge structure factor
        G = self.iG @ lattice.Gbasis.T
        Gsq = (G ** 2).sum(dim=-1)
        Eterm = (4 * np.pi / lattice.volume) * torch.exp((-0.5 * sigmaSq) * Gsq) / Gsq
        Erecip = 0.5 * (Eterm @ (SfZ.abs() ** 2))
        E += Erecip
        if positions.requires_grad:
            positions.grad += (
                (2 * np.pi)
                * (((Eterm * SfZ).conj()[:, None] * Sf).imag * Z[None, :]).T
                @ self.iG
            )
        if lattice.requires_grad:
            lattice.grad += 0.5 * torch.einsum(
                "g, ga, gb -> ab",
                (SfZ.abs() ** 2) * Eterm * (sigmaSq + 2.0 / Gsq),
                G,
                G,
            )
            lattice.grad -= Erecip * torch.eye(3, device=Z.device)
        return E


if __name__ == "__main__":

    def test_nyquist():
        """Test dependence of Nyquist frequency with broadening
        to inform heuristic for best ion width selection"""
        import matplotlib.pyplot as plt

        dx = 0.2  # Typical grid spacing at 100 Eh plane-wave spacing
        rTest = 3.0  # Worst-case ion-fluid spacing (H in H3O+, NonlinearPCM)
        N = 128
        x = dx * np.arange(N)
        L = dx * N
        G = (2 * np.pi / L) * np.concatenate(
            (np.arange(N // 2), np.arange(N // 2 - N, 0))
        )
        sigma_by_dx = np.arange(1.0, 4.0, 0.1)
        sigma = dx * sigma_by_dx
        fTilde = np.exp(-0.5 * (sigma[:, None] * G[None, :]) ** 2) * (1.0 / L)
        f = np.fft.fft(fTilde, axis=-1).real
        # Visualize f near test radius:
        for i, f_i in enumerate(f):
            plt.plot(x, np.abs(f_i), label=r"$\sigma$/dx=" + f"{sigma_by_dx[i]:.1f}")
        plt.axvline(rTest, color="k", ls="dotted", lw=1)
        plt.xlim(0, 2 * rTest)
        plt.ylim(1e-14, 1.0)
        plt.yscale("log")
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        # MAE between rTest and 2*rTest vs sigma:
        sel = np.where(np.logical_and(x >= rTest, x <= 2 * rTest))
        mae = np.abs(f[:, sel]).mean(axis=-1)
        plt.figure(2)
        plt.plot(sigma_by_dx, mae)
        plt.yscale("log")
        plt.ylabel("MAE")
        plt.xlabel(r"$\sigma$/dx")
        plt.show()

    test_nyquist()
