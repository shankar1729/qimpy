import qimpy as qp
import numpy as np
import torch
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from ._grid import Grid


class Coulomb:
    '''Coulomb interactions between fields and point charges.
    TODO: support non-periodic geometries (truncation).'''
    __slots__ = ('grid', 'sigma', 'iR', 'iG')
    grid: 'Grid'  #: Grid associated with fields for coulomb interaction
    sigma: float  #: Ewald range-separation parameter
    iR: torch.Tensor  #: Ewald real-space mesh points
    iG: torch.Tensor  #: Ewald reciprocal-space mesh points

    def __init__(self, grid: 'Grid', n_ions: int) -> None:
        '''Initialize coulomb interactions.

        Parameters
        ----------
        grid
            Fields for coulomb interaction will be on this grid.
        n_ions
            Number of point charges to optimize Ewald sums for.
        '''
        self.grid = grid
        rc = grid.rc
        lattice = grid.lattice

        # Determine optimum gaussian width for Ewald sums:
        # Uses fact that number of reciprocal cells is proportional to lattice
        # volume and number of real space cells inversely proportional to it.
        # Also accounts for real space cost ~ Natoms^2/cell,
        # while reciprocal space cost ~ Natoms/cell.
        self.sigma = (lattice.volume**2
                      / ((2*np.pi)**3 * max(1, n_ions))) ** (1./6)

        # Compute Ewald real and reciprocal meshes:
        def get_mesh(Rbasis: torch.Tensor, Gbasis: torch.Tensor, sigma: float,
                     include_margin: bool, exclude_zero: bool) -> torch.Tensor:
            '''Create mesh that includes all non-zero terms based on sigma,
            optionally including a margin of 1 in grid units,
            and optionaly excluding the zero (origin) point'''
            Rcut = qp.grid.N_SIGMAS_PER_WIDTH * sigma
            # Create parallelopiped mesh:
            RlengthInv = Gbasis.norm(dim=0).to(rc.cpu) / (2*np.pi)
            Ncut = 1 + torch.ceil(Rcut * RlengthInv).to(torch.int)
            iRgrids = [torch.arange(-N, N+1, device=rc.device,
                                    dtype=torch.double) for N in Ncut]
            iR = torch.stack(torch.meshgrid(*tuple(iRgrids))).flatten(1).T
            if exclude_zero:
                iR = iR[torch.where(iR.abs().sum(dim=-1))[0]]
            # Add margin mesh:
            marginGrid = torch.tensor([-1, +1] if include_margin else [0],
                                      device=rc.device)
            iMargin = torch.stack(torch.meshgrid(marginGrid, marginGrid,
                                                 marginGrid)).flatten(1).T
            iRmargin = iR[None, ...] + iMargin[:, None, :]
            # Select points within Rcut (including margin):
            RmagMin = (iRmargin @ Rbasis.T).norm(dim=-1).min(dim=0)[0]
            return iR[torch.where(RmagMin <= Rcut)[0]]
        # --- real-space mesh
        self.iR = get_mesh(lattice.Rbasis, lattice.Gbasis, self.sigma,
                           include_margin=True, exclude_zero=False)
        # --- reciprocal space mesh (flip R <-> G, sigma <-> 1/sigma)
        self.iG = get_mesh(lattice.Gbasis, lattice.Rbasis, 1./self.sigma,
                           include_margin=False, exclude_zero=True)
        qp.log.info(f'Ewald:  sigma: {self.sigma:f}'
                    f'  nR: {self.iR.shape[0]}  nG: {self.iG.shape[0]}')

    def ewald(self, positions: torch.Tensor, Z: torch.Tensor,
              compute_stress: bool = False
              ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        '''Compute Ewald energy, forces and optionally stress.

        Parameters
        ----------
        positions:
            Positions (fractional coordinates) of point charges
        Z:
            Charges of each point charge
        '''
        sigma = self.sigma
        sigmaSq = sigma**2
        eta = np.sqrt(0.5)/sigma
        lattice = self.grid.lattice

        # Position independent terms:
        Ztot = Z.sum()
        ZsqTot = (Z ** 2).sum()
        E_Gzero = 0.5 * ((4*np.pi) * (-0.5*sigmaSq*(Ztot**2))/lattice.volume)
        E_self = 0.5 * (-ZsqTot * eta * (2./np.sqrt(np.pi)))
        E = E_Gzero + E_self
        E_RRT = -E_Gzero * torch.eye(3, device=Z.device)

        # Real-space sum:
        rCut = 1e-6  # cutoff to detect self-term
        pos = positions - torch.floor(0.5 + positions)  # in [-0.5,0.5)
        x = self.iR.view(1, 1, -1, 3) + (pos.view(-1, 1, 1, 3)
                                         - pos.view(1, -1, 1, 3))
        rVec = x @ lattice.Rbasis.T  # Cartesian separations for all pairs
        r = rVec.norm(dim=-1)
        r[torch.where(r < rCut)] = qp.grid.N_SIGMAS_PER_WIDTH * sigma
        Zprod = Z.view(-1, 1, 1) * Z.view(1, -1, 1)
        Eterm = Zprod * torch.erfc(eta * r) / r
        minus_E_r_by_r = (Eterm + (2.*eta/np.sqrt(np.pi)) * Zprod
                          * torch.exp(-(eta*r)**2)) / (r**2)  # -(dE/dr)/r
        E += 0.5 * Eterm.sum()
        E_pos = ((rVec @ lattice.Rbasis)
                 * minus_E_r_by_r[..., None]).sum(dim=(1, 2))  # forces
        if compute_stress:
            E_RRT -= 0.5 * torch.einsum('rij, rija, rijb -> ab',
                                        minus_E_r_by_r, rVec, rVec)

        # Reciprocal space sum:
        Sf = torch.exp((-2j*np.pi) * (self.iG @ pos.T))  # Translation phases
        SfZ = (Sf * Z[None, :]).sum(dim=-1)  # Charge structure factor
        G = self.iG @ lattice.Gbasis.T
        Gsq = (G**2).sum(dim=-1)
        Eterm = (4*np.pi/lattice.volume) * torch.exp((-0.5*sigmaSq)*Gsq)/Gsq
        Erecip = 0.5 * (Eterm @ (SfZ.abs()**2))
        E += Erecip
        E_pos -= (2*np.pi) * (((Eterm * SfZ).conj()[:, None] * Sf).imag
                              * Z[None, :]).T @ self.iG
        if compute_stress:
            E_RRT += 0.5 * torch.einsum('g, ga, gb -> ab', (SfZ.abs()**2)
                                        * Eterm * (sigmaSq + 2./Gsq), G, G)
            E_RRT -= Erecip * torch.eye(3, device=Z.device)

        return E, E_pos, E_RRT
