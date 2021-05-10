from dataclasses import dataclass, InitVar
from ._basis import Basis
import qimpy as qp
import torch


@dataclass
class Wavefunction:
    '''TODO: document class Wavefunction'''

    basis: Basis
    coeff: torch.Tensor = None  # coefficients in specified basis
    proj: list = None  # projections to pseudopotential projectors
    is_split_basis: bool = True  # whether MPI-split over basis

    # Variables used for initialization only:
    n_bands: InitVar[int] = 0
    n_spins: InitVar[int] = 0  # optionally override n_spins from basis
    n_spinor: InitVar[int] = 0  # optionally override n_spinor from basis
    randomize: InitVar[bool] = False  # optionally random instead of zeroes

    def __post_init__(self, n_bands, n_spins, n_spinor, randomize):
        '''Create wavefunctions with specified number of bands and
        optionally randomize them  (only if coeff is None).
        Note that this does not initialize the projections.'''
        if (self.coeff is None) and n_bands:
            basis = self.basis
            nk_mine = basis.kpoints.n_mine
            n_spins = (n_spins if n_spins else basis.n_spins)
            n_spinor = (n_spinor if n_spinor else basis.n_spinor)
            n_basis_eff = (basis.n_each if self.is_split_basis
                           else basis.n_tot)
            self.coeff = torch.zeros(
                (n_spins, nk_mine, n_bands, n_spinor, n_basis_eff),
                dtype=torch.cdouble, device=basis.rc.device)
