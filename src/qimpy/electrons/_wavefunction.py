import qimpy as qp
import torch


class Wavefunction:
    '''TODO: document class Wavefunction'''

    def __init__(self, basis, coeff=None, proj=None, is_split_basis=True,
                 n_bands=0, n_spins=0, n_spinor=0, randomize=False):
        '''
        Parameters
        ----------
        basis: qimpy.electrons.Basis
            Basis set for wavefunction
        coeff: torch.Tensor
            Wavefunction coefficients in specified basis
        proj: list of torch.Tensor
            Projections to pseudopotential projectors for each ion species
        is_split_basis: bool, default: True
            Whether the wavefunctions are MPI-split over basis (default).
            If false, the wavefunctions have been rearranged to bring
            all basis coefficients together (splitting over bands instead)
        n_bands: int, default: 0
            Number of bands to initialize wavefunctions for.
            Used only if coeff is None, and does not initialize proj
        n_spins: int, default: 0
            If non-zero, override the number of spin channels in
            new wavefunctions (instead of the value in basis).
            Used only if coeff is None, and n_bands is non-zero
        n_spinor: int, default: 0
            If non-zero, override the number of spinor components
            in new wavefunctions (instead of the value in basis).
            Used only if coeff is None, and n_bands is non-zero
        randomize: bool, default: False
            If true, randomize new wavefunctions (see :meth:`randomize`),
            else initialize them to zero (default).
            Used only if coeff is None, and n_bands is non-zero
        '''
        self.basis = basis
        self.coeff = coeff
        self.proj = proj
        self.is_split_basis = is_split_basis
        # Optional initialization:
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
