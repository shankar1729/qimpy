import qimpy as qp
import numpy as np
import torch
from ._wavefunction_init import _randomize
from ._wavefunction_split import _split_bands, _split_basis


class Wavefunction:
    '''TODO: document class Wavefunction'''

    randomize = _randomize
    split_bands = _split_bands
    split_basis = _split_basis

    def __init__(self, basis, coeff=None, proj=None, band_division=None,
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
        band_division: qimpy.utils.TaskDivision or None, default: None
            If None, wavefunctions are MPI-split over basis coefficients,
            which is the default or "home" position for the wavefunctions.
            When not None, the wavefunctions are split over bands as
            specified by the TaskDivision object, with all basis coefficients
            for each band together on some process.
        n_bands: int, default: band_division.n_mine if available, else 0
            Number of bands to initialize wavefunctions for.
            Used only if coeff is None, and this does not initialize proj
        n_spins: int, default: 0
            If non-zero, override the number of spin channels in
            new wavefunctions (instead of the value in basis).
            Used only if coeff is None, and n_bands or band_division is given
        n_spinor: int, default: 0
            If non-zero, override the number of spinor components
            in new wavefunctions (instead of the value in basis).
            Used only if coeff is None, and n_bands or band_division is given
        randomize: bool, default: False
            If true, randomize new wavefunctions (see :meth:`randomize`),
            else initialize them to zero (default).
            Used only if coeff is None, and n_bands or band_division is given
        '''
        self.basis = basis
        self.coeff = coeff
        self.proj = proj
        self.band_division = band_division
        # Optional initialization:
        if (self.coeff is None) and (n_bands or band_division):
            basis = self.basis
            nk_mine = basis.kpoints.n_mine
            n_bands = (n_bands if n_bands else band_division.n_mine)
            n_spins = (n_spins if n_spins else basis.n_spins)
            n_spinor = (n_spinor if n_spinor else basis.n_spinor)
            n_basis = basis.n_each * (basis.n_procs if band_division else 1)
            self.coeff = torch.zeros(
                (n_spins, nk_mine, n_bands, n_spinor, n_basis),
                dtype=torch.cdouble, device=basis.rc.device)
            self.proj = None  # invalidate any previous projections
            if randomize:
                self.randomize()

    def norm(self):
        'Return overall Frobenius norm of wavefunction coefficients'
        return np.sqrt(self.basis.rc.comm_kb.allreduce(
            self.coeff.norm().item()**2, qp.MPI.SUM))

    def __sub__(self, other):
        assert(self.basis is other.basis)
        coeff = self.coeff - other.coeff
        proj = ((self.proj - other.proj) if (self.proj and other.proj)
                else None)
        return Wavefunction(self.basis, coeff, proj, self.band_division)
