import qimpy as qp
import numpy as np
import torch
from ._wavefunction_init import _randomize, \
    _randomize_selected, _RandomizeSelected
from ._wavefunction_split import _split_bands, _split_basis
from ._wavefunction_ops import _norm, _band_norm, _band_ke, \
    _dot, _dot_O, _overlap, _matmul, _orthonormalize, \
    _mul, _imul, _add, _iadd, _sub, _isub, _getitem, _cat
from typing import Callable, Optional, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import TaskDivision
    from ._basis import Basis


class Wavefunction:
    """Electronic wavefunctions including coefficients and projections"""
    basis: 'Basis'  #: Corresponding basis

    #: Wavefunction coefficients (n_spins x nk x n_bands x n_spinor x n_basis)
    coeff: torch.Tensor

    #: Projections of each band on all pseudopotential projectors
    #: (n_spins x nk x n_bands x n_spinor x n_projectors)
    proj: Optional[torch.Tensor]

    #: If present, wavefunctions are split along bands instead of
    #: along the 'home position' of split along basis.
    band_division: Optional['TaskDivision']

    def __init__(self, basis: 'Basis', coeff: Optional[torch.Tensor] = None,
                 proj: Optional[torch.Tensor] = None,
                 band_division: Optional['TaskDivision'] = None,
                 n_bands: int = 0, n_spins: int = 0, n_spinor: int = 0,
                 randomize: bool = False) -> None:
        """Initialize wavefunctions of specified size or with given
        coefficients.

        Parameters
        ----------
        basis
            Basis set for wavefunction
        coeff
            Wavefunction coefficients in specified basis.
            If not provided, one of `band_division` or `n_bands` must be
            provided to initialize zero wavefunctions of determined size.
        proj
            Projections to pseudopotential projectors for each ion species
        band_division
            If None, wavefunctions are MPI-split over basis coefficients,
            which is the default or "home" position for the wavefunctions.
            When not None, the wavefunctions are split over bands as
            specified by the TaskDivision object, with all basis coefficients
            for each band together on some process.
        n_bands: int, default: band_division.n_mine if available, else 0
            Number of bands to initialize zero wavefunctions for.
            Used only if coeff is None, and this does not initialize proj.
        n_spins
            If non-zero, override the number of spin channels in
            new wavefunctions (instead of the value in basis).
            Used only if coeff is None, and n_bands or band_division is given
        n_spinor
            If non-zero, override the number of spinor components
            in new wavefunctions (instead of the value in basis).
            Used only if coeff is None, and n_bands or band_division is given
        """
        self.basis = basis
        self.proj = proj
        self.band_division = band_division
        if coeff is None:
            # Initialize zero coefficients:
            assert(n_bands or band_division)
            basis = self.basis
            nk_mine = basis.kpoints.n_mine
            n_bands = (n_bands if (band_division is None)
                       else band_division.n_mine)
            n_spins = (n_spins if n_spins else basis.n_spins)
            n_spinor = (n_spinor if n_spinor else basis.n_spinor)
            n_basis = (basis.n_tot if band_division else basis.n_mine)
            self.coeff = torch.zeros(
                (n_spins, nk_mine, n_bands, n_spinor, n_basis),
                dtype=torch.cdouble, device=basis.rc.device)
            self.proj = None  # invalidate any previous projections
        else:
            # Set provided coefficients:
            self.coeff = coeff

    def zeros_like(self, n_bands: int = 0) -> 'Wavefunction':
        """Create a zero Wavefunction similar to the present one.
        Optionally override the number of bands with `n_bands`."""
        n_bands = (n_bands if n_bands else self.coeff.shape[2])
        return Wavefunction(self.basis, band_division=self.band_division,
                            n_bands=n_bands, n_spins=self.coeff.shape[0],
                            n_spinor=self.coeff.shape[3])

    def clone(self) -> 'Wavefunction':
        """Create a copy"""
        return Wavefunction(
            self.basis, self.coeff.clone().detach(),
            None if (self.proj is None) else self.proj.clone().detach(),
            band_division=self.band_division)

    def n_bands(self) -> int:
        """Get number of bands in wavefunction"""
        return 0 if (self.coeff is None) else self.coeff.shape[2]

    def randomize(self: 'Wavefunction', seed: int = 0, b_start: int = 0,
                  b_stop: Optional[int] = None) -> None:
        _randomize(self, seed, b_start, b_stop)

    # Function types for checking imported methods:
    UnaryOp = Callable[['Wavefunction'], 'Wavefunction']
    UnaryOpT = Callable[['Wavefunction'], torch.Tensor]
    UnaryOpS = Callable[['Wavefunction'], float]
    BinaryOp = Callable[['Wavefunction', 'Wavefunction'], 'Wavefunction']
    BinaryOpT = Callable[['Wavefunction', 'Wavefunction'], torch.Tensor]
    ScaleOp = Callable[['Wavefunction', Union[float, torch.Tensor]],
                       'Wavefunction']

    randomize.__doc__ = _randomize.__doc__  # randomize stub'd above
    randomize_selected: _RandomizeSelected = _randomize_selected
    split_bands: UnaryOp = _split_bands
    split_basis: UnaryOp = _split_basis
    norm: UnaryOpS = _norm
    band_norm: UnaryOpT = _band_norm
    band_ke: UnaryOpT = _band_ke
    dot_O: BinaryOpT = _dot_O  # dot product through overlap operator
    dot: BinaryOpT = _dot  # bare dot product
    __xor__: BinaryOpT = _dot  # convenient shorthand C1 ^ C2 for dot product
    overlap: UnaryOp = _overlap  # apply overlap operator to wavefunction
    matmul = _matmul  # transform by a matrix in band space
    __matmul__ = _matmul  # shorthand C @ M for matmul
    orthonormalize: UnaryOp = _orthonormalize
    __mul__: ScaleOp = _mul  # scalar multiply
    __rmul__: ScaleOp = _mul  # scalar multiply is commutative
    __imul__: ScaleOp = _imul  # scale
    __add__: BinaryOp = _add
    __iadd__: BinaryOp = _iadd
    __sub__: BinaryOp = _sub
    __isub__: BinaryOp = _isub
    __getitem__ = _getitem
    cat = _cat  # join wavefunctions
