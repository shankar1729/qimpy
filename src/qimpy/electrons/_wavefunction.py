from __future__ import annotations
import qimpy as qp
import torch
from ._wavefunction_init import _randomize, _randomize_selected, _RandomizeSelected
from ._wavefunction_split import _split_bands, _split_basis
from ._wavefunction_arithmetic import _mul, _imul, _add, _iadd, _sub, _isub
from ._wavefunction_bandwise import _band_norm, _band_ke, _band_spin, _constrain
from ._wavefunction_slice import _getitem, _setitem, _cat
from ._wavefunction_dot import _norm, _dot, _dot_O, _overlap, _matmul, _orthonormalize
from typing import Callable, Optional, Union


class Wavefunction(qp.utils.Gradable["Wavefunction"]):
    """Electronic wavefunctions including coefficients and projections"""

    basis: qp.electrons.Basis  #: Corresponding basis

    #: Wavefunction coefficients (n_spins x nk x n_bands x n_spinor x n_basis)
    coeff: torch.Tensor

    #: Projections of each band on all pseudopotential projectors
    #: (n_spins x nk x (n_spinor*n_projectors) x n_bands)
    #: Access this using property `proj` instead, which takes care of
    #: automatically calculating and invalidating this when necessary.
    _proj: Optional[torch.Tensor]
    _proj_version: int  #: `Ions.beta_version` for which `_proj` is valid

    #: If present, wavefunctions are split along bands instead of
    #: along the 'home position' of split along basis.
    band_division: Optional[qp.utils.TaskDivision]

    def __init__(
        self,
        basis: qp.electrons.Basis,
        *,
        coeff: Optional[torch.Tensor] = None,
        band_division: Optional[qp.utils.TaskDivision] = None,
        n_bands: int = 0,
        n_spins: int = 0,
        n_spinor: int = 0
    ) -> None:
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
        self.band_division = band_division
        self._proj_invalidate()
        if coeff is None:
            # Initialize zero coefficients:
            assert n_bands or band_division
            basis = self.basis
            nk_mine = basis.kpoints.division.n_mine
            n_bands = n_bands if (band_division is None) else band_division.n_mine
            n_spins = n_spins if n_spins else basis.n_spins
            n_spinor = n_spinor if n_spinor else basis.n_spinor
            n_basis = basis.n_tot if band_division else basis.division.n_each
            self.coeff = torch.zeros(
                (n_spins, nk_mine, n_bands, n_spinor, n_basis),
                dtype=torch.cdouble,
                device=qp.rc.device,
            )
            # Projections of zero wavefunctions are always zero:
            self._proj = torch.zeros(
                (n_spins, nk_mine, basis.ions.n_projectors, n_bands),
                dtype=torch.cdouble,
                device=qp.rc.device,
            )
            self._proj_version = basis.ions.beta_version
        else:
            # Set provided coefficients:
            self.coeff = coeff

    @property
    def proj(self) -> torch.Tensor:
        """Projection of wavefunction on current projector in :class:`Ions`.
        This is computed, cached and invalidated automatically."""
        if self._proj_is_valid():
            assert self._proj is not None
            return self._proj
        else:
            ions = self.basis.ions
            if self.band_division:
                assert ions.beta_full is not None
                self._proj = (ions.beta_full ^ self).wait()
            else:
                self._proj = (ions.beta ^ self).wait()
            self._proj_version = ions.beta_version
            return self._proj

    def _proj_invalidate(self) -> None:
        """Invalidate cached projections."""
        self._proj = None
        self._proj_version = -1

    def _proj_is_valid(self) -> bool:
        """Check whether cached projections are still valid."""
        return self._proj_version == self.basis.ions.beta_version

    def zeros_like(self, n_bands: int = 0) -> qp.electrons.Wavefunction:
        """Create a zero Wavefunction similar to the present one.
        Optionally override the number of bands with `n_bands`."""
        n_bands = n_bands if n_bands else self.coeff.shape[2]
        return Wavefunction(
            self.basis,
            band_division=self.band_division,
            n_bands=n_bands,
            n_spins=self.coeff.shape[0],
            n_spinor=self.coeff.shape[3],
        )

    def clone(self) -> qp.electrons.Wavefunction:
        """Create an independent copy (not a view / reference)."""
        result = Wavefunction(
            self.basis,
            coeff=self.coeff.clone().detach(),
            band_division=self.band_division,
        )
        if self._proj_is_valid():
            assert self._proj is not None
            result._proj = self._proj.clone().detach()
            result._proj_version = self._proj_version
        return result

    def n_bands(self) -> int:
        """Get number of bands in wavefunction"""
        return self.coeff.shape[2]

    def read(self, cp_path: qp.utils.CpPath) -> int:
        """Read wavefunctions from `cp_path`. Return number of bands read."""
        checkpoint, path = cp_path
        assert checkpoint is not None
        dset = checkpoint[path]
        n_bands_in = min(dset.shape[2], self.n_bands())
        basis = self.basis
        # Slice to be read on this process:
        if basis.n_max <= basis.division.i_start:
            return n_bands_in  # Only padded elements here; nothing to read
        basis_n_mine = (
            min(basis.n_max, basis.division.i_stop) - basis.division.i_start
        )  # size without padding
        # Read:
        k_division = basis.kpoints.division
        n_spins, nk_mine, _, n_spinor, _ = self.coeff.shape
        offset = (0, k_division.i_start, 0, 0, basis.division.i_start)
        size = (n_spins, nk_mine, n_bands_in, n_spinor, basis_n_mine)
        self.coeff[:, :, :n_bands_in, :, :basis_n_mine] = checkpoint.read_slice_complex(
            dset, offset, size
        )
        self._proj_invalidate()
        return n_bands_in

    def write(self, cp_path: qp.utils.CpPath) -> None:
        """Write wavefunctions to `cp_path`."""
        checkpoint, path = cp_path
        assert checkpoint is not None
        basis = self.basis
        k_division = basis.kpoints.division
        # Create dataset with overall shape:
        n_spins, _, n_bands, n_spinor, _ = self.coeff.shape
        shape = (n_spins, k_division.n_tot, n_bands, n_spinor, basis.n_max)
        dset = checkpoint.create_dataset_complex(
            path, shape=shape, dtype=self.coeff.dtype
        )
        # Slice to be written from this process:
        if basis.n_max <= basis.division.i_start:
            return  # Only padded elements on this process (not written)
        basis_n_mine = (
            min(basis.n_max, basis.division.i_stop) - basis.division.i_start
        )  # size without padding
        offset = (0, k_division.i_start, 0, 0, basis.division.i_start)
        checkpoint.write_slice_complex(dset, offset, self.coeff[..., :basis_n_mine])

    def randomize(
        self: qp.electrons.Wavefunction,
        seed: int = 0,
        b_start: int = 0,
        b_stop: Optional[int] = None,
    ) -> None:
        _randomize(self, seed, b_start, b_stop)

    @property
    def non_spinor(self) -> Wavefunction:
        """Return a non-spinorial view of this wavefunction.
        If spinorial, the result will have twice the bands instead.
        This is useful when dealing with projectors which remain
        non-spinorial, even in spinorial calculations.
        The view is identical to the input for non-spinorial wavefunctions.
        """
        if self.coeff.shape[-2] == 1:
            return self
        else:
            return Wavefunction(self.basis, coeff=self.coeff.flatten(2, 3).unsqueeze(3))

    # Function types for checking imported methods:
    UnaryOpAsync = Callable[["Wavefunction"], qp.utils.Waitable["Wavefunction"]]
    UnaryOp = Callable[["Wavefunction"], "Wavefunction"]
    UnaryOpT = Callable[["Wavefunction"], torch.Tensor]
    UnaryOpS = Callable[["Wavefunction"], float]
    BinaryOp = Callable[["Wavefunction", "Wavefunction"], "Wavefunction"]
    BinaryOpAsyncT = Callable[
        ["Wavefunction", "Wavefunction"], qp.utils.Waitable[torch.Tensor]
    ]  # binary operation that returns a tensor asynchronously
    ScaleOp = Callable[["Wavefunction", Union[float, torch.Tensor]], "Wavefunction"]

    randomize.__doc__ = _randomize.__doc__  # randomize stub'd above
    randomize_selected: _RandomizeSelected = _randomize_selected
    split_bands: UnaryOpAsync = _split_bands
    split_basis: UnaryOpAsync = _split_basis
    norm: UnaryOpS = _norm
    band_norm: UnaryOpT = _band_norm
    band_ke: UnaryOpT = _band_ke
    band_spin: UnaryOpT = _band_spin
    dot_O: BinaryOpAsyncT = _dot_O  # dot product through overlap operator
    dot: BinaryOpAsyncT = _dot  # bare dot product
    __xor__: BinaryOpAsyncT = _dot  # convenient shorthand C1 ^ C2 for dot product
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
    __setitem__ = _setitem
    cat = _cat  # join wavefunctions
    constrain = _constrain
