import qimpy as qp
import numpy as np
import torch
from typing import Callable, Union, Optional, Any, TYPE_CHECKING
from typing_extensions import Protocol
if TYPE_CHECKING:
    from ._wavefunction import Wavefunction


def _norm(self: 'Wavefunction') -> float:
    """Return overall norm of wavefunctions"""
    return np.sqrt(qp.utils.globalreduce.sum(qp.utils.abs_squared(self.coeff),
                                             self.basis.rc.comm_kb))


def _band_norms(self: 'Wavefunction', mode: str) -> torch.Tensor:
    """Shared implementation of _band_norm and _band_ke"""
    assert(mode in ('norm', 'ke'))
    assert(not self.band_division)
    basis = self.basis
    coeff_sq = qp.utils.abs_squared(self.coeff)
    if mode == 'ke':
        coeff_sq *= basis.get_ke(basis.mine)[None, :, None, None, :]
    result = ((coeff_sq @ basis.real.Gweight_mine).sum(dim=-1)
              if basis.real_wavefunctions
              else coeff_sq.sum(dim=(-2, -1)))
    if basis.division.n_procs > 1:
        basis.rc.comm_b.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(result),
                                  op=qp.MPI.SUM)
    return result if (mode == 'ke') else result.sqrt()


def _band_norm(self: 'Wavefunction') -> torch.Tensor:
    """Return per-band norm of wavefunctions"""
    return _band_norms(self, 'norm')


def _band_ke(self: 'Wavefunction') -> torch.Tensor:
    """Return per-band norm of wavefunctions"""
    return _band_norms(self, 'ke')


def _band_spin(self: 'Wavefunction') -> torch.Tensor:
    """Return per-band spin of wavefunctions (must be spinorial).
    Result dimensions are 3 x 1 x nk x n_bands."""
    assert(not self.band_division)
    basis = self.basis
    coeff = self.coeff
    assert(coeff.shape[-2] == 2)  # must be spinorial
    # Compute spin density matrix per band:
    rho_s = (torch.einsum('skbxg, g, skbyg -> skbxy',
                          coeff.conj(), basis.real.Gweight_mine, coeff)
             if basis.real_wavefunctions
             else torch.einsum('skbxg, skbyg -> skbxy', coeff.conj(), coeff))
    if basis.division.n_procs > 1:
        basis.rc.comm_b.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(rho_s),
                                  op=qp.MPI.SUM)
    # Convert to spin vector:
    result = torch.zeros((3,) + coeff.shape[:3], device=coeff.device)
    result[0] = 2. * rho_s[..., 1, 0].real  # Sx
    result[1] = 2. * rho_s[..., 1, 0].imag  # Sy
    result[2] = rho_s[..., 0, 0].real - rho_s[..., 1, 1].real  # Sz
    return result


def _dot(self: 'Wavefunction', other: 'Wavefunction') -> torch.Tensor:
    """Inner (dot) product of `self` with `other`.
    For convenience, this can also be invoked as `self` ^ `other`.
    Note that this means xor(`self`, `other`) also computes wavefunction dot
    instead of logical xor (which is meaningless for wavefunctions anyway).
    All dimensions except bands must match between `self` and `other`.

    Parameters
    ----------
    other: qimpy.electrons.Wavefunction
        Dimensions must match self for spinor and basis, can differ for bands,
        and must be broadcastable for preceding dimensions (spin and k)
    overlap: bool, default: False
        If True, include the overlap operator in the product. The overlap
        operator is identity for norm-conserving pseudopotentials, but
        includes augmentation for ultrasoft and PAW pseudopotentials

    Returns
    -------
    torch.Tensor
        Dimensions: n_spins x nk_mine x n_bands(self) x n_bands(other)
    """
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    basis = self.basis
    assert(basis is other.basis)
    assert(not self.band_division)
    watch = qp.utils.StopWatch('Wavefunction.dot', basis.rc)
    # Prepare left operand:
    C1 = (self.coeff * basis.real.Gweight_mine.view(1, 1, 1, 1, -1)
          if basis.real_wavefunctions
          else self.coeff).flatten(-2).conj()  # merge spinor & basis dims
    # Prepare right operand:
    C2 = other.coeff.flatten(-2).transpose(-2, -1)  # last dim now band2
    # Compute local inner product and reduce:
    result = (C1 @ C2)
    if basis.real_wavefunctions:
        result.imag *= 0.  # due to implicit +h.c. terms in C1 and C2
    if basis.division.n_procs > 1:
        basis.rc.comm_b.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(result),
                                  op=qp.MPI.SUM)
    watch.stop()
    return result


def _dot_O(self: 'Wavefunction', other: 'Wavefunction') -> torch.Tensor:
    """Dot product of `self` with O(`other`).
    Here, the overlap operator O is identity for norm-conserving
    pseudopotentials, but includes augmentation for ultrasoft
    and PAW pseudopotentials.

    Returns
    -------
    torch.Tensor
        Dimensions: n_spins x nk_mine x n_bands(self) x n_bands(other)
    """
    result = _dot(self, other)
    # Overlap augmentation:
    # TODO: augment overlap here when adding ultrasoft / PAW
    return result


def _overlap(self: 'Wavefunction') -> 'Wavefunction':
    """Return wavefunction with overlap operator applied.
    This is identity and returns a view of the original wavefunction
    for norm-conserving pseudopotentials, but includes augmentation terms
    for ultrasoft and PAW pseudopotentials"""
    return self  # TODO: augment overlap here when adding ultrasoft / PAW


def _matmul(self: 'Wavefunction', mat: torch.Tensor) -> 'Wavefunction':
    """Compute a matrix transformation (such as rotation) along the band
    dimension of wavefunction (self). For convenience, this can also be
    invoked as self @ other, using the standard matrix multiply operator.

    Parameters
    ----------
    mat : torch.Tensor
        Last two dimensions specify transformation in band space,
        so final dimension should match n_bands of the wavefunction
        and penultimate dimension determines n_bands of output.
        Preceding dimensions should be broadcastable with spin and k

    Returns
    -------
    qimpy.electrons.Wavefunction
        The result will have n_bands = mat.shape[-2], same basis and spinor as
        input, and spin and k determined by broadcasting with mat.shape[:-2]
    """
    if not isinstance(mat, torch.Tensor):
        return NotImplemented
    watch = qp.utils.StopWatch('Wavefunction.matmul', self.basis.rc)
    # Prepare input view:
    C = self.coeff.flatten(-2)  # merge spinor & basis dims at input
    C_out = mat.transpose(-2, -1) @ C
    C_out = C_out.view(C_out.shape[:-1] + self.coeff.shape[-2:])  # un-merge
    result = qp.electrons.Wavefunction(self.basis, coeff=C_out,
                                       band_division=self.band_division)
    if self._proj_is_valid():
        result._proj = self._proj @ mat
    watch.stop()
    return result


def _orthonormalize(self: 'Wavefunction') -> 'Wavefunction':
    """Return orthonormalized version of wavefunctions.
    This internally uses Gram-Schmidt scheme using a Cholesky decomposition,
    which is not differentiable. Use a symmetric orthonormalization scheme
    using :meth:`qimpy.utils.ortho_matrix` for a differentiable scheme."""
    return self @ qp.utils.ortho_matrix(self.dot_O(self), use_cholesky=True)


def _mul(self: 'Wavefunction',
         scale: Union[float, torch.Tensor]) -> 'Wavefunction':
    is_suitable = is_band_scale = isinstance(scale, float)
    if isinstance(scale, torch.Tensor) and (len(scale.shape) == 5):
        is_suitable = True
        is_band_scale = (scale.shape[-2:] == (1, 1))
    if not is_suitable:
        return NotImplemented
    result = qp.electrons.Wavefunction(self.basis, coeff=self.coeff * scale,
                                       band_division=self.band_division)
    if is_band_scale and self._proj_is_valid():
        assert self._proj is not None
        result._proj = self._proj * scale
        result._proj_version = self._proj_version
    return result


def _imul(self: 'Wavefunction',
          scale: Union[float, torch.Tensor]) -> 'Wavefunction':
    is_suitable = is_band_scale = isinstance(scale, float)
    if isinstance(scale, torch.Tensor) and (len(scale.shape) == 5):
        is_suitable = True
        is_band_scale = (scale.shape[-2:] == (1, 1))
    if not is_suitable:
        return NotImplemented
    self.coeff *= scale
    if is_band_scale and self._proj_is_valid():
        assert self._proj is not None
        self._proj *= scale
    else:
        self._proj_invalidate()
    return self


def _add(self: 'Wavefunction', other: 'Wavefunction') -> 'Wavefunction':
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert(self.basis is other.basis)
    result = qp.electrons.Wavefunction(self.basis,
                                       coeff=(self.coeff + other.coeff),
                                       band_division=self.band_division)
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        result._proj = self._proj + other._proj
        result._proj_version = self._proj_version
    return result


def _iadd(self: 'Wavefunction', other: 'Wavefunction') -> 'Wavefunction':
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert(self.basis is other.basis)
    self.coeff += other.coeff
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        self._proj += other._proj
    else:
        self._proj_invalidate()
    return self


def _sub(self: 'Wavefunction', other: 'Wavefunction') -> 'Wavefunction':
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert(self.basis is other.basis)
    result = qp.electrons.Wavefunction(self.basis,
                                       coeff=(self.coeff - other.coeff),
                                       band_division=self.band_division)
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        result._proj = self._proj - other._proj
        result._proj_version = self._proj_version
    return result


def _isub(self: 'Wavefunction', other: 'Wavefunction') -> 'Wavefunction':
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert(self.basis is other.basis)
    self.coeff -= other.coeff
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        self._proj -= other._proj
    else:
        self._proj_invalidate()
    return self


def _getitem(self: 'Wavefunction', index: Any) -> 'Wavefunction':
    """Propagate getting slices to coeff and proj if present"""
    result = qp.electrons.Wavefunction(self.basis, coeff=self.coeff[index],
                                       band_division=self.band_division)
    if self._proj_is_valid():
        assert self._proj is not None
        result._proj = self._proj[index]
        result._proj_version = self._proj_version
    return result


def _setitem(self: 'Wavefunction', index: Any, other: 'Wavefunction') -> None:
    """Propagate setting slices to `coeff` and `proj` if present"""
    self.coeff[index] = other.coeff
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        self._proj[index] = other._proj
    else:
        self._proj_invalidate()


def _cat(self: 'Wavefunction', other: 'Wavefunction',
         dim: int = 2) -> 'Wavefunction':
    """Join wavefunctions along specified dimension (default: 2 => bands)"""
    result = qp.electrons.Wavefunction(self.basis,
                                       coeff=torch.cat((self.coeff,
                                                        other.coeff), dim=dim),
                                       band_division=self.band_division)
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        result._proj = torch.cat((self._proj, other._proj), dim=dim)
        result._proj_version = self._proj_version
    return result


def _constrain(self: 'Wavefunction'):
    """Enforce basis constraints on wavefunction coefficients.
    This includes setting padded coefficients to zero, and imposing
    Hermitian symmetry in Gz = 0 coefficients for real wavefunctions.
    """
    basis = self.basis
    # Padded coefficients:
    pad_index = (basis.pad_index if self.band_division
                 else basis.pad_index_mine)
    self.coeff[pad_index] = 0.
    # Real wavefunction symmetry:
    if basis.real_wavefunctions:
        basis.real.symmetrize(self.coeff)
