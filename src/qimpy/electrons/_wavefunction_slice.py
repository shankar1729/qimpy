from __future__ import annotations
import qimpy as qp
import torch
from typing import Any


def _getitem(self: qp.electrons.Wavefunction, index: Any) -> qp.electrons.Wavefunction:
    """Propagate getting slices to coeff and proj if present"""
    result = qp.electrons.Wavefunction(
        self.basis, coeff=self.coeff[index], band_division=self.band_division
    )
    if self._proj_is_valid():
        assert self._proj is not None
        result._proj = self._proj[index]
        result._proj_version = self._proj_version
    return result


def _setitem(
    self: qp.electrons.Wavefunction, index: Any, other: qp.electrons.Wavefunction
) -> None:
    """Propagate setting slices to `coeff` and `proj` if present"""
    self.coeff[index] = other.coeff
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        self._proj[index] = other._proj
    else:
        self._proj_invalidate()


def _cat(
    self: qp.electrons.Wavefunction,
    other: qp.electrons.Wavefunction,
    dim: int = 2,
    clear: bool = False,
) -> qp.electrons.Wavefunction:
    """Join wavefunctions along specified dimension (default: 2 => bands).
    If `clear` is True, eagerly clear memory of the input operands.
    Note that this will leave `self` and `other` in a broken state,
    so use this only if they are deleted or replaced shortly thereafter.
    Despite this danger, this is often necessary becaause this operation will
    likely be near the peak memory usage eg. within `Davidson` and `CheFSI`."""
    result = qp.electrons.Wavefunction(
        self.basis,
        coeff=torch.cat((self.coeff, other.coeff), dim=dim),
        band_division=self.band_division,
    )
    if clear:
        del self.coeff
        del other.coeff
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        result._proj = torch.cat((self._proj, other._proj), dim=dim)
        result._proj_version = self._proj_version
        if clear:
            del self.proj
            del other.proj
    return result
