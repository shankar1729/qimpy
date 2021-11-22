from __future__ import annotations
import qimpy as qp
import torch
from typing import Union


def _mul(
    self: qp.electrons.Wavefunction, scale: Union[float, torch.Tensor]
) -> qp.electrons.Wavefunction:
    is_suitable = is_band_scale = isinstance(scale, float)
    if isinstance(scale, torch.Tensor) and (len(scale.shape) == 5):
        is_suitable = True
        is_band_scale = scale.shape[-2:] == (1, 1)
    if not is_suitable:
        return NotImplemented
    result = qp.electrons.Wavefunction(
        self.basis, coeff=self.coeff * scale, band_division=self.band_division
    )
    if is_band_scale and self._proj_is_valid():
        assert self._proj is not None
        result._proj = self._proj * scale
        result._proj_version = self._proj_version
    return result


def _imul(
    self: qp.electrons.Wavefunction, scale: Union[float, torch.Tensor]
) -> qp.electrons.Wavefunction:
    is_suitable = is_band_scale = isinstance(scale, float)
    if isinstance(scale, torch.Tensor) and (len(scale.shape) == 5):
        is_suitable = True
        is_band_scale = scale.shape[-2:] == (1, 1)
    if not is_suitable:
        return NotImplemented
    self.coeff *= scale
    if is_band_scale and self._proj_is_valid():
        assert self._proj is not None
        self._proj *= scale
    else:
        self._proj_invalidate()
    return self


def _add(
    self: qp.electrons.Wavefunction, other: qp.electrons.Wavefunction
) -> qp.electrons.Wavefunction:
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert self.basis is other.basis
    result = qp.electrons.Wavefunction(
        self.basis, coeff=(self.coeff + other.coeff), band_division=self.band_division
    )
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        result._proj = self._proj + other._proj
        result._proj_version = self._proj_version
    return result


def _iadd(
    self: qp.electrons.Wavefunction, other: qp.electrons.Wavefunction
) -> qp.electrons.Wavefunction:
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert self.basis is other.basis
    self.coeff += other.coeff
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        self._proj += other._proj
    else:
        self._proj_invalidate()
    return self


def _sub(
    self: qp.electrons.Wavefunction, other: qp.electrons.Wavefunction
) -> qp.electrons.Wavefunction:
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert self.basis is other.basis
    result = qp.electrons.Wavefunction(
        self.basis, coeff=(self.coeff - other.coeff), band_division=self.band_division
    )
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        result._proj = self._proj - other._proj
        result._proj_version = self._proj_version
    return result


def _isub(
    self: qp.electrons.Wavefunction, other: qp.electrons.Wavefunction
) -> qp.electrons.Wavefunction:
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert self.basis is other.basis
    self.coeff -= other.coeff
    if self._proj_is_valid() and other._proj_is_valid():
        assert self._proj is not None
        assert other._proj is not None
        self._proj -= other._proj
    else:
        self._proj_invalidate()
    return self
