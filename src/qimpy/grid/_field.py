import qimpy as qp
import numpy as np
import torch
from abc import ABCMeta, abstractmethod
from numbers import Number
from ._change import _change_real, _change_recip
from typing import TypeVar, Tuple, Optional, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from ._grid import Grid


FieldType = TypeVar('FieldType', bound='Field')  #: Type for field ops.


class Field(metaclass=ABCMeta):
    """Abstract base class for scalar/vector fields in real/reciprocal space.
    Provides common operators for fields in either space, but any fields
    used must specifically be in real (:class:`FieldR` and :class:`FieldC`),
    full-reciprocal (:class:`FieldG`) or half-reciprocal (:class:`FieldH`)
    space."""

    __slots__ = ('grid', 'data')
    grid: 'Grid'  #: Associated grid, which determines dimensions of the field
    data: torch.Tensor  #: Underlying data, with last three dimensions on grid

    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Expected data type for the Field type"""

    @abstractmethod
    def shape_grid(self) -> Tuple[int, ...]:
        """Expected grid shape (last 3 data dimension) for the Field type"""

    def __init__(self, grid: 'Grid', *,
                 shape_batch: Sequence[int] = tuple(),
                 data: Optional[torch.Tensor] = None) -> None:
        """Initialize to zeros or specified `data`.

        Parameters
        ----------
        grid
            Associated grid, which determines last three dimensions of data
        shape_batch
            Optional preceding batch dimensions for vector fields, arrays of
            scalar fields etc. Not used if data is provided.
        data
            Initial data if provided; initialize to zero otherwise
        """
        self.grid = grid
        shape_grid = self.shape_grid()
        dtype = self.dtype()
        if data is None:
            # Initialize to zero:
            self.data = torch.zeros(tuple(shape_batch) + shape_grid,
                                    dtype=dtype, device=grid.rc.device)
        else:
            # Initialize to provided data:
            assert data.shape[-3:] == shape_grid
            assert data.dtype == dtype
            self.data = data

    def __add__(self: FieldType, other: FieldType) -> FieldType:
        if not isinstance(other, type(self)):
            return NotImplemented
        assert self.grid is other.grid
        return self.__class__(self.grid, data=(self.data + other.data))

    def __iadd__(self: FieldType, other: FieldType) -> FieldType:
        if not isinstance(other, type(self)):
            return NotImplemented
        assert self.grid is other.grid
        self.data += other.data
        return self

    def __sub__(self: FieldType, other: FieldType) -> FieldType:
        if not isinstance(other, type(self)):
            return NotImplemented
        assert self.grid is other.grid
        return self.__class__(self.grid, data=(self.data - other.data))

    def __isub__(self: FieldType, other: FieldType) -> FieldType:
        if not isinstance(other, type(self)):
            return NotImplemented
        assert self.grid is other.grid
        self.data -= other.data
        return self

    def __mul__(self: FieldType, other: float) -> FieldType:
        if not isinstance(other, Number):
            return NotImplemented
        return self.__class__(self.grid, data=(self.data * other))

    __rmul__ = __mul__

    def __imul__(self: FieldType, other: float) -> FieldType:
        if not isinstance(other, Number):
            return NotImplemented
        self.data *= other
        return self

    def norm(self) -> float:
        norm_sq = self.data.norm().item() ** 2
        if self.grid.comm is not None:
            norm_sq = self.grid.comm.allreduce(norm_sq, qp.MPI.SUM)
        return np.sqrt(norm_sq)


class FieldR(Field):
    """Real fields in real space."""
    def dtype(self) -> torch.dtype:
        return torch.double

    def shape_grid(self) -> Tuple[int, ...]:
        return self.grid.shapeR_mine

    def __invert__(self) -> 'FieldH':
        """Fourier transform (enables the ~ operator)"""
        return FieldH(self.grid, data=self.grid.fft(self.data))

    def to(self, grid: 'Grid') -> 'FieldR':
        """Switch field to another `grid` with same `shape`.
        The new grid can only differ in the MPI split."""
        if grid is self.grid:
            return self
        return _change_real(self, grid)


class FieldC(Field):
    """Complex fields in real space."""
    def dtype(self) -> torch.dtype:
        return torch.cdouble

    def shape_grid(self) -> Tuple[int, ...]:
        return self.grid.shapeR_mine

    def __invert__(self) -> 'FieldG':
        """Fourier transform (enables the ~ operator)"""
        return FieldG(self.grid, data=self.grid.fft(self.data))

    def to(self, grid: 'Grid') -> 'FieldC':
        """Switch field to another `grid` with same `shape`.
        The new grid can only differ in the MPI split."""
        if grid is self.grid:
            return self
        return _change_real(self, grid)


class FieldH(Field):
    """Real fields in (half) reciprocal space. Note that the underlying
    data is complex in reciprocal space, but reduced to one half of
    reciprocal space using Hermitian symmetry."""
    def dtype(self) -> torch.dtype:
        return torch.cdouble

    def shape_grid(self) -> Tuple[int, ...]:
        return self.grid.shapeH_mine

    def __invert__(self) -> 'FieldR':
        """Fourier transform (enables the ~ operator)"""
        return FieldR(self.grid, data=self.grid.ifft(self.data))

    def to(self, grid: 'Grid') -> 'FieldH':
        """Switch field to another `grid` with possibly different `shape`.
        This routine will perform Fourier resampling and MPI rearrangements,
        as necessary."""
        if grid is self.grid:
            return self
        return _change_recip(self, grid)


class FieldG(Field):
    """Complex fields in (full) reciprocal space."""
    def dtype(self) -> torch.dtype:
        return torch.cdouble

    def shape_grid(self) -> Tuple[int, ...]:
        return self.grid.shapeG_mine

    def __invert__(self) -> 'FieldC':
        """Fourier transform (enables the ~ operator)"""
        return FieldC(self.grid, data=self.grid.ifft(self.data))

    def to(self, grid: 'Grid') -> 'FieldG':
        """Switch field to another `grid` with possibly different `shape`.
        This routine will perform Fourier resampling and MPI rearrangements,
        as necessary."""
        if grid is self.grid:
            return self
        return _change_recip(self, grid)
