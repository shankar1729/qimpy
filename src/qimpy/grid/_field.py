import qimpy as qp
import numpy as np
import torch
from abc import abstractmethod
from numbers import Number
from ._change import _change_real, _change_recip
from typing import TypeVar, Any, Tuple, Optional, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from ._grid import Grid


FieldType = TypeVar('FieldType', bound='Field')  #: Type for field ops.


class Field:
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

    def __neg__(self: FieldType) -> FieldType:
        return self.__class__(self.grid, data=(-self.data))

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

    def dot(self: FieldType, other: FieldType) -> torch.Tensor:
        r"""Compute broadcasted inner product :math:`\int a^\dagger b`.
        This includes appropriate volume / mesh prefactors to convert it
        to an integral. Batch dimensions must be broadcastable together.
        Note that `a ^ b` is exactly equivalent to `a.overlap(b)` for `Field`s.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        assert self.grid is other.grid
        data1 = self.data.conj() if self.data.is_complex() else self.data
        data2 = other.data
        if type(self) == FieldH:
            result = (data1 * data2).sum(dim=(-3, -2))  # z-axis not summed yet
            # Account for Hermitian symmetry weights:
            split2H = self.grid.split2H
            iz_start = max(1, split2H.i_start) - split2H.i_start
            iz_stop = min(split2H.n_tot - 1, split2H.i_stop) - split2H.i_start
            result[..., iz_start:iz_stop] *= 2
            result = result.sum(dim=-1).real  # z-axis summed here
        else:
            result = (data1 * data2).sum(dim=(-3, -2, -1))
        # Volume factor:
        result *= (self.grid.lattice.volume  # recip. space integration weight
                   if isinstance(self, (FieldG, FieldH))
                   else self.grid.dV)  # real space integration weight
        # Collect over MPI if needed:
        if self.grid.comm is not None:
            result = result.contiguous()
            self.grid.comm.Allreduce(qp.MPI.IN_PLACE,
                                     qp.utils.BufferView(result), qp.MPI.SUM)
        return result

    __xor__ = dot

    def overlap(self: FieldType, other: FieldType) -> float:
        """Compute net inner product i.e. `dot` summed over batch dimensions.
        The result is always real, which amounts to treating complex fields
        as two-component real fields, and is therefore useful for optimizing
        over `Field`s eg. with the `Pulay` or `Minimizer` algorithm templates.
        """
        result = self.dot(other).sum()
        return float(result.real.item() if result.is_complex()
                     else result.item())

    def norm(self) -> torch.Tensor:
        r"""Norm of a field, defined by :math:`\sqrt{\int |a|^2}`.
        Returns a real tensor with shape equal to batch dimensions."""
        norm_sq = self ^ self
        return (norm_sq.real.sqrt() if norm_sq.is_complex()
                else norm_sq.sqrt())

    def get_origin_index(self):
        """Return index into local data of the spatial index = 0 component(s),
        which corresponds to r = 0 for real-space fields and to the G = 0
        component for reciprocal-space fields. Returns an empty index
        if the origin component is not local to this process.
        The `o` property provides convenient access to data at this index.
        """
        return ((slice(None),) * (len(self.data.shape)-3)  # batch dimensions
                + (((), (), ()) if self.grid.i_proc else (0, 0, 0)))

    @property
    def o(self) -> torch.Tensor:
        """Slice of `data` corresponding to :meth:`get_origin_index`."""
        return self.data[self.get_origin_index()]

    @o.setter
    def o(self, other: torch.Tensor) -> None:
        self.data[self.get_origin_index()] = other

    def __getitem__(self: FieldType, index: Any) -> FieldType:
        """Slice on batch dimensions."""
        return self.__class__(self.grid, data=self.data[index])

    def __setitem__(self: FieldType, index: Any, value: FieldType) -> None:
        """Assign to slice on batch dimensions"""
        self.data[index] = value.data

    def gradient(self: FieldType, dim: int = 0) -> FieldType:
        """Gradient of field. A new batch dimension of length 3 is inserted
        at the location specified by `dim`, by default at the beginning."""
        field_type = type(self)
        if field_type in {FieldR, FieldC}:  # apply in reciprocal space
            return ~((~self).gradient(dim=dim))  # type: ignore
        op = self.grid.get_gradient_operator('H' if (field_type == FieldH)
                                             else 'G')
        shape_in = self.data.shape
        n_batch_dims = len(shape_in) - 3
        shape_data = shape_in[:dim] + (1,) + shape_in[dim:]
        shape_op = (1,)*dim + (3,) + (1,)*(n_batch_dims-dim) + op.shape[1:]
        return self.__class__(self.grid, data=(op.view(shape_op)
                                               * self.data.view(shape_data)))

    def divergence(self: FieldType, dim: int = 0) -> FieldType:
        """Divergence of field. A dimension of length 3 at `dim`, by default
        at the beginning, is contracted against the gradient operator."""
        field_type = type(self)
        if field_type in {FieldR, FieldC}:  # apply in reciprocal space
            return ~((~self).divergence(dim=dim))  # type: ignore
        op = self.grid.get_gradient_operator('H' if (field_type == FieldH)
                                             else 'G')
        n_batch_dims = len(self.data.shape) - 4  # other than contracted one
        shape_op = (1,)*dim + (3,) + (1,)*(n_batch_dims-dim) + op.shape[1:]
        return self.__class__(self.grid, data=(op.view(shape_op)
                                               * self.data).sum(dim=dim))

    def laplacian(self: FieldType) -> FieldType:
        """Laplacian of field."""
        field_type = type(self)
        if field_type in {FieldR, FieldC}:  # apply in reciprocal space
            return ~((~self).laplacian())  # type: ignore
        iG = self.grid.get_mesh('H' if (field_type == FieldH) else 'G')
        op = -((iG.to(torch.double) @ self.grid.lattice.Gbasis)**2).sum(dim=-1)
        return self.__class__(self.grid, data=(op * self.data))

    def zeros_like(self: FieldType) -> FieldType:
        """Create zero Field with same grid and batch dimensions."""
        return self.__class__(self.grid, shape_batch=self.data.shape[:-3])


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

    def symmetrize(self) -> None:
        """Symmetrize field in-place."""
        self.grid.field_symmetrizer(self)


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
