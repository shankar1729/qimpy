from __future__ import annotations
from typing import TypeVar, Any, Union, Optional, Sequence
from abc import abstractmethod

import numpy as np
import torch

from qimpy import rc, MPI
from qimpy.algorithms import Gradable
from qimpy.math import random
from qimpy.mpi import BufferView
from qimpy.io import CheckpointPath
from . import Grid
from ._change import _change_real, _change_recip


FieldType = TypeVar("FieldType", bound="Field")  #: Type for field ops.


class Field(Gradable[FieldType]):
    """Abstract base class for scalar/vector fields in real/reciprocal space.
    Provides common operators for fields in either space, but any fields
    used must specifically be in real (:class:`FieldR` and :class:`FieldC`),
    full-reciprocal (:class:`FieldG`) or half-reciprocal (:class:`FieldH`)
    space."""

    grid: Grid  #: Associated grid that determines dimensions of field
    data: torch.Tensor  #: Underlying data, with last three dimensions on grid

    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Data type for the Field type"""

    @abstractmethod
    def shape_grid(self) -> tuple[int, ...]:
        """Global grid shape for the Field type"""

    @abstractmethod
    def shape_grid_mine(self) -> tuple[int, ...]:
        """Local grid shape (last 3 data dimensions) for the Field type"""

    @abstractmethod
    def offset_grid_mine(self) -> tuple[int, ...]:
        """Offset of local grid dimensions into global grid for Field type"""

    @property
    @abstractmethod
    def is_complex(self) -> bool:
        """Whether this represents a complex scalar field.
        Note that a real scalar field has complex Fourier transform coefficients,
        but would still be considered real here (i.e. this is False for `FieldH`).
        """

    @property
    @abstractmethod
    def is_tilde(self) -> bool:
        """Whether this field is in reciprocal space.
        (Corresponding variable names are typically suffixed by '_tilde'.)
        """

    @abstractmethod
    def __invert__(self) -> Field[Any]:
        """Fourier transform (enables the ~ operator)"""

    @abstractmethod
    def to(self: FieldType, grid: Grid) -> FieldType:
        """Switch field to a grid that is different in shape or parallelization."""

    def __init__(
        self,
        grid: Grid,
        *,
        shape_batch: Sequence[int] = tuple(),
        data: Optional[torch.Tensor] = None,
    ) -> None:
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
        super().__init__()
        self.grid = grid
        shape_grid_mine = self.shape_grid_mine()
        dtype = self.dtype()
        if data is None:
            # Initialize to zero:
            self.data = torch.zeros(
                tuple(shape_batch) + shape_grid_mine, dtype=dtype, device=rc.device
            )
        else:
            # Initialize to provided data:
            assert data.shape[-3:] == shape_grid_mine
            assert data.dtype == dtype
            self.data = data

    def clone(self: FieldType) -> FieldType:
        """Create field with cloned data (deep copy)."""
        return self.__class__(self.grid, data=self.data.clone())

    def add_(
        self: FieldType, other: Union[FieldType, float], *, alpha: float = 1.0
    ) -> FieldType:
        """Add in-place with optional scale factor (Mirroring torch.Tensor.add_)."""
        if isinstance(other, float):
            if self.is_tilde:
                # Scalar shift only affects constant G=0 term:
                self.data[self.get_origin_index()] += other * alpha
            else:
                self.data += other * alpha
        elif isinstance(other, type(self)):
            assert self.grid is other.grid
            self.data.add_(other.data, alpha=alpha)
        else:
            return NotImplemented
        return self

    def __add__(self: FieldType, other: Union[FieldType, float]) -> FieldType:
        return self.clone().add_(other)

    __radd__ = __add__

    def __iadd__(self: FieldType, other: Union[FieldType, float]) -> FieldType:
        return self.add_(other)

    def __sub__(self: FieldType, other: Union[FieldType, float]) -> FieldType:
        return self.clone().add_(other, alpha=-1.0)

    def __rsub__(self: FieldType, other: Union[FieldType, float]) -> FieldType:
        return (-self).add_(other)

    def __isub__(self: FieldType, other: Union[FieldType, float]) -> FieldType:
        return self.add_(other, alpha=-1.0)

    def __neg__(self: FieldType) -> FieldType:
        return self.__class__(self.grid, data=(-self.data))

    def __mul__(self: FieldType, other: Union[FieldType, float]) -> FieldType:
        return self.clone().__imul__(other)

    __rmul__ = __mul__

    def __imul__(self: FieldType, other: Union[FieldType, float]) -> FieldType:
        if isinstance(other, float):
            self.data *= other
        elif isinstance(other, type(self)):
            self.data *= other.data
        else:
            return NotImplemented
        return self

    def __truediv__(self: FieldType, other: Union[FieldType, float]) -> FieldType:
        return self.clone().__itruediv__(other)

    def __rtruediv__(self: FieldType, other: float) -> FieldType:
        return self.__class__(self.grid, data=(other / self.data))

    def __itruediv__(self: FieldType, other: Union[FieldType, float]) -> FieldType:
        if isinstance(other, float):
            self.data /= other
        elif isinstance(other, type(self)):
            self.data /= other.data
        else:
            return NotImplemented
        return self

    def integral(self) -> torch.Tensor:
        r"""Compute integral over unit cell, retaining batch dimensions in output."""
        data = self.data
        grid = self.grid
        if self.is_tilde:
            # Integral depends on G=0 component alone:
            result = (
                self.o * grid.lattice.volume
                if (grid.i_proc == 0)
                else torch.zeros(data.shape[:-3], device=data.device, dtype=data.dtype)
            )
            if isinstance(self, FieldH):
                result = result.real  # due to Hermitian symmetry
        else:
            result = data.sum(dim=(-3, -2, -1)) * grid.dV
        # Collect over MPI if needed:
        if self.grid.comm is not None:
            result = result.contiguous()
            rc.current_stream_synchronize()
            self.grid.comm.Allreduce(MPI.IN_PLACE, BufferView(result), MPI.SUM)
        return result

    def dot(self: FieldType, other: FieldType) -> torch.Tensor:
        r"""Compute broadcasted inner product :math:`\int a^\dagger b`.
        This includes appropriate volume / mesh prefactors to convert it
        to an integral. Batch dimensions must be broadcastable together.
        Note that `a ^ b` is exactly equivalent to `a.dot(b)`.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        assert self.grid is other.grid
        data1 = self.data.conj() if self.data.is_complex() else self.data
        data2 = other.data
        if isinstance(self, FieldH):
            result = (data1 * data2).sum(dim=(-3, -2)).real @ self.grid.weight2H
        else:
            result = (data1 * data2).sum(dim=(-3, -2, -1))
        # Volume factor:
        if self.is_tilde:
            result *= self.grid.lattice.volume  # reciprocal space integration weight
        else:
            result *= self.grid.dV  # real space integration weight
        # Collect over MPI if needed:
        if self.grid.comm is not None:
            result = result.contiguous()
            rc.current_stream_synchronize()
            self.grid.comm.Allreduce(MPI.IN_PLACE, BufferView(result), MPI.SUM)
        return result

    __xor__ = dot

    def vdot(self: FieldType, other: FieldType) -> float:
        """Vector-space dot product of data summed over all dimensions.
        (Scalar contraction needed for the `Pulay` or `Minimizer` algorithm templates.)
        """
        result = torch.vdot(self.data.flatten(), other.data.flatten()).real
        if self.grid.comm is not None:
            self.grid.comm.Allreduce(MPI.IN_PLACE, BufferView(result), MPI.SUM)
        return result.item()

    def norm(self: FieldType) -> torch.Tensor:
        r"""Norm of a field, defined by :math:`\sqrt{\int |a|^2}`.
        Returns a real tensor with shape equal to batch dimensions."""
        norm_sq = self ^ self
        return norm_sq.real.sqrt() if norm_sq.is_complex() else norm_sq.sqrt()

    def get_origin_index(self):
        """Return index into local data of the spatial index = 0 component(s),
        which corresponds to r = 0 for real-space fields and to the G = 0
        component for reciprocal-space fields. Returns an empty index
        if the origin component is not local to this process.
        The `o` property provides convenient access to data at this index.
        """
        return (slice(None),) * (len(self.data.shape) - 3) + (  # batch dimensions
            ((), (), ()) if self.grid.i_proc else (0, 0, 0)
        )

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
        if not self.is_tilde:  # apply in reciprocal space
            return ~((~self).gradient(dim=dim))  # type: ignore
        op = self.grid.get_gradient_operator("G" if self.is_complex else "H")
        shape_in = self.data.shape
        n_batch_dims = len(shape_in) - 3
        shape_data = shape_in[:dim] + (1,) + shape_in[dim:]
        shape_op = (1,) * dim + (3,) + (1,) * (n_batch_dims - dim) + op.shape[1:]
        return self.__class__(
            self.grid, data=(op.view(shape_op) * self.data.view(shape_data))
        )

    def divergence(self: FieldType, dim: int = 0) -> FieldType:
        """Divergence of field. A dimension of length 3 at `dim`, by default
        at the beginning, is contracted against the gradient operator."""
        if not self.is_tilde:  # apply in reciprocal space
            return ~((~self).divergence(dim=dim))  # type: ignore
        op = self.grid.get_gradient_operator("G" if self.is_complex else "H")
        n_batch_dims = len(self.data.shape) - 4  # other than contracted one
        shape_op = (1,) * dim + (3,) + (1,) * (n_batch_dims - dim) + op.shape[1:]
        return self.__class__(
            self.grid, data=(op.view(shape_op) * self.data).sum(dim=dim)
        )

    def laplacian(self: FieldType) -> FieldType:
        """Laplacian of field."""
        if not self.is_tilde:  # apply in reciprocal space
            return ~((~self).laplacian())  # type: ignore
        iG = self.grid.get_mesh("G" if self.is_complex else "H")
        op = -((iG.to(torch.double) @ self.grid.lattice.Gbasis) ** 2).sum(dim=-1)
        return self.__class__(self.grid, data=(op * self.data))

    def convolve(
        self: FieldType, kernel_tilde: torch.Tensor, reduction: str = ""
    ) -> FieldType:
        """Convolve scalar field by reciprocal-space `kernel_tilde`.
        Amounts to an elementwise multiply for reciprocal space fields,
        but involve a forward and inverse fourier transform for real space fields.
        If specified, reduction must be an `einsum` index expression to contract
        certain indices of `self` and `kernel_tilde` (in that order).
        """
        if not self.is_tilde:  # apply in reciprocal space
            return ~((~self).convolve(kernel_tilde, reduction))  # type: ignore
        if reduction:
            data = torch.einsum(reduction, self.data, kernel_tilde)
        else:
            data = self.data * kernel_tilde
        return self.__class__(self.grid, data=data)

    def zeros_like(self: FieldType) -> FieldType:
        """Create zero Field with same grid and batch dimensions."""
        return self.__class__(self.grid, shape_batch=self.data.shape[:-3])

    def read(self, cp_path: CheckpointPath) -> None:
        """Read field from `cp_path`."""
        checkpoint, path = cp_path
        assert checkpoint is not None
        dset = checkpoint[path]
        shape_batch = self.data.shape[:-3]
        assert dset.shape == (shape_batch + self.shape_grid())
        offset = (0,) * len(shape_batch) + self.offset_grid_mine()
        size = shape_batch + self.shape_grid_mine()
        self.data = (
            checkpoint.read_slice_complex(dset, offset, size)
            if self.dtype().is_complex
            else checkpoint.read_slice(dset, offset, size)
        )

    def write(self, cp_path: CheckpointPath) -> None:
        """Write field to `cp_path`."""
        checkpoint, path = cp_path
        assert checkpoint is not None
        shape_batch = self.data.shape[:-3]
        dtype: torch.dtype = self.data.dtype
        shape = shape_batch + self.shape_grid()  # global dimensions
        offset = (0,) * len(shape_batch) + self.offset_grid_mine()
        if dtype.is_complex:
            dset = checkpoint.create_dataset_complex(path, shape=shape, dtype=dtype)
            checkpoint.write_slice_complex(dset, offset, self.data)
        else:
            dset = checkpoint.create_dataset_real(path, shape=shape, dtype=dtype)
            checkpoint.write_slice(dset, offset, self.data)

    def randomize(self, seed: int = 0) -> None:
        """Initialize to MPI-reproducible standard-normal numbers, based on `seed`."""
        # Initialize reproducible, split-independent random number state:
        # Ignore penultimate (y) dimension in state, since it is looped over later.
        # (The y direction is picked for looping since it is never split.)
        grid = self.grid
        shape_mine = self.data.shape
        shape_full = shape_mine[:-3] + grid.shape  # without split or H symmetry
        shape_mine = shape_mine[:-2] + shape_mine[-1:]  # drop y
        shape_full = shape_full[:-2] + shape_full[-1:]  # drop y
        offsets = [0] * len(shape_full)
        if self.is_tilde:
            offsets[-1] = (
                grid.split2.i_start if self.is_complex else grid.split2H.i_start
            )
        else:
            offsets[-2] = grid.split0.i_start  # note that dim 1 already removed
        seed_offset = 1 + seed * np.prod(shape_full)
        state = torch.full(shape_mine, seed_offset, device=rc.device, dtype=torch.int64)
        for i_dim, offset in enumerate(offsets):
            cur_offset = torch.arange(shape_mine[i_dim], device=rc.device) + offset
            stride = int(np.prod(shape_full[i_dim + 1 :]))
            bcast_shape = [1] * len(shape_full)
            bcast_shape[i_dim] = -1
            state += stride * cur_offset.view(bcast_shape)
        random.initialize_state(state)

        # Assign random numbers one slice of y at a time:
        n1 = grid.shape[1]
        if self.dtype().is_complex:
            for i1 in range(n1):
                self.data[..., i1, :] = random.randn(state)
        else:
            for i1 in range(0, n1, 2):
                data_complex = random.randn(state)
                self.data[..., i1, :] = data_complex.real
                if i1 + 1 < n1:
                    self.data[..., i1 + 1, :] = data_complex.imag

        # Enforce hermitian symmetry for FieldH:
        if self.is_tilde and (not self.is_complex):
            if grid.split2H.is_mine(0):
                self.data[..., 0].imag.zero_()
            half_n2 = grid.shape[2] // 2
            if (2 * half_n2 == grid.shape[2]) and grid.split2H.is_mine(half_n2):
                self.data[..., half_n2 - grid.split2H.i_start].imag.zero_()


class FieldR(Field["FieldR"]):
    """Real fields in real space."""

    def dtype(self) -> torch.dtype:
        return torch.double

    def shape_grid(self) -> tuple[int, ...]:
        return self.grid.shape

    def shape_grid_mine(self) -> tuple[int, ...]:
        return self.grid.shapeR_mine

    def offset_grid_mine(self) -> tuple[int, ...]:
        return self.grid.split0.i_start, 0, 0

    @property
    def is_complex(self) -> bool:
        return False

    @property
    def is_tilde(self) -> bool:
        return False

    def __invert__(self) -> FieldH:
        """Fourier transform (enables the ~ operator)"""
        return FieldH(self.grid, data=self.grid.fft(self.data))

    def to(self, grid: Grid) -> FieldR:
        """Switch field to another `grid` with same `shape`.
        The new grid can only differ in the MPI split."""
        if grid is self.grid:
            return self
        return _change_real(self, grid)

    def log(self) -> FieldR:
        return FieldR(self.grid, data=self.data.log())

    def exp(self) -> FieldR:
        return FieldR(self.grid, data=self.data.exp())


class FieldC(Field["FieldC"]):
    """Complex fields in real space."""

    def dtype(self) -> torch.dtype:
        return torch.cdouble

    def shape_grid(self) -> tuple[int, ...]:
        return self.grid.shape

    def shape_grid_mine(self) -> tuple[int, ...]:
        return self.grid.shapeR_mine

    def offset_grid_mine(self) -> tuple[int, ...]:
        return self.grid.split0.i_start, 0, 0

    @property
    def is_complex(self) -> bool:
        return True

    @property
    def is_tilde(self) -> bool:
        return False

    def __invert__(self) -> FieldG:
        """Fourier transform (enables the ~ operator)"""
        return FieldG(self.grid, data=self.grid.fft(self.data))

    def to(self, grid: Grid) -> FieldC:
        """Switch field to another `grid` with same `shape`.
        The new grid can only differ in the MPI split."""
        if grid is self.grid:
            return self
        return _change_real(self, grid)


class FieldH(Field["FieldH"]):
    """Real fields in (half) reciprocal space. Note that the underlying
    data is complex in reciprocal space, but reduced to one half of
    reciprocal space using Hermitian symmetry."""

    def dtype(self) -> torch.dtype:
        return torch.cdouble

    def shape_grid(self) -> tuple[int, ...]:
        return self.grid.shapeH

    def shape_grid_mine(self) -> tuple[int, ...]:
        return self.grid.shapeH_mine

    def offset_grid_mine(self) -> tuple[int, ...]:
        return 0, 0, self.grid.split2H.i_start

    @property
    def is_complex(self) -> bool:
        return False

    @property
    def is_tilde(self) -> bool:
        return True

    def __invert__(self) -> FieldR:
        """Fourier transform (enables the ~ operator)"""
        return FieldR(self.grid, data=self.grid.ifft(self.data))

    def to(self, grid: Grid) -> FieldH:
        """Switch field to another `grid` with possibly different `shape`.
        This routine will perform Fourier resampling and MPI rearrangements,
        as necessary."""
        if grid is self.grid:
            return self
        return _change_recip(self, grid)

    def symmetrize(self) -> None:
        """Symmetrize field in-place."""
        self.grid.field_symmetrizer(self)


class FieldG(Field["FieldG"]):
    """Complex fields in (full) reciprocal space."""

    def dtype(self) -> torch.dtype:
        return torch.cdouble

    def shape_grid(self) -> tuple[int, ...]:
        return self.grid.shape

    def shape_grid_mine(self) -> tuple[int, ...]:
        return self.grid.shapeG_mine

    def offset_grid_mine(self) -> tuple[int, ...]:
        return 0, 0, self.grid.split2.i_start

    @property
    def is_complex(self) -> bool:
        return True

    @property
    def is_tilde(self) -> bool:
        return True

    def __invert__(self) -> FieldC:
        """Fourier transform (enables the ~ operator)"""
        return FieldC(self.grid, data=self.grid.ifft(self.data))

    def to(self, grid: Grid) -> FieldG:
        """Switch field to another `grid` with possibly different `shape`.
        This routine will perform Fourier resampling and MPI rearrangements,
        as necessary."""
        if grid is self.grid:
            return self
        return _change_recip(self, grid)
