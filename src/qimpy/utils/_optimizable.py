import qimpy as qp
import torch
from dataclasses import dataclass
from typing import Protocol, TypeVar, Deque
from qimpy.rc import MPI

T = TypeVar("T")


class Optimizable(Protocol):
    """Class requirements for use as vector space in optimization algorithms.
    This is required in :class:`Pulay` and :class:`Minimize`, for example."""

    def __add__(self: T, other: T) -> T:
        ...

    def __iadd__(self: T, other: T) -> T:
        ...

    def __sub__(self: T, other: T) -> T:
        ...

    def __isub__(self: T, other: T) -> T:
        ...

    def __mul__(self: T, other: float) -> T:
        ...

    def __rmul__(self: T, other: float) -> T:
        ...

    def __imul__(self: T, other: float) -> T:
        ...

    def vdot(self: T, other: T) -> float:
        ...


class ConvergenceCheck(Deque[bool]):
    """Check quantity stays unchanged a certain number of times."""

    threshold: float  #: Convergence threshold
    n_check: int  #: Number of consecutive checks that must pass at convergence

    def __init__(self, threshold: float, n_check: int = 2) -> None:
        """Initialize convergence check to specified `threshold`.
        The check must pass `n_check` consecutive times."""
        self.threshold = threshold
        self.n_check = n_check
        super().__init__(maxlen=n_check)

    def check(self, v: float) -> bool:
        """Return if converged, given latest quantity `v` to check."""
        self.append(abs(v) < self.threshold)
        return all(converged for converged in self)


@dataclass
class MatrixArray:
    """Array of matrices implementing the `Optimizable` protocol.
    This is convenient as an independent variable for optimizing over
    subspace rotations, such as in `LCAO` and `Wannier`."""

    M: torch.Tensor  #: Array of matrices with dimension ..., N x N
    comm: MPI.Comm  #: Communicator where M is split on some dimension(s)

    def __add__(self, other: "MatrixArray") -> "MatrixArray":
        return MatrixArray(M=(self.M + other.M), comm=self.comm)

    def __iadd__(self, other: "MatrixArray") -> "MatrixArray":
        self.M += other.M
        return self

    def __sub__(self, other: "MatrixArray") -> "MatrixArray":
        return MatrixArray(M=(self.M - other.M), comm=self.comm)

    def __isub__(self, other: "MatrixArray") -> "MatrixArray":
        self.M -= other.M
        return self

    def __mul__(self, other: float) -> "MatrixArray":
        return MatrixArray(M=(self.M * other), comm=self.comm)

    __rmul__ = __mul__

    def __imul__(self, other: float) -> "MatrixArray":
        self.M *= other
        return self

    def vdot(self, other: "MatrixArray") -> float:
        """Global vector-space dot product collected over `comm`."""
        result = torch.vdot(self.M.flatten(), other.M.flatten()).real
        self.comm.Allreduce(MPI.IN_PLACE, qp.utils.BufferView(result), MPI.SUM)
        return result.item()
