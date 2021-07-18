from typing import Protocol, TypeVar, Deque

T = TypeVar('T')


class Optimizable(Protocol):
    """Class requirements for use as vector space in optimization algorithms.
    This is required in :class:`Pulay` and :class:`Minimize`, for example."""
    def __add__(self: T, other: T) -> T: ...
    def __iadd__(self: T, other: T) -> T: ...
    def __sub__(self: T, other: T) -> T: ...
    def __isub__(self: T, other: T) -> T: ...
    def __mul__(self: T, other: float) -> T: ...
    def __rmul__(self: T, other: float) -> T: ...
    def overlap(self: T, other: T) -> float: ...


class ConvergenceCheck(Deque[bool]):
    """Check quantity stays unchanged a certain number of times."""
    __slots__ = ('threshold', 'n_check')
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
