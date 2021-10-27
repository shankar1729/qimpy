import numpy as np
import torch
from typing import List, Tuple, TypeVar


def prime_factorization(N: int) -> List[int]:
    """Get list of prime factors of `N` in ascending order"""
    factors = []
    p = 2
    while p * p <= N:
        while N % p == 0:
            factors.append(p)
            N //= p
        p += 1
    if N > 1:  # any left-over factor must be prime itself
        factors.append(N)
    return factors


def fft_suitable(N: int) -> bool:
    """Check whether `N` has only small prime factors. Return True if
    the prime factorization of `N` is suitable for efficient FFTs,
    that is contains only 2, 3, 5 and 7."""
    for p in [2, 3, 5, 7]:
        while N % p == 0:
            N //= p
    # All suitable prime factors taken out
    # --- a suitable N should be left with just 1
    return N == 1


IntLike = TypeVar("IntLike", int, np.ndarray)


def ceildiv(num: IntLike, den: IntLike) -> IntLike:
    """Compute ceil(num/den) with purely integer operations"""
    return (num + den - 1) // den


def cis(x: torch.Tensor) -> torch.Tensor:
    """Compute complex exponential exp(i x)."""
    return torch.polar(torch.ones_like(x), x)


def dagger(x: torch.Tensor) -> torch.Tensor:
    """Conjugate transpose of a batch of matrices. Matrix dimensions are
    assumed to be the final two, with all preceding dimensions batched over."""
    return x.conj().transpose(-2, -1)


def abs_squared(x: torch.Tensor) -> torch.Tensor:
    """Compute absolute value squared of complex or real tensor.
    Avoids complex multiply and square root for efficiency."""
    if x.is_complex():
        return torch.view_as_real(x).square().sum(dim=-1)
    else:
        return x.square()


def accum_norm_(
    f: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    start_dim: int,
    safe_mode: bool = False,
) -> None:
    """Accumulate :math:`f |x|^2` to `out` in-place. The result is accumulated
    over dimensions `start_dim` to number of dimensions of `f`. Dimensions of
    `f` must match the starting dimensions of `x`. Dimensions of `out` must
    match `f` up to `start_dim`, and then equal to those of `x` beyond
    the dimensions of `f`.
    This is equivalent to performing:

    `out += (f[..., #] * abs_squared(x)).sum(dim=(start_dim ... len(f.shape)))`

    where `f[..., #]` indicates broadcasting `f` to match initial dimensions
    of `x`. This pattern is used in density computation, and is extremely
    inefficient as written due to multiple passes through the data of `x`.
    If `safe_mode` is set to True, this function returns the above naive
    implementation for checking correctness conveniently.
    This function optimizes this evaluation as much as possible while staying
    in pure Python, and is a prime candidate for C++ re-implementation."""
    stop_dim = len(f.shape)
    if safe_mode:
        n_extra = len(x.shape) - stop_dim
        out += (f.view(f.shape + (1,) * n_extra) * abs_squared(x)).sum(
            dim=tuple(range(start_dim, stop_dim))
        )
        return
    assert f.shape == x.shape[:stop_dim]
    assert f.shape[:start_dim] == out.shape[:start_dim]
    assert x.shape[stop_dim:] == out.shape[start_dim:]
    f_np = f.cpu().numpy()
    for index_in in np.argwhere(f_np):
        index = tuple(index_in)
        f_cur = f_np[index]  # this is a scalar
        x_cur = x[index]  # this will typically have extra dimenions
        out_cur = out[index[:start_dim]] if start_dim else out
        if x_cur.is_complex():
            for x_component in (x_cur.real, x_cur.imag):
                out_cur.addcmul_(x_component, x_component, value=f_cur)
        else:
            out_cur.addcmul_(x_cur, x_cur, value=f_cur)


def accum_prod_(
    f: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor,
    start_dim: int,
    safe_mode: bool = False,
) -> None:
    """Accumulate :math:`f x y` to `out` in-place.
    Similar to `accum_norm_`, except for a product of two tensors `x` and `y`,
    instead of the norm of a single tensor `x`. See :func:`accum_norm` for
    details on the indexing and dimensions that are summed."""
    stop_dim = len(f.shape)
    if safe_mode:
        n_extra = len(x.shape) - stop_dim
        out += (f.view(f.shape + (1,) * n_extra) * x * y).sum(
            dim=tuple(range(start_dim, stop_dim))
        )
        return
    assert f.shape == x.shape[:stop_dim]
    assert f.shape[:start_dim] == out.shape[:start_dim]
    assert x.shape[stop_dim:] == out.shape[start_dim:]
    assert x.shape == y.shape
    f_np = f.cpu().numpy()
    for index_in in np.argwhere(f_np):
        index = tuple(index_in)
        f_cur = f_np[index]  # this is a scalar
        x_cur = x[index]  # this will typically have extra dimenions
        y_cur = y[index]
        out_cur = out[index[:start_dim]] if start_dim else out
        out_cur.addcmul_(x_cur, y_cur, value=f_cur)


def ortho_matrix(O: torch.Tensor, use_cholesky: bool = True) -> torch.Tensor:
    """Return orthonormalization matrix of a basis.
    The basis is specified by its overlap matrix or metric, `O`.

    Parameters
    ----------
    O
        Overlap matrix / metric (Hermitian, positive definite) in last
        two dimensions, and batched over any preceding dimensions
    use_cholesky
        If True, use Cholesky decomposition followed by a triangular solve,
        essentially amounting to Gram-Schmidt orthonormalization.
        If False, return the symmetric orthonormalization matrix calculated
        by diagonalizing O, which may be more stable, but may be an order of
        magnitude slower than the default Cholesky method
    """
    assert O.shape[-2] == O.shape[-1]  # check square
    if use_cholesky:
        # Gram-Schmidt orthonormalization matrix:
        identity = torch.eye(O.shape[-1], device=O.device, dtype=O.dtype).view(
            (1,) * (len(O.shape) - 2) + O.shape[-2:]
        )
        return dagger(
            torch.triangular_solve(identity, torch.linalg.cholesky(O), upper=False)[0]
        )
    else:
        # Symmetric orthonormalization matrix:
        lbda, V = torch.linalg.eigh(O)
        return V @ ((1.0 / torch.sqrt(lbda))[..., None] * dagger(V))


def eighg(
    H: torch.Tensor, O: torch.Tensor, use_cholesky: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve Hermitian generalized eigenvalue problem.
    Specifically, find `E` and `V` that satisfy `H` @ `V` = `O` @ `V` @ `E`.

    Parameters
    ----------
    H
        Set of complex Hermitian (in last two dimensions) matrices
        to diagonalize, all dimensions before last two are batched over
    O
        Corresponding overlap (metric) matrices, with same size as H.
        Must additionally be positive-definite.
    use_cholesky
        See :meth:`qimpy.utils.ortho_matrix`

    Returns
    -------
    E : torch.Tensor
        Real eigenvalues (shape = H.shape[:-1])
    V : torch.Tensor
        Eigenvectors (same shape as H and O)
    """
    U = ortho_matrix(O, use_cholesky)
    E, V = torch.linalg.eigh(dagger(U) @ (H @ U))
    return E, U @ V  # transform eigenvectors back to original basis
