import numpy as np
import torch
from typing import List, Tuple


def prime_factorization(N: int) -> List[int]:
    'Get list of prime factors of `N` in ascending order'
    factors = []
    p = 2
    while p*p <= N:
        while N % p == 0:
            factors.append(p)
            N //= p
        p += 1
    if N > 1:  # any left-over factor must be prime itself
        factors.append(N)
    return factors


def fft_suitable(N: int) -> bool:
    '''Check whether the prime factorization of `N` is suitable for
    efficient FFTs, that is contains only 2, 3, 5 and 7.'''
    for p in [2, 3, 5, 7]:
        while N % p == 0:
            N /= p
    # All suitable prime factors taken out
    # --- a suitable N should be left with just 1
    return (N == 1)


def ceildiv(num: int, den: int) -> int:
    'Compute ceil(num/den) with purely integer operations'
    return (num + den-1) // den


def ortho_matrix(O: torch.Tensor, use_cholesky: bool = True) -> torch.Tensor:
    """Return orthonormalization matrix of a basis, given the overlap matrix
    (metric) `O` of that basis.

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
    assert(O.shape[-2] == O.shape[-1])  # check square
    if use_cholesky:
        # Gram-Schmidt orthonormalization matrix:
        identity = torch.eye(O.shape[-1], device=O.device,
                             dtype=O.dtype).view((1,) * (len(O.shape) - 2)
                                                 + O.shape[-2:])
        return torch.triangular_solve(identity, torch.linalg.cholesky(O),
                                      upper=False)[0].transpose(-2, -1).conj()
    else:
        # Symmetric orthonormalization matrix:
        lbda, V = torch.linalg.eigh(O)
        return V @ ((1./torch.sqrt(lbda))[..., None]
                    * V.transpose(-2, -1).conj())


def eighg(H: torch.Tensor, O: torch.Tensor,
          use_cholesky: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve Hermitian generalized eigenvalue problem
    `H` @ `V` = `O` @ `V` @ `E`

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
    E, V = torch.linalg.eigh(U.transpose(-2, -1).conj() @ (H @ U))
    return E, U @ V  # transform eigenvectors back to original basis
