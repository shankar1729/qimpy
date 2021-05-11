import numpy as np
import torch


def prime_factorization(N):
    '''Prime factorization of a number

    Parameters
    ----------
    N : int
        Number to factorize

    Returns
    -------
    list of int
        List of prime factors in ascending order'''
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


def fft_suitable(N):
    '''Determine whether the prime factorization of N is suitable for
    efficient FFTs, that is contains only 2, 3, 5 and 7.

    Parameters
    ----------
    N : int
        Candidate FFT dimension

    Returns
    -------
    bool
        True, if 2, 3, 5 and 7 are only prime factors, else False'''
    for p in [2, 3, 5, 7]:
        while N % p == 0:
            N /= p
    # All suitable prime factors taken out
    # --- a suitable N should be left with just 1
    return (N == 1)


def ceildiv(num, den):
    'Compute ceil(num/den) with int inputs, output and purely integer ops'
    return (num + den-1) // den


def ortho_matrix(O, use_cholesky=True):
    """Return orthonormalization matrix of a basis, given the overlap matrix
    (metric) of that basis.

    Parameters
    ----------
    O : torch.Tensor
        Overlap matrix / metric (Hermitian, positive definite) in last
        two dimensions, and batched over any preceding dimensions
    use_cholesky : bool, default: True
        If True, use Cholesky decomposition followed by a triangular solve,
        essentially amounting to Gram-Schmidt orthonormalization.
        If False, return the symmetric orthonormalization matrix calculated
        by diagonalizing O, which may be more stable, but may be an order of
        magnitude slower than the default Cholesky method

    Returns
    -------
    torch.Tensor (same dimensions as O)
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
