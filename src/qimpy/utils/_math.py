import numpy as np


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
