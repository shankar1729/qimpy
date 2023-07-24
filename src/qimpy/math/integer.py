from __future__ import annotations
from typing import overload, Union

import numpy as np


def prime_factorization(N: int) -> list[int]:
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


@overload
def ceildiv(num: int, den: int) -> int:
    ...


@overload
def ceildiv(num: np.ndarray, den: Union[int, np.ndarray]) -> np.ndarray:
    ...


@overload
def ceildiv(num: Union[int, np.ndarray], den: np.ndarray) -> np.ndarray:
    ...


def ceildiv(num, den):
    """Compute ceil(num/den) with purely integer operations"""
    return (num + den - 1) // den
