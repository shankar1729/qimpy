"""Coulomb interactions with support for partial periodicity"""
# List exported symbols for doc generation
__all__ = (
    "Coulomb",
    "Kernel",
    "Ewald",
    "N_SIGMAS_PER_WIDTH",
)

from ._coulomb import Coulomb, Kernel, Ewald
import numpy as np

N_SIGMAS_PER_WIDTH: float = 1.0 + np.sqrt(-2.0 * np.log(np.finfo(float).eps))
"""Gaussian negligible after this many standard deviations.
 Evaluated at double precision with 1 extra standard deviation for margin."""
