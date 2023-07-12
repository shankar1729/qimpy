"""Grids, fields and their operations"""
# List exported symbols for doc generation
__all__ = (
    "Grid",
    "FieldType",
    "Field",
    "FieldR",
    "FieldC",
    "FieldH",
    "FieldG",
    "Coulomb",
    "N_SIGMAS_PER_WIDTH",
)

from .grid import Grid
from .field import FieldType, Field, FieldR, FieldC, FieldH, FieldG
from .coulomb import Coulomb
import numpy as np

N_SIGMAS_PER_WIDTH: float = 1.0 + np.sqrt(-2.0 * np.log(np.finfo(float).eps))
"""Gaussian negligible after this many standard deviations.
 Evaluated at double precision with 1 extra standard deviation for margin."""
