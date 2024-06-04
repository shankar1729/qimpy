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
    "FieldSymmetrizer",
    "Coulomb",
    "Coulomb_Slab",
    "Coulomb_Isolated",
    "N_SIGMAS_PER_WIDTH",
)

from ._grid import Grid
from ._field import FieldType, Field, FieldR, FieldC, FieldH, FieldG
from ._field_symmetrizer import FieldSymmetrizer
from ._coulomb import Coulomb
from ._coulombslab import Coulomb_Slab
from ._coulombisolated import Coulomb_Isolated
import numpy as np

N_SIGMAS_PER_WIDTH: float = 1.0 + np.sqrt(-2.0 * np.log(np.finfo(float).eps))
"""Gaussian negligible after this many standard deviations.
 Evaluated at double precision with 1 extra standard deviation for margin."""
