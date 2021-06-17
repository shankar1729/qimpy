'''Grids, fields and their operations'''
# List exported symbols for doc generation
__all__ = [
    'Grid', 'Field', 'FieldR', 'FieldC', 'FieldH', 'FieldG', 'Coulomb'
    'N_SIGMAS_PER_WIDTH']

from ._grid import Grid
from ._field import Field, FieldR, FieldC, FieldH, FieldG
from ._coulomb import Coulomb
import numpy as np

N_SIGMAS_PER_WIDTH: float = 1.+np.sqrt(-2.*np.log(np.finfo(float).eps))
'''Gaussian negligible after this many standard deviations.
 Evaluated at double precision with 1 extra standard deviation for margin.'''
