import qimpy

from qimpy.grid.test_common import get_sequential_grid 
from qimpy.grid.test_common import get_reference_field

from . import Coulomb, FieldH
from . import Coulomb_Slab
def test_energy():
    grid = get_sequential_grid((10, 10, 10))
    #coulomb = Coulomb(grid, 5)
    coulomb = Coulomb_Slab(grid, 5, 2)
    fieldh = get_reference_field(FieldH, grid)
    #result = coulomb(fieldh)
    #return result

