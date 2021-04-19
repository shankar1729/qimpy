import qimpy as qp
import numpy as np


class System:

    def __init__(self, *, lattice, ions):

        # Initialize lattice:
        if isinstance(lattice, dict):
            self.lattice = qp.Lattice(**lattice)
        elif isinstance(lattice, qp.Lattice):
            self.lattice = lattice
        else:
            raise TypeError("lattice must be dict or qimpy.Lattice")

        # Initialize ions:
        if isinstance(ions, dict):
            self.ions = qp.ions.Ions(**ions)
        elif isinstance(ions, qp.ions.Ions):
            self.ions = ions
        else:
            raise TypeError("ions must be dict or qimpy.ions.Ions")


def fmt(tensor):
    'Standardized printing of arrays within QimPy'
    return np.array2string(
        tensor.numpy(),
        precision=8,
        separator=', ')
