import qimpy as qp
import numpy as np


class System:
    '''TODO: document class System'''

    def __init__(self, *, rc, lattice, ions=None, symmetries=None):
        '''
        Parameters
        ----------
        TODO
        '''
        self.rc = rc

        # Initialize lattice:
        if isinstance(lattice, dict):
            self.lattice = qp.lattice.Lattice(rc=rc, **lattice)
        elif isinstance(lattice, qp.lattice.Lattice):
            self.lattice = lattice
        else:
            raise TypeError("lattice must be dict or qimpy.lattice.Lattice")

        # Initialize ions:
        if ions is None:
            # Set-up default of no ions:
            ions = {
                'pseudopotentials': [],
                'coordinates': []}
        if isinstance(ions, dict):
            self.ions = qp.ions.Ions(rc=rc, **ions)
        elif isinstance(ions, qp.ions.Ions):
            self.ions = ions
        else:
            raise TypeError("ions must be dict or qimpy.ions.Ions")

        # Initialize symmetries:
        if symmetries is None:
            symmetries = {}
        if isinstance(symmetries, dict):
            self.symmetries = qp.symmetries.Symmetries(
                rc=rc, lattice=self.lattice, ions=self.ions,
                **symmetries)
        elif isinstance(symmetries, qp.symmetries.Symmetries):
            self.symmetries = symmetries
        else:
            raise TypeError(
                "symmetries must be dict or qimpy.symmetries.Symmetries")
