import qimpy as qp
from ._lattice import _get_lattice_point_group
from ._ions import _get_space_group


class Symmetries:
    '''TODO: document class Symmetries'''

    def __init__(self, *, rc, lattice, ions, tolerance=1e-6):
        '''
        Parameters
        ----------
        rc : qimpy.utils.RunConfig
            Current run configuration.
        lattice : qimpy.lattice.Lattice
            Bravais lattice / unit cell (determines initial point group).
        ions : qimpy.ions.Ions
            Ion specification that, with lattice, determines space group.
        '''
        self.rc = rc
        qp.log.info('\n--- Initializing Symmetries ---')
        lattice_sym = _get_lattice_point_group(lattice.Rbasis, tolerance)
        qp.log.info(
            'Found {:d} point group symmetries of Bravais lattice'.format(
                lattice_sym.shape[0]))
