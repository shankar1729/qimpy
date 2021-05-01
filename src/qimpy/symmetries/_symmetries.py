import qimpy as qp
import torch
from ._lattice import _get_lattice_point_group, _symmetrize_lattice
from ._ions import _get_space_group, _symmetrize_positions


class Symmetries:
    '''TODO: document class Symmetries'''

    symmetrize_lattice = _symmetrize_lattice
    symmetrize_positions = _symmetrize_positions

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

        # Lattice point group:
        lattice_sym = _get_lattice_point_group(lattice.Rbasis, tolerance)
        qp.log.info(
            'Found {:d} point-group symmetries of the Bravais lattice'.format(
                lattice_sym.shape[0]))

        # Space group:
        self.rot, self.trans, self.ion_map = _get_space_group(
            lattice_sym, lattice, ions, tolerance)
        self.n_sym = self.rot.shape[0]
        qp.log.info(
            'Found {:d} space-group symmetries with basis:'.format(self.n_sym))
        for i_sym in range(self.n_sym):
            sym_str = '- ['
            for row in range(3):
                sym_str += rc.fmt(self.rot[i_sym, row].to(int)) + ', '
            qp.log.info(sym_str + rc.fmt(self.trans[i_sym]) + ']')
        qp.log.debug('Ion map:\n' + rc.fmt(self.ion_map))

        # Enforce symmetries exactly on lattice:
        qp.log.info('Enforcing symmetries:')
        lattice.update(self.symmetrize_lattice(lattice.Rbasis))

        # Enforce symmetries exactly on ions:
        if ions.n_ions:
            positions_sym = self.symmetrize_positions(ions.positions)
            qp.log.info(
                'RMS change in fractional positions of ions: {:e}'.format(
                    ((positions_sym - ions.positions)**2).mean().sqrt()))
            ions.positions = positions_sym
