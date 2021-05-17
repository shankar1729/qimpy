import qimpy as qp
import numpy as np
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from .utils import RunConfig
    from .lattice import Lattice
    from .ions import Ions
    from .symmetries import Symmetries
    from .electrons import Electrons
    from .grid import Grid


class System:
    def __init__(self, *, rc: 'RunConfig',
                 lattice: Union['Lattice', dict],
                 ions: Union['Ions', dict, None] = None,
                 symmetries: Union['Symmetries', dict, None] = None,
                 electrons: Union['Electrons', dict, None] = None,
                 grid: Union['Grid', dict, None] = None):
        '''Compose a System to calculate from its pieces, each of which
        could be provided as an object or a dictionary of parameters
        suitable for initializing that object'''
        self.rc: 'RunConfig' = rc  #: current run configuration
        self.lattice: 'Lattice' = qp.construct(
            qp.lattice.Lattice, lattice, 'lattice',
            rc=rc)  #: lattice vectors / unit cell definition
        self.ions: 'Ions' = qp.construct(
            qp.ions.Ions, ions, 'ions',
            rc=rc)  #: ionic positions and pseudopotentials
        self.symmetries: 'Symmetries' = qp.construct(
            qp.symmetries.Symmetries, symmetries, 'symmetries',
            rc=rc, lattice=self.lattice,
            ions=self.ions)  #: point and space group symmetries
        self.electrons: 'Electrons' = qp.construct(
            qp.electrons.Electrons, electrons, 'electrons',
            rc=rc, lattice=self.lattice, ions=self.ions,
            symmetries=self.symmetries
        )  #: electronic wavefunctions and related quantities

        qp.log.info('\n--- Initializing Charge-Density Grid ---')
        self.grid: 'Grid' = qp.construct(
            qp.grid.Grid, grid, 'grid',
            rc=rc, lattice=self.lattice, symmetries=self.symmetries,
            comm=rc.comm_kb,  # parallelized on intra-replica comm
            ke_cutoff_wavefunction=self.electrons.basis.ke_cutoff
        )  #: charge-density grid

        qp.log.info(f'\nInitialization completed at t[s]: {rc.clock():.2f}\n')

    def run(self):
        'Run any actions specified in the input'
        # TODO: systematize selection of what actions to perform
        self.electrons.diagonalize()
        self.electrons.output()
