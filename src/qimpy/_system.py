import qimpy as qp
import numpy as np
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from ._energy import Energy
    from .utils import RunConfig
    from .lattice import Lattice
    from .ions import Ions
    from .symmetries import Symmetries
    from .electrons import Electrons
    from .grid import Grid, Coulomb
    from .utils import HDF5_io


class System:
    """Overall system to calculate within QimPy"""
    __slots__ = ('rc', 'chk', 'lattice', 'ions', 'symmetries', 'electrons',
                 'grid', 'coulomb', 'energy')
    rc: 'RunConfig'  #: Current run configuration
    chk: 'HDF5_io'  #: Checkpoint file handler
    lattice: 'Lattice'  #: Lattice vectors / unit cell definition
    ions: 'Ions'  #: Ionic positions and pseudopotentials
    symmetries: 'Symmetries'  #: Point and space group symmetries
    electrons: 'Electrons'  #: Electronic sub-system
    grid: 'Grid'  #: Charge-density grid
    coulomb: 'Coulomb'  #: Coulomb interactions on charge-density grid
    energy: 'Energy'  #: Energy components

    def __init__(self, *, rc: 'RunConfig',
                 chk: 'HDF5_io',
                 lattice: Union['Lattice', dict],
                 ions: Union['Ions', dict, None] = None,
                 symmetries: Union['Symmetries', dict, None] = None,
                 electrons: Union['Electrons', dict, None] = None,
                 grid: Union['Grid', dict, None] = None):
        """Compose a System to calculate from its pieces. Each piece
        could be provided as an object or a dictionary of parameters
        suitable for initializing that object"""
        self.rc = rc
        self.chk = chk
        self.lattice = qp.construct(qp.lattice.Lattice, lattice, 'lattice',
                                    rc=rc)
        self.ions = qp.construct(qp.ions.Ions, ions, 'ions', rc=rc)
        self.symmetries = qp.construct(
            qp.symmetries.Symmetries, symmetries, 'symmetries',
            rc=rc, lattice=self.lattice, ions=self.ions)
        self.electrons = qp.construct(
            qp.electrons.Electrons, electrons, 'electrons',
            rc=rc, lattice=self.lattice, ions=self.ions,
            symmetries=self.symmetries)

        qp.log.info('\n--- Initializing Charge-Density Grid ---')
        self.grid = qp.construct(
            qp.grid.Grid, grid, 'grid',
            rc=rc, lattice=self.lattice, symmetries=self.symmetries,
            comm=rc.comm_kb,  # parallelized on intra-replica comm
            ke_cutoff_wavefunction=self.electrons.basis.ke_cutoff)
        self.coulomb = qp.grid.Coulomb(self.grid, self.ions.n_ions)

        # Initialize ionic potentials and energies at initial configuration:
        self.energy = qp.Energy()
        self.ions.update(self)

        qp.log.info(f'\nInitialization completed at t[s]: {rc.clock():.2f}\n')

    def run(self):
        """Run any actions specified in the input."""
        # TODO: systematize selection of what actions to perform
        self.electrons.fillings.update(self)
        n_prev = None
        for i_scf in range(10):
            self.electrons.update_density(self)
            if n_prev is not None:
                self.electrons.n = 0.5 * (n_prev + self.electrons.n)
            n_prev = self.electrons.n
            self.electrons.update_potential(self)
            self.electrons.diagonalize(n_iterations=2)
            self.energy['KE'] = self.rc.comm_k.allreduce(
                (self.electrons.C.band_ke()[:, :, :self.electrons.f.shape[2]]
                 * self.electrons.basis.w_sk
                 * self.electrons.f).sum().item(), qp.MPI.SUM)
            self.electrons.fillings.update(self)
            qp.log.info(f'SCF:  Cycle: {i_scf}  {self.energy.name()}:'
                        f' {float(self.energy):.12f}')
        self.electrons.output()
        qp.log.info(f'\nEnergy components:\n{repr(self.energy)}')
