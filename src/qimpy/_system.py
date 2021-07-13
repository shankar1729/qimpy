import qimpy as qp
import numpy as np
from typing import Union, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ._energy import Energy
    from .utils import RunConfig
    from .lattice import Lattice
    from .ions import Ions
    from .symmetries import Symmetries
    from .electrons import Electrons
    from .grid import Grid, Coulomb
    from .utils import Checkpoint


class System(qp.Constructable):
    """Overall system to calculate within QimPy"""
    __slots__ = ('lattice', 'ions', 'symmetries', 'electrons',
                 'grid', 'coulomb', 'energy', 'checkpoint_out')
    lattice: 'Lattice'  #: Lattice vectors / unit cell definition
    ions: 'Ions'  #: Ionic positions and pseudopotentials
    symmetries: 'Symmetries'  #: Point and space group symmetries
    electrons: 'Electrons'  #: Electronic sub-system
    grid: 'Grid'  #: Charge-density grid
    coulomb: 'Coulomb'  #: Coulomb interactions on charge-density grid
    energy: 'Energy'  #: Energy components
    checkpoint_out: Optional[str]  #: Filename for output checkpoint

    def __init__(self, *, rc: 'RunConfig',
                 lattice: Union['Lattice', dict],
                 ions: Union['Ions', dict, None] = None,
                 symmetries: Union['Symmetries', dict, None] = None,
                 electrons: Union['Electrons', dict, None] = None,
                 grid: Union['Grid', dict, None] = None,
                 checkpoint: Optional[str] = None,
                 checkpoint_out: Optional[str] = None):
        """Compose a System to calculate from its pieces. Each piece
        could be provided as an object or a dictionary of parameters
        suitable for initializing that object"""

        # Set in and out checkpoints:
        try:
            checkpoint_in = (None if (checkpoint is None)
                             else qp.utils.Checkpoint(checkpoint,
                                                      rc=rc, mode='r'))
        except OSError:  # Raised by h5py when file not readable
            qp.log.info(f"Cannot load checkpoint file '{checkpoint}'")
            checkpoint_in = None
        self.checkpoint_out = (checkpoint if checkpoint_out is None
                               else checkpoint_out)

        super().__init__(qp.ConstructOptions(rc=rc,
                                             checkpoint_in=checkpoint_in))
        self.construct('lattice', qp.lattice.Lattice, lattice)
        self.construct('ions', qp.ions.Ions, ions)
        self.construct('symmetries', qp.symmetries.Symmetries, symmetries,
                       lattice=self.lattice, ions=self.ions)
        self.construct('electrons', qp.electrons.Electrons, electrons,
                       lattice=self.lattice, ions=self.ions,
                       symmetries=self.symmetries)

        qp.log.info('\n--- Initializing Charge-Density Grid ---')
        self.construct('grid', qp.grid.Grid, grid, lattice=self.lattice,
                       symmetries=self.symmetries, comm=rc.comm_kb,  # Parallel
                       ke_cutoff_wavefunction=self.electrons.basis.ke_cutoff)
        self.coulomb = qp.grid.Coulomb(self.grid, self.ions.n_ions)

        # Initialize ionic potentials and energies at initial configuration:
        self.energy = qp.Energy()
        self.ions.update(self)

        qp.log.info(f'\nInitialization completed at t[s]: {rc.clock():.2f}\n')

    def run(self) -> None:
        """Run any actions specified in the input."""
        # TODO: systematize selection of what actions to perform
        self.electrons.scf.update(self)
        self.electrons.scf.optimize()
        self.electrons.output()
        qp.log.info(f'\nEnergy components:\n{repr(self.energy)}')

        qp.log.info('')
        if self.checkpoint_out:
            self.save_checkpoint(qp.utils.Checkpoint(self.checkpoint_out,
                                                     rc=self.rc, mode='w'))
