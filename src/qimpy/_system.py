from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from typing import Union, Optional, Dict, List, Any


class System(qp.Constructable):
    """Overall system to calculate within QimPy"""
    __slots__ = ('lattice', 'ions', 'symmetries', 'electrons',
                 'grid', 'coulomb', 'energy', 'checkpoint_out')
    lattice: qp.lattice.Lattice  #: Lattice vectors / unit cell definition
    ions: qp.ions.Ions  #: Ionic positions and pseudopotentials
    symmetries: qp.symmetries.Symmetries  #: Point and space group symmetries
    electrons: qp.electrons.Electrons  #: Electronic sub-system
    grid: qp.grid.Grid  #: Charge-density grid
    coulomb: qp.grid.Coulomb  #: Coulomb interactions on charge-density grid
    energy: qp.Energy  #: Energy components
    checkpoint_out: Optional[str]  #: Filename for output checkpoint

    def __init__(self, *, rc: qp.utils.RunConfig,
                 lattice: Union[qp.lattice.Lattice, dict],
                 ions: Union[qp.ions.Ions, dict, None] = None,
                 symmetries: Union[qp.symmetries.Symmetries, dict,
                                   None] = None,
                 electrons: Union[qp.electrons.Electrons, dict, None] = None,
                 grid: Union[qp.grid.Grid, dict, None] = None,
                 checkpoint: Optional[str] = None,
                 checkpoint_out: Optional[str] = None):
        """Compose a System to calculate from its pieces. Each piece
        could be provided as an object or a dictionary of parameters
        suitable for initializing that object.

        Parameters
        ----------
        lattice
            :yaml:`Lattice vectors / unit cell definition.`
        ions
            :yaml:`Ionic positions and pseudopotentials.`
        symmetries
            :yaml:`Point and space group symmetries.`
        electrons
            :yaml:`Electronic sub-system.`
        grid
            :yaml:`Charge-density grid.`
        checkpoint
            :yaml:`Checkpoint file to read at start-up.`
        checkpoint_out
            :yaml:`Checkpoint file to write.`
            Defaults to `checkpoint` if unspecified.
        """

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

        # Determine any global axes that break symmetries:
        axes: Dict[str, np.ndarray] = {}
        _add_axis(axes, 'magnetization', electrons, ['fillings', 'M'])
        _add_axis(axes, 'magnetic field',  electrons, ['fillings', 'B'])
        # TODO: similarly account for applied electric fields

        super().__init__(qp.ConstructOptions(rc=rc,
                                             checkpoint_in=checkpoint_in))
        self.construct('lattice', qp.lattice.Lattice, lattice)
        self.construct('ions', qp.ions.Ions, ions)
        self.construct('symmetries', qp.symmetries.Symmetries, symmetries,
                       lattice=self.lattice, ions=self.ions, axes=axes)
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
        self.electrons.initialize_wavefunctions(self)  # LCAO / randomize

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


def _add_axis(axes: Dict[str, np.ndarray], name: str,
              obj: Any, path: List[str]) -> None:
    """Add `name`d axis that should reduce symmetries to `axes`. Check from
    `path` within object `obj`, which could be a dictionary of parameters."""
    if obj is None:
        return
    if len(path):
        # Recur down to the appropriate sub-object:
        if isinstance(obj, dict):
            if path[0] in obj:
                _add_axis(axes, name, obj[path[0]], path[1:])
        else:
            if hasattr(obj, path[0]):
                _add_axis(axes, name, getattr(obj, path[0]), path[1:])
    else:
        # Check if object itself is suitable as an axis:
        axis = np.array(obj, dtype=float).flatten()
        if len(axis) == 3:
            axes[name] = axis
