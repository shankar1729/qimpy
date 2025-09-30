from typing import Union, Optional, Any, Sequence

import numpy as np
import torch

from qimpy import rc, log, TreeNode, Energy, MPI
from qimpy.io import Checkpoint, CheckpointPath
from qimpy.mpi import ProcessGrid
from qimpy.lattice import Lattice
from qimpy.symmetries import Symmetries
from qimpy.grid import Grid
from qimpy.grid.coulomb import Coulomb
from .ions import Ions
from .electrons import Electrons
from .geometry import Geometry
from .export import Export
from qimpy.dft.fluid import LinearPCMFluidModel


class System(TreeNode):
    """Overall system to calculate within QimPy"""
    
    lattice: Lattice  #: Lattice vectors / unit cell definition
    ions: Ions  #: Ionic positions and pseudopotentials
    symmetries: Symmetries  #: Point and space group symmetries
    electrons: Electrons  #: Electronic sub-system
    grid: Grid  #: Charge-density grid
    coulomb: Coulomb  #: Coulomb interactions on charge-density grid
    geometry: Geometry  #: Geometry actions, e.g., relaxation / dynamics
    export: Export  #: Exporters to interface with other codes
    energy: Energy  #: Energy components
    checkpoint_in: CheckpointPath  #: Input checkpoint
    checkpoint_out: Optional[str]  #: Filename for output checkpoint
    process_grid: ProcessGrid  #: Process grid for parallelization
    fluid: LinearPCMFluidModel  #: fluid model for implicit solvent

    def __init__(
        self,
        *,
        lattice: Union[Lattice, dict, None] = None,
        ions: Union[Ions, dict, None] = None,
        symmetries: Union[Symmetries, dict, None] = None,
        electrons: Union[Electrons, dict, None] = None,
        grid: Union[Grid, dict, None] = None,
        coulomb: Union[Coulomb, dict, None] = None,
        geometry: Union[Geometry, dict, str, None] = None,
        export: Union[Export, dict, None] = None,
        checkpoint: Optional[str] = None,
        checkpoint_out: Optional[str] = None,
        comm: Optional[MPI.Comm] = None,
        process_grid_shape: Optional[Sequence[int]] = None,
        fluid: Union[LinearPCMFluidModel, dict, None] = None
    ):
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
        coulomb
            :yaml:`Coulomb interactions.`
        geometry
            :yaml:`Geometry actions such as relaxation and dynamics.`
            Specify name of geometry action eg. 'relax' if using default options for
            that action, and if not, specify an explicit  dictionary of parameters.
        export
            :yaml:`Export data for other codes.`
        checkpoint
            :yaml:`Checkpoint file to read at start-up.`
        checkpoint_out
            :yaml:`Checkpoint file to write.`
            Defaults to `checkpoint` if unspecified.
        comm
            Overall communicator for system. Defaults to `qimpy.rc.comm` if unspecified.
        process_grid_shape
            Parallelization dimensions over replicas, k-points and bands/basis, used
            to initialize a `qimpy.mpi.ProcessGrid`. Dimensions that are -1 will be
            auto-determined based on number of tasks available to split along them.
            Default: all process grid dimensions are auto-determined."""
        super().__init__()
        self.process_grid = ProcessGrid(
            comm if comm else rc.comm, "rkb", process_grid_shape
        )
        # Set in and out checkpoints:
        checkpoint_in = CheckpointPath()
        if checkpoint is not None:
            try:
                checkpoint_in = CheckpointPath(Checkpoint(checkpoint))
            except OSError:  # Raised by h5py when file not readable
                log.info(f"Cannot load checkpoint file '{checkpoint}'")
        self.checkpoint_out = checkpoint if checkpoint_out is None else checkpoint_out

        # Determine any global axes that break symmetries:
        axes: dict[str, np.ndarray] = {}
        _add_axis(axes, "magnetization", electrons, ["fillings", "M"])
        _add_axis(axes, "magnetic field", electrons, ["fillings", "B"])
        # TODO: similarly account for applied electric fields

        self.add_child("lattice", Lattice, lattice, checkpoint_in)
        self.add_child("ions", Ions, ions, checkpoint_in, lattice=self.lattice)
        self.process_grid.provide_n_tasks("r", self.ions.n_replicas)
        self.add_child(
            "symmetries",
            Symmetries,
            symmetries,
            checkpoint_in,
            lattice=self.lattice,
            labeled_positions=self.ions.labeled_positions,
            axes=axes,
        )
        self.add_child(
            "electrons",
            Electrons,
            electrons,
            checkpoint_in,
            process_grid=self.process_grid,
            lattice=self.lattice,
            ions=self.ions,
            symmetries=self.symmetries,
        )

        log.info("\n--- Initializing Charge-Density Grid ---")
        self.add_child(
            "grid",
            Grid,
            grid,
            checkpoint_in,
            lattice=self.lattice,
            symmetries=self.symmetries,
            comm=self.electrons.comm,  # Parallel
            ke_cutoff_wavefunction=self.electrons.basis.ke_cutoff,
        )

        self.add_child(
            "coulomb",
            Coulomb,
            coulomb,
            checkpoint_in,
            grid=self.grid,
            n_ions=self.ions.n_ions,
        )

        self.add_child(
            "geometry",
            Geometry,
            geometry,
            checkpoint_in,
            comm=self.electrons.comm,
            lattice=self.lattice,
        )
        self.add_child(
            "fluid",
            LinearPCMFluidModel,
            fluid,
            checkpoint_in,
            grid=self.grid,
            coulomb=self.coulomb,
            ions=self.ions,
        )

        self.add_child("export", Export, export, checkpoint_in, system=self)

        # Initialize ionic potentials and energies at initial configuration:
        self.energy = Energy()
        self.ions.update(self)

        log.info(f"\nInitialization completed at t[s]: {rc.clock():.2f}\n")

    def geometry_grad(self) -> None:
        """Update geometric gradients i.e. forces and optionally, stresses."""
        # Initialize gradients:
        self.ions.positions.requires_grad_(True)
        self.ions.positions.grad = torch.zeros_like(self.ions.positions)
        if self.lattice.compute_stress:
            self.lattice.requires_grad_(True, clear=True)

        # Compute gradients:
        self.ions.accumulate_geometry_grad(self)  # includes ionic and electronic pieces

        # Disable gradient computation flags:
        self.ions.positions.requires_grad_(False)
        self.lattice.requires_grad_(False)

    def run(self) -> None:
        """Run any actions specified in the input."""
        self.geometry.run(self)
        self.export(self)


def _add_axis(
    axes: dict[str, np.ndarray], name: str, obj: Any, path: list[str]
) -> None:
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
