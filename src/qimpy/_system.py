import qimpy as qp
import numpy as np
import torch
from typing import Union, Optional, Any, Sequence
from mpi4py import MPI


class System(qp.TreeNode):
    """Overall system to calculate within QimPy"""

    lattice: qp.lattice.Lattice  #: Lattice vectors / unit cell definition
    ions: qp.ions.Ions  #: Ionic positions and pseudopotentials
    symmetries: qp.symmetries.Symmetries  #: Point and space group symmetries
    electrons: qp.electrons.Electrons  #: Electronic sub-system
    grid: qp.grid.Grid  #: Charge-density grid
    coulomb: qp.grid.Coulomb  #: Coulomb interactions on charge-density grid
    geometry: qp.geometry.Geometry  #: Geometry actions, e.g., relaxation / dynamics
    export: qp.export.Export  #: Exporters to interface with other codes
    energy: qp.Energy  #: Energy components
    checkpoint_in: qp.utils.CpPath  #: Input checkpoint
    checkpoint_out: Optional[str]  #: Filename for output checkpoint
    process_grid: qp.utils.ProcessGrid  #: Process grid for parallelization

    def __init__(
        self,
        *,
        lattice: Union[qp.lattice.Lattice, dict],
        ions: Union[qp.ions.Ions, dict, None] = None,
        symmetries: Union[qp.symmetries.Symmetries, dict, None] = None,
        electrons: Union[qp.electrons.Electrons, dict, None] = None,
        grid: Union[qp.grid.Grid, dict, None] = None,
        geometry: Union[qp.geometry.Geometry, dict, str, None] = None,
        export: Union[qp.export.Export, dict, None] = None,
        checkpoint: Optional[str] = None,
        checkpoint_out: Optional[str] = None,
        comm: Optional[MPI.Comm] = None,
        process_grid_shape: Optional[Sequence[int]] = None,
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
            to initialize a `qimpy.utils.ProcessGrid`. Dimensions that are -1 will be
            auto-determined based on number of tasks available to split along them.
            Default: all process grid dimensions are auto-determined."""
        super().__init__()
        self.process_grid = qp.utils.ProcessGrid(
            comm if comm else qp.rc.comm, "rkb", process_grid_shape
        )
        # Set in and out checkpoints:
        checkpoint_in = qp.utils.CpPath()
        if checkpoint is not None:
            try:
                checkpoint_in = qp.utils.CpPath(qp.utils.Checkpoint(checkpoint))
            except OSError:  # Raised by h5py when file not readable
                qp.log.info(f"Cannot load checkpoint file '{checkpoint}'")
        self.checkpoint_out = checkpoint if checkpoint_out is None else checkpoint_out

        # Determine any global axes that break symmetries:
        axes: dict[str, np.ndarray] = {}
        _add_axis(axes, "magnetization", electrons, ["fillings", "M"])
        _add_axis(axes, "magnetic field", electrons, ["fillings", "B"])
        # TODO: similarly account for applied electric fields

        self.add_child("lattice", qp.lattice.Lattice, lattice, checkpoint_in)
        self.add_child(
            "ions",
            qp.ions.Ions,
            ions,
            checkpoint_in,
            lattice=self.lattice,
        )
        self.process_grid.provide_n_tasks("r", self.ions.n_replicas)
        self.add_child(
            "symmetries",
            qp.symmetries.Symmetries,
            symmetries,
            checkpoint_in,
            lattice=self.lattice,
            ions=self.ions,
            axes=axes,
        )
        self.add_child(
            "electrons",
            qp.electrons.Electrons,
            electrons,
            checkpoint_in,
            process_grid=self.process_grid,
            lattice=self.lattice,
            ions=self.ions,
            symmetries=self.symmetries,
        )

        qp.log.info("\n--- Initializing Charge-Density Grid ---")
        self.add_child(
            "grid",
            qp.grid.Grid,
            grid,
            checkpoint_in,
            lattice=self.lattice,
            symmetries=self.symmetries,
            comm=self.electrons.comm,  # Parallel
            ke_cutoff_wavefunction=self.electrons.basis.ke_cutoff,
        )
        self.coulomb = qp.grid.Coulomb(self.grid, self.ions.n_ions)

        self.add_child(
            "geometry",
            qp.geometry.Geometry,
            geometry,
            checkpoint_in,
            comm=self.electrons.comm,
            lattice=self.lattice,
        )

        self.add_child("export", qp.export.Export, export, checkpoint_in, system=self)

        # Initialize ionic potentials and energies at initial configuration:
        self.energy = qp.Energy()
        self.ions.update(self)

        qp.log.info(f"\nInitialization completed at t[s]: {qp.rc.clock():.2f}\n")

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
