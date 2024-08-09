from typing import Optional, Sequence, Union

from qimpy import rc, log, TreeNode
from qimpy.rc import MPI
from qimpy.io import CheckpointPath, Checkpoint, CheckpointContext
from qimpy.mpi import ProcessGrid
from qimpy.profiler import stopwatch
from .geometry import Geometry, PatchSet, ParameterGrid
from .material import Material, FermiCircle
from .material.ab_initio import AbInitio
from . import TimeEvolution


class Transport(TreeNode):
    material: Material
    geometry: Geometry
    time_evolution: TimeEvolution

    def __init__(
        self,
        *,
        ab_initio: Optional[Union[AbInitio, dict]] = None,
        fermi_circle: Optional[Union[FermiCircle, dict]] = None,
        patch_set: Optional[Union[PatchSet, dict]] = None,
        parameter_grid: Optional[Union[ParameterGrid, dict]] = None,
        time_evolution: Optional[Union[TimeEvolution, dict]] = None,
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
        ab_initio
            :yaml:`Ab-initio material.`
            Exactly one supported material type must be specified.
        fermi_circle
            :yaml:`Fermi-circle material for graphene/2DEG.`
            Exactly one supported material type must be specified.
        patch_set
            :yaml:`Geometry consisting of bicubic patches.`
            Exactly one supported geometry type must be specified.
        parameter_grid
            :yaml:`Virtual geometry of disconnected points for batched dynamics.`
            Exactly one supported geometry type must be specified.
        checkpoint
            :yaml:`Checkpoint file to read at start-up.`
        checkpoint_out
            :yaml:`Checkpoint file pattern to write at regular intervals.`
            The pattern should contain an integer format eg. '{:04d}'
            that can be replaced with the frame number.
        comm
            Overall communicator for system. Defaults to `qimpy.rc.comm` if unspecified.
        process_grid_shape
            Parallelization dimensions over replicas, k-points and bands/basis, used
            to initialize a `qimpy.mpi.ProcessGrid`. Dimensions that are -1 will be
            auto-determined based on number of tasks available to split along them.
            Default: all process grid dimensions are auto-determined."""
        super().__init__()
        self.process_grid = ProcessGrid(
            comm if comm else rc.comm, "rk", process_grid_shape
        )
        self.process_grid.provide_n_tasks("k", 1)  # prefer r-split if unspecified
        # Set in and out checkpoints:
        checkpoint_in = CheckpointPath()
        if checkpoint is not None:
            try:
                checkpoint_in = CheckpointPath(Checkpoint(checkpoint))
            except OSError:  # Raised by h5py when file not readable
                log.info(f"Cannot load checkpoint file '{checkpoint}'")
        self.checkpoint_out = checkpoint_out

        self.add_child_one_of(
            "material",
            checkpoint_in,
            TreeNode.ChildOptions(
                "ab-initio", AbInitio, ab_initio, process_grid=self.process_grid
            ),
            TreeNode.ChildOptions(
                "fermi-circle",
                FermiCircle,
                fermi_circle,
                process_grid=self.process_grid,
            ),
            have_default=False,
        )
        self.add_child_one_of(
            "geometry",
            checkpoint_in,
            TreeNode.ChildOptions(
                "patch_set",
                PatchSet,
                patch_set,
                material=self.material,
                process_grid=self.process_grid,
            ),
            TreeNode.ChildOptions(
                "parameter_grid",
                ParameterGrid,
                parameter_grid,
                material=self.material,
                process_grid=self.process_grid,
            ),
            have_default=False,
        )
        self.add_child(
            "time_evolution",
            TimeEvolution,
            time_evolution,
            checkpoint_in,
            geometry=self.geometry,
        )

    def run(self):
        self.time_evolution.run(self)

    @stopwatch
    def save(self, i_step: int) -> None:
        if self.checkpoint_out:
            filename = self.checkpoint_out.format(i_step)
            with Checkpoint(filename, writable=True, rotate=False) as cp:
                self.save_checkpoint(CheckpointPath(cp), CheckpointContext(""))
