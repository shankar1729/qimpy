from typing import Optional, Sequence, Union

from qimpy import rc, log, TreeNode
from qimpy.rc import MPI
from qimpy.io import CheckpointPath, Checkpoint
from qimpy.mpi import ProcessGrid
from .geometry import Geometry
from .material import Material, AbInitio, FermiCircle


class Transport(TreeNode):
    material: Material
    geometry: Geometry

    def __init__(
        self,
        *,
        geometry: Union[Geometry, dict],
        ab_initio: Optional[Union[AbInitio, dict]] = None,
        fermi_circle: Optional[Union[FermiCircle, dict]] = None,
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
        geometry
            :yaml:`Geometry specification.`
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
        self.checkpoint_out = checkpoint if checkpoint_out is None else checkpoint_out

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
        self.add_child(
            "geometry",
            Geometry,
            geometry,
            checkpoint_in,
            material=self.material,
            process_grid=self.process_grid,
        )

    def run(self):
        pass
