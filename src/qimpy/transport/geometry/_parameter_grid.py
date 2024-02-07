from __future__ import annotations

import torch

from qimpy.mpi import ProcessGrid
from qimpy.transport.material import Material
from . import Geometry


class ParameterGrid(Geometry):
    """Geometry specification."""

    # stash: ResultStash  #: Saved results for collating into fewer checkpoints

    def __init__(
        self,
        *,
        material: Material,
        process_grid: ProcessGrid,
        # checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize parameter grid parameters.

        Parameters
        ----------

        """
        super().__init__(
            material=material,
            process_grid=process_grid,
        )
        self.dt_max = 0

    def rho_dot(
        self,
        rho_list_eval: list[torch.Tensor],
        t: float,
    ) -> list[torch.Tensor]:
        return NotImplemented
