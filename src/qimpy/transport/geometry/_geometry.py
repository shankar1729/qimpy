from __future__ import annotations
from typing import Optional
from abc import abstractmethod

import torch

from qimpy import TreeNode, MPI, rc
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import Material


class Geometry(TreeNode):
    """Geometry specification."""

    comm: MPI.Comm  #: Communicator for real-space split over patches
    material: Material  #: Corresponding material
    dt_max: float  #: Maximum stable time step

    def __init__(
        self,
        *,
        material: Material,
        process_grid: ProcessGrid,
    ):
        """Initialize geometry parameters, typically used from a derived class."""
        super().__init__()
        self.comm = process_grid.get_comm("r")
        self.material = material

    @abstractmethod
    def rho_dot(self, rho_list_eval: list[torch.Tensor], t: float
                ) -> list[torch.Tensor]:
        """Return list of drho/dt from PatchSet or ParameterGrid"""
