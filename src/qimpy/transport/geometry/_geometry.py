from __future__ import annotations
from abc import abstractmethod

from qimpy import TreeNode, MPI
from qimpy.mpi import ProcessGrid
from ..material import Material
from . import TensorList


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
    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        """Return list of drho/dt from PatchSet or ParameterGrid"""

    @property
    @abstractmethod
    def rho(self) -> TensorList:
        """Get current values of density matrices."""

    @rho.setter
    @abstractmethod
    def rho(self, rho_new: TensorList) -> None:
        """Set current values of density matrices."""
