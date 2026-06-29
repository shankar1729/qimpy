from __future__ import annotations
from typing import Optional
from abc import abstractmethod

from qimpy import TreeNode, MPI
from ..material import Material
from ._tensor_list import TensorList


class Geometry(TreeNode):
    """Abstract base for a transport geometry.

    A geometry owns the spatial discretization and the material, and exposes the
    interface the time integrator drives: the current state ``rho``, its time
    derivative ``rho_dot`` (spatial advection + the material's local dynamics),
    the maximum stable time step ``dt_max``, and ``update_stash`` to accumulate
    per-step observables for checkpointing.  The concrete implementation is
    :class:`FiniteVolume` (cell-centered finite volume on triangle / line meshes).
    """

    comm: MPI.Comm  #: Communicator for the real-space (cell) split
    material: Material  #: Corresponding material
    contacts: dict[str, Optional[dict]]  #: contact names -> material parameters
    dt_max: float  #: Maximum stable time step
    save_rho: bool  #: whether to write rho (not just observables) to checkpoint

    @abstractmethod
    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        """Return list of drho/dt corresponding to each rho at time ``t``."""

    @property
    @abstractmethod
    def rho(self) -> TensorList:
        """Current values of the density matrices / distributions."""

    @rho.setter
    @abstractmethod
    def rho(self, rho_new: TensorList) -> None:
        """Set current values of the density matrices / distributions."""

    @abstractmethod
    def update_stash(self, i_step: int, t: float) -> None:
        """Stash this step's observables for a later ``save_checkpoint`` call."""
