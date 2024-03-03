from __future__ import annotations

from qimpy.mpi import ProcessGrid
from qimpy.transport.material import Material
from . import TensorList, Geometry


class ParameterGrid(Geometry):
    """Geometry specification."""

    # stash: ResultStash  #: Saved results for collating into fewer checkpoints

    def __init__(
        self,
        *,
        material: Material,
        process_grid: ProcessGrid,
    ) -> None:
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

    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        return NotImplemented

    @property
    def rho(self) -> TensorList:
        return NotImplemented

    @rho.setter
    def rho(self, rho_new: TensorList) -> None:
        raise NotImplementedError
