from __future__ import annotations

import numpy as np

from qimpy.io import CheckpointPath
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import Material
from . import TensorList, Geometry, QuadSet


class ParameterGrid(Geometry):
    """Geometry specification."""

    def __init__(
        self,
        *,
        material: Material,
        shape: tuple[int, int],
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize parameter grid parameters.

        Parameters
        ----------
        shape
            :yaml:`Dimensions of parameter grid (always 2D).`
        """
        assert len(shape) == 2
        # Create fake geometry for parameter grid
        quad_set = QuadSet(
            vertices=_GRID_VERTICES * np.array(shape),
            quads=_GRID_QUADS,
            adjacency=np.full((1, 4, 2), -1),
            displacements=np.zeros((1, 4, 2)),
            grid_size=np.array([shape]),
            contacts=np.zeros((0, 3)),
            apertures=np.zeros((0, 3)),
            aperture_names=[],
            has_apertures=np.full((1, 4), False),
        )
        super().__init__(
            material=material,
            process_grid=process_grid,
            quad_set=quad_set,
            grid_spacing=1,
            grid_size_max=0,
            contacts={},
        )
        self.dt_max = 0  # disable transport dt limit

    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        return TensorList(self.material.rho_dot(rho_i, t) for rho_i in rho)


_GRID_VERTICES = (1.0 / 3) * np.array(
    [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 3],
        [2, 3],
        [3, 3],
        [3, 2],
        [3, 1],
        [3, 0],
        [2, 0],
        [1, 0],
    ]
)
_GRID_QUADS = np.array([[[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [9, 10, 11, 0]]])
