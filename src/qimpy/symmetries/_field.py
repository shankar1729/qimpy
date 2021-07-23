import qimpy as qp
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..grid import Grid, FieldH


class FieldSymmetrizer:
    """Space group symmetrization of reciprocal-space :class:`FieldH`'s."""
    grid: 'Grid'

    def __init__(self, grid: 'Grid') -> None:
        """Initialize symmetrization for fields on `grid`."""
        self.grid = grid
        # Get reciprocal mesh of all processes:
        # TODO

        # Find symmetry-reduced set:
        # TODO

        # Set up indices, phases of orbits of each point in reduced set:
        # TODO

        # Set up MPI split over orbits:
        # TODO

    def __call__(self, v: 'FieldH') -> None:
        """Symmetrize field `v` in-place."""
        grid = self.grid
        assert v.grid == grid
        # Collect data by orbits, transfering over MPI as needed:
        # TODO

        # Symmetrize in each orbit:
        # TODO

        # Set results back to original grid, transfering over MPI as needed:
        # TODO
