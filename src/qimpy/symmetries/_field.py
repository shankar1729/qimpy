import qimpy as qp
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..grid import Grid, FieldH


class FieldSymmetrizer:
    """Space group symmetrization of reciprocal-space :class:`Field`'s."""
    grid: 'Grid'

    def __init__(self, grid: 'Grid') -> None:
        """Initialize symmetrization for fields on `grid`."""
        self.grid = grid
        qp.log.info('Will initialize symmetrization here.')

    def __call__(self, v: 'FieldH') -> None:
        """Symmetrize field `v` in-place."""
        grid = self.grid
        assert v.grid == grid
