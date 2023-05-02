from .. import TreeNode
from ..utils import CpPath
from typing import Sequence


class Geometry(TreeNode):
    """Geometry specification."""

    def __init__(
        self,
        *,
        vertices: Sequence[Sequence[float]],
        edges: Sequence[Sequence[int]],
        checkpoint_in: CpPath = CpPath(),
    ):
        """
        Initialize geometry parameters.

        Parameters
        ----------
        vertices
            :yaml:`Vertex coordinates.`
        edges
            :yaml:`Indices of vertices in edges.`
        """
        super().__init__()
