from typing import Sequence

from qimpy import TreeNode
from qimpy.io import CheckpointPath


class Material(TreeNode):
    """Material specification."""

    def __init__(
        self,
        *,
        fname: str,
        rotation: Sequence[Sequence[float]] = ((1, 0, 0), (0, 1, 0, (0, 0, 1))),
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize material parameters.

        Parameters
        ----------
        fname
            :yaml:`File name to load materials data from.`
        rotation
            :yaml:`3 x 3 rotation matrix from material to simulation frame.`
        """
        super().__init__()
