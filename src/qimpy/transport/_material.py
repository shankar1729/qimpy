from .. import TreeNode
from ..utils import CpPath


class Material(TreeNode):
    """Material specification."""

    def __init__(
        self,
        *,
        fname: str,
        checkpoint_in: CpPath = CpPath(),
    ):
        """
        Initialize material parameters.

        Parameters
        ----------
        fname
            :yaml:`File name to load materials data from.`
        """
        super().__init__()
