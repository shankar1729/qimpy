from .. import TreeNode
from ..rc import MPI
from typing import Optional, Sequence


class Transport(TreeNode):
    def __init__(
        self,
        *,
        checkpoint: Optional[str] = None,
        checkpoint_out: Optional[str] = None,
        comm: Optional[MPI.Comm] = None,
        process_grid_shape: Optional[Sequence[int]] = None,
    ):
        super().__init__()

    def run(self):
        pass
