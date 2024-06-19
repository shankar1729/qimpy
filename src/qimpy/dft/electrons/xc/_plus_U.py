from qimpy import TreeNode, log
from qimpy.io import CheckpointPath
from .. import Wavefunction


class PlusU(TreeNode):
    """DFT+U correction."""

    U_values: dict[str, dict[str, float]]

    def __init__(
        self, *, checkpoint_in: CheckpointPath = CheckpointPath(), **U_values
    ) -> None:
        """Initialize from components and/or dictionary of options.

        Parameters
        ----------
        U_values
            Dictionary of U values by species and orbital names.
        """
        super().__init__()
        self.U_values = U_values
        for specie, Us in U_values.items():
            log.info(f"  +U on {specie}: {Us}")

    def __bool__(self) -> bool:
        return bool(self.U_values)

    def __call__(self, C: Wavefunction) -> Wavefunction:
        """TODO."""
        raise NotImplementedError
