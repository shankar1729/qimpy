from qimpy import TreeNode, log
from qimpy.io import CheckpointPath, CheckpointContext
from .. import Wavefunction


class PlusU(TreeNode):
    """DFT+U correction."""

    U_values: dict[tuple[str, str], float]  #: map specie, orbital -> U value

    def __init__(
        self, *, checkpoint_in: CheckpointPath = CheckpointPath(), **U_values: dict
    ) -> None:
        """Initialize from components and/or dictionary of options.

        Parameters
        ----------
        U_values
            :yaml:`Dictionary of U values by species and orbital names.`
            For example, to add U to Cu d and O s and p, the yaml input would be:

            .. code-block:: yaml

                plus_U:
                  Cu d: 2.4 eV
                  O s: 0.1 eV
                  O p: 0.7 eV
        """
        super().__init__()
        self.U_values = {}
        for key, U in U_values:
            specie, orbital = key.split()
            # TODO: validate and map orbital codes, check against Ions
            log.info(f"  +U on {specie}: {U}")
            self.U_values[(specie, orbital)] = float(U)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        for (specie, orbital), U in self.U_values.items():
            attrs[f"{specie} {orbital}"] = U
        return list(attrs.keys())

    def __bool__(self) -> bool:
        return bool(self.U_values)

    def __call__(self, C: Wavefunction) -> Wavefunction:
        """TODO."""
        raise NotImplementedError
