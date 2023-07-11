from __future__ import annotations

from qimpy import log, TreeNode, dft
from qimpy.utils import CpPath


class BGW(TreeNode):
    """Fixed geometry, i.e. only optimize electronic degrees of freedom."""

    filename: str  #: BGW-format HDF5 file to output

    def __init__(
        self,
        *,
        system: dft.System,
        filename: str,
        checkpoint_in: CpPath = CpPath(),
    ) -> None:
        """Export data for BerkeleyGW.

        Parameters
        ----------
        system
            Overall electronic DFT system to export data for.
        filename
            :yaml:`Filename for BerkeleyGW output.`
        """
        super().__init__()
        self.filename = filename

    def export(self, system: dft.System) -> None:
        """Export BGW HDF5 file."""
        log.info("Will do BGW export here.")
        raise NotImplementedError
