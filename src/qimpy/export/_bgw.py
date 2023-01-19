from __future__ import annotations
import qimpy as qp


class BGW(qp.TreeNode):
    """Fixed geometry, i.e. only optimize electronic degrees of freedom."""

    filename: str  #: BGW-format HDF5 file to output

    def __init__(
        self,
        *,
        system: qp.System,
        filename: str,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
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

    def export(self, system: qp.System) -> None:
        """Export BGW HDF5 file."""
        qp.log.info("Will do BGW export here.")
        raise NotImplementedError
