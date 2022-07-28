from __future__ import annotations
import qimpy as qp


class Relax(qp.TreeNode):
    """Relax geometry of ions and/or lattice.
    Whether lattice changes is controlled by `lattice.movable`.
    """

    def __init__(self, *, checkpoint_in: qp.utils.CpPath = qp.utils.CpPath()) -> None:
        super().__init__()

    def run(self, system: qp.System) -> None:
        qp.log.info("\n--- Geometry relaxation ---\n")
        raise NotImplementedError
