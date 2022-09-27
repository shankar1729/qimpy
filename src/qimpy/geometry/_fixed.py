from __future__ import annotations
import qimpy as qp
from qimpy.rc import MPI
from ._relax import Relax


class Fixed(Relax):
    """Fixed geometry, i.e. only optimize electronic degrees of freedom."""

    def __init__(
        self,
        *,
        comm: MPI.Comm,
        lattice: qp.lattice.Lattice,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath()
    ) -> None:
        super().__init__(
            n_iterations=0, comm=comm, lattice=lattice, checkpoint_in=checkpoint_in
        )
