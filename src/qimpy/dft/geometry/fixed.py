from __future__ import annotations

from mpi4py import MPI

from qimpy.utils import CpPath
from qimpy.lattice import Lattice
from . import Relax


class Fixed(Relax):
    """Fixed geometry, i.e. only optimize electronic degrees of freedom."""

    def __init__(
        self, *, comm: MPI.Comm, lattice: Lattice, checkpoint_in: CpPath = CpPath()
    ) -> None:
        super().__init__(
            n_iterations=0, comm=comm, lattice=lattice, checkpoint_in=checkpoint_in
        )
