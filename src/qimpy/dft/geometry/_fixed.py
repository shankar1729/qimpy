from __future__ import annotations

from qimpy import MPI, dft
from qimpy.io import CheckpointPath, CheckpointContext, Checkpoint
from qimpy.lattice import Lattice
from . import Relax


class Fixed(Relax):
    """Fixed geometry, i.e. only optimize electronic degrees of freedom."""

    def __init__(
        self,
        *,
        comm: MPI.Comm,
        lattice: Lattice,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        super().__init__(
            n_iterations=0,
            save_history=False,
            comm=comm,
            lattice=lattice,
            checkpoint_in=checkpoint_in,
        )

    def run(self, system: dft.System) -> None:
        if system.electrons.fixed_H:
            # Bypass stepper and force calculations for non-SCF calculations:
            system.electrons.run(system)
            if system.checkpoint_out:
                with Checkpoint(system.checkpoint_out, writable=True) as cp:
                    system.save_checkpoint(CheckpointPath(cp), CheckpointContext("end"))
        else:
            Relax.run(self, system)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        pass  # just need to bypass setting incompatible attributes from Relax
