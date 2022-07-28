from __future__ import annotations
import qimpy as qp


class Fixed(qp.TreeNode):
    """Fixed geometry, i.e. only optimize electronic degrees of freedom."""

    def __init__(self, *, checkpoint_in: qp.utils.CpPath = qp.utils.CpPath()) -> None:
        super().__init__()

    def run(self, system: qp.System) -> None:
        qp.log.info("\n--- Electronic optimization at fixed geometry ---\n")
        system.electrons.run(system)
        qp.log.info(f"\nEnergy components:\n{repr(system.energy)}")
        qp.log.info("")
        if not system.electrons.fixed_H:
            system.geometry_grad()  # update forces / stress
            system.ions.report(report_grad=True)  # positions, forces
            if system.lattice.compute_stress:
                system.lattice.report(report_grad=True)  # lattice, stress
        if system.checkpoint_out:
            system.save_checkpoint(
                qp.utils.CpPath(
                    checkpoint=qp.utils.Checkpoint(system.checkpoint_out, mode="w")
                )
            )
