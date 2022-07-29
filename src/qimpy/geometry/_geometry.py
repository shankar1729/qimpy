from __future__ import annotations
import qimpy as qp
from typing import Protocol, Optional, Union


class Action(Protocol):
    """Class requirements to use as a geometry action."""

    def run(self, system: qp.System) -> None:
        ...


class Geometry(qp.TreeNode):
    """Select between possible geometry actions."""

    action: Action

    def __init__(
        self,
        *,
        comm: qp.MPI.Comm,
        lattice: qp.lattice.Lattice,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        fixed: Optional[Union[dict, qp.geometry.Fixed]] = None,
        relax: Optional[Union[dict, qp.geometry.Relax]] = None,
    ) -> None:
        """Specify one of the supported geometry actions.
        Defaults to `Fixed` if none specified.

        Parameters
        ----------
        fixed
            Electronic optimization only at a fixed geometry.
        relax
            :yaml:`Geometry relaxation of ions, and optionally, also the lattice.`
        """
        super().__init__()
        self.add_child_one_of(
            "action",
            checkpoint_in,
            qp.TreeNode.ChildOptions("fixed", qp.geometry.Fixed, fixed),
            qp.TreeNode.ChildOptions(
                "relax", qp.geometry.Relax, relax, comm=comm, lattice=lattice
            ),
            have_default=True,
        )

    def run(self, system: qp.System):
        """Run selected geometry action."""
        self.action.run(system)
