from __future__ import annotations
import qimpy as qp
from typing import Protocol, Optional, Union
from qimpy.rc import MPI


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
        comm: MPI.Comm,
        lattice: qp.lattice.Lattice,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        fixed: Optional[Union[dict, qp.geometry.Fixed]] = None,
        relax: Optional[Union[dict, qp.geometry.Relax]] = None,
        dynamics: Optional[Union[dict, qp.geometry.Dynamics]] = None,
    ) -> None:
        """Specify one of the supported geometry actions.
        Defaults to `Fixed` if none specified.

        Parameters
        ----------
        fixed
            :yaml:`Electronic optimization only at a fixed geometry.`
        relax
            :yaml:`Geometry relaxation of ions, and optionally, also the lattice.`
        dynamics
            :yaml:`Molecular dynamics of ions, and optionally, also the lattice.`
        """
        super().__init__()
        ChildOptions = qp.TreeNode.ChildOptions
        self.add_child_one_of(
            "action",
            checkpoint_in,
            ChildOptions("fixed", qp.geometry.Fixed, fixed, comm=comm, lattice=lattice),
            ChildOptions("relax", qp.geometry.Relax, relax, comm=comm, lattice=lattice),
            ChildOptions("dynamics", qp.geometry.Dynamics, dynamics, comm=comm),
            have_default=True,
        )

    def run(self, system: qp.System):
        """Run selected geometry action."""
        self.action.run(system)
