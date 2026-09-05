from __future__ import annotations
from typing import Protocol

import torch.distributed as dist

from qimpy import TreeNode, dft
from qimpy.io import CheckpointPath
from qimpy.lattice import Lattice
from . import Fixed, Relax, Dynamics


class Action(Protocol):
    """Class requirements to use as a geometry action."""

    def run(self, system: dft.System) -> None:
        """Run selected geometry action."""
        ...


class Geometry(TreeNode):
    """Select between possible geometry actions."""

    action: Action

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        lattice: Lattice,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        fixed: dict | Fixed | None = None,
        relax: dict | Relax | None = None,
        dynamics: dict | Dynamics | None = None,
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
        ChildOptions = TreeNode.ChildOptions
        self.add_child_one_of(
            "action",
            checkpoint_in,
            ChildOptions("fixed", Fixed, fixed, group=group, lattice=lattice),
            ChildOptions("relax", Relax, relax, group=group, lattice=lattice),
            ChildOptions("dynamics", Dynamics, dynamics, group=group),
            have_default=True,
        )

    def run(self, system: dft.System):
        """Run selected geometry action."""
        self.action.run(system)
