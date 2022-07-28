from __future__ import annotations
import qimpy as qp
import numpy as np
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
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        fixed: Optional[Union[dict, qp.geometry.Fixed]] = None,
        relax: Optional[Union[dict, qp.geometry.Relax]] = None,
    ) -> None:
        """Specify one of the supported geometry actions.
        Defaults to `Fixed` if none specified.
        """
        super().__init__()
        n_options = np.count_nonzero([(d is not None) for d in (fixed, relax)])
        if n_options == 0:
            fixed = {}
        if n_options > 1:
            raise ValueError("Cannot use both davidson and chefsi")
        if fixed is not None:
            self.add_child(
                "action",
                qp.geometry.Fixed,
                fixed,
                checkpoint_in,
                attr_version_name="fixed",
            )
        if relax is not None:
            self.add_child(
                "action",
                qp.geometry.Relax,
                relax,
                checkpoint_in,
                attr_version_name="relax",
            )

    def run(self, system: qp.System):
        """Run selected geometry action."""
        self.action.run(system)
