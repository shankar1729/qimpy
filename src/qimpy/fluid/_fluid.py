from __future__ import annotations
from typing import Protocol, Optional, Union

from qimpy import TreeNode, Energy
from qimpy.io import CheckpointPath
from qimpy.grid import Grid, FieldH
from qimpy.grid.coulomb import Coulomb
from . import Linear


class Model(Protocol):
    """Class requirements to use as a fluid model."""

    energy: Energy  #: energy components

    def update(self, n_tilde: FieldH, rho_tilde: FieldH) -> None:
        """Update fluid given electron density `n_tilde` to determine
        cavity and total solute charge density `rho_tilde`.
        Update `energy` and accumulate gradients to `n_tilde.grad`
        and `rho_tilde.grad` if corresponding requires_grad is set."""
        ...


class Fluid(TreeNode):
    """Select between possible fluid models."""

    model: Model
    enabled: bool  #: whether the fluid model is currently enabled

    def __init__(
        self,
        *,
        grid: Grid,
        coulomb: Coulomb,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        solvent: str = "",
        linear: Optional[Union[dict, Linear]] = None,
    ) -> None:
        """Specify one of the supported fluid models.
        Defaults to the Null model (no solvation) if none specified.

        Parameters
        ----------
        solvent
            :yaml:`Name of solvent to load default model parameters.`
        linear
            :yaml:`Linear-response solvation model.`
        """
        super().__init__()
        ChildOptions = TreeNode.ChildOptions
        self.add_child_one_of(
            "model",
            checkpoint_in,
            ChildOptions("null", Null, None),
            ChildOptions(
                "linear", Linear, linear, grid=grid, coulomb=coulomb, solvent=solvent
            ),
            have_default=True,
        )
        self.enabled = not isinstance(self.model, Null)


class Null(TreeNode):
    """No solvation model."""

    def __init__(self, *, checkpoint_in=None):
        super().__init__()

    def calculate(self, n_tilde: FieldH, rho_tilde: FieldH) -> float:
        return 0.0
