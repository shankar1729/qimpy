from __future__ import annotations
from typing import Protocol

import torch

from qimpy import log, rc, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.grid import Grid, FieldH
from ._periodic import KernelPeriodic, EwaldPeriodic
from ._slab import KernelSlab, EwaldSlab
from ._wire import KernelCylindrical, KernelWire, EwaldWire
from ._isolated import KernelSpherical, KernelIsolated, EwaldIsolated


class Kernel(Protocol):
    """Specification for Coulomb kernel."""

    def __call__(self, rho: FieldH, correct_G0_width: bool = False) -> FieldH:
        """
        Apply coulomb operator on charge density `rho`.
        If correct_G0_width = True, rho is a point charge distribution
        widened by `ion_width` and needs a corresponding G=0 correction.
        """

    def stress(self, rho1: FieldH, rho2: FieldH) -> torch.Tensor:
        """
        Return stress due to Coulomb interaction between `rho1` and `rho2`.
        The result has dimensions of energy, appropriate for adding to `lattice.grad`.
        """


class Ewald(Protocol):
    """Specification for Ewald sum."""

    def __call__(self, positions: torch.Tensor, Z: torch.Tensor) -> float:
        """Compute Ewald energy, and optionally accumulate gradients.
        Each gradient contribution is accumulated to a `grad` attribute,
        only if the corresponding `requires_grad` is enabled.
        Force contributions are collected in `positions.grad`.
        Stress contributions are collected in `self.grid.lattice.grad`.

        Parameters
        ----------
        positions
            Positions (fractional coordinates) of point charges
        Z
            Charges of each point charge
        """


class Coulomb(TreeNode):
    """Coulomb interactions between fields and point charges."""

    grid: Grid  #: Grid associated with fields for coulomb interaction
    embed: bool  #: Whether to embed the Coulomb interaction
    analytic: bool  #: Whether to use an analytic or a numerical truncation scheme
    radius: float  #: Radius for analytical truncation (use in-radius if zero)
    ion_width: float  #: Ion-charge gaussian width for embedding and solvation
    kernel: Kernel  #: Coulomb kernel
    ewald: Ewald  #: Ewald sum

    def __init__(
        self,
        *,
        grid: Grid,
        n_ions: int = 1,
        embed: bool = False,
        analytic: bool = False,
        radius: float = 0.0,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """Initialize coulomb interactions.

        Parameters
        ----------
        grid
            Fields for coulomb interaction will be on this grid.
        n_ions
            Number of point charges to optimize Ewald sums for.
        embed
            :yaml:`Whether to embed the Coulomb interaction.`
        analytic
            :yaml:`Whether to use an analytic or a numerical truncation scheme.`
            This only matters when 0 or 1 directions are periodic:
            selecting spherical or cylindrical truncation when True,
            and the numerical Wigner-Seitz kernel otherwise (default).
        """
        super().__init__()
        self.grid = grid
        self.n_ions = n_ions
        self.embed = embed
        self.analytic = analytic
        self.radius = radius
        self.update_lattice_dependent()

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["embed"] = self.embed
        attrs["analytic"] = self.analytic
        attrs["radius"] = self.radius
        return list(attrs.keys())

    def update_lattice_dependent(self) -> None:
        """Update all members due to change of lattice vectors."""
        # Determine ionic width from maximum grid spacing
        grid = self.grid
        lattice = grid.lattice
        h_max = (
            (lattice.Rbasis.norm(dim=0).to(rc.cpu) / torch.tensor(grid.shape))
            .max()
            .item()
        )
        self.ion_width = 2.2 * h_max  # Best balance from test_nyquist()
        log.info(f"Ionic width for embedding / fluids: {self.ion_width:f}")

        # For simplicity, just recreate kernel and ewald with new lattice
        if self.embed:
            raise NotImplementedError  # TODO

        i_periodic = [i for (i, x) in enumerate(lattice.periodic) if x]
        n_periodic = len(i_periodic)
        if n_periodic == 3:
            self.kernel = KernelPeriodic(self)
            self.ewald = EwaldPeriodic(lattice, self.n_ions)
        elif n_periodic == 2:
            i_dir = 3 - sum(i_periodic)  # only truncated direction
            self.kernel = KernelSlab(self, i_dir)
            self.ewald = EwaldSlab(lattice, i_dir)
        elif n_periodic == 1:
            i_dir = i_periodic[0]  # only periodic direction
            if self.analytic:
                self.kernel = KernelCylindrical(self, i_dir)
            else:
                self.kernel = KernelWire(self, i_dir)
            self.ewald = EwaldWire(lattice, i_dir)
        else:
            if self.analytic:
                self.kernel = KernelSpherical(self)
            else:
                self.kernel = KernelIsolated(self)
            self.ewald = EwaldIsolated(lattice)
