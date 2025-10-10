from typing import Protocol, Optional, Union

import numpy as np
import torch

from qimpy import log, Energy, TreeNode
from qimpy.algorithms import LinearSolve
from qimpy.grid import Grid, FieldH, FieldR
from qimpy.grid.coulomb import Coulomb
from qimpy.profiler import stopwatch
from . import variants


class Variant(Protocol):
    """Class requirements to use as a variant for Linear / Nonlinear fluid models."""

    shape: FieldR  #: cavity shape function

    def update_shape(self, n_tilde: FieldH) -> None:
        """Update `shape` from electron density `n_tilde`."""
        ...

    def propagate_shape_grad(self, n_tilde: FieldH) -> None:
        """Propagate gradient from `shape.grad` to `n_tilde.grad` (accumulate)."""
        ...

    def update_energy(self, energy: Energy) -> None:
        """Update shape-dependent energy terms, e.g., cavitation and dispersion.
        If `shape.requires_grad`, accumulate corresponding gradient to `shape.grad`."""
        ...


class Linear(LinearSolve[FieldH]):
    grid: Grid
    coulomb: Coulomb
    eps_bulk: float  #: Bulk dielectric constant
    variant: Variant  #: variant of cavity shape and cavitation model

    energy: Energy  #: energy components
    phi_tilde: FieldH  #: net electrostatic potential
    epsilon: FieldR  #: spatially varying dielectric constant

    def __init__(
        self,
        *,
        grid: Grid,
        coulomb: Coulomb,
        checkpoint_in=None,
        n_iterations=100,
        gradient_threshold=1e-8,
        eps_bulk=78.4,
        GLSSA13: Optional[Union[dict, variants.GLSSA13]] = None,
        LA12: Optional[Union[dict, variants.LA12]] = None,
    ):
        super().__init__(
            checkpoint_in=checkpoint_in,
            comm=grid.comm,
            n_iterations=n_iterations,
            gradient_threshold=gradient_threshold,
        )
        self.grid = grid
        self.coulomb = coulomb
        self.eps_bulk = eps_bulk
        self.add_child_one_of(
            "variant",
            checkpoint_in,
            TreeNode.ChildOptions("glssa13", variants.GLSSA13, GLSSA13),
            TreeNode.ChildOptions("la12", variants.LA12, LA12),
            have_default=True,
        )

        self.energy = Energy(name="Afluid")
        self.phi_tilde = FieldH(self.grid)

        # Initialize preconditioner:
        iG = grid.get_mesh("H").to(torch.double)
        Gsq = (iG @ grid.lattice.Gbasis.T).square().sum(dim=-1)
        GSQ_CUT = 1e-12  # regularization
        self.Kkernel = torch.clamp(Gsq, min=GSQ_CUT).reciprocal() / self.eps_bulk
        self.Kkernel[Gsq < GSQ_CUT] = 0.0  # project out null-space

    def hessian(self, phi_tilde: FieldH) -> FieldH:
        result = (~(~phi_tilde.gradient() * self.epsilon[None])).divergence()
        return (-1 / (4 * np.pi)) * result

    def precondition(self, vector: FieldH) -> FieldH:
        return vector.convolve(self.Kkernel)

    @stopwatch(name="Linear.calculate")
    def update(self, n_tilde: FieldH, rho_tilde: FieldH) -> None:
        self.variant.update_shape(n_tilde)
        shape = self.variant.shape
        self.epsilon = 1.0 + (self.eps_bulk - 1.0) * shape

        n_iter = self.solve(rho_tilde, self.phi_tilde)
        log.info(f"  Fluid: solve completed in {n_iter} iterations")

        # Electrostatic contributions:
        phi_ext_tilde = self.coulomb.kernel(rho_tilde)
        self.energy["Acoulomb"] = -0.5 * (
            self.phi_tilde ^ self.hessian(self.phi_tilde)
        ) + ((self.phi_tilde - 0.5 * phi_ext_tilde) ^ rho_tilde)
        if n_tilde.requires_grad:
            grad_phi_sq = (~self.phi_tilde.gradient()).data.square().sum(dim=0)
            shape.requires_grad_(True)
            shape.grad = FieldR(
                self.grid, data=(-(self.eps_bulk - 1) / (8 * np.pi)) * grad_phi_sq
            )

        # Cavitation terms:
        self.variant.update_energy(self.energy)

        # Propagate gradients as needed:
        if n_tilde.requires_grad:
            self.variant.propagate_shape_grad(n_tilde)
        if rho_tilde.requires_grad:
            rho_tilde.grad += self.phi_tilde - phi_ext_tilde
