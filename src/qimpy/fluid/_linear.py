from typing import Protocol

import numpy as np
import torch

from qimpy import log, Energy, TreeNode
from qimpy.io import CheckpointPath
from qimpy.algorithms import LinearSolve
from qimpy.grid import Grid, FieldH, FieldR
from qimpy.grid.coulomb import Coulomb
from qimpy.profiler import stopwatch
from . import variants, set_solvent_properties, DIELECTRIC_PROPERTIES


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
    epsilon_0: float  #: bulk static dielectric constant
    screening_length: float  #: fluid (Debye) screening length; None => no screening
    variant: Variant  #: variant of cavity shape and cavitation model
    zero_nyquist: bool  #: whether to zero Nyquist frequencies in Poisson equation

    energy: Energy  #: energy components
    phi_tilde: FieldH  #: net electrostatic potential
    epsilon: FieldR  #: spatially varying dielectric constant
    kappa_sq: float  #: screening strength (0 => disabled)
    Kkernel: torch.Tensor  #: preconditioner kernel

    def __init__(
        self,
        *,
        grid: Grid,
        coulomb: Coulomb,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        n_iterations: int = 100,
        threshold: float = 1e-8,
        epsilon_0: float | None = None,
        screening_length: float | None = None,
        solvent: str = "",
        GLSSA13: dict | variants.GLSSA13 | None = None,
        LA12: dict | variants.LA12 | None = None,
        zero_nyquist: bool = True,
        verbose: bool = False,
    ):
        super().__init__(
            checkpoint_in=checkpoint_in,
            group=grid.group,
            n_iterations=n_iterations,
            threshold=threshold,
            name=("  Fluid" if verbose else ""),
        )
        self.grid = grid
        self.coulomb = coulomb
        self.screening_length = screening_length
        set_solvent_properties(
            solvent, DIELECTRIC_PROPERTIES, dict(epsilon_0=epsilon_0), self
        )
        self.kappa_sq = (
            self.epsilon_0 / screening_length**2
            if screening_length is not None
            else 0.0
        )
        self.add_child_one_of(
            "variant",
            checkpoint_in,
            TreeNode.ChildOptions(
                "GLSSA13", variants.GLSSA13, GLSSA13, solvent=solvent
            ),
            TreeNode.ChildOptions("LA12", variants.LA12, LA12, solvent=solvent),
            have_default=True,
        )
        self.zero_nyquist = zero_nyquist

        self.energy = Energy(name="Afluid")
        self.phi_tilde = FieldH(self.grid)

        # Initialize preconditioner:
        Gsq = grid.get_gradient_operator("H", zero_nyquist).imag.square().sum(dim=0)
        Kinv = (self.epsilon_0 * Gsq + self.kappa_sq) / (4 * np.pi)
        KINV_CUT = 1e-12  # regularization
        self.Kkernel = torch.clamp(Kinv, min=KINV_CUT).reciprocal()
        self.Kkernel[Kinv < KINV_CUT] = 0.0  # project out null-space

    def hessian(self, phi_tilde: FieldH) -> FieldH:
        result = (
            ~(~phi_tilde.gradient(zero_nyquist=self.zero_nyquist) * self.epsilon[None])
        ).divergence(zero_nyquist=self.zero_nyquist)
        if self.kappa_sq:
            # Screening (ionic) term, per fluid screening length:
            kappa_sq_r = self.kappa_sq * self.variant.shape  # fieldR
            result -= ~(kappa_sq_r * (~phi_tilde))
        return (-1 / (4 * np.pi)) * result

    def precondition(self, vector: FieldH) -> FieldH:
        return vector.convolve(self.Kkernel)

    @stopwatch(name="Linear.calculate")
    def update(self, n_tilde: FieldH, rho_tilde: FieldH, phi_o_offset: float) -> None:
        self.variant.update_shape(n_tilde)
        shape = self.variant.shape
        self.epsilon = 1.0 + (self.epsilon_0 - 1.0) * shape

        if self.zero_nyquist:
            rho_tilde.zero_nyquist()
        n_iter = self.solve(rho_tilde, self.phi_tilde)
        log.info(f"  Fluid: solve completed in {n_iter} iterations")

        # Electrostatic contributions:
        phi_ext_tilde = self.coulomb.kernel(rho_tilde)
        self.energy["Acoulomb"] = -0.5 * (
            self.phi_tilde ^ self.hessian(self.phi_tilde)
        ) + ((self.phi_tilde - 0.5 * phi_ext_tilde) ^ rho_tilde)
        if n_tilde.requires_grad:
            grad_phi_sq = (
                (~self.phi_tilde.gradient(zero_nyquist=self.zero_nyquist))
                .data.square()
                .sum(dim=0)
            )
            shape.requires_grad_(True)
            shape.grad = FieldR(
                self.grid, data=(-(self.epsilon_0 - 1) / (8 * np.pi)) * grad_phi_sq
            )
            if self.kappa_sq:
                # Ionic contribution to the shape gradient (backprop):
                phi_sq = (~self.phi_tilde).data.square()
                shape.grad.data -= (self.kappa_sq / (8 * np.pi)) * phi_sq

        # Cavitation terms:
        self.variant.update_energy(self.energy)

        # Corrections due to ion width:
        phi_ext_tilde.o += phi_o_offset
        self.energy["muShift"] = -phi_o_offset * rho_tilde.integral()

        # Propagate gradients as needed:
        if n_tilde.requires_grad:
            self.variant.propagate_shape_grad(n_tilde)
        if rho_tilde.requires_grad:
            rho_tilde.grad += self.phi_tilde - phi_ext_tilde
