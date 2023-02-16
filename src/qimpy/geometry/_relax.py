from __future__ import annotations
import qimpy as qp
import torch
import os
from typing import Union, Optional
from qimpy.rc import MPI
from ._gradient import Gradient
from ._stepper import Stepper


class Relax(qp.utils.Minimize[Gradient]):
    """Relax geometry of ions and/or lattice.
    Whether lattice changes is controlled by `lattice.movable`.
    """

    latticeK: float  #: Preconditioning factor of lattice relative to ions
    drag_wavefunctions: bool  #: Whether to drag atomic components of wavefunctions
    stepper: Stepper  #: Interface to move ions/lattice and compute forces/stress
    checkpoint: Optional[qp.utils.Checkpoint]  #: Output checkpoint

    def __init__(
        self,
        *,
        comm: MPI.Comm,
        lattice: qp.lattice.Lattice,
        n_iterations: int = 20,
        energy_threshold: float = 5e-5,
        fmax_threshold: float = 5e-4,
        stress_threshold: float = 1e-5,
        n_consecutive: int = 1,
        method: str = "l-bfgs",
        cg_type: str = "polak-ribiere",
        line_minimize: str = "auto",
        n_history: int = 15,
        converge_on: Union[str, int] = "all",
        drag_wavefunctions: bool = True,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
    ) -> None:
        """
        Specify geometry relaxation algorithm and convergence parameters.

        Parameters
        ----------
        n_iterations
            :yaml:`Maximum number of iterations.`
        energy_threshold
            :yaml:`Convergence threshold on energy change in Eh.`
        fmax_threshold
            :yaml:`Convergence threshold on maximum force in Eh/a0.`
        stress_threshold
            :yaml:`Convergence threshold on |stress| (stress tensor norm) in Eh/a0^3.`
        n_consecutive
            :yaml:`Number of consecutive iterations each threshold must be satisfied.`
        method
            :yaml:`Relaxation algorithm: L-BFGS, CG or Gradient.`
            The default L-BFGS (limited-memory Broyden–Fletcher–Goldfarb–Shanno)
            method is strongly recommended as it requires the least number of force
            evaluations per line minimize step (with the `Wolfe` line minimize).
            Only use CG (conjugate gradients) if L-BFGS fails for some system.
            The steepest-descent `Gradient` method is only for special test cases.
        cg_type
            :yaml:`CG variant: Polak-Ribiere, Fletcher-Reeves or Hestenes-Stiefel.`
            Variant of conjugate gradients method (only matters if `method` is CG).
        line_minimize
            :yaml:`Line minimization scheme: Auto, Constant, Quadratic, Wolfe.`
            Auto matches the line minimization scheme to `method` (recommended).
            Constant is a constant step-size usable with the Gradient-descent method.
            Quadratic is best-suited for conjugate-gradients methods.
            Wolfe is a cubic line step best suited for L-BFGS.
        n_history
            :yaml:`Maximum history size (only used for L-BFGS).`
        converge_on
            :yaml:`Converge on 'any', 'all' or a specific number of thresholds.`
            If set to `any`, reaching threshold on any one of energy, force and stress
            (wherever applicable) will lead to convergence. When set to `all`, all
            applicable thresholds must be satisfied. If set to an integer between 1
            and the number of applicable thresholds, require that many thresholds to
            be satisfied simultaneously to achieve convergence.
        """
        extra_thresholds = {"fmax": fmax_threshold}
        if lattice.movable:
            extra_thresholds["|stress|"] = stress_threshold
        super().__init__(
            checkpoint_in=checkpoint_in,
            comm=comm,
            name="Relax",
            n_iterations=n_iterations,
            energy_threshold=energy_threshold,
            extra_thresholds=extra_thresholds,
            n_consecutive=n_consecutive,
            method=method,
            cg_type=cg_type,
            line_minimize=line_minimize,
            n_history=n_history,
            converge_on=converge_on,
        )
        self.drag_wavefunctions = drag_wavefunctions
        self.checkpoint = None

    def run(self, system: qp.System) -> None:
        qp.log.info(
            "\n--- Geometry relaxation ---\n"
            if self.n_iterations
            else "\n--- Fixed geometry ---\n"
        )
        self.stepper = Stepper(system, drag_wavefunctions=self.drag_wavefunctions)
        self.latticeK = (
            system.lattice.move_scale.mean() / system.lattice.volume ** (1.0 / 3)
        ) ** 2  # effectively bring lattice derivatives to same dimensions as forces
        if os.environ.get("QIMPY_FDTEST_RELAX", "0") in {"1", "yes"}:
            self._run_fd_test()  # finite difference test if needed
        if system.checkpoint_out:
            self.checkpoint = qp.utils.Checkpoint(system.checkpoint_out, mode="a")
        self.minimize()
        del self.stepper

        # Check point at end:
        if self.checkpoint is not None:
            system.save_checkpoint(
                qp.utils.CpPath(checkpoint=self.checkpoint), qp.utils.CpContext("end")
            )

    def step(self, direction: Gradient, step_size: float) -> None:
        """Update the geometry along `direction` by amount `step_size`"""
        self.stepper.step(direction, step_size)

    def compute(
        self, state: qp.utils.MinimizeState[Gradient], energy_only: bool
    ) -> None:
        """Update energy and/or gradients in `state`."""
        state.energy, gradient = self.stepper.compute(not energy_only)
        if gradient is not None:
            state.gradient = gradient
            state.K_gradient = gradient.clone()
            if state.K_gradient.lattice is not None:
                state.K_gradient.lattice *= self.latticeK
            # Extra convergence checks:
            system = self.stepper.system
            state.extra = [
                system.ions.forces.norm(dim=1).max().item()
                if system.ions.n_ions
                else 0.0
            ]  # fmax
            if system.lattice.movable:
                state.extra.append(system.lattice.stress.norm().item())  # |stress|

    def report(self, i_iter: int) -> bool:
        if self.checkpoint is not None:
            self.stepper.system.save_checkpoint(
                qp.utils.CpPath(checkpoint=self.checkpoint),
                qp.utils.CpContext("geometry", i_iter),
            )
        self.stepper.report()
        return False  # State not changed by report

    def _run_fd_test(self):
        """Run finite difference test."""

        def _randn_like(t: torch.Tensor) -> torch.Tensor:
            """Return an MPI-consistent random tensor with same shape as `t`."""
            result = torch.randn_like(t)
            if self.comm.size > 1:
                self.comm.Bcast(qp.utils.BufferView(result))
            return result

        # Prepare a random direction to test along:
        STD_FORCES = 1e-3  # Std. deviation of force components
        STD_STRESS = STD_FORCES * self.latticeK  # Std. deviation of stress components
        system = self.stepper.system
        lattice = system.lattice
        torch.manual_seed(0)
        direction = self.constrain(
            Gradient(
                ions=_randn_like(system.ions.positions) * STD_FORCES,
                lattice=(
                    _randn_like(lattice.Rbasis) * STD_STRESS
                    if lattice.movable
                    else None
                ),
            )
        )
        self.finite_difference_test(direction)
