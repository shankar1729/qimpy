from __future__ import annotations
from typing import Union, Optional
import os

import torch

from qimpy import log, dft, MPI
from qimpy.lattice import Lattice
from qimpy.io import Checkpoint, CheckpointPath, CheckpointContext
from qimpy.mpi import BufferView
from qimpy.algorithms import Minimize, MinimizeState
from ._gradient import Gradient
from ._stepper import Stepper
from ._history import History


class Relax(Minimize[Gradient]):
    """Relax geometry of ions and/or lattice.
    Whether lattice changes is controlled by `lattice.movable`.
    """

    latticeK: float  #: Preconditioning factor of lattice relative to ions
    drag_wavefunctions: bool  #: Whether to drag atomic components of wavefunctions
    history: Optional[History]  #: Utility to save trajectory data
    stepper: Stepper  #: Interface to move ions/lattice and compute forces/stress

    def __init__(
        self,
        *,
        comm: MPI.Comm,
        lattice: Lattice,
        i_iter: int = 0,
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
        save_history: bool = True,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Specify geometry relaxation algorithm and convergence parameters.

        Parameters
        ----------
        i_iter
            Iteration number to start with (used for continuing from checkpoint)
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
        drag_wavefunctions
            :yaml:`Whether to drag atomic components of wavefunctions.`
        save_history
            :yaml:`Whether to save history along the trajectory.`
            Saved quantities include positions, forces, energies,
            stress (if available) and lattice (if movable).
        """
        extra_thresholds = {"fmax": fmax_threshold}
        if lattice.movable:
            extra_thresholds["|stress|"] = stress_threshold
        super().__init__(
            checkpoint_in=checkpoint_in,
            comm=comm,
            name="Relax",
            i_iter_start=i_iter,
            n_iterations=n_iterations,
            energy_threshold=energy_threshold,
            extra_thresholds=extra_thresholds,
            n_consecutive=int(n_consecutive),
            method=method,
            cg_type=cg_type,
            line_minimize=line_minimize,
            n_history=int(n_history),
            converge_on=converge_on,
        )
        self.drag_wavefunctions = drag_wavefunctions
        if save_history:
            self.add_child(
                "history",
                History,
                {},
                checkpoint_in,
                comm=comm,
                n_max=(n_iterations + 1),
            )
        else:
            self.history = None

    def run(self, system: dft.System) -> None:
        assert not system.electrons.fixed_H
        log.info(
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
        self.minimize()
        del self.stepper

        # Check point at end:
        if system.checkpoint_out:
            with Checkpoint(system.checkpoint_out, writable=True) as cp:
                system.save_checkpoint(CheckpointPath(cp), CheckpointContext("end"))

    def step(self, direction: Gradient, step_size: float) -> None:
        """Update the geometry along `direction` by amount `step_size`"""
        self.stepper.step(direction, step_size)

    def compute(self, state: MinimizeState[Gradient], energy_only: bool) -> None:
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
        system = self.stepper.system
        if system.checkpoint_out:
            with Checkpoint(system.checkpoint_out, writable=True) as cp:
                system.save_checkpoint(
                    CheckpointPath(cp), CheckpointContext("geometry", i_iter)
                )
        self.stepper.report()
        return False  # State not changed by report

    def _run_fd_test(self):
        """Run finite difference test."""

        def _randn_like(t: torch.Tensor) -> torch.Tensor:
            """Return an MPI-consistent random tensor with same shape as `t`."""
            result = torch.randn_like(t)
            if self.comm.size > 1:
                self.comm.Bcast(BufferView(result))
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

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        stage, i_iter = context
        attrs = cp_path.attrs
        attrs["i_iter"] = i_iter if (stage == "geometry") else self.n_iterations
        attrs["n_iterations"] = self.n_iterations
        attrs["energy_threshold"] = self.energy_threshold
        attrs["fmax_threshold"] = self.extra_thresholds["fmax"]
        if "|stress|" in self.extra_thresholds:
            attrs["stress_threshold"] = self.extra_thresholds["|stress|"]
        attrs["n_consecutive"] = self.n_consecutive
        attrs["method"] = self.method
        attrs["cg_type"] = self.cg_type
        attrs["line_minimize"] = self.line_minimize
        attrs["n_history"] = self.n_history
        attrs["converge_on"] = self.converge_on
        attrs["drag_wavefunctions"] = self.drag_wavefunctions
        attrs["save_history"] = self.history is not None
        saved_list = list(attrs.keys())
        if stage == "geometry":
            # Prepare for trajectory output if needed:
            history = self.history
            if history is not None:
                system = self.stepper.system
                lattice = system.lattice
                ions = system.ions
                history.i_iter = i_iter
                history.add("energy", float(system.energy))
                history.add("positions", ions.positions.detach())
                history.add("forces", ions.forces)
                if lattice.movable:
                    history.add("Rbasis", lattice.Rbasis)
                if lattice.compute_stress:
                    history.add("stress", lattice.stress.detach())
        return saved_list
