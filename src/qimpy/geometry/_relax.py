from __future__ import annotations
import qimpy as qp
import torch
import os
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Gradient:
    """Geometry gradient used for"""

    ions: torch.Tensor  #: ionic gradient (forces)
    lattice: Optional[torch.Tensor]  #: lattice gradient (stress)

    def clone(self) -> "Gradient":
        return Gradient(
            ions=self.ions.clone().detach(),
            lattice=(None if (self.lattice is None) else self.lattice.clone().detach()),
        )

    def __add__(self, other: "Gradient") -> "Gradient":
        return Gradient(
            ions=(self.ions + other.ions),
            lattice=(None if (self.lattice is None) else self.lattice + other.lattice),
        )

    def __iadd__(self, other: "Gradient") -> "Gradient":
        self.ions += other.ions
        if self.lattice is not None:
            self.lattice += other.lattice
        return self

    def __sub__(self, other: "Gradient") -> "Gradient":
        return Gradient(
            ions=(self.ions - other.ions),
            lattice=(None if (self.lattice is None) else self.lattice - other.lattice),
        )

    def __isub__(self, other: "Gradient") -> "Gradient":
        self.ions -= other.ions
        if self.lattice is not None:
            self.lattice -= other.lattice
        return self

    def __mul__(self, other: float) -> "Gradient":
        return Gradient(
            ions=(self.ions * other),
            lattice=(None if (self.lattice is None) else self.lattice * other),
        )

    __rmul__ = __mul__

    def __imul__(self, other: float) -> "Gradient":
        self.ions *= other
        if self.lattice is not None:
            self.lattice *= other
        return self

    def overlap(self, other: "Gradient") -> float:
        """Global overlap collected over `comm`. For complex arrays,
        real and imaginary components are treated as independent."""
        result = self.ions.flatten() @ other.ions.flatten()
        if self.lattice is not None:
            assert other.lattice is not None
            result += self.lattice.flatten() @ other.lattice.flatten()
        return float(result.item())


class Relax(qp.utils.Minimize[Gradient]):
    """Relax geometry of ions and/or lattice.
    Whether lattice changes is controlled by `lattice.movable`.
    """

    system: qp.System  #: System being optimized currently
    invRbasis0: torch.Tensor  #: Initial lattice vectors inverse (used to define strain)
    latticeK: float  #: Preconditioning factor of lattice relative to ions

    def __init__(
        self,
        *,
        comm: qp.MPI.Comm,
        lattice: qp.lattice.Lattice,
        n_iterations: int,
        energy_threshold: float = 1e-5,
        fmax_threshold: float = 5e-4,
        stress_threshold: float = 1e-3,
        n_consecutive: int = 1,
        method: str = "l-bfgs",
        cg_type: str = "polak-ribiere",
        line_minimize: str = "auto",
        n_history: int = 15,
        converge_on: Union[str, int] = "any",
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

    def run(self, system: qp.System) -> None:
        qp.log.info("\n--- Geometry relaxation ---\n")
        self.system = system
        self.invRbasis0 = system.lattice.invRbasis
        self.latticeK = (
            system.lattice.move_scale.mean() / system.lattice.volume ** (1.0 / 3)
        ) ** 2  # effectively bring lattice derivatives to same dimensions as forces
        if os.environ.get("QIMPY_FDTEST_RELAX", "0") in {"1", "yes"}:
            self._run_fd_test()  # finite difference test if needed
        self.minimize()

    def step(self, direction: Gradient, step_size: float) -> None:
        """Update the geometry along `direction` by amount `step_size`"""
        lattice = self.system.lattice
        self.system.ions.positions += step_size * (direction.ions @ lattice.invRbasis)
        if lattice.movable:
            lattice.update(
                Rbasis=lattice.Rbasis
                + step_size * (direction.lattice @ lattice.Rbasis),
                report_change=False,
            )
            self.system.coulomb.update_lattice_dependent(self.system.ions.n_ions)

    def compute(
        self, state: qp.utils.MinimizeState[Gradient], energy_only: bool
    ) -> None:
        """Update energy and/or gradients in `state`."""
        system = self.system
        lattice = system.lattice
        # Update ionic potentials and energies:
        system.energy = qp.Energy()
        system.ions.update(system)
        # Optimize electrons:
        qp.log.info("\n--- Electronic optimization ---\n")
        system.electrons.run(system)
        state.energy = system.energy
        # Update forces / stresses if needed:
        if not energy_only:
            system.geometry_grad()  # update forces / stress
            state.gradient = self.constrain(
                Gradient(
                    ions=-system.ions.forces,
                    lattice=(lattice.grad if lattice.movable else None),
                )
            )
            state.K_gradient = state.gradient.clone()
            if state.K_gradient.lattice is not None:
                state.K_gradient.lattice *= self.latticeK
            # Extra convergence checks:
            state.extra = [system.ions.forces.norm(dim=1).max().item()]  # fmax
            if lattice.movable:
                state.extra.append(lattice.stress.norm().item())  # |stress|

    def report(self, i_iter: int) -> bool:
        qp.log.info(f"\nEnergy components:\n{repr(self.system.energy)}")
        qp.log.info("")
        self.system.ions.report(report_grad=True)  # positions, forces
        if self.system.lattice.compute_stress:
            self.system.lattice.report(report_grad=True)  # lattice, stress
            qp.log.info(f"Strain:\n{qp.utils.fmt(self.strain)}")
            qp.log.info("")
        return False  # State not changed by report

    @property
    def strain(self) -> torch.Tensor:
        eye = torch.eye(3, device=qp.rc.device)
        return self.system.lattice.Rbasis @ self.invRbasis0 - eye

    def constrain(self, v: Gradient) -> Gradient:
        """Impose fixed atom / lattice direction constraints."""
        self.symmetrize_(v)
        v.ions -= v.ions.mean(dim=0)
        self.symmetrize_(v)
        return v

    def symmetrize_(self, v: Gradient) -> None:
        """Symmetrize gradient in-place."""
        lattice = self.system.lattice
        v.ions = (
            self.system.symmetries.symmetrize_forces(v.ions @ lattice.Rbasis)
            @ lattice.invRbasis
        )
        if v.lattice is not None:
            v.lattice = self.system.symmetries.symmetrize_matrix(v.lattice)

    def _run_fd_test(self):
        """Run finite difference test."""
        # Prepare a random direction to test along:
        STD_FORCES = 1e-3  # Std. deviation of force components
        STD_STRESS = STD_FORCES * self.latticeK  # Std. deviation of stress components
        lattice = self.system.lattice
        direction = self.constrain(
            Gradient(
                ions=torch.randn_like(self.system.ions.positions) * STD_FORCES,
                lattice=(
                    torch.randn_like(lattice.Rbasis) * STD_STRESS
                    if lattice.movable
                    else None
                ),
            )
        )
        self.finite_difference_test(direction)
