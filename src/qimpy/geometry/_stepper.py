from __future__ import annotations
import qimpy as qp
import torch
from typing import Optional
from ._gradient import Gradient


class Stepper:
    """Shared interface of dynamics and optimization with electronic system."""

    system: qp.dft.System  #: System being optimized currently
    invRbasis0: torch.Tensor  #: Initial lattice vectors inverse (used to define strain)
    drag_wavefunctions: bool  #: Whether to drag atomic components of wavefunctions
    isotropic: bool  #: Whether to force lattice changes to be isotropic (NPT mode)
    _lowdin: Optional[qp.ions.Lowdin]  #: Lowdin and wavefunction drag shared data

    def __init__(
        self,
        system: qp.dft.System,
        *,
        drag_wavefunctions: bool = True,
        isotropic: bool = False,
    ) -> None:
        self.system = system
        self.drag_wavefunctions = drag_wavefunctions
        self.isotropic = isotropic
        self.invRbasis0 = system.lattice.invRbasis
        self._lowdin = None

    def step(self, direction: Gradient, step_size: float) -> None:
        """Update the geometry along `direction` by amount `step_size`"""
        lattice = self.system.lattice

        if self.drag_wavefunctions:
            if self._lowdin is None:
                self._lowdin = qp.ions.Lowdin(self.system.electrons.C)
            self._lowdin.remove_atomic_projections()

        delta_positions = step_size * (direction.ions @ lattice.invRbasis.T)
        self.system.ions.positions += delta_positions
        if lattice.movable:
            lattice.update(
                lattice.Rbasis + step_size * (direction.lattice @ lattice.Rbasis),
                report_change=False,
            )
            self.system.coulomb.update_lattice_dependent(self.system.ions.n_ions)

        if self.drag_wavefunctions:
            assert self._lowdin is not None
            self._lowdin.restore_atomic_projections(delta_positions)
            self._lowdin = None

    def compute(self, require_grad: bool) -> tuple[qp.Energy, Optional[Gradient]]:
        """Compute energy and optionally ionic/lattice gradient."""
        system = self.system
        lattice = system.lattice
        # Update ionic potentials and energies:
        system.energy = qp.Energy()
        system.ions.update(system)
        # Optimize electrons:
        qp.log.info("\n--- Electronic optimization ---\n")
        system.electrons.run(system)
        # Update forces / stresses if needed:
        if require_grad:
            system.geometry_grad()  # update forces / stress
            gradient = self.constrain(
                Gradient(
                    ions=-system.ions.forces,
                    lattice=(lattice.grad if lattice.movable else None),
                )
            )
            return system.energy, gradient
        else:
            return system.energy, None

    def report(self, total_stress: Optional[torch.Tensor] = None) -> None:
        system = self.system
        ions = system.ions
        electrons = system.electrons
        qp.log.info(f"\nEnergy components:\n{repr(system.energy)}")
        qp.log.info("")
        self._lowdin = qp.ions.Lowdin(electrons.C)
        ions.Q, ions.M = self._lowdin.analyze(
            electrons.fillings.f, electrons.spin_polarized
        )
        ions.report(report_grad=True)  # positions, forces, Lowdin Q/M
        if system.lattice.compute_stress:
            system.lattice.report(report_grad=True)  # lattice, stress
            if total_stress is not None:  # total stress including kinetic contributions
                qp.log.info(f"Stress+kinetic [Eh/a0^3]:\n{qp.utils.fmt(total_stress)}")
            qp.log.info(f"Strain:\n{qp.utils.fmt(self.strain)}")
            qp.log.info("")

    @property
    def strain(self) -> torch.Tensor:
        eye = torch.eye(3, device=qp.rc.device)
        return self.system.lattice.Rbasis @ self.invRbasis0 - eye

    def constrain(self, v: Gradient) -> Gradient:
        """Impose fixed atom / lattice direction constraints."""
        self.symmetrize_(v)
        v.ions -= v.ions.mean(dim=0)
        if self.isotropic and (v.lattice is not None):
            isotropic_strain = torch.trace(v.lattice) / 3.0
            v.lattice = isotropic_strain * torch.eye(3, device=qp.rc.device)
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
