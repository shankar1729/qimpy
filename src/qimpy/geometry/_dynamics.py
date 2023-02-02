from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from typing import Union, Optional, Callable
from qimpy.rc import MPI
from ._stepper import Stepper
from ._gradient import Gradient
from .thermostat import Thermostat
from qimpy.ions.symbols import ATOMIC_WEIGHTS, ATOMIC_NUMBERS
from qimpy.utils import Unit, UnitOrFloat


class Dynamics(qp.TreeNode):
    """Molecular dynamics of ions and/or lattice.
    Whether lattice changes is controlled by `lattice.movable`.
    """

    system: qp.System  #: System being optimized currently
    masses: torch.Tensor  #: Mass of each ion in system (Dim: n_ions x 1 for bcast)
    stepper: Stepper
    comm: MPI.Comm  #: Communictaor over which forces consistent
    dt: float  #: Time step
    n_steps: int  #: Number of MD steps
    thermostat: Thermostat  #: Thermostat/barostat method
    seed: int  #: Random seed for initial velocities
    T0: float  #: Initial temperature / temperature set point
    P0: Optional[float]  #: Pressure set point
    stress0: Optional[Union[np.ndarray, torch.Tensor]]  #: Stress set point
    t_damp_T: float  #: Thermostat damping time
    t_damp_P: float  #: Barostat damping time
    B0: float  #: Characteristic bulk modulus for Berendsen barostat
    langevin_gamma: float  #: Damping rate for Langevin thermostat
    drag_wavefunctions: bool  #: Whether to drag atomic components of wavefunctions
    P: Optional[float]  #: Current pressure (available if `lattice.compute_stress`)
    T: float  #: Current temperature
    KE: float  #: Current kinetic energy
    stress: Optional[torch.Tensor]  #: Current stress including kinetic contributions
    report_callback: Optional[Callable[[Dynamics, int], None]]  #: Callback from report

    def __init__(
        self,
        *,
        comm: MPI.Comm,
        dt: float,
        n_steps: int,
        thermostat: Union[Thermostat, dict, str, None] = None,
        seed: int = 1234,
        T0: UnitOrFloat = Unit(298.0, "K"),
        P0: Optional[float] = None,
        stress0: Optional[Union[np.ndarray, torch.Tensor]] = None,
        t_damp_T: UnitOrFloat = Unit(50.0, "fs"),
        t_damp_P: UnitOrFloat = Unit(100.0, "fs"),
        drag_wavefunctions: bool = True,
        report_callback: Optional[Callable[[Dynamics, int], None]] = None,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
    ) -> None:
        """
        Specify molecular dynamics parameters.

        Parameters
        ----------

        dt
            :yaml:`Time step.`
        n_steps
            :yaml:`Number of MD steps.`
        thermostat
            :yaml:`Thermostat/barostat method.`
            Specify name of thermostat eg. 'nose-hoover' if using default options
            for that thermostat method, and dictionary of parameters if not.
        seed
            :yaml:`Random seed for initial velocities.`
        T0
            :yaml:`Initial temperature / temperature set point.`
        P0
            :yaml:`Pressure set point.`
        stress0
            :yaml:`Stressthermostat method set point.`
        t_damp_T
            :yaml:`Thermostat damping time.`
        t_damp_P
            :yaml:`Barostat damping time.`
        drag_wavefunctions
            :yaml:`Whether to drag atomic components of wavefunctions.`
        report_callback
            Optional function to call at each step during `report`.
            Use this to perform additional reportig / data collection.
            The functional will be called as `report_callback(dynamics, i_iter)`.
        """
        super().__init__()
        self.comm = comm
        self.dt = dt
        self.n_steps = n_steps
        self.seed = seed
        self.T0 = float(T0)
        self.P0 = P0
        self.stress0 = stress0
        self.t_damp_T = float(t_damp_T)
        self.t_damp_P = float(t_damp_P)
        self.drag_wavefunctions = drag_wavefunctions
        self.report_callback = report_callback
        self.add_child(
            "thermostat", Thermostat, thermostat, checkpoint_in, dynamics=self
        )

    def run(self, system: qp.System) -> None:
        self.system = system
        self.masses = Dynamics.get_masses(system.ions)
        self.stepper = Stepper(self.system, drag_wavefunctions=self.drag_wavefunctions)

        # Initial velocity and acceleration:
        if not self.system.ions.velocities.norm().item():  # velocities not read in
            self.system.ions.velocities = self.thermal_velocities(self.T0, self.seed)
        velocity = Gradient(ions=self.system.ions.velocities)
        acceleration = self.get_acceleration()

        # MD loop
        for i_iter in range(self.n_steps + 1):
            self.report(i_iter, velocity)
            if i_iter == self.n_steps:
                break

            # First half-step velocity update
            velocity = self.thermostat.step(velocity, acceleration, 0.5 * self.dt)

            # Position and position-dependent acceleration update
            self.stepper.step(velocity, self.dt)
            acceleration = self.get_acceleration()

            # Second half-step velocity update
            velocity = self.thermostat.step(velocity, acceleration, 0.5 * self.dt)

    def thermal_velocities(self, T: float, seed: int) -> torch.Tensor:
        """Thermal velocity distribution at `T`, randomized with `seed`."""
        generator = torch.Generator(device=qp.rc.device)
        generator.manual_seed(seed)
        velocities = (
            torch.randn(
                *self.system.ions.velocities.shape,
                generator=generator,
                device=qp.rc.device,
            )
            / self.masses.sqrt()
        )
        self.comm.Bcast(qp.utils.BufferView(velocities))
        # TODO: account for center of mass, constraints
        # Normalize to set temperature:
        T_current = self.get_T(self.get_KE(velocities))
        velocities *= np.sqrt(T / T_current)
        return velocities

    def get_acceleration(self) -> Gradient:
        """Obtain forces using the stepper and calculate accelerations."""
        energy, gradient = self.stepper.compute(require_grad=True)
        assert gradient is not None
        return Gradient(ions=(-gradient.ions / self.masses))

    def report(self, i_iter: int, velocity: Gradient) -> None:
        # Update velocity-dependent quantities:
        self.KE = self.get_KE(velocity.ions)
        self.T = self.get_T(self.KE)
        self.stress = self.get_stress(velocity.ions)
        self.P = Dynamics.get_pressure(self.stress)
        # Report positions, forces, stresses etc.:
        self.stepper.report(total_stress=self.stress)
        if self.report_callback is not None:
            self.report_callback(self, i_iter)
        E = self.system.energy
        qp.log.info(
            f"Dynamics: {i_iter}  {E.name}: {float(E):+.11f}"
            f"  KE: {self.KE:.6f}  T: {Unit.convert(self.T, 'K')}"
            f"  P: {'null' if (self.P is None) else Unit.convert(self.P, 'bar')}"
            f"  t[s]: {qp.rc.clock():.2f}"
        )

    @staticmethod
    def get_masses(ions: qp.ions.Ions) -> torch.Tensor:
        """Collect the masses of all ions as an n_ions x 1 tensor."""
        atomic_weights = np.empty(ions.n_ions)
        for ion_slice, symbol in zip(ions.slices, ions.symbols):
            atomic_weights[ion_slice] = ATOMIC_WEIGHTS[ATOMIC_NUMBERS[symbol]]
        # Convert to atomic units (in terms of m_e):
        amu = float(Unit(1.0, "amu"))
        return torch.tensor(atomic_weights, device=qp.rc.device).unsqueeze(1) * amu

    def get_stress(self, velocity: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute total stress tensor including ion `velocity` contributions."""
        lattice = self.system.lattice
        if not lattice.compute_stress:
            return None
        kinetic_stress = (-1.0 / lattice.volume) * torch.einsum(
            "a, ai, aj -> ij", self.masses.squeeze(), velocity, velocity
        )
        return kinetic_stress + self.system.lattice.stress

    @staticmethod
    def get_pressure(stress: Optional[torch.Tensor]) -> Optional[float]:
        if stress is None:
            return None
        return (-1.0 / 3) * torch.trace(stress).item()

    def get_KE(self, velocity: torch.Tensor) -> float:
        """Compute kinetic energy from ion `velocity`."""
        return 0.5 * (self.masses * velocity.square()).sum().item()

    def get_T(self, KE: float) -> float:
        """Compute temperature from kinetic energy `KE`."""
        return KE / (0.5 * self.nDOF)

    @property
    def nDOF(self) -> int:
        """Number of degrees of freedom in the dynamics."""
        return 3 * len(self.masses)  # TODO: account for center of mass, constraints
