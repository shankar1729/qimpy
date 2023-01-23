from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from typing import Union, Optional
from qimpy.rc import MPI
from ._stepper import Stepper
from ._gradient import Gradient
from qimpy.ions.symbols import ATOMIC_WEIGHTS, ATOMIC_NUMBERS


class Dynamics(qp.TreeNode):
    """Molecular dynamics of ions and/or lattice.
    Whether lattice changes is controlled by `lattice.movable`.
    """

    system: qp.System  #: System being optimized currently
    dt: float  #: Time step
    n_steps: int  #: Number of MD steps
    thermostat: Optional[str]  #: Thermostat/barostat method
    T0: float = 298.0 / 3.157e5  #: Initial temperature / temperature set point
    P0: Optional[float] = None  #: Pressure set point
    stress0: Optional[Union[np.ndarray, torch.Tensor]]  #: Stress set point
    t_damp_T: float = 50.0 / 0.02419  #: Thermostat damping time
    t_damp_P: float = 100.0 / 0.02419  #: Barostat damping time
    chain_length_T: int = 3  #: Nose-Hoover chain length for thermostat
    chain_length_P: int = 3  #: Nose-Hoover chain length for barostat
    B0: float = 2.2e9 / 2.942e13  #: Characteristic bulk modulus for Berendsen barostat
    drag_wavefunctions: bool  #: Whether to drag atomic components of wavefunctions

    def __init__(
        self,
        *,
        comm: MPI.Comm,
        lattice: qp.lattice.Lattice,
        dt: float,
        n_steps: int,
        thermostat: Optional[str] = None,
        T0: float = 298.0 / 3.157e5,
        P0: Optional[float] = None,
        stress0: Optional[Union[np.ndarray, torch.Tensor]] = None,
        t_damp_T: float = 50.0 / 0.02419,
        t_damp_P: float = 100.0 / 0.02419,
        chain_length_T: int = 3,
        chain_length_P: int = 3,
        B0: float = 2.2e9 / 2.942e13,
        drag_wavefunctions: bool = True,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        langevin_gamma: Union[float, list[float], torch.Tensor] = 1.0,
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
        T0
            :yaml:`Initial temperature / temperature set point.`
        P0
            :yaml:`Pressure set point.`
        stress0
            :yaml:`Stress set point.`
        t_damp_T
            :yaml:`Thermostat damping time.`
        t_damp_P
            :yaml:`Barostat damping time.`
        chain_length_T
            :yaml:`Nose-Hoover chain length for thermostat.`
        chain_length_P
            :yaml:`Nose-Hoover chain length for barostat.`
        B0
            :yaml:`Characteristic bulk modulus for Berendsen barostat.`
        drag_wavefunctions
            :yaml:`Whether to drag atomic components of wavefunctions.`
        langevin_gamma
            :yaml:`Friction parameter for the Langevin thermostat method.`
        """
        super().__init__()
        self.dt = dt
        self.n_steps = n_steps
        self.thermostat = thermostat
        self.T0 = T0
        self.P0 = P0
        self.stress0 = stress0
        self.t_damp_T = t_damp_T
        self.t_damp_P = t_damp_P
        self.chain_length_T = chain_length_T
        self.chain_length_P = chain_length_P
        self.B0 = B0
        self.drag_wavefunctions = drag_wavefunctions
        self.langevin_gamma = langevin_gamma

        self.thermostat_methods = {"langevin": self.langevin_thermostat}

    def get_accel(self) -> torch.Tensor:
        """Obtain forces using the stepper and calculate acceleration."""
        energy, gradient = self.stepper.compute(require_grad=True)
        assert gradient is not None
        return -gradient.ions / self.atomic_weights

    def langevin_thermostat(self, vel: torch.Tensor) -> torch.Tensor:
        """Implement Langevin thermostat."""
        if isinstance(self.langevin_gamma, list):
            self.langevin_gamma = torch.unsqueeze(
                torch.as_tensor(self.langevin_gamma, device=qp.rc.device), dim=-1
            )
        prefactor = 2 * self.T0 / self.dt
        variances = prefactor * torch.ones_like(
            self.atomic_weights, device=qp.rc.device
        )
        variances *= self.atomic_weights
        variances *= self.langevin_gamma
        accel = (
            torch.normal(
                mean=torch.zeros_like(self.system.ions.velocities, device=qp.rc.device),
                std=torch.sqrt(variances),
            )
            / self.atomic_weights
        )
        accel = accel - self.langevin_gamma * (vel * self.atomic_weights)
        return accel

    def compute_thermostat(self, vel: torch.Tensor) -> torch.Tensor:
        """Compute thermostat for the system."""
        if self.thermostat is None:
            return torch.zeros_like(self.system.ions.velocities)  # Zero for now
        else:
            return self.thermostat_methods[self.thermostat](vel)

    def get_atomic_weights(self) -> torch.Tensor:
        """Initialize the atomic weights for the system."""
        amu = 1822.89  # (Temporary) AMU conversion for mass
        ions = self.system.ions
        atomic_weights = np.empty(ions.n_ions)
        for ion_slice, symbol in zip(ions.slices, ions.symbols):
            atomic_weights[ion_slice] = amu * ATOMIC_WEIGHTS[ATOMIC_NUMBERS[symbol]]
        return torch.tensor(atomic_weights, device=qp.rc.device).unsqueeze(1)

    def run(self, system: qp.System) -> None:
        self.system = system
        self.atomic_weights = self.get_atomic_weights()
        self.stepper = Stepper(self.system, drag_wavefunctions=self.drag_wavefunctions)

        vel = self.system.ions.velocities

        # Initial forces
        accel = self.get_accel()
        accel_thermostat_step1 = self.compute_thermostat(vel)

        # MD loop
        for i in range(self.n_steps):
            qp.log.info(f"Step {i}")

            self.report()

            # Compute first half step
            vel += 0.5 * self.dt * (accel + accel_thermostat_step1)

            # Position and position-dependent acceleration update
            self.stepper.step(Gradient(ions=vel, lattice=None), self.dt)
            accel = self.get_accel()

            # Second half-step estimator
            vel += 0.5 * self.dt * (accel + accel_thermostat_step1)

            # Second half-step correction
            accel_thermostat_step2 = self.compute_thermostat(vel)
            vel += 0.5 * self.dt * (accel_thermostat_step2 - accel_thermostat_step1)

            # Calculate forces for next step
            accel_thermostat_step1 = accel_thermostat_step2

        # Print final state
        self.report()

    def report(self):
        self.stepper.report()
