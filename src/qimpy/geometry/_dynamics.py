from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from typing import Union, Optional, Callable
from mpi4py import MPI
from ._stepper import Stepper
from ._history import History
from ._gradient import Gradient
from .thermostat import Thermostat
from qimpy.ions.symbols import ATOMIC_WEIGHTS, ATOMIC_NUMBERS
from qimpy.utils import Unit, UnitOrFloat, Checkpoint, CpPath, CpContext


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
    stress0: torch.Tensor  #: Stress set point (used only if `lattice.movable`)
    isotropic: bool  #: Whether lattice change is isotropic (NPT, vs. N-stress-T mode)
    t_damp_T: float  #: Thermostat damping time
    t_damp_P: float  #: Barostat damping time
    B0: float  #: Characteristic bulk modulus for Berendsen barostat
    langevin_gamma: float  #: Damping rate for Langevin thermostat
    drag_wavefunctions: bool  #: Whether to drag atomic components of wavefunctions
    P: Optional[float]  #: Current pressure (available if `lattice.compute_stress`)
    T: float  #: Current temperature
    KE: float  #: Current kinetic energy
    stress: Optional[torch.Tensor]  #: Current stress including kinetic contributions
    history: Optional[History]  #: Utility to save trajectory data
    report_callback: Optional[Callable[[Dynamics, int], None]]  #: Callback from report
    i_iter_start: int  #: Starting iteration number (when continuing from checkpoint)

    def __init__(
        self,
        *,
        comm: MPI.Comm,
        dt: float,
        n_steps: int,
        thermostat: Union[Thermostat, dict, str, None] = None,
        seed: int = 1234,
        T0: UnitOrFloat = Unit(298.0, "K"),
        P0: UnitOrFloat = Unit(1.0, "bar"),
        stress0: Optional[Union[np.ndarray, torch.Tensor]] = None,
        t_damp_T: UnitOrFloat = Unit(50.0, "fs"),
        t_damp_P: UnitOrFloat = Unit(100.0, "fs"),
        drag_wavefunctions: bool = True,
        save_history: bool = True,
        report_callback: Optional[Callable[[Dynamics, int], None]] = None,
        checkpoint_in: CpPath = CpPath(),
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
            :yaml:`Pressure set point for NPT, if lattice.movable is True.`
            Note that this is overridden by `stress0`, if that is specified.
        stress0
            :yaml:`Stress set point for N-stress-T, if lattice.movable is True.`
            If specified and lattice.movable, strain tensor will fluctuate
            during dynamics, instead of only volume in NPT mode.
            (Set to None and specify `P0` instead for NPT mode.)
        t_damp_T
            :yaml:`Thermostat damping time.`
        t_damp_P
            :yaml:`Barostat damping time.`
        drag_wavefunctions
            :yaml:`Whether to drag atomic components of wavefunctions.`
        save_history
            :yaml:`Whether to save history along the trajectory.`
            Saved quantities include positions, forces, velocities, temperature,
            pressure, potential and total stress (if available),
            and lattice (if movable).
        report_callback
            Optional function to call at each step during `report`.
            Use this to perform additional reporting / data collection.
            The functional will be called as `report_callback(dynamics, i_iter)`.
        """
        super().__init__()
        self.comm = comm
        self.dt = dt
        self.n_steps = n_steps
        self.seed = seed
        self.T0 = float(T0)
        if stress0 is None:
            self.isotropic = True
            self.stress0 = -float(P0) * torch.eye(3, device=qp.rc.device)
        else:
            self.isotropic = False
            self.stress0 = (
                stress0
                if isinstance(stress0, torch.Tensor)
                else torch.tensor(stress0, device=qp.rc.device)
            )
            assert self.stress0.shape == (3, 3)
        self.t_damp_T = float(t_damp_T)
        self.t_damp_P = float(t_damp_P)
        self.drag_wavefunctions = drag_wavefunctions
        if save_history:
            self.add_child(
                "history", History, {}, checkpoint_in, comm=comm, n_max=(n_steps + 1)
            )
        else:
            self.history = None
        self.report_callback = report_callback
        self.i_iter_start = checkpoint_in.attrs["i_iter"] if checkpoint_in else 0
        self.add_child(
            "thermostat", Thermostat, thermostat, checkpoint_in, dynamics=self
        )

    def run(self, system: qp.System) -> None:
        self.system = system
        self.masses = Dynamics.get_masses(system.ions)
        stepper = Stepper(
            self.system,
            drag_wavefunctions=self.drag_wavefunctions,
            isotropic=self.isotropic,
        )
        self.stepper = stepper
        thermostat_method = self.thermostat.method

        # Initial velocity and acceleration:
        if (ion_velocities := system.ions.velocities) is None:  # velocities not read in
            ion_velocities = self.thermal_velocities(self.T0, self.seed)
        velocity = self.create_gradient(ion_velocities)
        if (strain_rate := system.lattice.strain_rate) is not None:
            velocity.lattice = strain_rate
        thermostat_method.get_velocity(velocity)
        acceleration = self.get_acceleration()

        # MD loop
        for i_iter in range(self.i_iter_start, self.n_steps + 1):
            self.report(i_iter, velocity)
            if i_iter == self.n_steps:
                break

            # First half-step velocity update
            velocity = thermostat_method.step(velocity, acceleration, 0.5 * self.dt)
            velocity = stepper.constrain(velocity)

            # Position and position-dependent acceleration update
            stepper.step(velocity, self.dt)
            acceleration = self.get_acceleration()

            # Second half-step velocity update
            velocity = thermostat_method.step(velocity, acceleration, 0.5 * self.dt)
            velocity = stepper.constrain(velocity)

        # Check point at end:
        if system.checkpoint_out:
            with Checkpoint(system.checkpoint_out, writable=True) as cp:
                system.save_checkpoint(CpPath(cp), CpContext("end"))

    def thermal_velocities(self, T: float, seed: int) -> torch.Tensor:
        """Thermal velocity distribution at `T`, randomized with `seed`."""
        generator = torch.Generator(device=qp.rc.device)
        generator.manual_seed(seed)
        velocities = (
            torch.randn(
                *self.system.ions.positions.shape,
                generator=generator,
                device=qp.rc.device,
            )
            / self.masses.sqrt()
        )
        self.comm.Bcast(qp.utils.BufferView(velocities))
        velocities = self.stepper.constrain(self.create_gradient(velocities)).ions
        # Normalize to set temperature:
        T_current = self.get_T(self.get_KE(velocities))
        velocities *= np.sqrt(T / T_current)
        return velocities

    def get_acceleration(self) -> Gradient:
        """Acceleration due to ionic forces."""
        energy, gradient = self.stepper.compute(require_grad=True)
        assert gradient is not None
        return self.create_gradient(-gradient.ions / self.masses)

    def report(self, i_iter: int, velocity: Gradient) -> None:
        # Update velocities stored within each component:
        system = self.system
        system.ions.velocities = velocity.ions
        system.lattice.strain_rate = velocity.lattice
        self.thermostat.method.set_velocity(velocity)

        # Update velocity-dependent quantities:
        self.KE = self.get_KE(velocity.ions)
        self.T = self.get_T(self.KE)
        self.stress = self.get_stress(velocity.ions)
        self.P = Dynamics.get_pressure(self.stress)

        # Checkpoint:
        if system.checkpoint_out:
            with Checkpoint(system.checkpoint_out, writable=True) as cp:
                system.save_checkpoint(CpPath(cp), CpContext("geometry", i_iter))

        # Report positions, forces, stresses etc.:
        self.stepper.report(total_stress=self.stress)
        if self.report_callback is not None:
            self.report_callback(self, i_iter)
        E = system.energy
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
        return kinetic_stress + self.system.lattice.stress.detach()

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
        return 3 * len(self.masses) - 3  # TODO: account for constraints

    def create_gradient(self, ions: torch.Tensor) -> Gradient:
        """Create gradient from ionic part, initializing optional parts correctly."""
        gradient = Gradient(ions=ions)
        if self.system.lattice.movable:
            gradient.lattice = torch.zeros((3, 3), device=qp.rc.device)
        self.thermostat.method.initialize_gradient(gradient)
        return gradient

    def _save_checkpoint(self, cp_path: CpPath, context: CpContext) -> list[str]:
        stage, i_iter = context
        saved_list = ["i_iter"]
        cp_path.attrs["i_iter"] = i_iter if (stage == "geometry") else self.n_steps
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
                assert ions.velocities is not None
                history.add("velocities", ions.velocities)
                history.add("T", self.T)
                if lattice.movable:
                    history.add("Rbasis", lattice.Rbasis)
                if self.stress is not None:
                    assert self.P is not None
                    history.add("P", self.P)
                    history.add("stress", lattice.stress.detach())  # potential only
                    history.add("stress_total", self.stress)  # including kinetic
        return saved_list
