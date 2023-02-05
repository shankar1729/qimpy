"""Geometry actions: relaxation and dynamics."""
from __future__ import annotations
from typing import Union, Protocol, Callable
import torch
import qimpy as qp
from qimpy.utils import Unit, UnitOrFloat
from ._gradient import Gradient

# List exported symbols for doc generation
__all__ = [
    "Thermostat",
    "ThermostatMethod",
    "NVE",
    "NoseHoover",
    "Berendsen",
    "Langevin",
]


class Thermostat(qp.TreeNode):
    """Select between possible geometry actions."""

    method: ThermostatMethod

    def __init__(
        self,
        *,
        dynamics: qp.geometry.Dynamics,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        nve: Union[dict, NVE, None] = None,
        nose_hoover: Union[dict, NoseHoover, None] = None,
        berendsen: Union[dict, Berendsen, None] = None,
        langevin: Union[dict, Langevin, None] = None,
    ) -> None:
        """Specify one of the supported thermostat methods.
        Defaults to `NVE` if none specified.

        Parameters
        ----------
        nve
            :yaml:`No thermostat (or barostat), i.e. NVE ensemble.`
        nose_hoover
            :yaml:`Nose-Hoover thermostat and/or barostat.`
        berendsen
            :yaml:`Berendsen velocity-rescaling thermostat and/or barostat.`
        langevin
            :yaml:`Langevin stochastic thermostat and/or barostat.`
        """
        super().__init__()
        ChildOptions = qp.TreeNode.ChildOptions
        self.add_child_one_of(
            "method",
            checkpoint_in,
            ChildOptions("nve", NVE, nve, dynamics=dynamics),
            ChildOptions("nose_hoover", NoseHoover, nose_hoover, dynamics=dynamics),
            ChildOptions("berendsen", Berendsen, berendsen, dynamics=dynamics),
            ChildOptions("langevin", Langevin, langevin, dynamics=dynamics),
            have_default=True,
        )


class ThermostatMethod(Protocol):
    """Class requirements to use as a thermostat method."""

    def step(self, velocity: Gradient, acceleration: Gradient, dt: float) -> Gradient:
        """Return velocity after `dt`, given current `velocity` and `acceleration`."""
        ...

    def initialize_gradient(self, gradient: Gradient, lattice_movable: bool) -> None:
        """Initialize optional terms in `gradient` needed by this thermostat to zero."""
        ...


class NVE(qp.TreeNode):
    """No thermostat (or barostat), i.e. NVE ensemble."""

    dynamics: qp.geometry.Dynamics

    def __init__(
        self,
        *,
        dynamics: qp.geometry.Dynamics,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
    ) -> None:
        super().__init__()
        self.dynamics = dynamics

    def step(self, velocity: Gradient, acceleration: Gradient, dt: float) -> Gradient:
        """Return velocity after `dt`, given current `velocity` and `acceleration`."""
        return velocity + acceleration * dt

    def initialize_gradient(self, gradient: Gradient, lattice_movable: bool) -> None:
        """No optional `gradient` terms for this thermostat method."""


class NoseHoover(qp.TreeNode):
    """Nose-Hoover thermostat and/or barostat."""

    dynamics: qp.geometry.Dynamics
    chain_length_T: int  #: Nose-Hoover chain length for thermostat
    chain_length_P: int  #: Nose-Hoover chain length for barostat

    def __init__(
        self,
        *,
        dynamics: qp.geometry.Dynamics,
        chain_length_T: int = 3,
        chain_length_P: int = 3,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
    ) -> None:
        """
        Specify thermostat parameters.

        Parameters
        ----------

        chain_length_T
            :yaml:`Nose-Hoover chain length for thermostat.`
        chain_length_P
            :yaml:`Nose-Hoover chain length for barostat.`
        """
        super().__init__()
        self.dynamics = dynamics
        self.chain_length_T = chain_length_T
        self.chain_length_P = chain_length_P

    def step(self, velocity: Gradient, acceleration: Gradient, dt: float) -> Gradient:
        """Return velocity after `dt`, given current `velocity` and `acceleration`."""
        raise NotImplementedError

    def initialize_gradient(self, gradient: Gradient, lattice_movable: bool) -> None:
        """Initialize `thermostat` and, if needed, `barostat` terms in `gradient`."""
        gradient.thermostat = torch.zeros(self.chain_length_T, device=qp.rc.device)
        if lattice_movable:
            gradient.barostat = torch.zeros(self.chain_length_P, device=qp.rc.device)


class Berendsen(qp.TreeNode):
    """Berendsen velocity-rescaling thermostat and/or barostat."""

    dynamics: qp.geometry.Dynamics

    def __init__(
        self,
        *,
        dynamics: qp.geometry.Dynamics,
        B0: UnitOrFloat = Unit(2.2, "GPa"),
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
    ) -> None:
        """
        Specify thermostat parameters.

        Parameters
        ----------

        B0
            :yaml:`Characteristic bulk modulus for Berendsen barostat.`
            Default value corresponds to water, which should be acceptable for
            typical liquid simulations, but maybe too small for most solids.
            Make sure to set correct order of magnitude of `B0` in order for
            pressure to be equilibrated on the expected `t_damp_P` time scale.
        """
        super().__init__()
        self.dynamics = dynamics
        self.B0 = float(B0)

    def extra_acceleration(self, velocity: Gradient) -> Gradient:
        """Extra velocity-dependent acceleration due to thermostat."""
        dynamics = self.dynamics
        T = dynamics.get_T(dynamics.get_KE(velocity.ions))
        gamma = 0.5 * (T / dynamics.T0 - 1.0) / dynamics.t_damp_T
        acceleration = Gradient(ions=(-gamma * velocity.ions))
        # Optional barostat contributions:
        if dynamics.system.lattice.movable:
            dstress = dynamics.stress0 - dynamics.get_stress(velocity.ions)
            velocity.lattice = dstress / (dynamics.t_damp_P * self.B0)
            acceleration.ions -= velocity.ions @ velocity.lattice.T
            acceleration.lattice = torch.zeros_like(velocity.lattice)
        return acceleration

    def step(self, velocity: Gradient, acceleration: Gradient, dt: float) -> Gradient:
        """Return velocity after `dt`, given current `velocity` and `acceleration`."""
        return second_order_step(velocity, acceleration, self.extra_acceleration, dt)

    def initialize_gradient(self, gradient: Gradient, lattice_movable: bool) -> None:
        """No optional `gradient` terms for this thermostat method."""


class Langevin(qp.TreeNode):
    """Langevin stochastic thermostat and/or barostat."""

    dynamics: qp.geometry.Dynamics

    def __init__(
        self,
        *,
        dynamics: qp.geometry.Dynamics,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
    ) -> None:
        super().__init__()
        self.dynamics = dynamics

    def extra_acceleration(self, velocity: Gradient) -> Gradient:
        """Extra velocity-dependent acceleration due to thermostat."""
        return velocity * (-1.0 / self.dynamics.t_damp_T)

    def step(self, velocity: Gradient, acceleration: Gradient, dt: float) -> Gradient:
        """Return velocity after `dt`, given current `velocity` and `acceleration`."""
        dynamics = self.dynamics
        # Generate MPI-consistent stochastic acceleration (not velocity dependent):
        rand = torch.randn_like(velocity.ions)
        self.dynamics.comm.Bcast(qp.utils.BufferView(rand))
        variances = 2 * dynamics.T0 / (dynamics.masses * (dynamics.t_damp_T * dt))
        acceleration_noise = Gradient(ions=(rand * variances.sqrt()))
        # Take step including velocity-dependent damping:
        return second_order_step(
            velocity, acceleration + acceleration_noise, self.extra_acceleration, dt
        )

    def initialize_gradient(self, gradient: Gradient, lattice_movable: bool) -> None:
        """No optional `gradient` terms for this thermostat method."""


def second_order_step(
    velocity: Gradient,
    acceleration0: Gradient,
    acceleration: Callable[[Gradient], Gradient],
    dt: float,
) -> Gradient:
    """
    Integrate dv/dt = acceleration0 + acceleration(v) over dt to second order.
    Start from v = velocity at time t, and return velocity at t+dt.
    """
    velocity_half = velocity + (acceleration0 + acceleration(velocity)) * (0.5 * dt)
    return velocity + (acceleration0 + acceleration(velocity_half)) * dt
