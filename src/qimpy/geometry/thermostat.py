"""Geometry actions: relaxation and dynamics."""
from __future__ import annotations
from typing import Union, Protocol
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

    thermostat_method: ThermostatMethod

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
            "thermostat_method",
            checkpoint_in,
            ChildOptions("nve", NVE, nve, dynamics=dynamics),
            ChildOptions("nose_hoover", NoseHoover, nose_hoover, dynamics=dynamics),
            ChildOptions("berendsen", Berendsen, berendsen, dynamics=dynamics),
            ChildOptions("langevin", Langevin, langevin, dynamics=dynamics),
            have_default=True,
        )

    def step(self, velocity: Gradient, acceleration: Gradient, dt: float) -> Gradient:
        """Return velocity after `dt`, given current `velocity` and `acceleration`."""
        return self.thermostat_method.step(velocity, acceleration, dt)


class ThermostatMethod(Protocol):
    """Class requirements to use as a thermostat method."""

    def step(self, velocity: Gradient, acceleration: Gradient, dt: float) -> Gradient:
        """Return velocity after `dt`, given current `velocity` and `acceleration`."""
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
        """
        super().__init__()
        self.dynamics = dynamics
        self.B0 = float(B0)

    def step(self, velocity: Gradient, acceleration: Gradient, dt: float) -> Gradient:
        """Return velocity after `dt`, given current `velocity` and `acceleration`."""
        dynamics = self.dynamics
        nDOF = 3 * len(dynamics.masses)  # TODO
        KE_target = 0.5 * nDOF * dynamics.T0
        KE = 0.5 * (dynamics.masses * velocity.ions.square()).sum()
        gamma = 0.5 * (KE / KE_target - 1.0) / dynamics.t_damp_T
        accel_thermo = Gradient(ions=(-gamma * velocity.ions))
        return velocity + (acceleration + accel_thermo) * dt


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

    def step(self, velocity: Gradient, acceleration: Gradient, dt: float) -> Gradient:
        """Return velocity after `dt`, given current `velocity` and `acceleration`."""
        # Generate MPI-consistent random numbers for noise term:
        dynamics = self.dynamics
        rand = torch.randn_like(velocity.ions)
        self.dynamics.comm.Bcast(qp.utils.BufferView(rand))
        # Compute acceleration from stochastic and damping terms:
        gamma = 1.0 / dynamics.t_damp_T
        variances = 2 * gamma * dynamics.T0 / (dt * dynamics.masses)
        accel_thermo = Gradient(ions=(rand * variances.sqrt() - gamma * velocity.ions))
        return velocity + (acceleration + accel_thermo) * dt
