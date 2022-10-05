from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from typing import Union, Optional
from qimpy.rc import MPI


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

    def run(self, system: qp.System) -> None:
        self.system = system
        # TODO: actually implement dynamics
