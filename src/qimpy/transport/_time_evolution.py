from __future__ import annotations

import numpy as np

import qimpy
from qimpy import TreeNode, log, rc
from qimpy.io import CheckpointPath, InvalidInputException
from .geometry import Geometry


class TimeEvolution(TreeNode):
    """Time evolution parameters."""

    t: float  #: Current time
    dt: float  #: Time step (set automatically if zero)
    i_step: int  #: Current step number
    n_steps: int  #: Number of steps
    save_interval: int  #: Save results every so many steps
    n_collate: int  #: Collect these many save steps into a single checkpoint

    def __init__(
        self,
        *,
        dt: float = 0.0,
        dt_save: float,
        t_max: float,
        n_collate: int,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        geometry: Geometry,
    ) -> None:
        """
        Initialize time evolution parameters

        Parameters
        ----------
        dt
            :yaml:`Time step for evolution.`
            If zero, this is set to the maximum stable time step for advection.
        dt_save
            :yaml:`Time interval at which to save results.`
            This will be rounded to the nearest multiple of `dt` to ensure
            that the results are written at uniform intervals.
        n_collate
            :yaml:`Number of save-steps to collect into each checkpoint file.`
            Collecting together several saves can substantially improve performance
            by amortizing the latency associated with disk I/O and GPU transfers.
            The results in the checkpoint have an additional outermost dimension
            corresponding to the number of collated steps.
        t_max
            :yaml:`Stop evolution at this time.`
        geometry
            Corresponding geometry from which maximum time step is determined
        """
        super().__init__()
        self.t = 0.0
        dt_max = geometry.dt_max
        if dt == 0.0:
            dt = dt_max
            log.info(f"Setting time step dt = {dt_max = :.4g}")
        elif dt > geometry.dt_max:
            raise InvalidInputException(f"{dt = } must be smaller than {dt_max = }")
        self.dt = dt
        self.i_step = 0
        self.n_steps = max(1, int(np.round(t_max / self.dt)))
        self.save_interval = max(1, int(np.round(dt_save / self.dt)))
        self.n_collate = n_collate

    def time_step(self, geometry: Geometry) -> None:
        """Second-order correct time step."""
        dt = self.dt
        rho0 = geometry.rho
        rho_half = rho0 + (0.5 * dt) * geometry.rho_dot(rho0, self.t)
        geometry.rho = rho0 + dt * geometry.rho_dot(rho_half, self.t + 0.5 * dt)

    def run(self, transport: qimpy.transport.Transport) -> None:
        """Run time evolution loop, checkpointing at regular intervals."""
        i_collate = 0
        while self.i_step <= self.n_steps:
            if self.i_step % self.save_interval == 0:
                transport.geometry.update_stash(self.i_step, self.t)
                i_collate += 1
                log.info(f"Stashed results of step {self.i_step}")
                if i_collate == self.n_collate:
                    transport.save(self.i_step)
                    i_collate = 0

            if self.i_step == self.n_steps:
                if i_collate:
                    transport.save(self.i_step)
                break

            self.time_step(transport.geometry)

            log.info(
                f"Step {self.i_step} done of {self.n_steps} at t[s]: {rc.clock():.2f}"
            )
            self.i_step += 1
            self.t += self.dt
