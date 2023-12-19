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
    save_interval: int  #: Save checkpoint every so many steps
    geometry: Geometry

    def __init__(
        self,
        *,
        dt: float = 0.0,
        dt_save: float,
        t_max: float,
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
            :yaml:`Time interval at which to save checkpoints.`
            This will be rounded to the nearest multiple of `dt` to ensure
            that the checkpoints are written at uniform intervals.
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
        elif dt < geometry.dt_max:
            raise InvalidInputException(f"{dt = } must be smaller than {dt_max = }")
        self.dt = dt
        self.i_step = 0
        self.n_steps = max(1, int(np.round(t_max / self.dt)))
        self.save_interval = max(1, int(np.round(dt_save / self.dt)))
        self.geometry = geometry

    def run(self, transport: qimpy.transport.Transport) -> None:
        """Run time evolution loop, checkpointing at regular intervals."""
        while self.i_step <= self.n_steps:
            if self.i_step % self.save_interval == 0:
                transport.save(self.i_step)

            if self.i_step == self.n_steps:
                break

            transport.geometry.time_step(self.dt)

            log.info(
                f"Step {self.i_step} done of {self.n_steps} at t[s]: {rc.clock():.2f}"
            )
            self.i_step += 1
            self.t += self.dt
