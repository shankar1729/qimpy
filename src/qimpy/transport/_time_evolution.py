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
    save_rho_interval: int  #: Save rho every so many steps
    write_rho: bool  #: whether to write rho to checkpoint file
    n_collate: int  #: Collect these many save steps into a single checkpoint
    integrator: str  #: Time-step style used for integration

    def __init__(
        self,
        *,
        dt: float = 0.0,
        dt_save: float,
        t_max: float,
        n_collate: int,
        integrator: str = "RK2",
        write_rho: bool = False,
        dt_save_rho: float = 0.0,
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
        t_max
            :yaml:`Stop evolution at this time.`
        n_collate
            :yaml:`Number of save-steps to collect into each checkpoint file.`
            Collecting together several saves can substantially improve performance
            by amortizing the latency associated with disk I/O and GPU transfers.
            The results in the checkpoint have an additional outermost dimension
            corresponding to the number of collated steps.
        integrator
            :yaml:`Integrator for time-stepping: RK2 or RK4.`
        dt_save_rho
            :yaml:`Time interval at which to save rho.`
        geometry
            Corresponding geometry from which maximum time step is determined
        """
        super().__init__()
        if checkpoint_in:
            i_step_initial = checkpoint_in[0]["/geometry"].attrs["i_step"][-1]
            t_initial = checkpoint_in[0]["/geometry"].attrs["t"][-1]
            log.info(f"Setting initial step to {i_step_initial}, initial time to {t_initial}")
        else:
            i_step_initial,t_initial = 0, 0.0
        self.t = t_initial
        dt_max = geometry.dt_max
        if dt == 0.0:
            if dt_max == 0.0:
                raise InvalidInputException(
                    "Specify dt explicitly, because dt_max is not available"
                )
            dt = dt_max
            log.info(f"Setting time step dt = {dt_max = :.4g}")
        elif dt_max and (dt > dt_max):
            raise InvalidInputException(f"{dt = } must be smaller than {dt_max = }")
        self.dt = dt
        self.i_step = i_step_initial
        self.n_steps = max(1, int(np.round(t_max/ self.dt))) + i_step_initial
        self.save_interval = max(1, int(np.round(dt_save / self.dt)))
        self.write_rho = write_rho
        if write_rho: 
            if dt_save_rho > 0:
                self.save_rho_interval = max(self.save_interval, int(np.round(dt_save_rho / self.dt)))
                if self.save_rho_interval % self.save_interval != 0:
                    raise InvalidInputException(
                        "save_rho_interval should be divisible by save_interval"
                    )
            else:
                self.save_rho_interval = self.save_interval * n_collate # Only save one rho in a file
                log.info(f"rho saving interval is set to {self.save_rho_interval}")
            if (self.n_steps // self.save_interval) * self.save_interval % self.save_rho_interval != 0:
                log.info("The rho at the last save step will not be saved. " 
                         "For a proper restart, please let "
                         "the last save step also divisible by save_rho_interval.")
        self.n_collate = n_collate
        self.integrator = integrator
        if integrator not in {"RK2", "RK4"}:
            raise InvalidInputException(f"Unrecognized {integrator = }")

    def time_step(self, geometry: Geometry) -> None:
        """Second-order correct time step."""
        t = self.t
        dt = self.dt
        rho0 = geometry.rho
        if self.integrator == "RK2":
            rho_half = rho0 + (0.5 * dt) * geometry.rho_dot(rho0, t)
            geometry.rho = rho0 + dt * geometry.rho_dot(rho_half, t + 0.5 * dt)
        elif self.integrator == "RK4":
            k1 = geometry.rho_dot(rho0, t)
            k2 = geometry.rho_dot(rho0 + (0.5 * dt) * k1, t + 0.5 * dt)
            k3 = geometry.rho_dot(rho0 + (0.5 * dt) * k2, t + 0.5 * dt)
            k4 = geometry.rho_dot(rho0 + dt * k3, t + dt)
            geometry.rho = rho0 + (dt / 6.0) * (k1 + 2 * (k2 + k3) + k4)
        else:
            raise KeyError(f"Unrecognized integrator = {self.integrator}")

    def run(self, transport: qimpy.transport.Transport) -> None:
        """Run time evolution loop, checkpointing at regular intervals."""
        i_collate = 0
        while self.i_step <= self.n_steps:
            if self.i_step % self.save_interval == 0:
                update_rho = (self.write_rho and self.i_step % self.save_rho_interval == 0)
                transport.geometry.update_stash(self.i_step, self.t, update_rho)
                i_collate += 1
                log.info(f"Stashed results of step {self.i_step}")
                if update_rho:
                    log.info(f"Stashed rho of step {self.i_step}")
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
