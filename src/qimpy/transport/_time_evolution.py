from __future__ import annotations
from typing import Union
from dataclasses import dataclass

import numpy as np
import torch
import h5py
from scipy import optimize

import qimpy
from qimpy import TreeNode, log, rc
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from .geometry import Geometry, TensorList


class TimeEvolution(TreeNode):
    """Time evolution parameters."""

    t: float  #: Current time
    dt: float  #: Time step (set automatically if zero)
    i_step: int  #: Current step number
    i_step_initial: int  #: Initial step number for current job (not zero if continued)
    n_steps: int  #: Number of steps
    save_interval: int  #: Save results every so many steps
    n_collate: int  #: Collect these many save steps into a single checkpoint
    integrator: str  #: Time-step style used for integration
    steady_state: dict[str, Union[str, float]]

    def __init__(
        self,
        *,
        i_step: int = 0,
        t: float = 0.0,
        dt: float = 0.0,
        dt_save: float = 0.0,
        t_max: float = 0.0,
        n_collate: int = 0,
        integrator: str = "RK2",
        steady_state: dict[str, Union[str, float]] = None,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        geometry: Geometry,
    ) -> None:
        """
        Initialize time evolution parameters

        Parameters
        ----------
        i_step
            Initial step index, used for continuing from checkpoint.
        t
            Initial time, used for continuing from checkpoint.
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
        geometry
            Corresponding geometry from which maximum time step is determined
        """
        super().__init__()
        self.steady_state = steady_state
        if self.steady_state:
            self.rho0_path = self.steady_state["rho0_path"]
            self.method = self.steady_state["method"]
            self.nit = self.steady_state["nit"]
            self.nit_save = self.steady_state["nit_save"]
            self.t = 0.0
            log.info("Steady state mode")
        else:
            self.i_step_initial = int(i_step)
            self.i_step = self.i_step_initial
            self.t = float(t)
            if i_step:
                log.info(f"Continuing from step {i_step}")
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
            self.dt = float(dt)
            self.n_steps = max(1, int(np.round(t_max / self.dt)))
            self.save_interval = max(1, int(np.round(dt_save / self.dt)))
            self.n_collate = int(n_collate)
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

    def steady_state_sol(
        self, transport: qimpy.transport.Transport, geometry: Geometry
    ) -> None:
        with h5py.File(
            self.rho0_path,
            "r",
        ) as cp:
            cp_geom = cp["/geometry"]
            cp_quad = cp_geom["quad0"]
            rho_f = np.array(cp_quad["rho"])
            t_f = cp["/time_evolution"].attrs["t"]
        rho_f = torch.from_numpy(rho_f).to(rc.device)
        if isinstance(transport.material, qimpy.transport.material.ab_initio.AbInitio):
            ph = transport.material.packed_hermitian
            phase = transport.material.schrodingerV(t_f)
            rho_f = rho_f.unflatten(
                -1,
                (
                    transport.material.nk_mine,
                    transport.material.n_bands,
                    transport.material.n_bands,
                ),
            )
            rho0_I = ph.unpack(rho_f)  # interaction picture, unpacked to complex
            rho0_S = rho0_I * phase
            rho0 = ph.pack(rho0_S).flatten(-3, -1)
            rho0 = rho0.to(rc.cpu).numpy()
            rho0 = rho0.flatten()

        rho_shape = geometry.patches[0].rho_shape
        rho = TensorList(
            v.view(rho_shape) for v in [torch.from_numpy(rho0).to(rc.device)]
        )
        rho_dot = geometry.rho_dot(rho, t=0.0)
        RHO_SCALE = np.abs(rho0).max()
        T_SCALE = 1.0 / (torch.max(torch.abs(rho_dot[0])).item() / RHO_SCALE)

        steady_state_root_fn = SteadyStateRootFunction(
            geometry, RHO_SCALE, T_SCALE, self.nit, self.nit_save
        )
        optimizer = optimize.root(
            steady_state_root_fn,
            rho0 / RHO_SCALE,
            method=self.method,
            callback=steady_state_root_fn.callback_fn,
            options={"disp": True, "nit": self.nit},
        )
        log.info(optimizer)
        log.info(f"{steady_state_root_fn.n_calls = }")
        geometry.rho = TensorList(
            v.view(rho_shape)
            for v in [torch.from_numpy(optimizer.x * RHO_SCALE).to(rc.device)]
        )

    def run(self, transport: qimpy.transport.Transport) -> None:
        """Run time evolution loop, checkpointing at regular intervals."""
        if self.steady_state:
            transport.geometry.update_stash(0, self.t)
            log.info("Stashed results of iteration 0")
            if isinstance(
                transport.material, qimpy.transport.material.ab_initio.AbInitio
            ):
                transport.material.include_coherent = True
            self.steady_state_sol(transport, transport.geometry)
            if self.nit % self.nit_save > 0:
                transport.geometry.update_stash(self.nit, self.t)
                log.info(f"Stashed results of iteration {self.nit}")
            transport.save(self.nit)
        else:
            i_collate = 0
            while self.i_step <= self.n_steps:
                should_save = (self.i_step > self.i_step_initial) or (self.i_step == 0)
                if self.i_step % self.save_interval == 0 and should_save:
                    transport.geometry.update_stash(self.i_step, self.t)
                    i_collate += 1
                    log.info(f"Stashed results of step {self.i_step}")
                    if i_collate == self.n_collate or self.i_step == 0:
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

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["t"] = self.t
        if self.steady_state:
            attrs["rho0_path"] = self.rho0_path
            attrs["method"] = self.method
            attrs["nit"] = self.nit
            attrs["nit_save"] = self.nit_save
        else:
            attrs["i_step"] = self.i_step
            attrs["dt"] = self.dt
            attrs["dt_save"] = self.save_interval * self.dt
            attrs["t_max"] = self.n_steps * self.dt
            attrs["n_collate"] = self.n_collate
            attrs["integrator"] = self.integrator
        return list(attrs.keys())


@dataclass
class SteadyStateRootFunction:
    geometry: Geometry
    RHO_SCALE: float = 1.0e-7
    T_SCALE: float = 1.0e4
    nit: int = 0
    nit_save: int = 0
    n_calls: int = 0
    iter: int = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # TODO: general tensor list case
        shape = self.geometry.patches[0].rho_shape
        rho = TensorList(
            v.view(shape) for v in [torch.from_numpy(x * self.RHO_SCALE).to(rc.device)]
        )
        rho_dot = self.geometry.rho_dot(rho, t=0.0)
        self.n_calls += 1
        result = rho_dot[0].flatten().to(rc.cpu).numpy() / (
            self.RHO_SCALE / self.T_SCALE
        )
        log.info(
            f"Norm(f(x): {np.linalg.norm(result)}, Max(f(x)): {np.abs(result).max()}, n_calls: {self.n_calls} at t[s]: {rc.clock():.2f}"
        )
        return result

    def callback_fn(self, x: np.ndarray, f: np.ndarray):
        self.iter += 1
        if (self.iter % self.nit_save) == 0:
            shape = self.geometry.patches[0].rho_shape
            self.geometry.rho = TensorList(
                v.view(shape)
                for v in [torch.from_numpy(x * self.RHO_SCALE).to(rc.device)]
            )
            self.geometry.update_stash(self.iter, 0.0)
            log.info(f"Stashed results of iteration {self.iter}")
