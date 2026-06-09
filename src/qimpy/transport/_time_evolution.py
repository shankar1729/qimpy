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
        positivity: bool = False,
        steady_state: dict[str, Union[str, float]] = None,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        dt_max_sources: list,
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
            :yaml:`Integrator for time-stepping: RK2, RK4 or SSPRK3.`
            SSPRK3 is the 3-stage strong-stability-preserving Runge-Kutta scheme
            required for the positivity guarantee below.
        positivity
            :yaml:`Enforce a non-negative density via a Zhang-Shu scaling limiter.`
            Applied to the m=0 (density) channel after each stage; conservative
            (preserves cell averages) and order-preserving. The rigorous
            maximum-principle guarantee holds with integrator SSPRK3 and a
            positivity-preserving CFL; with RK2/RK4 it is applied best-effort to
            the end-of-step state only.
        steady_state
            :yaml:`Steady state options.`
            EXPERIMENTAL: works only with a single process and geometry domain for now.
            Specify a dictionary of `rho0_path` for initial state, `method` for solver,
            `nit` for number of iterations and `nit_save` for iterations to save at.
        dt_max_sources
            List of objects with dt_max that determine maximum safe time step.
        """
        super().__init__()
        self.steady_state = steady_state
        if self.steady_state:
            # rho0_path is optional: absent => cold start from the geometry's
            # equilibrium-initialized state (works for every material). When set,
            # it warm-starts from a raw finite-volume state saved with save_rho.
            self.rho0_path = self.steady_state.get("rho0_path", "")
            self.method = self.steady_state.get("method", "df-sane")
            self.nit = int(self.steady_state.get("nit", 100))
            self.nit_save = int(self.steady_state.get("nit_save", 10))
            # Explicit warm-up: integrate this many steps before root-finding to
            # develop a nonzero, well-scaled seed from the contacts. Needed when
            # cold-starting from an empty (zero) field, where |rho| = 0 would make
            # the residual scale ill-defined.
            self.warmup_steps = int(self.steady_state.get("warmup_steps", 0))
            self.integrator = self.steady_state.get("integrator", "RK2")
            self.positivity = False
            self.dt = 0.0
            self.t = 0.0
            log.info("Steady state mode")
        else:
            self.i_step_initial = int(i_step)
            self.i_step = self.i_step_initial
            self.t = float(t)
            if i_step:
                log.info(f"Continuing from step {i_step}")
            dt_max = min(source.dt_max for source in dt_max_sources)
            if dt == 0.0:
                if not np.isfinite(dt_max):
                    raise InvalidInputException(
                        "Specify dt explicitly, because dt_max is not available"
                    )
                dt = dt_max
                log.info(f"Setting time step dt = {dt_max = :.4g}")
            elif dt > dt_max:
                if i_step:
                    # Continuing from a checkpoint whose dt is no longer valid for
                    # the current discretization (e.g. the DG order was increased,
                    # so the explicit-CFL limit dt_max ~ 1/(N+1)^2 tightened). The
                    # restored dt is a stale continuation default, so reduce it to
                    # the new dt_max automatically rather than failing.
                    log.info(f"Reducing restored time step dt = {dt:.4g} to "
                             f"{dt_max = :.4g} for the current discretization")
                    dt = dt_max
                else:
                    raise InvalidInputException(
                        f"{dt = } must be smaller than {dt_max = }")
            self.dt = float(dt)
            self.n_steps = max(1, int(np.round(t_max / self.dt)))
            self.save_interval = max(1, int(np.round(dt_save / self.dt)))
            self.n_collate = int(n_collate)
            self.integrator = integrator
            if integrator not in {"RK2", "RK4", "SSPRK3"}:
                raise InvalidInputException(f"Unrecognized {integrator = }")
            self.positivity = bool(positivity)
            if self.positivity and integrator != "SSPRK3":
                log.info("positivity=True is only rigorously guaranteed with "
                         "integrator=SSPRK3; applying the limiter best-effort to "
                         f"the end-of-step state with {integrator}.")

    def time_step(self, geometry: Geometry) -> None:
        """Advance one step (RK2/RK4, or SSPRK3 for positivity preservation)."""
        t = self.t
        dt = self.dt
        rho0 = geometry.rho
        _limit = getattr(geometry, "limit_positivity", None)
        if not getattr(self, "positivity", False):
            _limit = None

        def lim(rho):
            return _limit(rho) if _limit is not None else rho

        if self.integrator == "RK2":
            rho_half = rho0 + (0.5 * dt) * geometry.rho_dot(rho0, t)
            geometry.rho = lim(rho0 + dt * geometry.rho_dot(rho_half, t + 0.5 * dt))
        elif self.integrator == "RK4":
            k1 = geometry.rho_dot(rho0, t)
            k2 = geometry.rho_dot(rho0 + (0.5 * dt) * k1, t + 0.5 * dt)
            k3 = geometry.rho_dot(rho0 + (0.5 * dt) * k2, t + 0.5 * dt)
            k4 = geometry.rho_dot(rho0 + dt * k3, t + dt)
            geometry.rho = lim(rho0 + (dt / 6.0) * (k1 + 2 * (k2 + k3) + k4))
        elif self.integrator == "SSPRK3":
            # Shu-Osher SSPRK3: each stage is a convex combination of forward-Euler
            # steps, so applying the (convexity-based) limiter after every stage
            # preserves the maximum-principle guarantee.
            rho1 = lim(rho0 + dt * geometry.rho_dot(rho0, t))
            rho2 = lim(0.75 * rho0
                       + 0.25 * (rho1 + dt * geometry.rho_dot(rho1, t + dt)))
            geometry.rho = lim((1.0 / 3.0) * rho0
                               + (2.0 / 3.0) * (rho2 + dt * geometry.rho_dot(
                                   rho2, t + 0.5 * dt)))
        else:
            raise KeyError(f"Unrecognized integrator = {self.integrator}")

    def steady_state_sol(
        self, transport: qimpy.transport.Transport, geometry: Geometry
    ) -> None:
        """Solve rho_dot(rho) = 0 directly with a Newton-free root finder.

        The unknown is the flattened finite-volume state ``geometry.rho[0]`` of
        shape ``(n_cells, n_channels)``. The initial guess is the geometry's
        equilibrium-initialized state, optionally warm-started from a raw state
        saved by an earlier run with ``save_rho: true``.
        """
        rho_shape = geometry.rho[0].shape  # (n_cells, n_channels)
        if self.rho0_path:
            with h5py.File(self.rho0_path, "r") as cp:
                rho_f = np.array(cp["/geometry"]["fv_rho"])
                t_f = cp["/time_evolution"].attrs["t"]
            rho_f = torch.from_numpy(rho_f).to(rc.device, geometry.rho[0].dtype)
            material = transport.material
            if isinstance(material, qimpy.transport.material.ab_initio.AbInitio):
                # Rotate each cell's saved interaction-picture density into the
                # Schrodinger picture at the saved time, matching the live state.
                ph = material.packed_hermitian
                phase = material.schrodingerV(t_f)
                rho_f = rho_f.unflatten(
                    -1, (material.nk_mine, material.n_bands, material.n_bands)
                )
                rho_f = ph.pack(ph.unpack(rho_f) * phase).flatten(-3, -1)
            geometry.rho = TensorList([rho_f.reshape(rho_shape)])

        # Develop a nonzero seed from the contacts when cold-starting from an
        # empty field (otherwise the residual has no characteristic scale).
        if self.warmup_steps and not self.rho0_path:
            self.dt = float(geometry.dt_max)
            log.info(f"Steady-state warm-up: {self.warmup_steps} steps "
                     f"at dt = {self.dt:.4g}")
            for _ in range(self.warmup_steps):
                self.time_step(geometry)

        # Seed the root finder from the (possibly warm-started) live state.
        rho0 = geometry.rho[0].flatten().to(rc.cpu).numpy()
        rho_dot = geometry.rho_dot(geometry.rho, t=0.0)
        rho_scale = float(np.abs(rho0).max())
        rho_dot_scale = float(torch.max(torch.abs(rho_dot[0])))
        if rho_scale == 0.0:  # still empty: fall back to the drive scale
            rho_scale = max(rho_dot_scale, 1e-300)
        RHO_SCALE = rho_scale
        T_SCALE = RHO_SCALE / max(rho_dot_scale, 1e-300)

        steady_state_root_fn = SteadyStateRootFunction(
            geometry, rho_shape, RHO_SCALE, T_SCALE, self.nit, self.nit_save
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
            [torch.from_numpy(optimizer.x * RHO_SCALE).to(rc.device).reshape(rho_shape)]
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
    rho_shape: tuple  #: shape of the per-domain finite-volume state (n_cells, n_channels)
    RHO_SCALE: float = 1.0e-7
    T_SCALE: float = 1.0e4
    nit: int = 0
    nit_save: int = 0
    n_calls: int = 0
    iter: int = 0

    def _rho(self, x: np.ndarray) -> TensorList:
        v = torch.from_numpy(x * self.RHO_SCALE).to(rc.device).reshape(self.rho_shape)
        return TensorList([v])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rho_dot = self.geometry.rho_dot(self._rho(x), t=0.0)
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
            self.geometry.rho = self._rho(x)
            self.geometry.update_stash(self.iter, 0.0)
            log.info(f"Stashed results of iteration {self.iter}")
