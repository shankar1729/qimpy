from __future__ import annotations
from typing import Union
from dataclasses import dataclass
import os

import numpy as np
import torch
import h5py
from scipy import optimize

import qimpy
from qimpy import TreeNode, log, rc
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from .geometry import Geometry, TensorList


def _amax(x) -> float:
    """max |.| over a TensorList (or a bare Tensor). Forces one sync."""
    if torch.is_tensor(x):
        return float(x.abs().max())
    return max((float(xi.abs().max()) for xi in x), default=0.0)


def _patches(x) -> list:
    """View a TensorList (or bare Tensor) as a list of patch tensors."""
    return [x] if torch.is_tensor(x) else list(x)


def _clone(x):
    """Deep copy of a TensorList (or bare Tensor), preserving the type."""
    if torch.is_tensor(x):
        return x.detach().clone()
    return type(x)(xi.detach().clone() for xi in x)


def _amax_where(x) -> tuple[float, int, tuple]:
    """(max|.|, patch index, unraveled index) -- WHERE the extremum sits.

    The whole point of the divergence dump: a scalar max says the state blew
    up, this says which cell did.  Only called on the dump path, so the extra
    argmax never costs anything in the substep loop.
    """
    best = (-1.0, -1, ())
    for ip, xi in enumerate(_patches(x)):
        a = xi.abs()
        flat = int(a.argmax())
        val = float(a.reshape(-1)[flat])
        if val > best[0]:
            best = (val, ip, tuple(int(j) for j in np.unravel_index(flat, a.shape)))
    return best


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
    collision_interval: int  #: Strang-split the collision over this many streaming steps
    collision_fuse: bool  #: merge adjacent half-kicks across windows (2x fewer applies)
    collision_s_max: float  #: Adaptive-substep bound on the measured collision rate
    collision_rho_max: float  #: Abort if the collision kick leaves max|rho| above this
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
        collision_interval: int = 1,
        collision_fuse: bool = True,
        collision_s_max: float = 0.0,
        collision_rho_max: float = 0.0,
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
        collision_fuse
            :yaml:`Merge adjacent half-kicks across Strang windows.`
            Halves the number of collision applies per window. Note the merged
            kick spans twice the interval, so it doubles the stability number
            `s` below.
        collision_s_max
            :yaml:`Adaptive bound on the measured collision stability number.`
            If > 0, each Strang collision kick is subdivided so that

                s = 0.5 * h * max|C[rho]| / max|rho|     (~ gamma_eff * h / 2)

            stays below this on every substep; s is re-measured each substep
            from k1, which the midpoint has already evaluated, so the control
            costs one reduction per substep and no extra applies. Zero
            (default) disables it and takes the original single-midpoint path
            exactly.

            Size this from a MEASUREMENT, not from the linear gamma_max: C[f]
            is cubic, so gamma_eff grows with amplitude and a step sized once
            from gamma_max is not safe for a run whose amplitude evolves.
            Measured on the M=32/Nr=6 mixer at 17.66 uA: s = 0.19 diverges
            (gain 1.002 per kick early, 3.7e7 by the end, and |df| > 1 some
            700 steps before the overflow), while s <= 0.040 is flat over the
            same window. Recommended: 0.05.
        collision_rho_max
            :yaml:`Abort if a collision kick leaves max|rho| above this.`
            rho is the deviation df about f0, so |df| <= 1 identically for a
            Fermi occupation and anything above 1 is unphysical. Setting this
            to ~1 turns a slow numerical divergence into an immediate, located
            failure instead of an overflow thousands of steps later. Zero
            (default) disables the check. NaN also trips it.
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
            self.collision_interval = 1        # root-finding: no splitting
            self.collision_fuse = False
            self.collision_s_max = 0.0
            self.collision_rho_max = float(collision_rho_max)
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
            self.collision_interval = max(1, int(collision_interval))
            self.collision_fuse = bool(collision_fuse)
            self.collision_s_max = float(collision_s_max)
            self.collision_rho_max = float(collision_rho_max)
            self._half_pending = False
            if self.collision_interval > 1:
                N = self.collision_interval
                if self.collision_fuse and self.save_interval % N:
                    # A fused window defers its trailing half-kick, so a
                    # checkpoint is only EXACT at a window boundary.  Round the
                    # save interval up so saves land there.
                    old = self.save_interval
                    self.save_interval = ((old + N - 1) // N) * N
                    log.info(f"save_interval {old} -> {self.save_interval}"
                             f" (multiple of collision_interval {N}, so"
                             f" checkpoints land on window boundaries)")
                log.info(
                    f"Strang-splitting the collision every {N} steps"
                    f" (dt_coll = {N * self.dt:.4g}),"
                    f" {'FUSED' if self.collision_fuse else 'unfused'}"
                    f" -> {2 if self.collision_fuse else 4} applies per window")
            if self.collision_s_max > 0.0:
                log.info(f"Adaptive collision substepping: s <= "
                         f"{self.collision_s_max:g} (measured per substep)")
            if self.collision_rho_max > 0.0:
                log.info(f"Collision validity gate: max|rho| <= "
                         f"{self.collision_rho_max:g}")
            self.positivity = bool(positivity)
            if self.positivity and integrator != "SSPRK3":
                log.info("positivity=True is only rigorously guaranteed with "
                         "integrator=SSPRK3; applying the limiter best-effort to "
                         f"the end-of-step state with {integrator}.")

    def time_step(self, geometry: Geometry) -> None:
        """Advance one step of dt.

        With ``collision_interval = N > 1`` the collision is Strang-split out
        of the streaming: each window of N steps is bracketed by two half-kicks
        of the collision alone over ``N*dt/2``, and the N streaming steps in
        between run with the collision suppressed.  This is second order in the
        splitting, and the state is a physical whole-step state at every step
        boundary (the half-kicks are NOT fused across windows), so checkpoints
        and observables stay exact.

        Why this is worth doing: dt is set by the streaming CFL on the k-grid,
        which for the Cartesian representation is far finer than any collision
        timescale -- the collision ends up applied O(100) times per 1/gamma_max
        while costing ~50x the streaming step.  N must be laddered against a
        converged N=1 answer, not assumed: the relevant number is
        ``material.gamma_max() * N * dt``, and gamma_max is the fastest mode,
        NOT 1/tau_ee from l_ee.

        N = 1 takes the original code path exactly (bit-identical).
        """
        N = self.collision_interval
        if N > 1:
            if not hasattr(geometry, "collision_dot"):
                raise InvalidInputException(
                    "collision_interval > 1 needs a geometry implementing"
                    " collision_dot (finite_volume)")
            dt_half = 0.5 * N * self.dt
            j = self.i_step % N
            if self.collision_fuse:
                # Fused: the trailing half-kick of one window and the leading
                # half of the next are adjacent (no streaming between them), so
                # apply them as ONE full kick -- 2 applies per window instead of
                # 4.  The cost is that mid-window the state carries a pending
                # trailing half-kick; flush_collision() settles it, and saves are
                # aligned to window boundaries so checkpoints stay exact.
                if j == 0:
                    if self._half_pending:
                        self._collision_kick(geometry, 2.0 * dt_half)
                    else:
                        self._collision_kick(geometry, dt_half)
                        self._half_pending = True
            else:
                if j == 0:
                    self._collision_kick(geometry, dt_half)
            geometry.collision_enabled = False
            try:
                self._rk_step(geometry)
            finally:
                geometry.collision_enabled = True
            if (not self.collision_fuse) and j == N - 1:
                self._collision_kick(geometry, dt_half)
            return
        self._rk_step(geometry)

    def flush_collision(self, geometry: Geometry) -> None:
        """Settle the deferred trailing half-kick of a fused window.

        A no-op unless fusing is on and a half-kick is outstanding.  Must be
        called before anything reads the state as physical -- checkpoints,
        stashed observables -- otherwise the snapshot is short by half a
        collision sub-step.
        """
        if self.collision_interval > 1 and self.collision_fuse \
                and getattr(self, "_half_pending", False):
            self._collision_kick(geometry, 0.5 * self.collision_interval * self.dt)
            self._half_pending = False

    #: Absolute floor on max|rho| in the substep control, so the ratio is
    #: defined at a cold start where rho is identically zero. Far below fp64
    #: noise on an O(1) occupation, so it never binds on a live state.
    _RHO_ATOL = 1e-12
    #: Refuse to subdivide a single kick further than this (fail loudly rather
    #: than spin forever if the rate estimate blows up).
    #: Env-overridable so a DIAGNOSTIC rerun can fail fast. At ~15 s/substep on
    #: the M=32/Nr=6 mixer the default budget costs ~17 h of grinding before it
    #: raises, which is exactly the wait that made the 2026-08-18 failure so
    #: expensive; a reproduce-the-divergence run wants ~96, not 4096.
    _MAX_SUBSTEPS = int(os.environ.get("QIMPY_MAX_SUBSTEPS", 4096))
    #: Dump the state once when a kick first needs this many substeps. Normal
    #: operation on the M=32/Nr=6 mixer is 1-2, so this only fires on trouble.
    _WARN_SUBSTEPS = int(os.environ.get("QIMPY_WARN_SUBSTEPS", 64))
    #: Log every kick at or above this count, not just on a change in count.
    _LOUD_SUBSTEPS = 4

    def _collision_kick(self, geometry: Geometry, dt_coll: float) -> None:
        """Advance the collision sub-flow ALONE over dt_coll (explicit midpoint).

        Second order, two collision applies per substep.

        With ``collision_s_max = 0`` (default) this is a single midpoint step
        over the whole of dt_coll, bit-identical to the original scheme.

        With ``collision_s_max > 0`` the interval is subdivided so that the
        MEASURED stability number

            s = 0.5 * h * max|C[rho]| / max|rho|          (~ gamma_eff * h / 2)

        stays below ``collision_s_max`` on every substep.  s is re-measured
        each substep from k1 -- which the midpoint has already evaluated -- so
        the control costs one reduction per substep and no extra applies.

        Do NOT size this step from ``gamma_max * dt_coll < 2``, the linear
        criterion this docstring used to quote.  C[f] is cubic, so gamma_eff
        rises with amplitude: on the M=32/Nr=6 mixer at 17.66 uA a kick that
        started at s = 0.19 was already growing by 1.002 per kick, reached
        |df| > 1 (unphysical) ~700 steps later and overflowed to 1e156 by step
        124,099, while the same state at s <= 0.040 was flat over the same
        window.  Hence: measure s, do not assume it, and keep it <= 0.05.

        Note ``collision_fuse`` doubles dt_coll on the merged kicks, hence
        doubles s -- another reason to bound the measurement rather than a
        nominal N.
        """
        rho = geometry.rho
        s_max = self.collision_s_max
        if s_max <= 0.0:
            k1 = geometry.collision_dot(rho)
            k2 = geometry.collision_dot(rho + (0.5 * dt_coll) * k1)
            geometry.rho = rho + dt_coll * k2
            # rho is untouched here (all ops out-of-place), so it IS the entry
            # state -- no clone, so this path stays bit-identical.
            self._check_collision_rho(geometry.rho, rho)
            return

        t_left = dt_coll
        k1 = geometry.collision_dot(rho)
        n_sub = 0
        # Snapshot the state ENTERING the kick.  This is the object worth
        # having: the kick is where the runaway happens, so replaying it in
        # 0-D from rho_entry reproduces the divergence with no streaming and
        # no multi-hour march.  One device-side copy per kick (kicks are every
        # collision_interval steps and the state is small next to the e-e
        # vertices), so the cost is noise against the 4 applies per window.
        rho_entry = _clone(rho)
        hist: list[tuple[int, float, float, float, float]] = []
        while t_left > 0.0:
            n_sub += 1
            # Rate measured from the state the substep actually starts at. The
            # maxima of |k1| and |rho| may sit in different cells, which only
            # makes h more conservative -- and it matches the quantity the
            # instability was diagnosed with.
            a_rho = _amax(rho)
            a_k1 = _amax(k1)
            rate = 0.5 * a_k1 / max(a_rho, self._RHO_ATOL)
            h = t_left if rate <= 0.0 else min(t_left, s_max / rate)
            hist.append((n_sub, a_rho, a_k1, rate, h))
            # Physicality guard, INSIDE the loop.  It used to run only after
            # the loop returned, so a kick that diverged internally ground
            # through the full substep budget (~17 h at ~15 s/substep on the
            # M=32/Nr=6 mixer) instead of failing in seconds.  a_rho is
            # already synced for the rate above, so this costs nothing.
            if self.collision_rho_max > 0.0 and not (a_rho <= self.collision_rho_max):
                self._dump_collision_state(
                    "rho_max", rho_entry, rho, k1, hist, dt_coll, t_left, n_sub)
                raise RuntimeError(
                    f"collision kick at step {self.i_step} reached "
                    f"max|rho| = {a_rho:.6e} at substep {n_sub}, above "
                    f"collision_rho_max = {self.collision_rho_max:g}. rho is "
                    "the deviation df about f0, so |df| <= 1 identically: the "
                    "state is no longer physical.")
            if n_sub > self._MAX_SUBSTEPS:
                self._dump_collision_state(
                    "max_substeps", rho_entry, rho, k1, hist, dt_coll, t_left, n_sub)
                raise RuntimeError(
                    f"collision kick at step {self.i_step} still needs "
                    f"substeps after {self._MAX_SUBSTEPS}: the collision rate "
                    "is diverging, not merely stiff. Inspect the state rather "
                    "than raising the substep cap.")
            # One early snapshot, long before the cap, so a run that recovers
            # still leaves evidence of what a hard kick looked like.
            if n_sub == self._WARN_SUBSTEPS and not getattr(self, "_warned_sub", False):
                self._warned_sub = True
                log.info(f"Collision substeps passed {self._WARN_SUBSTEPS} at step "
                         f"{self.i_step} (rate {rate:.6e}, max|rho| {a_rho:.6e}) "
                         "-- dumping state")
                self._dump_collision_state(
                    "warn", rho_entry, rho, k1, hist, dt_coll, t_left, n_sub)
            k2 = geometry.collision_dot(rho + (0.5 * h) * k1)
            rho = rho + h * k2
            t_left -= h  # h is min(t_left, .), so the last substep lands exactly
            if t_left > 0.0:
                k1 = geometry.collision_dot(rho)
        geometry.rho = rho
        self._check_collision_rho(rho, rho_entry)
        # Log on change (as before), but ALSO whenever the kick was hard: the
        # count is a ceil, so a rate climbing within the 1->2 band is invisible
        # until it crosses, and the run that died went 1,2,1,2,... then >4096
        # with nothing in between.
        if n_sub != getattr(self, "_n_sub_last", 0) or n_sub >= self._LOUD_SUBSTEPS:
            log.info(f"Collision substeps: {n_sub} (dt_coll = {dt_coll:.4g}, "
                     f"step {self.i_step}, max rate {max(r[3] for r in hist):.4e})")
            self._n_sub_last = n_sub

    def _check_collision_rho(self, rho, rho_entry=None) -> None:
        """Trip on an unphysical state as soon as the kick produces one."""
        if self.collision_rho_max > 0.0:
            a = _amax(rho)
            if not (a <= self.collision_rho_max):  # also catches NaN
                self._dump_collision_state(
                    "rho_max_post", rho_entry, rho, None, [], float("nan"),
                    0.0, -1)
                raise RuntimeError(
                    f"collision kick at step {self.i_step} left "
                    f"max|rho| = {a:.6e}, above collision_rho_max = "
                    f"{self.collision_rho_max:g}. rho is the deviation df "
                    "about f0, so |df| <= 1 identically: the state is no "
                    "longer physical. Reduce collision_s_max (or "
                    "collision_interval) rather than raising this bound.")

    def _dump_collision_state(
        self, reason, rho_entry, rho_now, k1_now, hist, dt_coll, t_left, n_sub
    ) -> str:
        """Write everything needed to replay a diverging kick offline.

        Losing this is what made the 2026-08-18 failure undiagnosable: the run
        raised after ~17 h inside one kick, the newest checkpoint was ~19k
        steps upstream, and FINAL.h5 was a byte copy of it -- so the diverging
        state was never on disk at all.

        ``rho_entry`` is the state ENTERING the kick and is the useful one:
        feeding it to collision_dot alone reproduces the runaway in 0-D, which
        is what distinguishes a genuine ODE blow-up (blow-up time invariant
        under h -> h/10) from a mere step-size instability.

        Never allowed to mask the RuntimeError it accompanies: any failure to
        write is logged and swallowed.
        """
        prefix = os.environ.get("QIMPY_COLLISION_DUMP", "collision_divergence")
        path = f"{prefix}_step{self.i_step:07d}_{reason}.h5"
        try:
            with h5py.File(path, "w") as fp:
                fp.attrs["reason"] = reason
                fp.attrs["i_step"] = int(self.i_step)
                fp.attrs["t"] = float(self.t)
                fp.attrs["dt"] = float(self.dt)
                fp.attrs["dt_coll"] = float(dt_coll)
                fp.attrs["t_left"] = float(t_left)
                fp.attrs["n_sub"] = int(n_sub)
                fp.attrs["collision_s_max"] = float(self.collision_s_max)
                fp.attrs["collision_rho_max"] = float(self.collision_rho_max)
                fp.attrs["collision_interval"] = int(self.collision_interval)
                fp.attrs["collision_fuse"] = bool(self.collision_fuse)
                for name, x in (("rho_entry", rho_entry),
                                ("rho", rho_now), ("k1", k1_now)):
                    if x is None:
                        continue
                    g = fp.create_group(name)
                    for ip, xi in enumerate(_patches(x)):
                        g.create_dataset(f"patch{ip}",
                                         data=xi.detach().cpu().numpy())
                    val, ip, idx = _amax_where(x)
                    g.attrs["amax"] = val
                    g.attrs["amax_patch"] = ip
                    g.attrs["amax_index"] = np.asarray(idx, dtype=np.int64)
                if hist:
                    # (substep, max|rho|, max|k1|, rate, h) -- the only place
                    # the runaway is visible, since the substep COUNT is a
                    # ceil and stays flat until the rate crosses an integer.
                    fp.create_dataset("substep_history",
                                      data=np.asarray(hist, dtype=np.float64))
                    fp["substep_history"].attrs["columns"] = \
                        "n_sub,amax_rho,amax_k1,rate,h"
            log.info(f"Collision divergence dump written to {path}")
        except Exception as exc:  # never mask the real error
            log.warning(f"FAILED to write collision divergence dump {path}: {exc}")
        return path

    def _rk_step(self, geometry: Geometry) -> None:
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
                rho_f = np.array(cp["/geometry"]["rho"])
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
                    self.flush_collision(transport.geometry)
                    transport.geometry.update_stash(self.i_step, self.t)
                    i_collate += 1
                    log.info(f"Stashed results of step {self.i_step}")
                    if i_collate == self.n_collate or self.i_step == 0:
                        transport.save(self.i_step)
                        i_collate = 0

                if self.i_step == self.n_steps:
                    self.flush_collision(transport.geometry)
                    if i_collate:
                        transport.save(self.i_step)
                    break

                self.time_step(transport.geometry)
                probe = getattr(transport.geometry, "maybe_probe", None)
                if probe is not None:
                    probe(self.i_step + 1, self.t + self.dt)

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
