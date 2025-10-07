from __future__ import annotations

import numpy as np

from qimpy import log, rc, algorithms, Energy
from ._minimize_line import LINE_MINIMIZE, Vector


def cg(self: algorithms.Minimize[Vector]) -> Energy:
    """Conjugate gradients implementation for `Minimize.minimize`.
    Also supports steepest descents (Gradient method) as a special case."""
    assert self.method in {"cg", "gradient"}

    # Initial energy and gradients:
    state = algorithms.MinimizeState[Vector]()
    E = self._compute(state, energy_only=False)
    E_prev = 0.0
    along_gradient = True  # Whether current search direction is along gradient
    direction: Vector  #: Search direction (initialized at 0th iteration)
    g_prev: Vector  #: Previous gradient (initialized at 0th iteration)
    g_prev_Kg_prev: float  #: Preconditioned norm of g_prev
    g_prev_used = not (
        (self.method == "gradient") or (self.cg_type == "fletcher-reeves")
    )  # whether previous gradient used in direction update
    step_size_test = self.step_size.initial  # test step size
    line_minimize = LINE_MINIMIZE[self.line_minimize]
    checks = initialize_convergence_checks(self, state)

    # Iterate till convergence (or iteration limit):
    for i_iter in range(self.i_iter_start, self.n_iterations + 1):
        # Optional reporting:
        if self.report(i_iter):
            log.info(
                f"{self.name}: State modified externally:"
                " resetting search direction."
            )
            E = self._compute(state, energy_only=False)
            along_gradient = True

        # Check and report convergence:
        E, E_prev, should_exit = check_convergence(
            self, state, i_iter, checks, E, E_prev
        )
        if should_exit:
            return state.energy

        # Direction update:
        g_Kg = self._sync(state.gradient.vdot(state.K_gradient))
        if along_gradient or self.method == "gradient":
            beta = 0.0  # weight of previous search direction
        else:
            # Conjugate gradient update:
            g_d = self._sync(state.gradient.vdot(direction))
            g_prev_Kg = (
                self._sync(g_prev.vdot(state.K_gradient)) if g_prev_used else 0.0
            )
            # --- nonlinear CG variants:
            if self.cg_type == "polak-ribiere":
                beta = (g_Kg - g_prev_Kg) / g_prev_Kg_prev
            elif self.cg_type == "hestenes-stiefel":
                g_prev_d = self._sync(g_prev.vdot(direction))
                beta = (g_Kg - g_prev_Kg) / (g_d - g_prev_d)
            else:  # self.cg_type == 'fletcher-reeves':
                beta = g_Kg / g_prev_Kg_prev
            # --- reset if needed:
            if beta < 0.0:
                log.info(f"{self.name}: Encountered beta < 0:" " resetting CG.")
                beta = 0.0
        along_gradient = False
        direction = self.constrain(
            (beta * direction - state.K_gradient)
            if beta
            else ((-1.0) * state.K_gradient)
        )
        g_prev_Kg_prev = g_Kg
        if g_prev_used:
            g_prev = state.gradient  # for next direction update

        # Line minimization:
        step_size_test = min(step_size_test, self.safe_step_size(direction))
        E, step_size, success = line_minimize(self, direction, step_size_test, state)
        if success:
            if self.step_size.should_update:
                step_size_test = (
                    step_size  # use if reasonable
                    if (step_size >= self.step_size.minimum)
                    else self.step_size.initial
                )  # else reset
        else:
            log.info(f"{self.name}: Undoing step.")
            self.step(direction, -step_size)
            E = self._compute(state, energy_only=False)
            if beta:
                # Step failed, but not along gradient direction:
                log.info(f"{self.name}: Step failed:" " resetting search direction.")
                along_gradient = True  # forget current search direction
            else:
                # Step failed along gradient direction:
                log.info(
                    f"{self.name}: Step failed along gradient: likely"
                    " at roundoff / inner-solve accuracy limit."
                )
                return state.energy

    log.info(f"{self.name}: Not converged in {self.n_iterations} iterations.")
    return state.energy


def initialize_convergence_checks(
    self: algorithms.Minimize[Vector], state: algorithms.MinimizeState[Vector]
) -> dict[str, algorithms.ConvergenceCheck]:
    """Initialize convergence checkers for energy and `extra_thresholds`."""
    dEname = f"d{state.energy.name}"
    checks = {
        dEname: algorithms.ConvergenceCheck(self.energy_threshold, self.n_consecutive)
    }
    for extra_name, extra_threshold in self.extra_thresholds.items():
        checks[extra_name] = algorithms.ConvergenceCheck(
            extra_threshold, self.n_consecutive
        )
    return checks


def check_convergence(
    self: algorithms.Minimize[Vector],
    state: algorithms.MinimizeState[Vector],
    i_iter: int,
    checks: dict[str, algorithms.ConvergenceCheck],
    E: float,
    E_prev: float,
) -> tuple[float, float, bool]:
    """Check and report convergence progress."""
    # Report convergence progress:
    Ename = state.energy.name
    line = f"{self.name}: {i_iter}  {Ename}: {E:+.11f}  "
    dE = E - E_prev
    E_prev = E
    values = [dE] + [self._sync(v) for v in state.extra]
    converged = []
    for i_check, (check_name, check) in enumerate(checks.items()):
        if not (i_check or i_iter):
            continue  # dE not available at first iteration
        value = values[i_check]
        value_str = f"{value:.3e}" if (check_name[0] == "|") else f"{value:+.3e}"
        line += f"  {check_name:s}: {value_str}"
        if check.check(value):
            converged.append(check_name)
    line += f"  t[s]: {rc.clock():.2f}"
    log.info(line)

    # Stopping criteria:
    if len(converged) >= self.n_converge:
        log.info(f"{self.name}: Converged on " f'{", ".join(converged)} criteria.')
        return E, E_prev, True
    if not np.isfinite(E):
        log.info(f"{self.name}: Stopping due to non-finite energy.")
        return E, E_prev, True
    if i_iter == self.n_iterations:
        return E, E_prev, True
    return E, E_prev, False
