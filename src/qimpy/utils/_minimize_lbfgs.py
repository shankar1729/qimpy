from __future__ import annotations
import qimpy as qp
from collections import deque
from dataclasses import dataclass
from typing import Generic, Deque
from ._minimize_line import LINE_MINIMIZE, Vector
from ._minimize_cg import _initialize_convergence_checks, _check_convergence


@dataclass
class _HistoryEntry(Generic[Vector]):
    """History of state and gradient changes within `_lbfgs`"""

    s: "Vector"  #: Change in state (step_size * direction)
    Ky: "Vector"  #: Corresponding change in preconditioned gradient
    rho: float  #: Inverse state-gradient overlap, 1/(s.y)


def _lbfgs(self: qp.utils.Minimize[Vector]) -> qp.Energy:
    """L-BFGS implementation for `Minimize.minimize`"""
    assert self.method == "l-bfgs"

    # Initial energy and gradients:
    state = qp.utils.MinimizeState[Vector]()
    E = self._compute(state, energy_only=False)
    E_prev = 0.0
    line_minimize = LINE_MINIMIZE[self.line_minimize]
    checks = _initialize_convergence_checks(self, state)
    history: Deque[_HistoryEntry] = deque(maxlen=self.n_history)
    gamma = 0.0  # current scaling factor, updated each iteration

    # Iterate till convergence (or iteration limit):
    for i_iter in range(self.n_iterations + 1):
        # Optional reporting:
        if self.report(i_iter):
            qp.log.info(
                f"{self.name}: State modified externally:" " resetting history."
            )
            E = self._compute(state, energy_only=False)
            history.clear()

        # Check and report convergence:
        E, E_prev, should_exit = _check_convergence(
            self, state, i_iter, checks, E, E_prev
        )
        if should_exit:
            return state.energy

        # Compute search direction:
        direction = (-1.0) * state.K_gradient
        alpha: Deque[float] = deque()  # scale factors to each history entry
        for h in reversed(history):
            alpha_i = h.rho * self._sync(h.s.overlap(direction))
            direction -= alpha_i * h.Ky
            alpha.append(alpha_i)
        if gamma:
            direction *= gamma  # scaling to keep reasonable step size ~ 1
        for h in history:
            alpha_i = alpha.pop()
            beta = h.rho * self._sync(h.Ky.overlap(direction))
            direction += (alpha_i - beta) * h.s
        direction = self.constrain(direction)
        if len(history) == self.n_history:
            history.popleft()  # save memory by removing oldest here, when full

        # Line minimization:
        step_size_test = min(self.step_size.initial, self.safe_step_size(direction))
        g_prev, Kg_prev = state.gradient, state.K_gradient
        E, step_size, success = line_minimize(self, direction, step_size_test, state)
        if not success:
            qp.log.info(f"{self.name}: Undoing step.")
            self.step(direction, -step_size)
            E = self._compute(state, energy_only=False)
            if len(history):
                # Step failed, but not along gradient direction:
                qp.log.info(f"{self.name}: Step failed: resetting history.")
                history.clear()
                gamma = 0.0
                continue
            else:
                # Step failed along gradient direction:
                qp.log.info(
                    f"{self.name}: Step failed along gradient: likely"
                    " at roundoff / inner-solve accuracy limit."
                )
                return state.energy

        # Update history:
        y = state.gradient - g_prev
        Ky = state.K_gradient - Kg_prev
        del g_prev, Kg_prev  # minimize # of gradient-like objects in memory
        direction *= step_size  # Now equal to s, the change of state in step
        y_s = self._sync(y.overlap(direction))
        gamma = y_s / self._sync(y.overlap(Ky))
        history.append(_HistoryEntry(s=direction, Ky=Ky, rho=1.0 / y_s))
        del y, Ky, direction  # minimize # of gradient-like objects in memory

    qp.log.info(f"{self.name}: Not converged in {self.n_iterations}" " iterations.")
    return state.energy
