from __future__ import annotations
import qimpy as qp
import numpy as np
from ._optimizable import Optimizable
from typing import TypeVar, Callable, Dict, Tuple


Vector = TypeVar('Vector', bound=Optimizable)


LineMinimize = Callable[['qp.utils.Minimize[Vector]', Vector, float,
                         'qp.utils.MinimizeState[Vector]'],
                        Tuple[float, float, bool]]


def _constant(self: qp.utils.Minimize[Vector], direction: Vector,
              step_size_test: float, state: qp.utils.MinimizeState[Vector]
              ) -> Tuple[float, float, bool]:
    """Take a pre-specified step size."""
    step_size = step_size_test  # Constant specified step size
    self.step(direction, step_size)
    E = self._compute(state, energy_only=False)
    if not np.isfinite(E):
        qp.log.info(f'{self.name}: Constant step failed with'
                    f' {state.energy.name} = {E}')
        return E, step_size, False
    return E, step_size, True


def _quadratic(self: qp.utils.Minimize[Vector], direction: Vector,
               step_size_test: float, state: qp.utils.MinimizeState[Vector]
               ) -> Tuple[float, float, bool]:
    """Take a quadratic step calculated from an energy-only test step.
    Adjusts step size to back off if energy increases."""

    # Check initial point:
    step_size_prev = 0.  # cumulative progress along direction
    E = self._sync(float(state.energy))
    E_orig = E
    g_d = self._sync(state.gradient.overlap(direction))
    if g_d >= 0.:
        qp.log.info(f'{self.name}: Bad step direction with positive'
                    ' gradient component')
        return E_orig, step_size_prev, False

    # Test step and quadratic step size prediction:
    for i_step in range(self.step_size.n_adjust):
        # Check test step size:
        if step_size_test < self.step_size.minimum:
            qp.log.info(f'{self.name}: Test step size below threshold.')
            return E, step_size_prev, False
        # Try test step:
        self.step(direction, step_size_test - step_size_prev)
        step_size_prev = step_size_test
        E_test = self._compute(state, energy_only=True)  # gradient not needed
        # Check if step left valid domain:
        if not np.isfinite(E_test):
            # Back off from difficult region
            step_size_test *= self.step_size.reduce_factor
            qp.log.info(f'{self.name}: Test step failed with'
                        f' {state.energy.name} = {E_test:.3e};'
                        f' reducing test step size to {step_size_test:.3e}.')
            continue
        # Predict step size (quadratic based on gradient and two energies):
        step_size = (0.5 * (step_size_test**2) * g_d
                     / (step_size_test * g_d + E - E_test))
        # Check reasonableness of predicted step:
        if step_size < 0.:
            # Curvature has wrong sign, but E_test < E, so accept step
            # for now and try descending further next time:
            step_size_test *= self.step_size.grow_factor
            qp.log.info(f'{self.name}: Wrong curvature in test step,'
                        f' growing test step size to {step_size_test:.3e}.')
            E = self._compute(state, energy_only=False)
            return E, step_size_prev, True
        if step_size / step_size_test > self.step_size.grow_factor:
            step_size_test *= self.step_size.grow_factor
            qp.log.info(f'{self.name}: Predicted step size growth'
                        f' > {self.step_size.grow_factor},'
                        f' growing test step size to {step_size_test:.3e}.')
            continue
        if step_size / step_size_test < self.step_size.reduce_factor:
            step_size_test *= self.step_size.reduce_factor
            qp.log.info(f'{self.name}: Predicted step size reduction'
                        f' < {self.step_size.reduce_factor},'
                        f' reducing test step size to {step_size_test:.3e}.')
            continue
        # Successful test step:
        break
    if not np.isfinite(E_test):
        qp.log.info(f'{self.name}: Test step failed {self.step_size.n_adjust}'
                    ' times. Quitting step.')
        return E_orig, step_size_prev, False

    # Actual step:
    for i_step in range(self.step_size.n_adjust):
        # Try the step:
        self.step(direction, step_size - step_size_prev)
        step_size_prev = step_size
        E = self._compute(state, energy_only=False)
        if not np.isfinite(E):
            step_size *= self.step_size.reduce_factor
            qp.log.info(f'{self.name}: Step failed with'
                        f' {state.energy.name} = {E:.3e};'
                        f' reducing step size to {step_size:.3e}.')
            continue
        if E > E_orig + self.energy_threshold:
            step_size *= self.step_size.reduce_factor
            qp.log.info(f'{self.name}: Step increased'
                        f' {state.energy.name} by {E - E_orig:.3e};'
                        f' reducing step size to {step_size:.3e}.')
            continue
        # Step successful:
        break
    if (not np.isfinite(E)) or (E > E_orig + self.energy_threshold):
        qp.log.info(f'{self.name}: Step failed to reduce {state.energy.name}'
                    f' after {self.step_size.n_adjust} attempts.'
                    ' Quitting step.')
        return E_orig, step_size_prev, False

    return E, step_size_prev, True


def _wolfe(self: qp.utils.Minimize[Vector], direction: Vector,
           step_size_test: float, state: qp.utils.MinimizeState[Vector]
           ) -> Tuple[float, float, bool]:
    """Take cubic steps till the Wolfe energy and gradient criteria are
    satisfied. This uses a full energy and gradient test step, and does not
    take a cubic step at all if the Wolfe criteria is satisfied at first."""

    # Check initial point:
    step_size_prev = 0.  # cumulative progress along direction
    E = self._sync(float(state.energy))
    g_d = self._sync(state.gradient.overlap(direction))
    if g_d >= 0.:
        qp.log.info(f'{self.name}: Bad step direction with positive'
                    ' gradient component')
        return E, step_size_prev, False
    # --- remember initial energy and gradient projection for Wolfe check:
    E0 = E_prev = E
    g_d0 = g_d_prev = g_d

    # Cubic steps till Wolfe criteria are satisfied:
    step_size = step_size_test  # initial tentative step
    step_size_prev = 0.  # other point in cubic interval
    step_size_state = 0.  # where state is along the line
    for i_step in range(self.step_size.n_adjust + 1):
        self.step(direction, step_size - step_size_state)
        step_size_state = step_size
        E = self._compute(state, energy_only=False)
        g_d = self._sync(state.gradient.overlap(direction))
        if i_step == self.step_size.n_adjust:
            break  # Reached step limit of line minimize
        # Make sure within domain:
        if not np.isfinite(E):
            step_size = step_size_prev + (self.step_size.reduce_factor
                                          * (step_size - step_size_prev))
            qp.log.info(f'{self.name}: Step failed with {state.energy.name}'
                        f' = {E:.3e}; reducing step size to {step_size:.3e}.')
            continue
        # Check Wolfe criteria:
        wolfe_E = (E - E0) / (abs(step_size) * g_d0)
        wolfe_g = g_d / g_d0
        if (wolfe_E >= self.wolfe.energy) and (wolfe_g <= self.wolfe.gradient):
            return E, step_size_state, True
        qp.log.info(f'{self.name}: Wolfe criteria failed at step size'
                    f' = {step_size:.3e}: reduction {wolfe_E:.3e} in energy'
                    f' and {wolfe_g:.3f} in gradient.')
        # Cubic step:
        # independent coordinates: step_size_prev, step_size
        # corresponding energies E_prev, E and derivatives g_d_prev, g_d
        # transform [step_size_prev, step_size] to unit interval in t:
        Eprime_prev = g_d_prev * (step_size - step_size_prev)
        Eprime = g_d * (step_size - step_size_prev)
        deltaE = E - E_prev
        # dE/dt is a quadratic A t^2 - 2B t + C with coefficients:
        A = 3*(Eprime_prev + Eprime - 2*deltaE)
        B = 2*Eprime_prev + Eprime - 3*deltaE
        C = Eprime_prev
        # Solve quadratic:
        t_min = np.nan  # location of minimum of E (NAN if no minimum)
        Dsq = B*B - A*C  # discriminant^2
        if Dsq >= 0.:  # dE/dt has at least one root
            t_opt = ((B + np.sqrt(Dsq))/A if (B > 0)
                     else C/(B - np.sqrt(Dsq)))  # only root with E''(t) > 0
            E_opt = E_prev + t_opt*(C + t_opt*(-B + t_opt*A/3))
            if np.isfinite(t_opt) and (E_opt < min(E, E_prev)):
                t_min = t_opt  # well-defined minimum lower than endpoints
        # Pick best, bounded step:
        if np.isfinite(t_min):  # local minimum available
            t_min = min(t_min, self.step_size.grow_factor)  # forward bound
            t_min = max(t_min, 1.-self.step_size.grow_factor)  # reverse bound
        else:  # no local minimum lower than endpoints within interval:
            # therefore E(t) must decrease away from at least one endpoint
            if Eprime <= 0.:  # E(t) decreases for t >~ 1
                t_min = self.step_size.grow_factor
            else:  # E(t) decreases for t <~ 0
                t_min = 1. - self.step_size.grow_factor
        step_size_new = step_size_prev + t_min*(step_size - step_size_prev)
        # Pick next interval:
        if E < E_prev:
            step_size_prev = step_size
            E_prev = E
            g_d_prev = g_d
        step_size = step_size_new

    if (not np.isfinite(E)) or (E > E0):
        return E0, step_size_prev, False  # minimize will roll back state
    return E, step_size_prev, True


LINE_MINIMIZE: Dict[str, LineMinimize] = {
    'constant': _constant,
    'quadratic': _quadratic,
    'wolfe': _wolfe
}
