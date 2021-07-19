import qimpy as qp
import numpy as np
from typing import Callable, Dict, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from ._minimize import Minimize, MinimizeState, Vector


LineMinimize = Callable[['Minimize[Vector]', 'Vector', float,
                         'MinimizeState[Vector]'], Tuple[float, float, bool]]


def _constant(self: 'Minimize[Vector]', direction: 'Vector',
              step_size_test: float, state: 'MinimizeState[Vector]'
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


def _quadratic(self: 'Minimize[Vector]', direction: 'Vector',
               step_size_test: float, state: 'MinimizeState[Vector]'
               ) -> Tuple[float, float, bool]:
    """Take a quadratic step calculated from an energy-only test step.
    Adjusts step size to back off if energy increases."""

    # Check initial point:
    step_size_prev = 0.  # cumulative progress along direction
    E = self._sync(float(state.energy))
    E_orig = E
    g_d = self._sync(state.gradient.overlap(direction))
    if g_d > 0.:
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
        if E > E_orig:
            step_size *= self.step_size.reduce_factor
            qp.log.info(f'{self.name}: Step increased'
                        f' {state.energy.name} by {E - E_orig:.3e};'
                        f' reducing step size to {step_size:.3e}.')
            continue
        # Step successful:
        break
    if (not np.isfinite(E)) or (E > E_orig):
        qp.log.info(f'{self.name}: Step failed to reduce {state.energy.name}'
                    f' after {self.step_size.n_adjust} attempts.'
                    ' Quitting step.')
        return E_orig, step_size_prev, False

    return E, step_size_prev, True


LINE_MINIMIZE: Dict[str, LineMinimize] = {
    'constant': _constant,
    'quadratic': _quadratic
}
