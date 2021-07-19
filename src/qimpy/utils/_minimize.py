import qimpy as qp
import numpy as np
from ._optimizable import Optimizable, ConvergenceCheck
from ._minimize_lbfgs import _lbfgs
from ._minimize_cg import _cg
from ._minimize_line import LINE_MINIMIZE
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import TypeVar, Generic, Sequence, Dict, NamedTuple, Optional, \
    TYPE_CHECKING
if TYPE_CHECKING:
    from .._energy import Energy

Vector = TypeVar('Vector', bound=Optimizable)


class MinimizeState(Generic[Vector]):
    """Current energies and gradients of `Minimize` algorithm."""
    energy: 'Energy'  #: Current energy (objective function)
    extra: Sequence[float]  #: Extra convergence quantities
    gradient: Vector  #: Gradient of energy w.r.t. parameters
    K_gradient: Vector  #: Preconditioned version of `gradient`

    def __init__(self):
        self.energy = qp.Energy()
        self.extra = []


class Minimize(Generic[Vector], ABC, qp.Constructable):
    """Abstract base class implementing large-scale minimization algorithms.
    The `Vector` that is minimized over must support vector-space operators
    as specified by the `Optimizable` abstract base class."""

    class StepSize(NamedTuple):
        """Parameters for controlling line-minimize step size."""
        initial: float = 1.  #: Initial step size
        minimum: float = 1e-10  #: Smallest step size at which to give up
        should_update: bool = True  #: Whether to update test step size
        reduce_factor: float = 0.1  #: Reduction factor when a step fails
        grow_factor: float = 3.  #: Maximum step size growth at one time
        n_adjust: int = 10  #: Number of step changes in one line minimize

    class Wolfe(NamedTuple):
        """Wolfe line minimize stopping conditions."""
        energy: float = 1e-4  #: Dimensionless minimum energy reduction in step
        gradient: float = 0.9  #: Required reduction of projected gradient

    comm: qp.MPI.Comm  #: Communicator over which algorithm operates in unison
    name: str  #: Name of algorithm instance used in reporting eg. 'Ionic'
    n_iterations: int  #: Maximum number of iterations
    energy_threshold: float  #: Convergence threshold on energy change
    method: str  #: CG, L-BFGS or Gradient (i.e steepest descent)
    cg_type: str  #: Polak-Ribiere, Fletcher-Reeves or Hestenes-Stiefel
    line_minimize: str  #: Line minimization: Auto, Constant, Quadratic, Wolfe
    step_size: StepSize  #: Step size options
    wolfe: Wolfe  #: Wolfe line minimize stopping conditions

    #: Names and thresholds for any additional convergence quantities. These
    #: are in addition to energy, included by default. Use a name bracketed by
    #: | | for always-positive norm-like quantities for clarity in output.
    #: These must correspond (in order) to the outputs of :meth:`compute`.
    extra_thresholds: Dict[str, float]

    def __init__(self, *, co: qp.ConstructOptions,
                 comm: qp.MPI.Comm, name: str, n_iterations: int,
                 energy_threshold: float, extra_thresholds: Dict[str, float],
                 method: str, cg_type: str = 'polak-ribiere',
                 line_minimize: str = 'auto', step_size: Optional[dict] = None,
                 wolfe: Optional[dict] = None) -> None:
        """Initialize minimization algorithm parameters."""
        super().__init__(co=co)
        self.comm = comm
        self.name = name
        self.n_iterations = n_iterations
        self.energy_threshold = energy_threshold
        self.extra_thresholds = extra_thresholds
        self.step_size = Minimize.StepSize(**qp.dict_input_cleanup(
            {} if (step_size is None) else step_size))
        self.wolfe = Minimize.Wolfe(**qp.dict_input_cleanup(
            {} if (wolfe is None) else wolfe))

        # Validate direction update options:
        preferred_lm = {'cg': 'quadratic',  # prefered line minimize for each
                        'l-bfgs': 'wolfe',
                        'gradient': 'constant'}
        method_options = preferred_lm.keys()
        self.method = method.lower()
        if self.method not in method_options:
            raise KeyError(f'method must be one of {method_options}')

        # Validate CG type:
        cg_options = {'polak-ribiere', 'fletcher-reeves', 'hestenes-stiefel'}
        self.cg_type = cg_type.lower()
        if self.cg_type not in cg_options:
            raise KeyError(f'cg_type must be one of {cg_options}')

        # Validate line minimize options:
        self.line_minimize = line_minimize.lower()
        if self.line_minimize not in LINE_MINIMIZE:
            if self.line_minimize != 'auto':
                raise KeyError('line_minimize must be "auto" or one of'
                               f' {LINE_MINIMIZE.keys()}')
            self.line_minimize = preferred_lm[self.method]

    @abstractmethod
    def step(self, direction: Vector, step_size: float) -> None:
        """Move the state along `direction` by amount `step_size`"""

    @abstractmethod
    def compute(self, state: MinimizeState[Vector], energy_only: bool) -> None:
        """Update energy and/or gradients in `state`.
        If `energy_only` is True, only update the energy, else update
        all entries including extra convergence checks, gradients and
        preconditioned gradient."""

    def report(self, i_iter: int) -> bool:
        """Override to perform optional reporting / processing every few steps.
        Return True if the state was modified in the process eg. to perform
        some kind of reset to stabilize the system. (This will be used to
        correspondingly reset search directions.)"""
        return False

    def constrain(self, v: Vector) -> Vector:
        """Override to impose any constraints, restricting to allowed subspace.
        The input may be modified in-place and returned for efficiency. """
        return v

    def safe_step_size(self, direction: Vector) -> float:
        """Override to return maximum safe step size along `direction`, if any.
        By default, there is no upper bound on step size."""
        return np.finfo(np.float64).max

    def minimize(self) -> 'Energy':
        """Minimize, and return optimized energy of system"""
        return _lbfgs(self) if (self.method == "l-bfgs") else _cg(self)

    def finite_difference_test(self, direction: Vector,
                               step_sizes: Optional[Sequence[float]] = None
                               ) -> None:
        """Check gradient implementation by taking steps along `direction`.
        This will print ratio of actual energy differences along steps of
        various sizes in `step_sizes` and the expected energy difference
        based on the gradient. A correct implementation should show a ratio
        approaching 1 for a range of step sizes, with deviations at lower
        step sizes due to round off error and at higher step sizes due to
        nonlinearity."""
        qp.log.info(f'{self.name}: {"-"*12} Finite difference test {"-"*12}')
        if step_sizes is None:
            step_sizes = np.logspace(-9, 1, 11).tolist()
        # Initial state with gradient:
        state = qp.utils.MinimizeState['Vector']()
        self.compute(state, energy_only=False)
        E0 = self._sync(float(state.energy))
        dE_step = self._sync(state.gradient.overlap(direction)
                             )  # directional derivative along step direction
        # Finite difference derivatives:
        step_size_prev = 0.  # cumulative progress along step:
        for step_size in sorted(step_sizes):
            self.step(direction, step_size - step_size_prev)
            step_size_prev = step_size
            self.compute(state, energy_only=True)
            deltaE = self._sync(float(state.energy)) - E0
            dE_expected = dE_step * step_size
            qp.log.info(f'{self.name}: step size: {step_size:.3e}'
                        f'  d{state.energy.name}'
                        f' ratio: {deltaE/dE_expected:.11f}')
        qp.log.info(f'{self.name}: {"-"*48}')
        # Restore original position:
        self.step(direction, -step_size_prev)

    def _sync(self, v: float) -> float:
        """Ensure `v` is consistent on `comm`."""
        return self.comm.bcast(v)
