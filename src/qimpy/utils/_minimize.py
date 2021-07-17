import qimpy as qp
import numpy as np
from ._optimizable import Optimizable, ConvergenceCheck
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, NamedTuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from .._energy import Energy

Variable = TypeVar('Variable', bound=Optimizable)


class StepSize(NamedTuple):
    initial: float = 1.  #: Initial step size
    minimum: float = 1e-10  #: Smallest step size at which to give up
    should_update: bool = True  #: Whether to update step size each iteration
    reduce_factor: float = 0.1  #: Factor to reduce step size by when needed
    grow_factor: float = 3.  #: Maximum factor step size can grow by at once
    n_adjust: int = 10  #: Number of step changes in one line minimize


class Minimize(Generic[Variable], ABC, qp.Constructable):
    """Abstract base class implementing large-scale minimization algorithms.
    The `Variable` that is minimized over must support vector-space operators
    as specified by the `Optimizable` abstract base class."""

    comm: qp.MPI.Comm  #: Communicator over which algorithm operates in unison
    name: str  #: Name of algorithm instance used in reporting eg. 'Ionic'
    n_iterations: int  #: Maximum number of iterations
    energy_threshold: float  #: Convergence threshold on energy change
    direction_update: str  #: Direction update scheme: CG, L-BFGS or Gradient
    line_minimize: str  #: Line minimization: Auto, Constant, Quadratic, Cubic
    step_size: StepSize  #: Step size options

    def __init__(self, *, co: qp.ConstructOptions,
                 comm: qp.MPI.Comm, name: str,
                 n_iterations: int, energy_threshold: float,
                 direction_update: str, line_minimize: str = 'auto',
                 step_size: Optional[dict] = None):
        """Initialize minimization algorithm parameters."""
        super().__init__(co=co)
        self.comm = comm
        self.name = name
        self.n_iterations = n_iterations
        self.energy_threshold = energy_threshold
        self.step_size = (StepSize() if (step_size is None)
                          else StepSize(**step_size))

        # Validate direction update options:
        preferred_lm = {'cg': 'quadratic',  # prefered line minimize for each
                        'l-bfgs': 'cubic',
                        'gradient': 'constant'}
        du_options = preferred_lm.keys()
        self.direction_update = direction_update.lower()
        if self.direction_update not in du_options:
            raise KeyError(f'direction_update must be one of {du_options}')

        # Validate line minimize options:
        lm_options = {'auto', 'constant', 'quadratic', 'cubic'}
        self.line_minimize = line_minimize.lower()
        if self.line_minimize not in lm_options:
            raise KeyError(f'line_minimize must be one of {lm_options}')
        if self.line_minimize == 'auto':
            self.line_minimize = preferred_lm[self.direction_update]
