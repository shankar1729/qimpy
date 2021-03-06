from __future__ import annotations
import qimpy as qp
import numpy as np
from ._optimizable import Optimizable, ConvergenceCheck
from collections import deque
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Sequence, Dict, Deque


Variable = TypeVar("Variable", bound=Optimizable)


class Pulay(Generic[Variable], ABC, qp.TreeNode):
    """Abstract base class implementing the Pulay mixing algorithm.
    The mixed `Variable` must support vector-space operators as specified
    by the `Optimizable` abstract base class.
    """

    comm: qp.MPI.Comm  #: Communicator over which algorithm operates in unison
    name: str  #: Name of algorithm instance used in reporting eg. 'SCF'.
    n_iterations: int  #: Maximum number of iterations
    energy_threshold: float  #: Convergence threshold on energy change
    residual_threshold: float  #: Covergence threshold on residual
    n_history: int  #: Number of past variables / residuals to retain
    mix_fraction: float  #: Variable mixing fraction between cycles
    _variables: Deque[Variable]  #: History of most recent variables
    _residuals: Deque[Variable]  #: History of most recent residuals
    _overlaps: np.ndarray  #: Overlap matrix of previous residuals

    #: Names and thresholds for any additional convergence quantities.
    #: These are in addition to energy and residual, included by default.
    #: Use a name bracketed by | | for always-positive norm-like
    #: quantities for clarity in output. These must correspond (in order)
    #: to the extra values output by :meth:`cycle`.
    extra_thresholds: Dict[str, float]

    def __init__(
        self,
        *,
        checkpoint_in: qp.utils.CpPath,
        comm: qp.MPI.Comm,
        name: str,
        n_iterations: int,
        energy_threshold: float,
        residual_threshold: float,
        extra_thresholds: Dict[str, float],
        n_history: int,
        mix_fraction: float,
    ) -> None:
        """Initialize Pulay algorithm parameters."""
        super().__init__()
        self.comm = comm
        self.name = name
        self.n_iterations = n_iterations
        self.energy_threshold = energy_threshold
        self.residual_threshold = residual_threshold
        self.extra_thresholds = extra_thresholds
        self.n_history = n_history
        self.mix_fraction = mix_fraction
        self._variables = deque(maxlen=n_history)
        self._residuals = deque(maxlen=n_history)
        self._overlaps = np.zeros((0, 0), dtype=float)

    @abstractmethod
    def cycle(self, dEprev: float) -> Sequence[float]:
        """Single cycle of the Pulay-mixed self-consistent iteration.
        In each subsequent cycle, Pulay will try to zero the difference
        between get_variable() before and after the cycle. The implementation
        should only do the work of computing the updated variable;
        debug printing and I/O, if any, should occur in report() instead.

        :param dEprev: Energy change at previous cycle, which may be used
            to adjust accuracy of any inner optimizations
        :return: Any quantities beyond energy and residual that require
            convergence checking, corresponding to `extra_thresholds`.
        """

    def report(self, i_iter: int) -> None:
        """Override to perform optional reporting."""

    @property  # type: ignore
    @abstractmethod
    def energy(self) -> qp.Energy:
        """Current energy components of the system (read-only)."""

    @property  # type: ignore
    @abstractmethod
    def variable(self) -> Variable:
        """Current variable in the state of the system."""

    @variable.setter  # type: ignore
    @abstractmethod
    def variable(self, v: Variable) -> None:
        ...

    @property
    def residual(self) -> Variable:
        """Get the current residual from state of system (read-only).
        Override this only if this Pulay mixing is not for a self-consistent
        iteration i.e. the residual is not the change of `variable`.
        """
        return self.variable - self._variables[-1]

    @abstractmethod
    def precondition(self, v: Variable) -> Variable:
        """Apply preconditioner to variable/residual."""

    @abstractmethod
    def metric(self, v: Variable) -> Variable:
        """Apply metric to variable/residual."""

    def optimize(self) -> None:
        """Minimize residual using a Pulay-mixing / DIIS algorithm."""

        # Initial energy and difference:
        energy = self.energy
        E = self._sync(float(energy))
        Eprev = 0.0
        dE = E - Eprev

        # Initialize convergence checks:
        Ename = energy.name
        checks = {
            "d" + Ename: ConvergenceCheck(self.energy_threshold),
            "|residual|": ConvergenceCheck(self.residual_threshold),
        }
        for extra_name, extra_threshold in self.extra_thresholds.items():
            checks[extra_name] = ConvergenceCheck(extra_threshold)

        for i_iter in range(self.n_iterations):
            # Cache variable:
            self._variables.append(self.variable)

            # Perform cycle:
            extra_values = self.cycle(dE)
            energy = self.energy
            Eprev = E
            E = self._sync(float(energy))
            dE = E - Eprev

            # Cache residual:
            residual = self.residual
            res_norm = self._sync(np.sqrt(residual.overlap(residual)))
            self._residuals.append(residual)

            # Check and report convergence:
            line = f"{self.name}: {i_iter}  {Ename}: {E:+.11f}  "
            values = [dE, res_norm] + [self._sync(v) for v in extra_values]
            converged = []
            for i_check, (check_name, check) in enumerate(checks.items()):
                value = values[i_check]
                value_str = (
                    f"{value:.3e}" if (check_name[0] == "|") else f"{value:+.3e}"
                )
                line += f"  {check_name:s}: {value_str}"
                if check.check(value):
                    converged.append(check_name)
            line += f"  t[s]: {qp.rc.clock():.2f}"
            qp.log.info(line)
            # --- optional reporting:
            self.report(i_iter)
            # --- stopping criteria:
            if converged:
                qp.log.info(
                    f"{self.name}: Converged on " f'{", ".join(converged)} criteria.'
                )
                break
            if not np.isfinite(E):
                qp.log.info(f"{self.name}: Stopping due to non-finite energy.")
                break

            # Pulay mixing / DIIS step:
            # --- update the overlap matrix
            Mresidual = self.metric(residual)
            new_overlaps = np.array([r.overlap(Mresidual) for r in self._residuals])
            N = len(new_overlaps)
            self._overlaps = np.vstack(
                (
                    np.hstack(
                        (
                            self._overlaps[-(N - 1) :, -(N - 1) :],
                            new_overlaps[:-1, None],
                        )
                    ),
                    new_overlaps[None, :],
                )
            )
            # --- invert overlap matrix to minimize residual
            overlapC = np.ones((N + 1, N + 1))  # extra row/col for norm constraint
            overlapC[:-1, :-1] = self._overlaps
            overlapC[-1, -1] = 0.0
            alpha = np.linalg.inv(overlapC)[N, :-1]  # optimum coefficients
            # --- update variable
            v = 0.0 * residual
            for i_hist, alpha_i in enumerate(alpha):
                v += alpha_i * (
                    self._variables[i_hist]
                    + self.mix_fraction * self.precondition(self._residuals[i_hist])
                )
            self.variable = v  # type: ignore

    def _sync(self, v: float) -> float:
        """Ensure `v` is consistent on `comm`."""
        return self.comm.bcast(v)
