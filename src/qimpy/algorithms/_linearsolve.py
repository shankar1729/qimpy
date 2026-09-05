from __future__ import annotations
from typing import Generic
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

from qimpy import log, rc, TreeNode
from qimpy.io import CheckpointPath
from ._minimize_line import Vector


class LinearSolve(Generic[Vector], ABC, TreeNode):
    group: dist.ProcessGroup | None  #: Process group over which to operate in unison
    n_iterations: int  #: Maximum number of iterations
    threshold: float  #: Convergence threshold on dot(residual, preconditioned residual)
    name: str  #: Line prefix in log for convergence progress; don't log if empty

    def __init__(
        self,
        *,
        checkpoint_in: CheckpointPath,
        group: dist.ProcessGroup | None,
        n_iterations: int,
        threshold: float,
        name: str = "",
    ) -> None:
        """Initialize minimization algorithm parameters."""
        super().__init__()
        self.group = group
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.name = name

    @abstractmethod
    def hessian(self, v: Vector) -> Vector:
        """Multiply by the hessian of the objective function.
        Derived classes must override this to specify the objective function."""

    def precondition(self, v: Vector) -> Vector:
        """Multiply by the preconditioner.
        Derived classes may override this to specify a preconditioner (defaullt: none).
        """
        return v

    def solve(self, rhs: Vector, x: Vector) -> int:
        """Solve `hessian`(`x`) = `rhs` by the conjugate-gradients method.
        Start from initial guess in `x` and return the result in place.
        Return the number of iterations taken to converge."""

        # Compute and check initial residual:
        r = rhs - self.hessian(x)  # residual
        Kr = self.precondition(r)  # preconditioned residual
        d = Kr  # search direction
        r_dot_Kr = self._sync(r.vdot(Kr)).item()
        r_dot_Kr_prev = 0.0
        if self.name:
            log.info(f"{self.name}: Initial r.Kr: {r_dot_Kr:12.6e}")
        if r_dot_Kr < self.threshold:
            if self.name:
                log.info(f"{self.name}: Converged r.Kr < {self.threshold:e}")
            return 0  # converged as-is

        for i_iter in range(self.n_iterations):
            # Update search direction:
            beta = (r_dot_Kr / r_dot_Kr_prev) if r_dot_Kr_prev else 0.0
            d = Kr + beta * d

            # Step:
            w = self.hessian(d)
            alpha = r_dot_Kr / self._sync(w.vdot(d)).item()
            x += alpha * d
            r -= alpha * w
            Kr = self.precondition(r)
            r_dot_Kr_prev = r_dot_Kr
            r_dot_Kr = self._sync(r.vdot(Kr)).item()

            # Report and check convergence:
            if self.name:
                log.info(
                    f"{self.name}: {i_iter}  r.Kr: {r_dot_Kr:12.6e}"
                    f" alpha: {alpha:12.6e} beta: {beta:13.6e}  t[s]: {rc.clock():.2f}"
                )
            if r_dot_Kr < self.threshold:
                if self.name:
                    log.info(f"{self.name}: Converged r.Kr < {self.threshold:e}")
                return i_iter

        # Did not converge
        if self.name:
            log.info(f"{self.name}: Not converged in {self.n_iterations} iterations.")
        return self.n_iterations

    def _sync(self, v: torch.Tensor) -> torch.Tensor:
        """Ensure `v` is consistent on `group`."""
        if (self.group is not None) and (self.group.size() > 1):
            dist.broadcast(v, group=self.group, group_src=0)
        return v
