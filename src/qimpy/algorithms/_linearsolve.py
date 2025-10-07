from __future__ import annotations
from typing import Generic
from abc import ABC, abstractmethod

import numpy as np

from qimpy import log, Energy, TreeNode, MPI
from ._minimize_line import Vector


class LinearSolve(Generic[Vector], ABC, TreeNode):
    comm: MPI.Comm  #: Communicator over which algorithm operates in unison
    n_iterations: int  #: Maximum number of iterations
    gradient_threshold: float  #: Convergence threshold on preconditioned residual norm
    name: str  #: Line prefix in log for convergence progress; don't log if empty

    def __init__(
        self,
        *,
        checkpoint_in: CheckpointPath,
        comm: MPI.Comm,
        n_iterations: int,
        gradient_threshold: float,
        name: str = "",
    ) -> None:
        """Initialize minimization algorithm parameters."""
        super().__init__()
        self.comm = comm
        self.n_iterations = n_iterations
        self.gradient_threshold = gradient_threshold
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

    def solve(self, rhs: FieldH, x: FieldH) -> int:
        """Solve `hessian`(`x`) = `rhs` by the conjugate-gradients method.
        Start from initial guess in `x` and return the result in place.
        Return the number of iterations taken to converge."""

        # Compute and check initial residual:
        r = rhs - self.hessian(x)  # residual
        z = self.precondition(r)  # preconditioned residual
        d = z  # search direction
        r_dot_z = self.comm.bcast(r.vdot(z))
        r_dot_z_prev = 0.0
        rz_norm = np.sqrt(abs(r_dot_z))
        if self.name:
            log.info(f"{self.name}: Initial |r|: {rz_norm:12.6e}")
        if rz_norm < self.gradient_threshold:
            if self.name:
                log.info(f"{self.name}: Converged |r| < {self.gradient_threshold:e}")
            return 0  # converged as-is

        for i_iter in range(self.n_iterations):
            # Update search direction:
            beta = (r_dot_z / r_dot_z_prev) if r_dot_z_prev else 0.0
            d = z + beta * d

            # Step:
            w = self.hessian(d)
            alpha = r_dot_z / self.comm.bcast(w.vdot(d))
            x += alpha * d
            r -= alpha * w
            z = self.precondition(r)
            r_dot_z_prev = r_dot_z
            r_dot_z = self.comm.bcast(r.vdot(z))

            # Report and check convergence:
            rz_norm = np.sqrt(abs(r_dot_z))
            if self.name:
                log.info(
                    f"{self.name}: {i_iter}  |r|: {rz_norm:12.6e}"
                    f" alpha: {alpha:12.6e} beta: {beta:13.6e}  t[s]: {rc.clock():.2f}"
                )
            if rz_norm < self.gradient_threshold:
                if self.name:
                    log.info(
                        f"{self.name}: Converged |r| < {self.gradient_threshold:e}"
                    )
                return i_iter

        # Did not converge
        if self.name:
            log.info(f"{self.name}: Not converged in {self.n_iterations} iterations.")
        return self.n_iterations
