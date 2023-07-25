from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from qimpy import dft
from qimpy.io import CheckpointPath
from qimpy.mpi import globalreduce
from qimpy.math import eighg
from ._davidson import Davidson


class CheFSI(Davidson):
    """Chebyshev Filter Subspace Iteration (CheFSI) diagonalization."""

    filter_order: int  #: Order of Chebyshev filter
    init_threshold: float  #: Threshold for Davidson initialization

    def __init__(
        self,
        *,
        electrons: dft.electrons.Electrons,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        n_iterations: int = 100,
        eig_threshold: float = 1e-8,
        filter_order: int = 10,
        init_threshold: float = 1e-1,
    ) -> None:
        """Initialize with stopping criteria and filter order.

        Parameters
        ----------
        n_iterations
            :yaml:`Number of diagonalization iterations.`
            This only affects fixed-Hamiltonian calculations because the
            self-consistent field method overrides this when diagonalizing
            in an inner loop.
        eig_threshold
            :yaml:`Convergence threshold on eigenvalues in Hartrees.`
            Stop when the maximum change in any eigenvalue between iterations
            falls below this threshold. This only affects fixed-Hamiltonian
            calculations because the self-consistent field method overrides
            this when diagonalizing in an inner loop.
        filter_order
            :yaml:`Order of the Chebyshev filter.`
            This amounts to the number of Hamiltonian evaluations per band
            per eigenvalue iteration.
        init_threshold
            :yaml:`Eigenvalue threshold for initial Davidson steps.`
            This controls the accuracy of a coarse Davidson diagonalization
            that creates a reasonable starting point for CheFSI.
        """
        super().__init__(
            electrons=electrons,
            checkpoint_in=checkpoint_in,
            n_iterations=n_iterations,
            eig_threshold=eig_threshold,
        )
        self.filter_order = filter_order
        self.init_threshold = init_threshold
        self._line_prefix = "CheFSI"

    def __repr__(self) -> str:
        return (
            f"CheFSI(n_iterations: {self.n_iterations},"
            f" eig_threshold: {self.eig_threshold:g},"
            f" filter_order: {self.filter_order},"
            f" init_threshold: {self.init_threshold:g})"
        )

    def __call__(
        self, n_iterations: Optional[int] = None, eig_threshold: Optional[float] = None
    ) -> None:
        """Diagonalize Kohn-Sham Hamiltonian in electrons.
        Also available as :meth:`__call__` to make `CheFSI` callable.
        """
        el = self.electrons
        n_bands = el.fillings.n_bands
        helper = type(self) != CheFSI
        inner_loop = not (
            helper or ((n_iterations is None) and (eig_threshold is None))
        )
        n_iterations = n_iterations if n_iterations else self.n_iterations
        eig_threshold = eig_threshold if eig_threshold else self.eig_threshold
        if el.deig_max > self.init_threshold:
            # Get initial wavefunctions and energies from Davidson:
            self._line_prefix = "ChefSI_init"
            super().__call__(n_iterations, self.init_threshold)
            self._line_prefix = "CheFSI"
            HC = self._HC
            del self._HC
        else:
            # Initialize subspace:
            HC = el.hamiltonian(el.C)
            el.eig, V = torch.linalg.eigh((el.C ^ HC).wait())  # subspace eigs
            el.deig_max = np.inf  # don't know eig accuracy yet
            el.C = el.C @ V  # switch to eigen-basis
            HC = HC @ V  # switch to eigen-basis
            self._i_iter = 0
            self._report(inner_loop=inner_loop)

        KEmax = el.basis.ke_cutoff  # upper bound on KE
        PEmax = globalreduce.max((~el.n_tilde.grad).data, el.comm)
        n_eigs_done = 0

        # Subspace iteration loop:
        for self._i_iter in range(self._i_iter + 1, n_iterations + 1):

            # Filter parameters:
            b_up = KEmax + PEmax  # Upper end of filter suppression
            b_lo = el.eig.max(dim=2)[0]  # Lower end of filter suppression
            a_lo = el.eig.min(dim=2)[0]  # Point that sets filter scaling
            b_shape = a_lo.shape + (1, 1, 1)  # for broadcasting with wavefunctions

            # Seperate converged eigenstates to 'C0', if any:
            if n_eigs_done:
                C0, el.C = el.C[:, :, :n_eigs_done], el.C[:, :, n_eigs_done:]
                HC0, HC = HC[:, :, :n_eigs_done], HC[:, :, n_eigs_done:]

            # Apply scaled Chebyshev filter:
            b_diff = 0.5 * (b_up - b_lo).view(b_shape)
            b_mid = 0.5 * (b_lo + b_up).view(b_shape)
            sigma = b_diff / (b_mid - a_lo.view(b_shape))
            tau = 2 / sigma
            Y = (HC - b_mid * el.C) * (sigma / b_diff)
            del HC
            Y = Y.split_bands().wait()  # start of operations in band-split mode
            el.C = el.C.split_bands().wait()
            for i_filter in range(self.filter_order - 2):
                sigma_new = 1 / (tau - sigma)
                HY = el.hamiltonian(Y)
                Yt = HY - b_mid * Y
                Yt *= 2 * sigma_new / b_diff
                Yt -= (sigma * sigma_new) * el.C
                # cycle for next filter iteration:
                HC = HY
                el.C = Y
                Y = Yt
                sigma = sigma_new
            del Yt, HY, HC, el.C
            # Note that Y is the highest order of the filter above
            # Return to basis-split mode here
            HC = el.hamiltonian(Y).split_basis().wait()
            el.C = Y.split_basis().wait()
            del Y

            # Rejoin converged eigenstates, if any:
            if n_eigs_done:
                el.C = C0.cat(el.C, clear=True)
                HC = HC0.cat(HC, clear=True)
                del C0, HC0  # Note: memory of these already destroyed above

            # Subspace orthonormalization and diagonalization:
            eig_prev = el.eig
            el.eig, V = eighg(el.C ^ HC, el.C.dot_O(el.C).wait())
            el.C = el.C @ V
            HC = HC @ V

            # Print and test convergence:
            deig = torch.abs(el.eig - eig_prev)
            el.deig_max, n_eigs_done = self._check_deig(deig, eig_threshold)
            converged = n_eigs_done == n_bands
            converge_failed = (self._i_iter == n_iterations) and (
                not (helper or converged)
            )
            self._report(
                inner_loop=inner_loop,
                n_eigs_done=n_eigs_done,
                converged=converged,
                converge_failed=converge_failed,
            )
            if converged:
                break
        if helper:
            self._HC = HC

    diagonalize = __call__
