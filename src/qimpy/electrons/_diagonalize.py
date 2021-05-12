import qimpy as qp
import numpy as np
import torch


class Davidson:
    '''TODO: document class Davidson'''

    def __init__(self, *, electrons, n_iterations=100, eig_threshold=1E-8):
        '''
        Parameters
        ----------
        n_iterations: int, default: 100
            Number of diagonalization iterations in fixed-Hamiltonian
            calculations; the self-consistent field method overrides this
            when diagonalizing in an inner loop
        eig_threshold: float, default: 1E-9
            Maximum change in any eigenvalue from the previous iteration
            to consider as converged for fixed-Hamiltonian calculations;
            the self-consistent field method overrides this when
            diagonalizing in an inner loop
        '''
        self.electrons = electrons
        self.n_iterations = n_iterations
        self.eig_threshold = eig_threshold

    def __repr__(self):
        return 'Davidson(n_iterations: {:d}, eig_threshold: {:g})'.format(
            self.n_iterations, self.eig_threshold)


class CheFSI(Davidson):
    '''TODO: document class CheFSI'''

    def __init__(self, *, electrons, n_iterations=100, eig_threshold=1E-8,
                 filter_order=10):
        '''
        Parameters
        ----------
        n_iterations: int, default: 100
            See :class:`Davidson`
        eig_threshold: float, default: 1E-9
            See :class:`Davidson`
        filter_order: int, default: 10
            Order of the Chebyshev filter, which amountd to the number of
            Hamiltonian evaluations per band per eigenvalue iteration
        '''
        super().__init__(electrons=electrons, n_iterations=n_iterations,
                         eig_threshold=eig_threshold)
        self.filter_order = filter_order

    def __repr__(self):
        return ('CheFSI(n_iterations: {:d}, eig_threshold: {:g}, filter_order:'
                ' {:d})'.format(self.n_iterations, self.eig_threshold,
                                self.filter_order))
