from ._davidson import Davidson


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
        self.line_prefix = 'CheFSI'

    def __repr__(self):
        return ('CheFSI(n_iterations: {:d}, eig_threshold: {:g}, filter_order:'
                ' {:d})'.format(self.n_iterations, self.eig_threshold,
                                self.filter_order))
