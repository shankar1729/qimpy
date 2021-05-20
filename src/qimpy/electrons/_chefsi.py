import qimpy as qp
import torch
from ._davidson import Davidson
from typing import Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from ._electrons import Electrons
    from ._wavefunction import Wavefunction


class CheFSI(Davidson):
    '''Chebyshev Filter Subspace Iteration (CheFSI) diagonalization.'''
    __slots__ = ('filter_order',)
    filter_order: int  #: Order of Chebyshev filter

    def __init__(self, *, electrons: 'Electrons', n_iterations: int = 100,
                 eig_threshold: float = 1E-8, filter_order: int = 10) -> None:
        '''Initialize with stopping criteria and filter order.

        Parameters
        ----------
        n_iterations
            See :class:`Davidson`
        eig_threshold
            See :class:`Davidson`
        filter_order
            Order of the Chebyshev filter, which amounts to the number of
            Hamiltonian evaluations per band per eigenvalue iteration
        '''
        super().__init__(electrons=electrons, n_iterations=n_iterations,
                         eig_threshold=eig_threshold)
        self.filter_order = filter_order
        self._line_prefix = 'CheFSI'

    def __repr__(self) -> str:
        return (f'CheFSI(n_iterations: {self.n_iterations},'
                f' eig_threshold: {self.eig_threshold:g},'
                f' filter_order: {self.filter_order})')

    def __call__(self, n_iterations: Optional[int] = None,
                 eig_threshold: Optional[float] = None,
                 helper: bool = False) -> Tuple[torch.Tensor, 'Wavefunction',
                                                'Wavefunction']:
        '''Diagonalize Kohn-Sham Hamiltonian in electrons.
        Also available as :meth:`__call__` to make `CheFSI` callable.'''
        electrons = self.electrons
        n_spins = electrons.n_spins
        nk_mine = electrons.kpoints.n_mine
        w_sk = electrons.w_spin * electrons.basis.wk.view(1, -1, 1)
        n_bands = electrons.n_bands
        inner_loop = not (helper or ((n_iterations is None)
                                     and (eig_threshold is None)))
        n_iterations = n_iterations if n_iterations else self.n_iterations
        eig_threshold = eig_threshold if eig_threshold else self.eig_threshold
        if electrons.E is None:
            # Get initial wavefunctions and energies from Davidson:
            self._line_prefix = 'Davidson'
            E, C, HC = super().__call__(1, eig_threshold, helper=True)
            self._line_prefix = 'CheFSI'
            n_iterations_done = 1
        else:
            # Initialize subspace:
            C, electrons.C = electrons.C, None  # to save memory
            HC = electrons.hamiltonian(C)
            E, V = torch.linalg.eigh(C ^ HC)  # subspace Hamiltonian eigs
            C = C @ V  # switch to eigen-basis
            HC = HC @ V  # switch to eigen-basis
            Eband = self._get_Eband(E)
            self._report(0, Eband, inner_loop=inner_loop)
            n_iterations_done = 0

        n_eigs_done = 0

        # Subspace iteration loop:
        for i_iter in range(n_iterations_done+1, n_iterations+1):

            # Filter parameters:
            b_up = (electrons.basis.ke_cutoff  # upper bound on KE
                    + electrons.V_ks.max())  # upper bound on PE
            b_lo = E.max(dim=2)[0]  # Lower end of filter suppression interval
            a_lo = E.min(dim=2)[0]  # Point that sets filter scaling

            # Seperate converged eigenstates, if any:
            if n_eigs_done:
                Cdone, C = C[:, :, :n_eigs_done], C[:, :, n_eigs_done:]
                HCdone, HC = HC[:, :, :n_eigs_done], HC[:, :, n_eigs_done:]

            # Apply scaled Chebyshev filter:
            b_diff = 0.5 * (b_up - b_lo).view(1, -1, 1, 1, 1)
            b_mid = 0.5 * (b_lo + b_up).view(1, -1, 1, 1, 1)
            sigma = b_diff/(b_mid - a_lo.view(1, -1, 1, 1, 1))
            tau = 2/sigma
            Y = (HC - b_mid*C) * (sigma/b_diff)
            del HC
            Y = Y.split_bands()  # start of operations in band-split mode
            C = C.split_bands()
            for i_filter in range(self.filter_order - 2):
                sigma_new = 1/(tau - sigma)
                HY = electrons.hamiltonian(Y)
                Yt = HY - b_mid * Y
                Yt *= (2*sigma_new/b_diff)
                Yt -= (sigma*sigma_new) * C
                # cycle for next filter iteration:
                HC = HY
                C = Y
                Y = Yt
                sigma = sigma_new
            del Yt, HY, HC, C
            # Note that Y is the highest order of the filter above
            # Return to basis-split mode here
            HC = electrons.hamiltonian(Y).split_basis()
            C = Y.split_basis()
            del Y

            # Rejoin converged eigenstates, if any:
            if n_eigs_done:
                C, Cdone = Cdone.cat(C), None
                HC, HCdone = HCdone.cat(HC), None

            # Subspace orthonormalization and diagonalization:
            E_prev = E
            E, V = qp.utils.eighg(C ^ HC, C.dot_O(C))
            C = C @ V
            HC = HC @ V

            # Print and test convergence:
            Eband = self._get_Eband(E)
            dE = torch.abs(E - E_prev)
            deig_max, n_eigs_done = self._check_deigs(dE, eig_threshold)
            converged = (n_eigs_done == n_bands)
            converge_failed = ((i_iter == n_iterations)
                               and (not (inner_loop or helper or converged)))
            self._report(i_iter, Eband, inner_loop=inner_loop,
                         deig_max=deig_max, n_eigs_done=n_eigs_done,
                         converged=converged, converge_failed=converge_failed)
            if converged:
                break

        # Store results:
        if not helper:
            electrons.C = C
            electrons.E = E

        return E, C, HC

    diagonalize = __call__
