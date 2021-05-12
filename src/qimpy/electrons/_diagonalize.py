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
        self.rc = electrons.rc
        self.electrons = electrons
        self.n_iterations = n_iterations
        self.eig_threshold = eig_threshold
        self.line_prefix = 'Davidson'

    def __repr__(self):
        return 'Davidson(n_iterations: {:d}, eig_threshold: {:g})'.format(
            self.n_iterations, self.eig_threshold)

    def _report(self, iteration, Eband, deig_max=None, n_eigs_done=None,
                inner_loop=False, converged=False, converge_failed=False):
        'Report iteration progress / convergence in standardized form'
        line_prefix = ('  ' if inner_loop else '') + self.line_prefix
        line = line_prefix + ':  Iter: {:d}'.format(iteration)
        line += '  Eband: {:+.11f}'.format(Eband)
        if deig_max:
            line += '  deig_max: {:.2e}'.format(deig_max)
        if n_eigs_done:
            line += '  n_eigs_done: {:d}'.format(n_eigs_done)
        line += '  t[s]: {:.2f}'.format(self.rc.clock())
        qp.log.info(line)
        if converged:
            qp.log.info('{:s}: Converged'.format(line_prefix))
        if converge_failed:
            qp.log.info('{:s}: Failed to converge'.format(line_prefix))

    def _precondition(self, Cerr, KEref):
        '''Inverse-kinetic preconditioner on the Cerr in eigenpairs,
        using the per-band kinetic energy KEref'''
        basis = self.electrons.basis
        x = (basis.get_ke(basis.mine)[None, :, None, None, :]
             / KEref[..., None, None])
        x += torch.exp(-x)  # don't modify x ~ 0
        return Cerr / x

    def _regularize(self, C, C_norm):
        'Regularize low-norm columns by setting to coordinate vectors'
        pass  # TODO

    def __call__(self, n_iterations=None, eig_threshold=None):
        'Diagonalize Kohn-Sham Hamiltonian in electrons'
        electrons = self.electrons
        C, electrons.C = electrons.C, None  # don't keep copy to save memory
        E = electrons.E
        n_spins = electrons.n_spins
        nk_mine = electrons.kpoints.n_mine
        w_sk = electrons.w_spin * electrons.basis.wk.view(1, -1, 1)
        n_bands = electrons.n_bands
        n_bands_max = electrons.n_bands + electrons.n_bands_extra
        inner_loop = n_iterations or eig_threshold
        n_iterations = n_iterations if n_iterations else self.n_iterations
        eig_threshold = eig_threshold if eig_threshold else self.eig_threshold

        # Initialize subspace:
        if(2 * n_bands_max >= electrons.basis.n_min):
            raise ValueError('n_bands + n_bands_extra = {:d} exceeds '
                             'min(n_basis)/2 = {:d} in Davidson'.format(
                                 n_bands_max, electrons.basis.n_min//2))
        HC = electrons.hamiltonian(C)
        E, V = torch.linalg.eigh(C ^ HC)  # diagonalize subspace Hamiltonian
        C = C @ V  # switch to eigen-basis
        HC = HC @ V  # switch to eigen-basis
        Eband = self.rc.comm_k.allreduce(
            (w_sk * E[..., :n_bands]).sum().item(), qp.MPI.SUM)
        self._report(0, Eband, inner_loop=inner_loop)
        n_eigs_done = 0

        for i_iter in range(1, n_iterations + 1):
            n_bands_cur = C.n_bands()

            # Compute subspace expansion after dropping converged eigenpairs:
            # --- select unconverged eigenpairs
            E_sel = E[:, :, n_eigs_done:, None, None]
            C_sel = C[:, :, n_eigs_done:] if n_eigs_done else C
            HC_sel = HC[:, :, n_eigs_done:] if n_eigs_done else HC
            # --- compute subspace expansion
            KEref = C_sel.norm('ke')  # reference KE for preconditioning
            Cexp = self._precondition(HC_sel - C_sel.overlap() * E_sel, KEref)
            Cexp_norm = Cexp.norm('band')
            self._regularize(Cexp, Cexp_norm)
            Cexp *= (1./torch.sqrt(Cexp_norm[..., None, None]))
            n_bands_new = n_bands_cur + Cexp.n_bands()

            # Expansion subspace overlaps:
            C_OC = torch.eye(n_bands_cur)[None, None]  # already orthonormal
            C_OCexp = C.dot(Cexp, overlap=True)
            Cexp_OC = C_OCexp.conj().transpose(-2, -1)
            Cexp_OCexp = Cexp.dot(Cexp, overlap=True)
            dims_new = (n_spins, nk_mine, n_bands_new, n_bands_new)
            C_OC_new = torch.zeros(dims_new, device=V.device, dtype=V.dtype)
            C_OC_new[:, :, :n_bands_cur, :n_bands_cur] += C_OC
            C_OC_new[:, :, :n_bands_cur, n_bands_cur:] = C_OCexp
            C_OC_new[:, :, n_bands_cur:, :n_bands_cur] = Cexp_OC
            C_OC_new[:, :, n_bands_cur:, n_bands_cur:] = Cexp_OCexp

            # Expansion subspace Hamiltonian:
            HCexp = electrons.hamiltonian(Cexp)
            C_HC = torch.diag_embed(E)
            C_HCexp = C ^ HCexp
            Cexp_HC = C_HCexp.conj().transpose(-2, -1)
            Cexp_HCexp = Cexp ^ HCexp
            C_HC_new = torch.zeros(dims_new, device=V.device, dtype=V.dtype)
            C_HC_new[:, :, :n_bands_cur, :n_bands_cur] = C_HC
            C_HC_new[:, :, :n_bands_cur, n_bands_cur:] = C_HCexp
            C_HC_new[:, :, n_bands_cur:, :n_bands_cur] = Cexp_HC
            C_HC_new[:, :, n_bands_cur:, n_bands_cur:] = Cexp_HCexp

            # Solve expanded subspace generalized eigenvalue problem:
            E_new, V_new = qp.utils.eighg(C_HC_new, C_OC_new)
            n_bands_next = min(n_bands_new, n_bands_max)  # number to retain
            Vcur = V_new[:, :, :n_bands_cur, :n_bands_next]  # cur -> next C
            Vexp = V_new[:, :, n_bands_cur:, :n_bands_next]  # exp -> next C

            # Update C to optimum n_bands_next subspace from [C, Cexp]:
            C = C @ Vcur
            C += Cexp @ Vexp
            del Cexp
            HC = HC @ Vcur
            HC += HCexp @ Vexp
            del HCexp
            dE = torch.abs(E - E_new[..., :n_bands_cur])  # change in eigs
            E = E_new[..., :n_bands_next]
            # print(torch.count_nonzero(dE[..., :n_bands] < eig_threshold,
            #                          dim=-1), self.rc.i_proc)

            # Test convergence and report:
            Eband = self.rc.comm_k.allreduce((w_sk * E[..., :n_bands]).sum()
                                             .item(), qp.MPI.SUM)
            deig_max = self.rc.comm_kb.allreduce(dE[..., :n_bands].max()
                                                 .item(), qp.MPI.MAX)
            eig_done = (dE[..., :n_bands]
                        < eig_threshold).flatten(0, 1).all(dim=0)
            n_eigs_done = self.rc.comm_kb.allreduce(np.where(np.concatenate((
                [True], eig_done.to(self.rc.cpu).numpy())))[0][-1], qp.MPI.MIN)
            converged = (n_eigs_done == n_bands)
            converge_failed = ((not inner_loop) and (not converged)
                               and (i_iter == n_iterations))
            self._report(i_iter, Eband, inner_loop=inner_loop,
                         deig_max=deig_max, n_eigs_done=n_eigs_done,
                         converged=converged, converge_failed=converge_failed)
            if converged:
                break

        # Store results:
        electrons.C = C
        electrons.E = E


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
