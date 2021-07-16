import qimpy as qp
import numpy as np
import torch
from typing import Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import RunConfig
    from ._electrons import Electrons
    from ._wavefunction import Wavefunction


class Davidson(qp.Constructable):
    """Davidson diagonalization of Hamiltonian in `electrons`."""
    __slots__ = ('electrons', 'n_iterations', 'eig_threshold',
                 '_line_prefix', '_norm_cut', '_i_iter', '_HC')
    electrons: 'Electrons'  #: Electronic system to diagonalize
    n_iterations: int  #: Number of diagonalization iterations
    eig_threshold: float  #: Eigenvalue convergence threshold (in :math:`E_h`)
    _line_prefix: str
    _norm_cut: float
    _i_iter: int
    _HC: 'Wavefunction'  # Used only for coordination with sub-classes

    def __init__(self, *, co: qp.ConstructOptions,
                 electrons: 'Electrons', n_iterations: int = 100,
                 eig_threshold: float = 1E-8) -> None:
        """Initialize diagonalizer with stopping criteria.

        Parameters
        ----------
        n_iterations
            Number of diagonalization iterations in fixed-Hamiltonian
            calculations; the self-consistent field method overrides this
            when diagonalizing in an inner loop
        eig_threshold
            Maximum change in any eigenvalue from the previous iteration
            to consider as converged for fixed-Hamiltonian calculations;
            the self-consistent field method overrides this when
            diagonalizing in an inner loop
        """
        super().__init__(co=co)
        self.electrons = electrons
        self.n_iterations = n_iterations
        self.eig_threshold = eig_threshold
        self._line_prefix = 'Davidson'
        self._norm_cut = np.sqrt(electrons.basis.n_tot  # estimate round-off
                                 * 1E-15)  # to spot null bands in _regularize
        self._i_iter = 0

    def __repr__(self) -> str:
        return (f'Davidson(n_iterations: {self.n_iterations},'
                f' eig_threshold: {self.eig_threshold:g})')

    def _report(self, n_eigs_done: int = 0,
                inner_loop: bool = False, converged: bool = False,
                converge_failed: bool = False) -> None:
        """Report iteration progress / convergence in standardized form"""
        if inner_loop and (not self._i_iter):
            return  # skip zero'th iteration when Eband is not printed
        line_prefix = ('  ' if inner_loop else '') + self._line_prefix
        line = f'{line_prefix}: {self._i_iter}'
        if not inner_loop:
            line += f'  Eband: {self._get_Eband():+.11f}'
        if self.electrons.deig_max != np.inf:
            line += f'  deig_max: {self.electrons.deig_max:.2e}'
        if n_eigs_done:
            line += f'  n_eigs_done: {n_eigs_done}'
        line += f'  t[s]: {self.rc.clock():.2f}'
        qp.log.info(line)
        if converged and (not inner_loop):
            qp.log.info(f'{line_prefix}: Converged')
        if converge_failed and (not inner_loop):
            qp.log.info(f'{line_prefix}: Failed to converge')

    def _precondition(self, Cerr: 'Wavefunction',
                      KEref: torch.Tensor) -> 'Wavefunction':
        """Inverse-kinetic preconditioner on the Cerr in eigenpairs,
        using the per-band kinetic energy KEref"""
        watch = qp.utils.StopWatch('Davidson.precondition', self.rc)
        basis = self.electrons.basis
        x = (basis.get_ke(basis.mine)[None, :, None, None, :]
             / KEref[..., None, None])
        x += torch.exp(-x)  # don't modify x ~ 0
        result = Cerr / x
        watch.stop()
        return result

    def _regularize(self, C: 'Wavefunction', norm: torch.Tensor,
                    i_iter: int) -> None:
        """Regularize low-norm bands of C by randomizing them,
        using seed based on current iteration number i_iter"""
        # Find low-norm bands:
        if self.rc.n_procs_b > 1:
            # guard against machine-precision differences between procs
            self.rc.comm_b.Bcast(qp.utils.BufferView(norm))
        low_norm = (norm < self._norm_cut)
        i_spin, i_k, i_band = torch.where(low_norm)
        if not len(i_spin):
            return  # no regularization needed
        # Randomize select and update the norm (just an estimate):
        basis = self.electrons.basis
        C.randomize_selected(i_spin, i_k, i_band, i_iter)  # seeded by i_iter
        norm[i_spin, i_k, i_band] = 1.

    def _get_Eband(self) -> float:
        """Compute the sum over band eigenvalues, averaged over k"""
        electrons = self.electrons
        n_bands = electrons.fillings.n_bands
        return qp.utils.globalreduce.sum(electrons.basis.w_sk
                                         * electrons.eig[..., :n_bands],
                                         self.rc.comm_k)

    def _check_deig(self, deig: torch.Tensor,
                    eig_threshold: float) -> Tuple[float, int]:
        """Return maximum change in eigenvalues and how many
        eigenvalues are converged at all spin and k"""
        n_bands = self.electrons.fillings.n_bands
        deig_max = qp.utils.globalreduce.max(deig[..., :n_bands],
                                             self.rc.comm_kb)
        pending = torch.where((deig[..., :n_bands]
                               > eig_threshold).flatten(0, 1).any(dim=0))[0]
        n_eigs_done = self.rc.comm_kb.allreduce(pending[0].item()
                                                if len(pending) else n_bands,
                                                qp.MPI.MIN)
        return deig_max, n_eigs_done

    def __call__(self, n_iterations: Optional[int] = None,
                 eig_threshold: Optional[float] = None) -> None:
        """Diagonalize Kohn-Sham Hamiltonian in electrons.
        Also available as :meth:`__call__` to make `Davidson` callable.
        """
        el = self.electrons
        n_spins = el.n_spins
        nk_mine = el.kpoints.division.n_mine
        n_bands = el.fillings.n_bands
        n_bands_max = n_bands + el.fillings.n_bands_extra
        helper = (type(self) != Davidson)
        inner_loop = not (helper or ((n_iterations is None)
                                     and (eig_threshold is None)))
        n_iterations = n_iterations if n_iterations else self.n_iterations
        eig_threshold = eig_threshold if eig_threshold else self.eig_threshold

        # Initialize subspace:
        if 2 * n_bands_max >= el.basis.n_min:
            raise ValueError(
                f'n_bands + n_bands_extra = {n_bands_max} exceeds'
                f' min(n_basis)/2 = {el.basis.n_min//2} in Davidson')
        HC = el.hamiltonian(el.C)
        el.eig, V = torch.linalg.eigh(el.C ^ HC)  # subspace eigs
        el.deig_max = np.inf  # don't know eig accuracy yet
        el.C = el.C @ V  # switch to eigen-basis
        HC = HC @ V  # switch to eigen-basis
        self._i_iter = 0
        self._report(inner_loop=inner_loop)
        n_eigs_done = 0

        for self._i_iter in range(self._i_iter+1, n_iterations+1):
            n_bands_cur = el.C.n_bands()

            # Compute subspace expansion after dropping converged eigenpairs:
            # --- select unconverged eigenpairs
            eig_sel = el.eig[:, :, n_eigs_done:, None, None]
            C_sel = el.C[:, :, n_eigs_done:] if n_eigs_done else el.C
            HC_sel = HC[:, :, n_eigs_done:] if n_eigs_done else HC
            # --- compute subspace expansion
            KEref = C_sel.band_ke()  # reference KE for preconditioning
            Cexp = self._precondition(HC_sel - C_sel.overlap()*eig_sel, KEref)
            norm_exp = Cexp.band_norm()
            self._regularize(Cexp, norm_exp, self._i_iter)
            Cexp *= (1./norm_exp[..., None, None])
            Cexp.constrain()
            n_bands_new = n_bands_cur + Cexp.n_bands()

            # Expansion subspace overlaps:
            C_OC = torch.eye(n_bands_cur, device=V.device)[None, None]
            C_OCexp = el.C.dot_O(Cexp)
            Cexp_OC = C_OCexp.conj().transpose(-2, -1)
            Cexp_OCexp = Cexp.dot_O(Cexp)
            dims_new = (n_spins, nk_mine, n_bands_new, n_bands_new)
            C_OC_new = torch.zeros(dims_new, device=V.device, dtype=V.dtype)
            C_OC_new[:, :, :n_bands_cur, :n_bands_cur] += C_OC
            C_OC_new[:, :, :n_bands_cur, n_bands_cur:] = C_OCexp
            C_OC_new[:, :, n_bands_cur:, :n_bands_cur] = Cexp_OC
            C_OC_new[:, :, n_bands_cur:, n_bands_cur:] = Cexp_OCexp

            # Expansion subspace Hamiltonian:
            HCexp = el.hamiltonian(Cexp)
            C_HC = torch.diag_embed(el.eig)
            C_HCexp = el.C ^ HCexp
            Cexp_HC = C_HCexp.conj().transpose(-2, -1)
            Cexp_HCexp = Cexp ^ HCexp
            C_HC_new = torch.zeros(dims_new, device=V.device, dtype=V.dtype)
            C_HC_new[:, :, :n_bands_cur, :n_bands_cur] = C_HC
            C_HC_new[:, :, :n_bands_cur, n_bands_cur:] = C_HCexp
            C_HC_new[:, :, n_bands_cur:, :n_bands_cur] = Cexp_HC
            C_HC_new[:, :, n_bands_cur:, n_bands_cur:] = Cexp_HCexp

            # Solve expanded subspace generalized eigenvalue problem:
            eig_new, V_new = qp.utils.eighg(C_HC_new, C_OC_new)
            n_bands_next = min(n_bands_new, n_bands_max)  # number to retain
            Vcur = V_new[:, :, :n_bands_cur, :n_bands_next]  # cur -> next C
            Vexp = V_new[:, :, n_bands_cur:, :n_bands_next]  # exp -> next C

            # Update C to optimum n_bands_next subspace from [C, Cexp]:
            el.C = el.C @ Vcur
            el.C += Cexp @ Vexp
            del Cexp
            HC = HC @ Vcur
            HC += HCexp @ Vexp
            del HCexp
            deig = torch.abs(el.eig - eig_new[..., :n_bands_cur])
            el.eig = eig_new[..., :n_bands_next]

            # Test convergence and report:
            el.deig_max, n_eigs_done = self._check_deig(deig, eig_threshold)
            converged = (n_eigs_done == n_bands)
            converge_failed = ((self._i_iter == n_iterations)
                               and (not (helper or converged)))
            self._report(inner_loop=inner_loop, n_eigs_done=n_eigs_done,
                         converged=converged, converge_failed=converge_failed)
            if converged:
                break
        if helper:
            self._HC = HC  # continue efficiently with another algorithm

    diagonalize = __call__
