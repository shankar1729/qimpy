from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


class Davidson(qp.TreeNode):
    """Davidson diagonalization of Hamiltonian in `electrons`."""

    __slots__ = (
        "rc",
        "electrons",
        "n_iterations",
        "eig_threshold",
        "_line_prefix",
        "_norm_cut",
        "_i_iter",
        "_HC",
    )
    rc: qp.utils.RunConfig
    electrons: qp.electrons.Electrons  #: Electronic system to diagonalize
    n_iterations: int  #: Number of diagonalization iterations
    eig_threshold: float  #: Eigenvalue convergence threshold (in :math:`E_h`)
    _line_prefix: str
    _norm_cut: float
    _i_iter: int
    _HC: qp.electrons.Wavefunction  #: Used for coordination with sub-classes

    def __init__(
        self,
        *,
        rc: qp.utils.RunConfig,
        electrons: qp.electrons.Electrons,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        n_iterations: int = 100,
        eig_threshold: float = 1e-8,
    ) -> None:
        """Initialize diagonalizer with stopping criteria.

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
        """
        super().__init__()
        self.rc = rc
        self.electrons = electrons
        self.n_iterations = n_iterations
        self.eig_threshold = eig_threshold
        self._line_prefix = "Davidson"
        self._norm_cut = np.sqrt(
            electrons.basis.n_tot * 1e-15  # estimate round-off
        )  # to spot null bands in _regularize
        self._i_iter = 0

    def __repr__(self) -> str:
        return (
            f"Davidson(n_iterations: {self.n_iterations},"
            f" eig_threshold: {self.eig_threshold:g})"
        )

    def _report(
        self,
        n_eigs_done: int = 0,
        inner_loop: bool = False,
        converged: bool = False,
        converge_failed: bool = False,
    ) -> None:
        """Report iteration progress / convergence in standardized form"""
        if inner_loop and (not self._i_iter):
            return  # skip zero'th iteration when Eband is not printed
        line_prefix = ("  " if inner_loop else "") + self._line_prefix
        line = f"{line_prefix}: {self._i_iter}"
        if not inner_loop:
            line += f"  Eband: {self.get_Eband():+.11f}"
        if self.electrons.deig_max != np.inf:
            line += f"  deig_max: {self.electrons.deig_max:.2e}"
        if n_eigs_done:
            line += f"  n_eigs_done: {n_eigs_done}"
        line += f"  t[s]: {self.rc.clock():.2f}"
        qp.log.info(line)
        if converged and (not inner_loop):
            qp.log.info(f"{line_prefix}: Converged")
        if converge_failed and (not inner_loop):
            qp.log.info(f"{line_prefix}: Failed to converge")

    def _precondition(
        self, Cerr: qp.electrons.Wavefunction, KEref: torch.Tensor
    ) -> qp.electrons.Wavefunction:
        """Inverse-kinetic preconditioner on the Cerr in eigenpairs,
        using the per-band kinetic energy KEref"""
        watch = qp.utils.StopWatch("Davidson.precondition", self.rc)
        basis = self.electrons.basis
        x = basis.get_ke(basis.mine)[None, :, None, None, :] / KEref[..., None, None]
        x += torch.exp(-x)  # don't modify x ~ 0
        result = Cerr / x
        watch.stop()
        return result

    def _regularize(
        self, C: qp.electrons.Wavefunction, norm: torch.Tensor, i_iter: int
    ) -> None:
        """Regularize low-norm bands of C by randomizing them,
        using seed based on current iteration number i_iter"""
        # Find low-norm bands:
        if self.rc.n_procs_b > 1:
            # guard against machine-precision differences between procs
            self.rc.current_stream_synchronize()
            self.rc.comm_b.Bcast(qp.utils.BufferView(norm))
        low_norm = norm < self._norm_cut
        i_spin, i_k, i_band = torch.where(low_norm)
        if not len(i_spin):
            return  # no regularization needed
        # Randomize select and update the norm (just an estimate):
        C.randomize_selected(i_spin, i_k, i_band, i_iter)  # seeded by i_iter
        norm[i_spin, i_k, i_band] = 1.0

    def get_Eband(self) -> float:
        """Compute the sum over band eigenvalues, averaged over k"""
        electrons = self.electrons
        n_bands = electrons.fillings.n_bands
        return qp.utils.globalreduce.sum(
            electrons.basis.w_sk * electrons.eig[..., :n_bands], self.rc.comm_k
        )

    def _check_deig(
        self, deig: torch.Tensor, eig_threshold: float
    ) -> Tuple[float, int]:
        """Return maximum change in eigenvalues and how many
        eigenvalues are converged at all spin and k"""
        n_bands = self.electrons.fillings.n_bands
        deig_max = qp.utils.globalreduce.max(deig[..., :n_bands], self.rc.comm_kb)
        pending = torch.where(
            (deig[..., :n_bands] > eig_threshold).flatten(0, 1).any(dim=0)
        )[0]
        n_eigs_done = self.rc.comm_kb.allreduce(
            pending[0].item() if len(pending) else n_bands, qp.MPI.MIN
        )
        return deig_max, n_eigs_done

    def __call__(
        self, n_iterations: Optional[int] = None, eig_threshold: Optional[float] = None
    ) -> None:
        """Diagonalize Kohn-Sham Hamiltonian in electrons.
        Also available as :meth:`__call__` to make `Davidson` callable.
        """
        el = self.electrons
        n_bands = el.fillings.n_bands
        n_bands_max = n_bands + el.fillings.n_bands_extra
        helper = type(self) != Davidson
        inner_loop = not (
            helper or ((n_iterations is None) and (eig_threshold is None))
        )
        n_iterations = n_iterations if n_iterations else self.n_iterations
        eig_threshold = eig_threshold if eig_threshold else self.eig_threshold

        # Initialize subspace:
        if 2 * n_bands_max >= el.basis.n_min:
            raise ValueError(
                f"n_bands + n_bands_extra = {n_bands_max} exceeds"
                f" min(n_basis)/2 = {el.basis.n_min//2} in Davidson"
            )
        HC = el.hamiltonian(el.C)
        el.eig, V = torch.linalg.eigh((el.C ^ HC).wait())  # subspace eigs
        el.deig_max = np.inf  # don't know eig accuracy yet
        el.C = el.C @ V  # switch to eigen-basis
        HC = HC @ V  # switch to eigen-basis
        self._i_iter = 0
        self._report(inner_loop=inner_loop)
        n_eigs_done = 0

        for self._i_iter in range(self._i_iter + 1, n_iterations + 1):
            n_bands_cur = el.C.n_bands()

            # Compute subspace expansion after dropping converged eigenpairs:
            # --- select unconverged eigenpairs
            eig_sel = el.eig[:, :, n_eigs_done:, None, None]
            C_sel = el.C[:, :, n_eigs_done:] if n_eigs_done else el.C
            HC_sel = HC[:, :, n_eigs_done:] if n_eigs_done else HC
            # --- compute subspace expansion
            KEref = C_sel.band_ke()  # reference KE for preconditioning
            Cexp = self._precondition(HC_sel - C_sel.overlap() * eig_sel, KEref)
            norm_exp = Cexp.band_norm()
            self._regularize(Cexp, norm_exp, self._i_iter)
            Cexp *= 1.0 / norm_exp[..., None, None]
            Cexp.constrain()
            n_bands_exp = Cexp.n_bands()
            n_bands_new = n_bands_cur + n_bands_exp

            # Combine current and expansion subspace:
            Cnew = el.C.cat(Cexp, clear=True)  # this clears el.C and Cexp memory
            Cexp = Cnew[:, :, n_bands_cur:]  # re-set to a view of the concatenation

            # Expansion subspace overlaps:
            C_OC = torch.eye(n_bands_cur, device=V.device)[None, None]
            C_OC_new = TileExpansion(C_OC, Cnew.dot_O(Cexp))

            # Expansion subspace Hamiltonian:
            HCexp = el.hamiltonian(Cexp)
            del Cexp
            C_HC = torch.diag_embed(el.eig)
            C_HC_new = TileExpansion(C_HC, Cnew ^ HCexp)

            # Solve expanded subspace generalized eigenvalue problem:
            eig_new, V_new = qp.utils.eighg(C_HC_new, C_OC_new.wait(), self.rc)
            n_bands_next = min(n_bands_new, n_bands_max)  # number to retain
            V_new = V_new[..., :n_bands_next]  # drop extra bands
            Vcur, Vexp = V_new.split((n_bands_cur, n_bands_exp), dim=2)

            # Update C and HC to optimum n_bands_next subspace from Cnew:
            el.C = Cnew @ V_new
            del Cnew
            HC = HC @ Vcur
            HC += HCexp @ Vexp
            del HCexp
            deig = torch.abs(el.eig - eig_new[..., :n_bands_cur])
            el.eig = eig_new[..., :n_bands_next]

            # Test convergence and report:
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
            self._HC = HC  # continue efficiently with another algorithm

    diagonalize = __call__


@dataclass
class TileExpansion:
    """Helper class to tile current and expansion subspace matrices for Davidson.
    Implements Waitable protocol to support delayed evaluation."""

    C_XC: torch.Tensor  #: C^X(C) for operator X (typically O or H)
    Cnew_XCexp: qp.utils.Waitable[
        torch.Tensor
    ]  #: future result of Cnew^X(Cexp), where Cnew = cat(C, Cexp)

    def wait(self) -> torch.Tensor:
        Cnew_XCexp = self.Cnew_XCexp.wait()
        n_spins, nk_mine, n_bands_new, n_bands_exp = Cnew_XCexp.shape
        n_bands_cur = n_bands_new - n_bands_exp
        C_XCexp, Cexp_XCexp = Cnew_XCexp.split((n_bands_cur, n_bands_exp), dim=2)
        result = torch.zeros(
            (n_spins, nk_mine, n_bands_new, n_bands_new),
            device=C_XCexp.device,
            dtype=C_XCexp.dtype,
        )
        result[:, :, :n_bands_cur, :n_bands_cur] += self.C_XC  # add to broadcast
        result[:, :, :n_bands_cur, n_bands_cur:] = C_XCexp
        result[:, :, n_bands_cur:, :n_bands_cur] = qp.utils.dagger(C_XCexp)
        result[:, :, n_bands_cur:, n_bands_cur:] = Cexp_XCexp
        return result
