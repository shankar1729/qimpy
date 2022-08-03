from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from typing import NamedTuple, Optional


class LowdinResults(NamedTuple):
    Q: torch.Tensor  #: Lowdin charges
    M: Optional[torch.Tensor] = None  #: Lowdin magnetizations in spin-polarized cases


class Lowdin:
    """Lowdin analysis and atomic manipulation of wavefunctions."""

    C: qp.electrons.Wavefunction  #: Wavefunction being analyzed / manipulated
    psi: qp.electrons.Wavefunction  #: Atomic orbitals
    psi_Opsi: torch.Tensor  #: Self-overlap of atomic orbitals
    psi_OC: torch.Tensor  #: Overlap of atomic orbitals with current wavefunction `C`
    coeff: Optional[torch.Tensor]  #: Best-fit coefficents of C on psi used for dragging

    def __init__(self, C: qp.electrons.Wavefunction) -> None:
        """Prepare to analyze / manipulate wavefunction `C`."""
        ions = C.basis.ions
        psi = ions.get_atomic_orbitals(C.basis)
        psi_Opsi = psi.dot_O(psi)
        psi_OC = psi.dot_O(C)
        self.C = C
        self.psi = psi
        self.psi_Opsi = psi_Opsi.wait()
        self.psi_OC = psi_OC.wait()

    @qp.utils.stopwatch(name="Lowdin.analyze")
    def analyze(self, f: torch.Tensor, spin_polarized: bool) -> LowdinResults:
        """Calculate Lowdin charges and magnetizations (if `spin_polarized`).
        Here, `f` are the occupation factors corresponding to wavefunctions `C`."""
        basis = self.C.basis
        ions = basis.ions
        index = ions.get_atomic_orbital_index(basis)
        i_ion = index[:, 0]  # atom index by atomic orbital
        Z = ions.Z[ions.types]  # neutral electron count per atom
        lowdin = qp.utils.ortho_matrix(self.psi_Opsi, use_cholesky=False) @ self.psi_OC
        lowdin = lowdin[..., : f.shape[-1]]  # drop extra empty bands
        wf = f * basis.w_sk
        if spin_polarized and self.C.spinorial:
            # Need off-diagonal density matrix components for spinorial magnetization:
            Rho = torch.einsum("skab, skb, skAb -> aA", lowdin, wf, lowdin.conj())
            basis.kpoints.comm.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(Rho))
            result = torch.empty((4, ions.n_ions), device=qp.rc.device)
            i_psi_start = 0
            for slice_i, ps in zip(ions.slices, ions.pseudopotentials):
                pauli = ps.pqn_psi.pauli_expectation()
                n_psi_each = pauli.shape[1]
                n_ions_i = slice_i.stop - slice_i.start
                i_psi_stop = i_psi_start + n_ions_i * n_psi_each
                # Fetch portion of density matrix on current species:
                psi_slice = slice(i_psi_start, i_psi_stop)
                Rho_i = Rho[psi_slice, psi_slice].reshape(
                    (n_ions_i, n_psi_each, n_ions_i, n_psi_each)
                )
                result[:, slice_i] = torch.einsum("iaib, dba -> di", Rho_i, pauli).real
                # Advance to next species:
                i_psi_start = i_psi_stop
            Q = Z - result[0]
            Mvec = result[1:].T  # vector M per atom
            return LowdinResults(Q, Mvec)
        else:
            # Diagonal components of density matrix suffice:
            Rho = torch.einsum("skb, skab -> sa", wf, qp.utils.abs_squared(lowdin))
            basis.kpoints.comm.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(Rho))
            # Reduce to (spin)-number on each atom:
            Ns = torch.zeros((Rho.shape[0], ions.n_ions), device=qp.rc.device)
            Ns.index_add_(1, i_ion, Rho)
            Q = Z - Ns.sum(dim=0)
            M = (Ns[0] - Ns[1]) if spin_polarized else None  # scalar or no M per atom
            return LowdinResults(Q, M)

    def remove_atomic_projections(self) -> None:
        """Subtract best-fit atomic projections of wavefunctions."""
        self.coeff = torch.linalg.inv(self.psi_Opsi) @ self.psi_OC
        self.C -= self.psi @ self.coeff

    def restore_atomic_projections(self, delta_positions: torch.Tensor) -> None:
        """Restore atomic projections with updated atomic orbitals."""
        assert self.coeff is not None
        basis = self.C.basis
        # Recompute / drag atomic orbitals:
        if basis.lattice.movable:
            # Recompute orbitals (since lattice dilation not straightforward):
            self.psi = basis.ions.get_atomic_orbitals(basis)
        else:
            # Drag orbitals with appropriate translation phase:
            iGk = basis.iG[:, basis.mine] + basis.k[:, None]  # fractional G + k
            phase = qp.utils.cis((-2 * np.pi) * (iGk @ delta_positions.T))
            i_ion = basis.ions.get_atomic_orbital_index(basis)[:, 0]
            self.psi *= phase[..., i_ion].transpose(1, 2)[None, :, :, None, :]
        # Restore atomic projections:
        self.C += self.psi @ self.coeff
        self.C.orthonormalize()
