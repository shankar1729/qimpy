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

    def analyze(self, spin_polarized: bool) -> LowdinResults:
        """Calculate Lowdin charges and magnetizations (if `spin_polarized`)."""
        ions = self.C.basis.ions
        Q = torch.zeros(ions.n_ions, device=qp.rc.device)  # TODO
        M = None  # TODO
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
