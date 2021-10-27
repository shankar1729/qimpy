from __future__ import annotations
import qimpy as qp


def _hamiltonian(
    self: qp.electrons.Electrons, C: qp.electrons.Wavefunction
) -> qp.electrons.Wavefunction:
    """Apply electronic Hamiltonian on wavefunction `C`"""
    basis = C.basis
    ions = basis.ions
    HC = basis.apply_ke(C)
    HC += basis.apply_potential(self.V_ks_t, C)
    # Nonlocal ps:
    beta_C = C.proj
    HC += ions.beta @ (ions.D_all @ beta_C)
    return HC
