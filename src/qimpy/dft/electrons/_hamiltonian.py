from __future__ import annotations
import qimpy as qp


def _hamiltonian(
    self: qp.electrons.Electrons, C: qp.electrons.Wavefunction
) -> qp.electrons.Wavefunction:
    """Apply electronic Hamiltonian on wavefunction `C`"""
    basis = C.basis
    ions = basis.ions
    HC = basis.apply_ke(C)
    HC += basis.apply_potential(self.n_tilde.grad, C)
    if self.xc.need_tau:
        for i_dir in range(3):
            HC -= 0.5 * self.basis.apply_gradient(
                basis.apply_potential(
                    self.tau_tilde.grad, self.basis.apply_gradient(C, i_dir)
                ),
                i_dir,
            )
    # Nonlocal ps:
    beta_C = C.proj
    beta = ions.beta_full if C.band_division else ions.beta
    HC += beta @ (ions.D_all @ beta_C)
    return HC
