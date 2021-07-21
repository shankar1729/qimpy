import qimpy as qp
import numpy as np
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._electrons import Electrons
    from ._wavefunction import Wavefunction


def _hamiltonian(self: 'Electrons', C: 'Wavefunction') -> 'Wavefunction':
    """Apply electronic Hamiltonian on wavefunction `C`"""
    basis = C.basis
    ions = basis.ions
    HC = basis.apply_ke(C)
    HC += basis.apply_potential(self.V_ks, C)
    # Nonlocal ps:
    beta_C = C.proj
    HC += ions.beta @ (ions.D_all @ beta_C)
    return HC
