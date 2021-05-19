import qimpy as qp
import numpy as np
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._electrons import Electrons
    from ._wavefunction import Wavefunction


def _hamiltonian(self: 'Electrons', C: 'Wavefunction') -> 'Wavefunction':
    '''Apply electronic Hamiltonian applied on wavefunctions `C`'''
    basis = C.basis
    HC = basis.apply_ke(C)
    HC += basis.apply_potential(self.V_ks, C)
    return HC
