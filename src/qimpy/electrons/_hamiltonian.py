import qimpy as qp
import numpy as np
import torch


def _hamiltonian(self, C):
    '''Return H(C) = electronic Hamiltonian applied on wavefunctions C'''
    basis = C.basis
    basis_slice = (slice(None) if C.band_division else basis.mine)
    ke_op = (basis.lattice.volume
             * basis.get_ke(basis_slice)[None, :, None, None, :])
    return qp.electrons.Wavefunction(basis, coeff=(C.coeff * ke_op),
                                     band_division=C.band_division)
    # TODO: add a potential!!!
