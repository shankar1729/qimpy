import qimpy as qp
import numpy as np
import torch


def _hamiltonian(self, C):
    '''Return H(C) = electronic Hamiltonian applied on wavefunctions C'''
    basis = C.basis
    HC = basis.apply_ke(C)
    HC += basis.apply_potential(self.V_ks, C)
    return HC
