import qimpy as qp
import numpy as np
import torch


def _norm(self, mode='all'):
    '''Return overall, per-band or weighted norm of wavefunctions

    Parameters
    ----------
    mode: {'all', 'band', 'ke'}, default: 'all'
        If 'all', return a single scalar with the overall Frobenius norm of
        the wavefunction coefficients (across all spin, k-points and bands).
        If 'band', return a n_spins x nk_mine x n_bands tensor of floats
        containing the norm of each band.
        If 'ke', return a n_spins x nk_mine x n_bands tensor of floats
        containing the kinetic energy of each band.

    Returns
    -------
    float or torch.Tensor
    '''
    basis = self.basis
    if mode == 'all':
        # Overall Frobenius norm:
        return np.sqrt(basis.rc.comm_kb.allreduce(self.coeff.norm().item()**2,
                                                  qp.MPI.SUM))

    assert((mode == 'band') or (mode == 'ke'))
    assert(not self.band_division)
    coeff_sq = torch.abs(self.coeff)**2
    if mode == 'ke':
        coeff_sq *= basis.get_ke(basis.mine)[None, :, None, None, :]

    result = ((coeff_sq @ basis.Gweight_mine).sum(dim=-1)
              if basis.real_wavefunctions
              else coeff_sq.sum(dim=(-2, -1))) * basis.lattice.volume
    if basis.n_procs > 1:
        basis.rc.comm_b.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(result),
                                  op=qp.MPI.SUM)
    return result


def _sub(self, other):
    assert(self.basis is other.basis)
    coeff = self.coeff - other.coeff
    proj = ((self.proj - other.proj) if (self.proj and other.proj) else None)
    return qp.electrons.Wavefunction(self.basis, coeff, proj,
                                     self.band_division)
