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


def _overlap(self, other):
    '''Compute the overlap of this wavefunction (self) with other.
    For convenience, this can also be invoked as self ^ other.
    Note that this means xor(self, other) also computes wavefunction overlap
    instead of logical xor (which is meaningless for wavefunctions anyway)

    Returns
    -------
    torch.Tensor with dimensions
    n_spins x nk_mine x n_bands(self) x n_bands(other)
    '''
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    basis = self.basis
    assert(basis is other.basis)
    assert(not self.band_division)
    watch = qp.utils.StopWatch('Wavefunction.overlap', basis.rc)
    # Prepare left operand:
    C1 = (self.coeff * basis.Gweight_mine.view(1, 1, 1, 1, -1)
          if basis.real_wavefunctions else self.coeff)
    C1 = C1.view(C1.shape[:-2] + (-1,)).conj()  # merge spinor & basis dims
    # Prepare right operand:
    C2 = other.coeff
    C2 = C2.view(C2.shape[:-2] + (-1,)).transpose(-2, -1)  # last dim now band2
    # Compute local overlap and reduce:
    result = (C1 @ C2) * basis.lattice.volume
    # TODO: overlap augmentation goes here when adding ultrasoft / PAW
    if basis.real_wavefunctions:
        result.imag *= 0.  # due to implicit +h.c. terms in C1 and C2
    if basis.n_procs > 1:
        basis.rc.comm_b.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(result),
                                  op=qp.MPI.SUM)
    watch.stop()
    return result


def _mul(self, scale):
    if not isinstance(scale, float):
        return NotImplemented
    coeff = self.coeff * scale
    proj = ((self.proj * scale) if self.proj else None)
    return qp.electrons.Wavefunction(self.basis, coeff, proj,
                                     self.band_division)


def _imul(self, scale):
    if not isinstance(scale, float):
        return NotImplemented
    self.coeff *= scale
    if self.proj:
        self.proj *= scale
    return self


def _add(self, other):
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert(self.basis is other.basis)
    coeff = self.coeff + other.coeff
    proj = ((self.proj + other.proj) if (self.proj and other.proj) else None)
    return qp.electrons.Wavefunction(self.basis, coeff, proj,
                                     self.band_division)


def _iadd(self, other):
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert(self.basis is other.basis)
    self.coeff += other.coeff
    if self.proj and other.proj:
        self.proj += other.proj
    else:
        self.proj = None
    return self


def _sub(self, other):
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert(self.basis is other.basis)
    coeff = self.coeff - other.coeff
    proj = ((self.proj - other.proj) if (self.proj and other.proj) else None)
    return qp.electrons.Wavefunction(self.basis, coeff, proj,
                                     self.band_division)


def _isub(self, other):
    if not isinstance(other, qp.electrons.Wavefunction):
        return NotImplemented
    assert(self.basis is other.basis)
    self.coeff -= other.coeff
    if self.proj and other.proj:
        self.proj -= other.proj
    else:
        self.proj = None
    return self
