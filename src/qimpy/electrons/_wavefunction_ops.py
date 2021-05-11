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

    Parameters
    ----------
    other: qimpy.electrons.Wavefunction
        Dimensions must match self for spinor and basis, can differ for bands,
        and must be broadcastable for preceding dimensions (spin and k).

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
          if basis.real_wavefunctions
          else self.coeff).flatten(-2).conj()  # merge spinor & basis dims
    # Prepare right operand:
    C2 = other.coeff.flatten(-2).transpose(-2, -1)  # last dim now band2
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


def _matmul(self, mat):
    '''Compute a matrix transformation (such as rotation) along the band
    dimension of wavefunction (self). For convenience, this can also be
    invoked as self @ other, using the standard matrix multiply operator.

    Parameters
    ----------
    mat : torch.Tensor
        Last two dimensions specify transformation in band space,
        so final dimension should match n_bands of the wavefunction
        and penultimate dimension determines n_bands of output.
        Preceding dimensions should be broadcastable with spin and k

    Returns
    -------
    qimpy.electrons.Wavefunction
        The result will have n_bands = mat.shape[-2], same basis and spinor as
        input, and spin and k determined by broadcasting with mat.shape[:-2]
    '''
    if not isinstance(mat, torch.Tensor):
        return NotImplemented
    watch = qp.utils.StopWatch('Wavefunction.matmul', self.basis.rc)
    # Prepare input view:
    C = self.coeff.flatten(-2)  # merge spinor & basis dims at input
    C_out = mat.transpose(-2, -1) @ C
    C_out = C_out.view(C_out.shape[:-1] + self.coeff.shape[-2:])  # un-merge
    proj_out = ((self.proj @ mat) if self.proj else None)
    watch.stop()
    return qp.electrons.Wavefunction(self.basis, C_out, proj_out,
                                     self.band_division)


def _orthonormalize(self, use_cholesky=True):
    '''Return an orthonormalized version of present wavefunctions,
    using either a Gram-Schmidt scheme (faster) if use_cholesky=True,
    or using symmetric orthonormalization (stabler) if use_cholesky=False.
    See :meth:`qimpy.utils.ortho_matrix` for details'''
    return self @ qp.utils.ortho_matrix(self ^ self, use_cholesky)


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
