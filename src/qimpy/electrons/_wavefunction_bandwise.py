from __future__ import annotations
import qimpy as qp
import torch


def _band_norms(self: qp.electrons.Wavefunction, mode: str) -> torch.Tensor:
    """Shared implementation of _band_norm and _band_ke"""
    assert mode in ("norm", "ke")
    assert not self.band_division
    basis = self.basis
    coeff_sq = qp.utils.abs_squared(self.coeff)
    if mode == "ke":
        coeff_sq *= basis.get_ke(basis.mine)[None, :, None, None, :]
    result = (
        (coeff_sq @ basis.real.Gweight_mine).sum(dim=-1)
        if basis.real_wavefunctions
        else coeff_sq.sum(dim=(-2, -1))
    )
    if basis.division.n_procs > 1:
        qp.rc.current_stream_synchronize()
        basis.comm.Allreduce(
            qp.MPI.IN_PLACE, qp.utils.BufferView(result), op=qp.MPI.SUM
        )
    return result if (mode == "ke") else result.sqrt()


def _band_norm(self: qp.electrons.Wavefunction) -> torch.Tensor:
    """Return per-band norm of wavefunctions"""
    return _band_norms(self, "norm")


def _band_ke(self: qp.electrons.Wavefunction) -> torch.Tensor:
    """Return per-band norm of wavefunctions"""
    return _band_norms(self, "ke")


def _band_spin(self: qp.electrons.Wavefunction) -> torch.Tensor:
    """Return per-band spin of wavefunctions (must be spinorial).
    Result dimensions are 3 x 1 x nk x n_bands."""
    assert not self.band_division
    basis = self.basis
    coeff = self.coeff
    assert coeff.shape[-2] == 2  # must be spinorial
    # Compute spin density matrix per band:
    rho_s = (
        torch.einsum(
            "skbxg, g, skbyg -> skbxy", coeff, basis.real.Gweight_mine, coeff.conj()
        )
        if basis.real_wavefunctions
        else torch.einsum("skbxg, skbyg -> skbxy", coeff, coeff.conj())
    )
    if basis.division.n_procs > 1:
        qp.rc.current_stream_synchronize()
        basis.comm.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(rho_s), op=qp.MPI.SUM)
    # Convert to spin vector:
    result = torch.zeros((3,) + coeff.shape[:3], device=coeff.device)
    result[0] = 2.0 * rho_s[..., 1, 0].real  # Sx
    result[1] = 2.0 * rho_s[..., 1, 0].imag  # Sy
    result[2] = rho_s[..., 0, 0].real - rho_s[..., 1, 1].real  # Sz
    return result


def _constrain(self: qp.electrons.Wavefunction) -> None:
    """Enforce basis constraints on wavefunction coefficients.
    This includes setting padded coefficients to zero, and imposing
    Hermitian symmetry in Gz = 0 coefficients for real wavefunctions.
    """
    basis = self.basis
    # Padded coefficients:
    pad_index = basis.pad_index if self.band_division else basis.pad_index_mine
    self.coeff[pad_index] = 0.0
    # Real wavefunction symmetry:
    if basis.real_wavefunctions:
        basis.real.symmetrize(self.coeff)
