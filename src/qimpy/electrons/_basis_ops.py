import qimpy as qp
import numpy as np
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._basis import Basis
    from ._wavefunction import Wavefunction
    from ..grid import FieldR, FieldH


def _apply_ke(self: 'Basis', C: 'Wavefunction') -> 'Wavefunction':
    'Apply kinetic energy (KE) operator to wavefunction `C`'
    watch = qp.utils.StopWatch('Basis.apply_ke', self.rc)
    basis_slice = (slice(None) if C.band_division else self.mine)
    coeff = C.coeff * self.get_ke(basis_slice)[None, :, None, None, :]
    watch.stop()
    return qp.electrons.Wavefunction(self, coeff,
                                     band_division=C.band_division)


def _apply_potential(self: 'Basis', V: 'FieldH',
                     C: 'Wavefunction') -> 'Wavefunction':
    'Apply potential `V` to wavefunction `C`'
    Vdata_in = (~(V.to(self.grid))).data  # change to real space on basis grid
    n_densities = Vdata_in.shape[0]
    spin_dm_mode = (n_densities == 4)  # spin density-matrix mode
    if spin_dm_mode:
        assert C.coeff.shape[-2] == 2  # must be spinorial
        Vdata = torch.empty((2, 2) + Vdata_in.shape[1:], dtype=C.coeff.dtype,
                            device=self.rc.device)
        Vdata[0, 0] = (Vdata_in[0] + Vdata_in[3])
        Vdata[1, 1] = (Vdata_in[0] - Vdata_in[3])
        Vdata[1, 0] = Vdata_in[1] + 1j*Vdata_in[2]
        Vdata[0, 1] = Vdata_in[1] - 1j*Vdata_in[2]
    elif n_densities == 2:
        Vdata = torch.empty((2, 1, 1, 1) + Vdata_in.shape[1:],
                            dtype=C.coeff.dtype, device=self.rc.device)
        Vdata[0, 0, 0, 0] = (Vdata_in[0] + Vdata_in[1])
        Vdata[1, 0, 0, 0] = (Vdata_in[0] - Vdata_in[1])
    else:  # n_densities == 1:
        Vdata = Vdata_in[:, None, None, None]  # broadcast for spin

    # Move wavefunctions to band-split, basis-together position:
    need_move = (self.rc.n_procs_b > 1) and (C.band_division is None)
    VC = (C.split_bands() if need_move  # C with all G-vectors local
          else C.clone())  # copy, since V applied in place below
    coeff = VC.coeff

    # Determine FFT type and dimensions:
    watch = qp.utils.StopWatch('Basis.apply_potential', self.rc)
    shapeG = (self.grid.shapeH if self.real_wavefunctions else self.grid.shape)
    fft_nG = int(np.prod(shapeG))  # total reciprocal space points in FFT grid
    n_spins, nk, n_bands_mine, n_spinor = coeff.shape[:-1]
    ik = torch.arange(nk, device=coeff.device)[:, None]
    index = (slice(None), ik, slice(None), slice(None), self.fft_index)

    # Apply potential with blocked FFTs:
    n_blocks = qp.utils.ceildiv(n_bands_mine, self.fft_block_size)
    b_start = 0
    b_stop = self.fft_block_size
    b_size = self.fft_block_size
    Cb = torch.zeros((n_spins, nk, self.fft_block_size, n_spinor, fft_nG),
                     dtype=coeff.dtype, device=coeff.device)  # FFT buffer

    for iBlock in range(n_blocks):
        if b_stop > n_bands_mine:
            b_stop = n_bands_mine
            b_size = b_stop - b_start
            Cb = Cb[:, :, :b_size]
        # Expand -> ifft -> multiply V -> fft -> reduce back (on block)
        Cb[index] = coeff[:, :, b_start:b_stop].permute(1, 4, 0, 2, 3)
        VCb = self.grid.ifft(Cb.view((n_spins, nk, b_size, n_spinor) + shapeG))
        if spin_dm_mode:
            VCb = torch.einsum('uvxyz, skbvxyz -> skbuxyz', Vdata, VCb)
        else:
            VCb *= Vdata
        VCb = self.grid.fft(VCb).flatten(-3)
        coeff[:, :, b_start:b_stop] = VCb[index].permute(2, 0, 3, 4, 1)
        # Advance to next block of data:
        b_start = b_stop
        b_stop += b_size

    # Enforce hermitian conjugacy for real wavefunctions:
    if self.real_wavefunctions:
        coeff[..., self.index_z0] += coeff[..., self.index_z0_conj].conj()
        coeff[..., self.index_z0] *= 0.5
    watch.stop()

    # Restore V*C to the same configuration (basis or band-split) as C:
    return (VC.split_basis() if need_move else VC)


def _collect_density(self: 'Basis', C: 'Wavefunction', f: torch.Tensor,
                     need_Mvec: bool) -> 'FieldR':
    r"""Collect density contributions given wavefunction `C` and occupations
    `f`. The result is in real-space on `basis.grid`.
    If the wavefunction has two spin channels, the two components of
    the resulting density correspond to total density and magentization.
    If `need_Mvec` = True i.e. magnetization vector (`C` must be spinorial for
    this), the result contains 4 components: (n_tot, Mx, My, Mz).
    Here, (n_tot +/- Mz)/2 yield the :math:`\rho_{\uparrow\uparrow}` and
    :math:`\rho_{\downarrow\downarrow}` components of the spin density matrix,
    while (Mx +/- i My)/2 yield the :math:`\rho_{\uparrow\downarrow}` and
    :math:`\rho_{\downarrow\uparrow}` components of the spin density matrix.
    """
    assert(f.shape == C.coeff.shape[:3])
    C = C.split_bands()  # bring all G-vectors of each band together
    coeff = C.coeff
    if need_Mvec:
        assert coeff.shape[-2] == 2  # must be spinorial
    if C.band_division is not None:
        f = f[:, :, C.band_division.i_start:C.band_division.i_stop]
    prefac = (f * (self.w_sk / self.lattice.volume)).view(f.shape
                                                          + (1, 1, 1, 1))

    # Determine FFT type and dimensions:
    watch = qp.utils.StopWatch('Basis.collect_density', self.rc)
    shapeG = (self.grid.shapeH if self.real_wavefunctions else self.grid.shape)
    fft_nG = int(np.prod(shapeG))  # total reciprocal space points in FFT grid
    n_spins, nk, n_bands_mine, n_spinor = coeff.shape[:-1]
    ik = torch.arange(nk, device=coeff.device)[:, None]
    index = (slice(None), ik, slice(None), slice(None), self.fft_index)

    # Collect density with blocked FFTs:
    n_blocks = qp.utils.ceildiv(n_bands_mine, self.fft_block_size)
    b_start = 0
    b_stop = self.fft_block_size
    b_size = self.fft_block_size
    Cb = torch.zeros((n_spins, nk, self.fft_block_size, n_spinor, fft_nG),
                     dtype=coeff.dtype, device=coeff.device)  # FFT buffer
    rho_diag = torch.zeros((2 if need_Mvec else n_spins,)
                           + self.grid.shapeR_mine, device=coeff.device)
    if need_Mvec:
        rho_dn_up = torch.zeros(self.grid.shapeR_mine, dtype=coeff.dtype,
                                device=coeff.device)
    for iBlock in range(n_blocks):
        if b_stop > n_bands_mine:
            b_stop = n_bands_mine
            b_size = b_stop - b_start
            Cb = Cb[:, :, :b_size]
        # Expand -> ifft -> collect | |^2
        Cb[index] = coeff[:, :, b_start:b_stop].permute(1, 4, 0, 2, 3)
        ICb = self.grid.ifft(Cb.view((n_spins, nk, b_size, n_spinor) + shapeG))
        if need_Mvec:
            rho_diag += (prefac[:, :, b_start:b_stop]
                         * qp.utils.abs_squared(ICb)).sum(dim=(0, 1, 2))
            rho_dn_up += (prefac[:, :, b_start:b_stop, 0]
                          * ICb[:, :, :, 1].conj()
                          * ICb[:, :, :, 0]).sum(dim=(0, 1, 2))
        else:
            rho_diag += (prefac[:, :, b_start:b_stop]
                         * qp.utils.abs_squared(ICb)).sum(dim=(1, 2, 3))
        # Advance to next block of data:
        b_start = b_stop
        b_stop += b_size

    # Convert density matrix components to dneisty, magnetization:
    n_densities = (4 if need_Mvec else n_spins)
    density = qp.grid.FieldR(self.grid, shape_batch=(n_densities,))
    density.data[0] = rho_diag.sum(dim=0)  # n_tot
    if n_densities >= 2:
        density.data[-1] = rho_diag[0] - rho_diag[1]  # Mz
    if need_Mvec:
        density.data[1] = 2. * rho_dn_up.real  # Mx
        density.data[2] = 2. * rho_dn_up.imag  # My

    # Collect over MPI:
    if self.rc.n_procs_kb > 1:
        self.rc.comm_kb.Allreduce(qp.MPI.IN_PLACE,
                                  qp.utils.BufferView(density.data),
                                  qp.MPI.SUM)
    watch.stop()
    return density
