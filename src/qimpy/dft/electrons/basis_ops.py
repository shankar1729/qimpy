from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from mpi4py import MPI

from qimpy import rc
from qimpy.profiler import stopwatch
from qimpy.math import ceildiv, accum_prod_, accum_norm_
from qimpy.grid import FieldR, FieldH
from qimpy.mpi import get_block_slices, Waitable, Iallreduce_in_place
from qimpy.dft import electrons


@stopwatch(name="Basis.apply_gradient")
def _apply_gradient(
    self: electrons.Basis,
    C: electrons.Wavefunction,
    i_dir: int,
) -> electrons.Wavefunction:
    "Apply gradient operator to wavefunction `C`"
    basis_slice = slice(None) if C.band_division else self.mine
    coeff = torch.einsum(
        "sknpb,kb->sknpb", C.coeff, self.get_gradient(basis_slice)[:, :, i_dir]
    )
    return electrons.Wavefunction(self, coeff=coeff, band_division=C.band_division)


@stopwatch(name="Basis.apply_ke")
def _apply_ke(
    self: electrons.Basis, C: electrons.Wavefunction
) -> electrons.Wavefunction:
    "Apply kinetic energy (KE) operator to wavefunction `C`"
    basis_slice = slice(None) if C.band_division else self.mine
    coeff = C.coeff * self.get_ke(basis_slice)[None, :, None, None, :]
    return electrons.Wavefunction(self, coeff=coeff, band_division=C.band_division)


@stopwatch(name="Basis.apply_potential")
def _apply_potential(
    self: electrons.Basis, V: FieldH, C: electrons.Wavefunction
) -> electrons.Wavefunction:
    "Apply potential `V` to wavefunction `C`"
    Vdata_in = (~(V.to(self.grid))).data  # change to real space on basis grid
    n_densities = Vdata_in.shape[0]
    spin_dm_mode = n_densities == 4  # spin density-matrix mode
    if spin_dm_mode:
        assert C.coeff.shape[-2] == 2  # must be spinorial
        Vdata = torch.empty(
            (2, 2) + Vdata_in.shape[1:], dtype=C.coeff.dtype, device=rc.device
        )
        Vdata[0, 0] = Vdata_in[0] + Vdata_in[3]
        Vdata[1, 1] = Vdata_in[0] - Vdata_in[3]
        Vdata[1, 0] = Vdata_in[1] + 1j * Vdata_in[2]
        Vdata[0, 1] = Vdata[1, 0].conj()
    elif n_densities == 2:
        Vdata = torch.empty(
            (2, 1, 1, 1) + Vdata_in.shape[1:],
            dtype=Vdata_in.dtype,
            device=rc.device,
        )
        Vdata[0, 0, 0, 0] = Vdata_in[0] + Vdata_in[1]
        Vdata[1, 0, 0, 0] = Vdata_in[0] - Vdata_in[1]
    else:  # n_densities == 1:
        Vdata = Vdata_in[:, None, None, None]  # broadcast for spin
    apply_potential_kernel = _ApplyPotentialKernel(C, spin_dm_mode, Vdata)

    # Apply potential, moving data between processes as needed:
    if apply_potential_kernel.need_move:
        # Transfer and compute in blocks to allow communication-compute overlap:
        mpi_block_slices = apply_potential_kernel.mpi_block_slices
        VC = C.zeros_like()

        # Prepare first input block ('g' indicates G-vectors of basis together):
        Cg = C[:, :, mpi_block_slices[0]].split_bands().wait()
        VCg: electrons.Wavefunction  # created at end of first iteration below

        for mpi_block_slice_prev, mpi_block_slice_next in zip(
            (None, *mpi_block_slices[:-1]), (*mpi_block_slices[1:], None)
        ):

            # Start communication of previous output block:
            if mpi_block_slice_prev:
                VC_prev = VCg.split_basis()

            # Start communication of next input block:
            if mpi_block_slice_next:  # get started on next block
                Cg_next = C[:, :, mpi_block_slice_next].split_bands()

            # Start compute:
            apply_potential_kernel(Cg)

            # Finish communication of previous output block:
            if mpi_block_slice_prev:
                VC[:, :, mpi_block_slice_prev] = VC_prev.wait()

            # Finish communication of next input block:
            if mpi_block_slice_next:
                Cg = Cg_next.wait()

            # Finish compute:
            VCg = apply_potential_kernel.wait()

        # Finish final output block:
        VC[:, :, mpi_block_slices[-1]] = VCg.split_basis().wait()

    else:
        # Basis together already => no transfers needed
        VC = C.clone()
        apply_potential_kernel(VC).wait()

    VC.constrain()  # project out spurious entries (padding and real symmetry)
    return VC


class _KernelCommon:
    """Common functionality between _ApplyPotentialKernel and _CollectDensityKernel."""

    def __init__(self, C_tot: electrons.Wavefunction) -> None:
        # Determine FFT type and dimensions:
        basis = C_tot.basis
        coeff = C_tot.coeff
        self.grid = basis.grid
        shapeG = self.grid.shapeH if basis.real_wavefunctions else self.grid.shape
        n_spins, nk, _, n_spinor, _ = coeff.shape
        ik = torch.arange(nk, device=coeff.device)[:, None]
        self.index = (slice(None), ik, slice(None), slice(None), basis.fft_index)

        # Initialize FFT buffer
        n_bands_mine_tot = (
            C_tot.n_bands()
            if C_tot.band_division
            else ceildiv(C_tot.n_bands(), basis.division.n_procs)
        )
        self.fft_block_size = basis.get_fft_block_size(n_spins * nk, n_bands_mine_tot)
        self.fft_shape = (n_spins, nk, self.fft_block_size, n_spinor) + shapeG
        self.Cb = torch.zeros(
            (n_spins, nk, self.fft_block_size, n_spinor, int(np.prod(shapeG))),
            dtype=coeff.dtype,
            device=coeff.device,
        )

        # Initialize MPI blocks (if needed):
        self.need_move = (basis.division.n_procs > 1) and (C_tot.band_division is None)
        if self.need_move:
            n_bands = C_tot.n_bands()
            mpi_block_size = basis.get_mpi_block_size(
                int(np.prod(coeff.shape[:2])), n_bands, self.fft_block_size
            )
            self.mpi_block_slices = get_block_slices(n_bands, mpi_block_size)

    def expand_ifft(self, coeff: torch.Tensor, block_slice: slice) -> torch.Tensor:
        """Expand and ifft `block_slice` of wavefunction coefficients `coeff`."""
        # Get slice of fft buffer and shape suitable for block_slice:
        b_size = block_slice.stop - block_slice.start
        if b_size < self.fft_block_size:
            Cb = self.Cb[:, :, :b_size]
            fft_shape = self.fft_shape[:2] + (b_size,) + self.fft_shape[3:]
        else:
            Cb, fft_shape = self.Cb, self.fft_shape
        # Expand slice of coeff into Cb and ifft:
        Cb[self.index] = coeff[:, :, block_slice].permute(1, 4, 0, 2, 3)
        return self.grid.ifft(Cb.view(fft_shape))


class _ApplyPotentialKernel(_KernelCommon):
    """Internal compute kernel of Basis.apply_potential."""

    def __init__(
        self,
        C_tot: electrons.Wavefunction,
        spin_dm_mode: bool,
        Vdata: torch.Tensor,
    ) -> None:
        """Initialize parameters given overall wavefunction `C_tot` to be worked with.
        C_tot is used only for determining sizes and is not stored.
        Subsequently, __call__ can be used with slices of C_tot.
        """
        super().__init__(C_tot)
        self.spin_dm_mode = spin_dm_mode
        self.Vdata = Vdata

    def __call__(self, C: electrons.Wavefunction) -> Waitable[electrons.Wavefunction]:
        """Apply potential to C in-place. C must be in bands-divided mode.
        Note that C could have a subset of bands of C_tot passed to __init__."""
        fft_block_slices = get_block_slices(C.n_bands(), self.fft_block_size)
        rc.compute_stream_wait_current()
        with torch.cuda.stream(rc.compute_stream):
            for fft_block_slice in fft_block_slices:
                # Expand -> ifft -> multiply V -> fft -> reduce back (on block)
                VCb = self.expand_ifft(C.coeff, fft_block_slice)
                if self.spin_dm_mode:
                    VCb = torch.einsum("uvxyz, skbvxyz -> skbuxyz", self.Vdata, VCb)
                else:
                    VCb *= self.Vdata
                VCb = self.grid.fft(VCb).flatten(-3)
                C.coeff[:, :, fft_block_slice] = VCb[self.index].permute(2, 0, 3, 4, 1)
        self.result = C  # return in wait() when above is asynchronous
        return self  # so that the output is Waitable

    def wait(self) -> electrons.Wavefunction:
        """Wait for completion (if running in separate stream)."""
        rc.current_stream_wait_compute()
        return self.result


@stopwatch(name="Basis.collect_density")
def _collect_density(
    self: electrons.Basis,
    C: electrons.Wavefunction,
    f: torch.Tensor,
    need_Mvec: bool,
) -> FieldR:
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
    assert f.shape == C.coeff.shape[:3]
    n_spins, _, _, n_spinor, _ = C.coeff.shape
    if need_Mvec:
        assert n_spinor == 2
    prefac = f * (self.w_sk / self.lattice.volume)
    if not need_Mvec:
        # Make fillings prefactor broadcast with spinor (summed over below):
        prefac = prefac[..., None]
        if n_spinor > 1:
            prefac = prefac.tile((1, 1, 1, n_spinor))
    prefac = prefac.cpu()  # accum_prod / accum_norm use prefac on cpu

    # Prepare outputs and compute kernrl:
    rho_diag = torch.zeros(
        (2 if need_Mvec else n_spins,) + self.grid.shapeR_mine, device=C.coeff.device
    )
    rho_dn_up = (
        torch.zeros(self.grid.shapeR_mine, dtype=C.coeff.dtype, device=C.coeff.device)
        if need_Mvec
        else None
    )
    collect_density_kernel = _CollectDensityKernel(C, rho_diag, rho_dn_up)

    # Collect density, moving data between processes as needed:
    if collect_density_kernel.need_move:
        # Transfer and compute in blocks to allow communication-compute overlap:
        mpi_block_slices = collect_density_kernel.mpi_block_slices

        # Prepare first input block ('g' indicates G-vectors of basis together):
        Cg = C[:, :, mpi_block_slices[0]].split_bands().wait()
        prefac_cur = prefac[:, :, mpi_block_slices[0]]

        for mpi_block_slice_next in (*mpi_block_slices[1:], None):

            # Start communication of next input block:
            if mpi_block_slice_next:  # get started on next block
                Cg_next = C[:, :, mpi_block_slice_next].split_bands()

            # Start compute:
            collect_density_kernel(Cg, prefac_cur)

            # Finish communication of next input block:
            if mpi_block_slice_next:
                Cg = Cg_next.wait()
                prefac_cur = prefac[:, :, mpi_block_slice_next]

            # Finish compute:
            collect_density_kernel.wait()
    else:
        collect_density_kernel(C, prefac).wait()

    # Convert density matrix components to density, magnetization:
    n_densities = 4 if need_Mvec else n_spins
    density = FieldR(self.grid, shape_batch=(n_densities,))
    density.data[0] = rho_diag.sum(dim=0)  # n_tot
    if n_densities >= 2:
        density.data[-1] = rho_diag[0] - rho_diag[1]  # Mz
    if rho_dn_up is not None:
        density.data[1] = 2.0 * rho_dn_up.real  # Mx
        density.data[2] = 2.0 * rho_dn_up.imag  # My

    # Collect over MPI:
    if self.comm_kb.size > 1:
        rc.current_stream_synchronize()
        Iallreduce_in_place(self.comm_kb, density.data, MPI.SUM).wait()
    return density


class _CollectDensityKernel(_KernelCommon):
    """Internal compute kernel of Basis.collect_density."""

    def __init__(
        self,
        C_tot: electrons.Wavefunction,
        rho_diag: torch.Tensor,
        rho_dn_up: Optional[torch.Tensor],
    ) -> None:
        """Initialize parameters given overall wavefunction `C_tot` to be worked with.
        C_tot is used only for determining sizes and is not stored.
        Subsequently, __call__ can be used with slices of C_tot.
        """
        super().__init__(C_tot)
        self.rho_diag = rho_diag
        self.rho_dn_up = rho_dn_up

    def __call__(
        self, C: electrons.Wavefunction, prefac: torch.Tensor
    ) -> Waitable[None]:
        """Collect density from wave-function `C` with prefactors `prefac`
        (related to fillings). C must be in bands-divided mode.
        Note that C could have a subset of bands of C_tot passed to __init__."""
        fft_block_slices = get_block_slices(C.n_bands(), self.fft_block_size)
        rc.compute_stream_wait_current()
        with torch.cuda.stream(rc.compute_stream):
            prefac_mine = (
                prefac[:, :, C.band_division.i_start : C.band_division.i_stop]
                if C.band_division
                else prefac
            )
            for fft_block_slice in fft_block_slices:
                # Expand -> ifft -> collect | |^2
                ICb = self.expand_ifft(C.coeff, fft_block_slice)
                prefac_cur = prefac_mine[:, :, fft_block_slice]
                if self.rho_dn_up is not None:  # vector-magnetization mode
                    accum_norm_(prefac_cur, ICb, self.rho_diag, start_dim=0)
                    accum_prod_(
                        prefac_cur,
                        ICb[:, :, :, 1],
                        ICb[:, :, :, 0].conj(),
                        self.rho_dn_up,
                        start_dim=0,
                    )
                else:
                    accum_norm_(prefac_cur, ICb, self.rho_diag, start_dim=1)
        return self  # so that the output is Waitable

    def wait(self) -> None:
        """Wait for completion (if running in separate stream)."""
        rc.current_stream_wait_compute()
