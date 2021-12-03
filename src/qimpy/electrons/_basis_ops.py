from __future__ import annotations
import qimpy as qp
import numpy as np
import torch


def _apply_ke(
    self: qp.electrons.Basis, C: qp.electrons.Wavefunction
) -> qp.electrons.Wavefunction:
    "Apply kinetic energy (KE) operator to wavefunction `C`"
    watch = qp.utils.StopWatch("Basis.apply_ke", self.rc)
    basis_slice = slice(None) if C.band_division else self.mine
    coeff = C.coeff * self.get_ke(basis_slice)[None, :, None, None, :]
    watch.stop()
    return qp.electrons.Wavefunction(self, coeff=coeff, band_division=C.band_division)


def _apply_potential(
    self: qp.electrons.Basis, V: qp.grid.FieldH, C: qp.electrons.Wavefunction
) -> qp.electrons.Wavefunction:
    "Apply potential `V` to wavefunction `C`"
    watch = qp.utils.StopWatch("Basis.apply_potential", self.rc)
    Vdata_in = (~(V.to(self.grid))).data  # change to real space on basis grid
    n_densities = Vdata_in.shape[0]
    spin_dm_mode = n_densities == 4  # spin density-matrix mode
    if spin_dm_mode:
        assert C.coeff.shape[-2] == 2  # must be spinorial
        Vdata = torch.empty(
            (2, 2) + Vdata_in.shape[1:], dtype=C.coeff.dtype, device=self.rc.device
        )
        Vdata[0, 0] = Vdata_in[0] + Vdata_in[3]
        Vdata[1, 1] = Vdata_in[0] - Vdata_in[3]
        Vdata[1, 0] = Vdata_in[1] + 1j * Vdata_in[2]
        Vdata[0, 1] = Vdata[1, 0].conj()
    elif n_densities == 2:
        Vdata = torch.empty(
            (2, 1, 1, 1) + Vdata_in.shape[1:],
            dtype=Vdata_in.dtype,
            device=self.rc.device,
        )
        Vdata[0, 0, 0, 0] = Vdata_in[0] + Vdata_in[1]
        Vdata[1, 0, 0, 0] = Vdata_in[0] - Vdata_in[1]
    else:  # n_densities == 1:
        Vdata = Vdata_in[:, None, None, None]  # broadcast for spin
    apply_potential_kernel = _ApplyPotentialKernel(C, spin_dm_mode, Vdata)

    # Apply potential, moving data between processes as needed:
    need_move = (self.rc.n_procs_b > 1) and (C.band_division is None)
    if need_move:
        # Transfer and compute in blocks to allow communication-compute overlap:
        mpi_block_size = apply_potential_kernel.fft_block_size * self.rc.n_procs_b
        mpi_block_slices = qp.utils.get_block_slices(C.n_bands(), mpi_block_size)
        VC = C.zeros_like()

        # Prepare first input block ('g' indicates G-vectors of basis together):
        Cg = C[:, :, mpi_block_slices[0]].split_bands().wait()
        VCg: qp.electrons.Wavefunction  # created at end of first iteration below

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
        apply_potential_kernel(VC)

    watch.stop()
    return VC


class _ApplyPotentialKernel:
    """Internal compute kernel of Basis.apply_potential."""

    def __init__(
        self,
        C_tot: qp.electrons.Wavefunction,
        spin_dm_mode: bool,
        Vdata: torch.Tensor,
    ) -> None:
        """Initialize parameters given overall wavefunction `C` to be worked with.
        C is used only for determining sizes and is not stored.
        Subsequently, __call__ can be used with slices of C.
        """
        basis = C_tot.basis
        self.spin_dm_mode = spin_dm_mode
        self.Vdata = Vdata

        # Determine FFT type and dimensions:
        self.grid = basis.grid
        coeff = C_tot.coeff
        shapeG = self.grid.shapeH if basis.real_wavefunctions else self.grid.shape
        n_spins, nk, _, n_spinor, _ = coeff.shape
        ik = torch.arange(nk, device=coeff.device)[:, None]
        self.index = (slice(None), ik, slice(None), slice(None), basis.fft_index)

        # Initialize FFT buffer
        n_bands_mine_tot = (
            C_tot.n_bands()
            if C_tot.band_division
            else qp.utils.ceildiv(C_tot.n_bands(), basis.division.n_procs)
        )
        self.fft_block_size = basis.get_fft_block_size(n_spins * nk, n_bands_mine_tot)
        self.Cb = torch.zeros(
            (n_spins, nk, self.fft_block_size, n_spinor, int(np.prod(shapeG))),
            dtype=coeff.dtype,
            device=coeff.device,
        )
        self.fft_shape = (n_spins, nk, self.fft_block_size, n_spinor) + shapeG

    def __call__(self, C: qp.electrons.Wavefunction) -> None:
        """Apply potential to C in-place. C must be in bands-divided mode.
        Note that C could have a subset of bands of C_tot passed to __init__."""
        fft_block_slices = qp.utils.get_block_slices(C.n_bands(), self.fft_block_size)
        for fft_block_slice in fft_block_slices:
            b_size = fft_block_slice.stop - fft_block_slice.start
            if b_size < self.fft_block_size:
                Cb = self.Cb[:, :, :b_size]
                fft_shape = self.fft_shape[:2] + (b_size,) + self.fft_shape[3:]
            else:
                Cb, fft_shape = self.Cb, self.fft_shape
            # Expand -> ifft -> multiply V -> fft -> reduce back (on block)
            Cb[self.index] = C.coeff[:, :, fft_block_slice].permute(1, 4, 0, 2, 3)
            VCb = self.grid.ifft(Cb.view(fft_shape))
            if self.spin_dm_mode:
                VCb = torch.einsum("uvxyz, skbvxyz -> skbuxyz", self.Vdata, VCb)
            else:
                VCb *= self.Vdata
            VCb = self.grid.fft(VCb).flatten(-3)
            C.coeff[:, :, fft_block_slice] = VCb[self.index].permute(2, 0, 3, 4, 1)
        C.constrain()  # project out spurious entries (padding and real symmetry)
        self.result = C  # return in wait() when above is asynchronous

    def wait(self) -> qp.electrons.Wavefunction:
        return self.result


def _collect_density(
    self: qp.electrons.Basis,
    C: qp.electrons.Wavefunction,
    f: torch.Tensor,
    need_Mvec: bool,
) -> qp.grid.FieldR:
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
    watch = qp.utils.StopWatch("Basis.collect_density", self.rc)
    assert f.shape == C.coeff.shape[:3]
    C = C.split_bands().wait()  # bring all G-vectors of each band together
    coeff = C.coeff
    if need_Mvec:
        assert coeff.shape[-2] == 2  # must be spinorial
    if C.band_division is not None:
        f = f[:, :, C.band_division.i_start : C.band_division.i_stop]
    prefac = f * (self.w_sk / self.lattice.volume)

    # Determine FFT type and dimensions:
    shapeG = self.grid.shapeH if self.real_wavefunctions else self.grid.shape
    fft_nG = int(np.prod(shapeG))  # total reciprocal space points in FFT grid
    n_spins, nk, n_bands_mine, n_spinor = coeff.shape[:-1]
    ik = torch.arange(nk, device=coeff.device)[:, None]
    index = (slice(None), ik, slice(None), slice(None), self.fft_index)
    if not need_Mvec:
        # Make fillings prefactor broadcast with spinor (summed over below):
        prefac = prefac[..., None]
        if n_spinor > 1:
            prefac = prefac.tile((1, 1, 1, n_spinor))

    # Collect density with blocked FFTs:
    fft_block_size = self.get_fft_block_size(n_spins * nk, n_bands_mine)
    fft_block_slices = qp.utils.get_block_slices(n_bands_mine, fft_block_size)
    Cb = torch.zeros(
        (n_spins, nk, fft_block_size, n_spinor, fft_nG),
        dtype=coeff.dtype,
        device=coeff.device,
    )  # FFT buffer
    rho_diag = torch.zeros(
        (2 if need_Mvec else n_spins,) + self.grid.shapeR_mine, device=coeff.device
    )
    if need_Mvec:
        rho_dn_up = torch.zeros(
            self.grid.shapeR_mine, dtype=coeff.dtype, device=coeff.device
        )
    for fft_block_slice in fft_block_slices:
        b_size = fft_block_slice.stop - fft_block_slice.start
        if b_size < fft_block_size:
            Cb = Cb[:, :, :b_size]
        # Expand -> ifft -> collect | |^2
        Cb[index] = coeff[:, :, fft_block_slice].permute(1, 4, 0, 2, 3)
        ICb = self.grid.ifft(Cb.view((n_spins, nk, b_size, n_spinor) + shapeG))
        prefac_cur = prefac[:, :, fft_block_slice]
        if need_Mvec:
            qp.utils.accum_norm_(prefac_cur, ICb, out=rho_diag, start_dim=0)
            qp.utils.accum_prod_(
                prefac_cur,
                ICb[:, :, :, 1],
                ICb[:, :, :, 0].conj(),
                out=rho_dn_up,
                start_dim=0,
            )
        else:
            qp.utils.accum_norm_(prefac_cur, ICb, out=rho_diag, start_dim=1)

    # Convert density matrix components to dneisty, magnetization:
    n_densities = 4 if need_Mvec else n_spins
    density = qp.grid.FieldR(self.grid, shape_batch=(n_densities,))
    density.data[0] = rho_diag.sum(dim=0)  # n_tot
    if n_densities >= 2:
        density.data[-1] = rho_diag[0] - rho_diag[1]  # Mz
    if need_Mvec:
        density.data[1] = 2.0 * rho_dn_up.real  # Mx
        density.data[2] = 2.0 * rho_dn_up.imag  # My

    # Collect over MPI:
    if self.rc.n_procs_kb > 1:
        self.rc.comm_kb.Allreduce(
            qp.MPI.IN_PLACE, qp.utils.BufferView(density.data), qp.MPI.SUM
        )
    watch.stop()
    return density
