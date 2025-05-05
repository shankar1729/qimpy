from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from qimpy.profiler import stopwatch
from qimpy.dft import electrons
from qimpy.math import random


@stopwatch(name="Wavefunction.randomize")
def _randomize(
    self: electrons.Wavefunction,
    seed: int = 0,
    b_start: int = 0,
    b_stop: Optional[int] = None,
) -> None:
    """Set wavefunction coefficients to bandwidth-limited random numbers.
    This is done reproducibly, regardless of MPI configuration of the run,
    by running a separate xor-shift random number generator with a different
    seed at each combination of spin, k, spinor and G-vector,
    looping only over the bands to generate the random number.

    Parameters
    ----------
    seed
        Seed offset at each k and iG for xor-shift random number generator
    b_start
        Starting band index (global index if MPI-split over bands)
    b_stop
        Stopping band index (global index if MPI-split over bands)
    """
    basis = self.basis

    # Range of bands and basis to operate on:
    if self.band_division is None:
        b_start_local = b_start
        b_stop_local = b_stop if b_stop else self.coeff.shape[2]
        n_bands_prev = b_start
        basis_start = basis.division.i_start
        basis_stop = basis.division.i_stop
        pad_index = basis.pad_index_mine
    else:
        b_start_local = max(b_start - self.band_division.i_start, 0)
        b_stop_local = (
            min(b_stop - self.band_division.i_start, self.band_division.n_mine)
            if b_stop
            else self.band_division.n_mine
        )
        n_bands_prev = self.band_division.i_start + b_start_local
        basis_start = 0
        basis_stop = basis.n_tot
        pad_index = basis.pad_index
    if b_start_local >= b_stop_local:
        return  # no bands to randomize on this process
    coeff_cur = self.coeff[:, :, b_start_local:b_stop_local]

    # Initialize random number state based on global coeff index and seed:
    def init_state(basis_index: torch.Tensor) -> torch.Tensor:
        n_spinor = basis.n_spinor
        i_spinor = torch.arange(n_spinor, device=self.coeff.device).view(1, 1, -1, 1)
        k_division = basis.kpoints.division
        i_k = torch.arange(
            k_division.i_start, k_division.i_stop, device=self.coeff.device
        ).view(1, -1, 1, 1)
        n_spins = basis.n_spins
        i_spin = torch.arange(n_spins, device=self.coeff.device).view(-1, 1, 1, 1)
        n_per_band = n_spins * k_division.n_tot * n_spinor * basis.n_max
        return (
            (1 + seed * n_per_band)
            + basis_index.view(1, 1, 1, -1)
            + basis.n_max * (i_spinor + n_spinor * (i_k + k_division.n_tot * i_spin))
        )

    # Create random complex numbers for each band with index-based seed:
    i_basis = torch.arange(basis_start, basis_stop, device=self.coeff.device)
    x = init_state(i_basis)
    random.initialize_state(x)
    for i_discard in range(n_bands_prev):
        random.randn(x)  # get RNG to position appropriate for starting band
    for b_local in range(b_stop_local - b_start_local):
        coeff_cur[:, :, b_local] = random.randn(x)

    # Enforce Hermitian symmetry in real case:
    if basis.real_wavefunctions:
        # Pick subset of iG_z = 0 in current process range:
        if self.band_division is None:
            iz0 = basis.real.iz0_mine_local
            iz0_conj = basis.real.iz0_mine_conj
        else:
            iz0 = basis.real.iz0
            iz0_conj = basis.real.iz0_conj
        # Create random complex numbers based on conjugate-index seed:
        x = init_state(iz0_conj)
        random.initialize_state(x)
        for i_discard in range(n_bands_prev):
            random.randn(x)  # get RNG to position appropriate for starting band
        for b_local in range(b_stop_local - b_start_local):
            coeff_cur[:, :, b_local, :, iz0] += random.randn(x).conj()
        coeff_cur[..., iz0] *= np.sqrt(0.5)  # keep variance = 1

    # Mask out inactive basis elements:
    coeff_cur[pad_index] = 0.0

    # Bandwidth limit:
    ke = basis.get_ke(slice(basis_start, basis_stop))[None, :, None, None, :]
    coeff_cur *= 1.0 / (1.0 + ((4.0 / 3) * ke) ** 6)  # damp-out high-KE coefficients


@stopwatch(name="Wavefunction.randomize_sel")
def _randomize_selected(
    self: electrons.Wavefunction,
    i_spin: torch.Tensor,
    i_k: torch.Tensor,
    i_band: torch.Tensor,
    seed: int,
) -> None:
    """Randomize wavefunction coefficients of selected bands.
    The bands are indexed by a tuple (`i_spin`, `i_k`, `i_band`) of 1D
    index tensors with same shape. This is only supported for wavefunctions
    split over basis (i.e. no band_division).
    """
    assert self.band_division is None
    basis = self.basis
    n_bands = self.n_bands()

    # Initialize random number state based on global coeff index and seed:
    def init_state(basis_index: torch.Tensor) -> torch.Tensor:
        i_spinor = torch.arange(basis.n_spinor, device=self.coeff.device).view(1, -1, 1)
        k_division = basis.kpoints.division
        i_k_global = (k_division.i_start + i_k).view(-1, 1, 1)
        n_per_band = basis.n_spins * k_division.n_tot * basis.n_spinor * basis.n_max
        return (
            (1 + seed * n_per_band)
            + basis_index.view(1, 1, -1)
            + basis.n_max
            * (
                i_spinor
                + basis.n_spinor
                * (
                    i_band.view(-1, 1, 1)
                    + n_bands * (i_k_global + k_division.n_tot * i_spin.view(-1, 1, 1))
                )
            )
        )

    # Create random complex numbers for each band with index-based seed:
    i_basis = torch.arange(
        basis.division.i_start, basis.division.i_stop, device=self.coeff.device
    )
    x = init_state(i_basis)
    random.initialize_state(x)
    self.coeff[(i_spin, i_k, i_band)] = random.randn(x)

    # Enforce Hermitian symmetry in real case:
    if basis.real_wavefunctions:
        # Create random complex numbers based on conjugate-index seed:
        iz0_local = basis.real.iz0_mine_local
        x = init_state(basis.real.iz0_mine_conj)
        random.initialize_state(x)
        self.coeff[(i_spin, i_k, i_band)][..., iz0_local] += random.randn(x).conj()

    # Mask out inactive basis elements:
    self.coeff[basis.pad_index_mine] = 0.0

    # Bandwidth limit:
    ke = basis.get_ke(basis.mine)[i_k, None, :]
    self.coeff[(i_spin, i_k, i_band)] *= 1.0 / (1.0 + ((4.0 / 3) * ke) ** 6)
