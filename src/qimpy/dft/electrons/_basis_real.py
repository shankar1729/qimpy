from __future__ import annotations
import numpy as np
import torch

from qimpy import rc, log
from qimpy.mpi import get_comm, all_gather_padded
from qimpy.dft import electrons


class BasisReal:
    """Extra book-keeping for real basis"""

    basis: electrons.Basis
    iz0: torch.Tensor  #: Index of Gz = 0 points
    iz0_conj: torch.Tensor  #: Hermitian conjugate points of `iz0`
    iz0_conj_self: torch.Tensor  #: Conjugate indices within Gz = 0 set
    iz0_mine_local: torch.Tensor  #: Local Gz = 0 indices on current process
    iz0_mine_conj: torch.Tensor  #: Global conjugates of `iz0_mine_local`
    nz0_each: np.ndarray  #: Number of Gz = 0 at each process
    nz0_prev: np.ndarray  #: Number of Gz = 0 entries before each process
    Gweight: torch.Tensor  #: Weight of all plane waves
    Gweight_mine: torch.Tensor  #: Weight of local plane waves
    Gweight_tot: float  #: Total weight of all plane waves

    def __init__(self, basis: electrons.Basis):
        """Initialize extra indexing required for real wavefunctions,
        if needed."""
        assert basis.real_wavefunctions and basis.kpoints.division.n_mine
        self.basis = basis
        div = basis.division

        # Find conjugate pairs with iG_z = 0:
        iGz = basis.iG[0, :, 2]
        self.iz0 = torch.where(iGz == 0)[0]
        # --- compute index of each point and conjugate in iG_z = 0 plane:
        shapeH = basis.grid.shapeH_mine
        plane_index = basis.fft_index[0, self.iz0].div(shapeH[2], rounding_mode="floor")
        iG_conj = (-basis.iG[0, self.iz0, :2]) % torch.tensor(
            shapeH[:2], device=rc.device
        )[None, :]
        plane_index_conj = iG_conj[:, 0] * shapeH[1] + iG_conj[:, 1]
        # --- map plane_index_conj to basis using full plane for look-up:
        plane = torch.zeros(
            shapeH[0] * shapeH[1], dtype=self.iz0.dtype, device=rc.device
        )
        plane[plane_index] = self.iz0
        self.iz0_conj = plane[plane_index_conj].clone().detach()
        # --- similar mapping within the Gz = 0 set:
        plane[plane_index] = torch.arange(len(plane_index), device=rc.device)
        self.iz0_conj_self = plane[plane_index_conj].clone().detach()

        # Extract local portions of above:
        mine = torch.where(
            torch.logical_and(self.iz0 >= div.i_start, self.iz0 < div.i_stop)
        )[0]
        self.iz0_mine_local = self.iz0[mine] - div.i_start
        self.iz0_mine_conj = self.iz0_conj[mine]
        self.nz0_each = np.array(get_comm(basis.group).allgather(len(mine)))
        self.nz0_prev = np.cumsum(np.concatenate(([0], self.nz0_each)))

        # Weight by element for overlaps:
        self.Gweight = torch.where(iGz == 0, 1.0, 2.0)
        self.Gweight[basis.n_max :] = 0.0  # padded elements
        self.Gweight_mine = self.Gweight[div.i_start : div.i_stop]
        self.Gweight_tot = self.Gweight.sum().item()
        log.info(f"real basis weight sum: {self.Gweight_tot:g}")

    def symmetrize(self, coeff: torch.Tensor) -> None:
        """Impose Hermitian symmetry constraint on Gz = 0 coefficients."""
        basis = self.basis

        # Collect all the z0 coefficients:
        is_split = not (coeff.shape[-1] == basis.n_tot)
        if is_split:
            # Bring basis to front for gather:
            coeff_z0_mine = coeff[..., self.iz0_mine_local].permute(4, 0, 1, 2, 3)
            coeff_z0 = all_gather_padded(coeff_z0_mine, self.nz0_each, basis.group)
            coeff_z0 = coeff_z0.permute(1, 2, 3, 4, 0)  # put basis back at end
        else:  # All coefficients local already:
            coeff_z0 = coeff[..., self.iz0]

        # Symmetrize:
        coeff_z0 = 0.5 * (coeff_z0 + coeff_z0[..., self.iz0_conj_self].conj())

        # Set the symmetrized coefficients:
        if is_split:
            z0_start = self.nz0_prev[basis.division.i_proc]
            z0_stop = self.nz0_prev[basis.division.i_proc + 1]
            coeff[..., self.iz0_mine_local] = coeff_z0[..., z0_start:z0_stop]
        else:
            coeff[..., self.iz0] = coeff_z0
