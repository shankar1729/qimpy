import qimpy as qp
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._basis import Basis


class BasisReal:
    """Extra book-keeping for real basis"""
    __slots__ = ('index_z0', 'index_z0_conj', 'Gweight_mine')
    index_z0: torch.Tensor  #: Index of Gz = 0 points
    index_z0_conj: torch.Tensor  #: Hermitian conjugate points of `index_z0`
    Gweight_mine: torch.Tensor  #: Weight of local plane waves

    def __init__(self, basis: 'Basis'):
        """Initialize extra indexing required for real wavefunctions,
        if needed."""
        assert basis.real_wavefunctions and basis.kpoints.division.n_mine
        div = basis.division
        rc = basis.rc

        # Find conjugate pairs with iG_z = 0:
        iGz = basis.iG[0, :, 2]
        self.index_z0 = torch.where(iGz == 0)[0]
        # --- compute index of each point and conjugate in iG_z = 0 plane:
        shapeH = basis.grid.shapeH_mine
        plane_index = basis.fft_index[0, self.index_z0].div(
            shapeH[2], rounding_mode='floor')
        iG_conj = (-basis.iG[0, self.index_z0, :2]) % torch.tensor(
            shapeH[:2], device=rc.device)[None, :]
        plane_index_conj = iG_conj[:, 0] * shapeH[1] + iG_conj[:, 1]
        # --- map plane_index_conj to basis using full plane for look-up:
        plane = torch.zeros(shapeH[0] * shapeH[1],
                            dtype=self.index_z0.dtype, device=rc.device)
        plane[plane_index] = self.index_z0
        self.index_z0_conj = plane[plane_index_conj].clone().detach()

        # Weight by element for overlaps (only for this process portion):
        iGz_mine = iGz[div.i_start:div.i_stop]
        self.Gweight_mine = torch.zeros(div.n_each, device=rc.device)
        self.Gweight_mine[:div.n_mine] = torch.where(iGz_mine == 0, 1., 2.)
        Gweight_sum = qp.utils.globalreduce.sum(self.Gweight_mine, rc.comm_b)
        qp.log.info(f'real basis weight sum: {Gweight_sum:g}')
