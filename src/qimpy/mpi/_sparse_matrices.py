from typing import Sequence, Optional

import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc
from qimpy.mpi import TaskDivision, all_gather_padded
from torch import sparse_coo_tensor


class SparseMatrixRight:
    split: TaskDivision
    indices: torch.Tensor
    values: torch.Tensor
    size: tuple  # size of dense matrix
    iRow_mine: torch.Tensor
    iCol_mine: torch.Tensor
    value_mine: torch.Tensor
    M_mine: torch.Tensor
    group: Optional[dist.ProcessGroup]  #: Process group to split matrix over
    n_procs: int  #: Size of comm
    i_proc: int  #: Rank within comm

    def __init__(
        self,
        indices: Sequence[int],
        values: torch.Tensor,
        *,
        group: Optional[dist.ProcessGroup],
    ) -> None:
        self.indices = indices
        self.values = values
        self.group = group
        self.n_procs = 1 if group is None else group.size()
        self.i_proc = 0 if group is None else group.rank()
        iRow, iCol = indices
        self.size = (iRow.max() + 1, iCol.max() + 1)
        self.split = TaskDivision(
            n_tot=self.size[1].cpu(), n_procs=self.n_procs, i_proc=self.i_proc
        )
        split = self.split
        sel = torch.nonzero(
            torch.logical_and(iCol >= split.i_start, iCol < split.i_stop)
        ).flatten()

        self.iRow_mine = iRow[sel]
        self.iCol_mine = iCol[sel] - split.i_start
        self.value_mine = values[sel]
        indices_mine = torch.stack([self.iRow_mine, self.iCol_mine])
        counts = np.diff(self.split.n_prev)
        nCols_mine = counts[self.i_proc]
        self.M_mine = sparse_coo_tensor(
            indices_mine,
            self.value_mine,
            size=(iRow.max() + 1, nCols_mine),
            device=rc.device,
        ).to_sparse_csr()

    def getM(self):
        return sparse_coo_tensor(
            self.indices, self.values, device=rc.device
        )  # .to_sparse_csr()

    def vecTimesMatrix(self, vec: torch.Tensor) -> torch.Tensor:
        assert len(vec.shape) == 1, "Need to pass 1D vector to vecTimesMatrix"
        result = vec @ self.M_mine
        if self.n_procs == 1:
            return result
        else:
            return all_gather_padded(result, np.diff(self.split.n_prev), self.group)
