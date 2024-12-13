from typing import Sequence

import numpy as np
import scipy as sp
import torch

from qimpy import log, MPI, rc
from qimpy.grid import Grid
from qimpy.mpi import TaskDivision, BufferView
from qimpy.grid._change import gather
from torch import sparse_coo_tensor


class SparseMatrixRight:
    split: TaskDivision
    indices: torch.Tensor
    values: torch.Tensor
    size: tuple # size of dense matrix
    iRow_mine: torch.Tensor
    iCol_mine: torch.Tensor
    value_mine: torch.Tensor
    M_mine: torch.Tensor
    comm: MPI.Comm  #: Communicator to split matrix over
    n_procs: int  #: Size of comm
    i_proc: int  #: Rank within comm

    def __init__(
        self,
        indices: Sequence[int],
        values: torch.Tensor,
        *,
        comm: MPI.Comm,
    ) -> None:
        self.indices = indices
        self.values = values
        self.comm = comm
        self.n_procs, self.i_proc = (
            (1, 0) if (comm is None) else (comm.Get_size(), comm.Get_rank())
        )
        iRow, iCol = indices
        self.size = (iRow.max() + 1, iCol.max() + 1)
        self.split = TaskDivision(
            n_tot=self.size[1], n_procs=self.n_procs, i_proc=self.i_proc
        )
        split = self.split
        sel = torch.nonzero(
            torch.logical_and(iCol >= split.i_start, iCol < split.i_stop)).flatten()

        self.iRow_mine = iRow[sel]
        self.iCol_mine = iCol[sel] - split.i_start
        self.value_mine = values[sel]
        indices_mine = torch.stack([self.iRow_mine,self.iCol_mine])
        counts = np.diff(self.split.n_prev)
        nCols_mine = counts[self.i_proc]
        self.M_mine = sparse_coo_tensor(indices_mine, self.value_mine,
                                        size=(iRow.max()+1, nCols_mine),
                                        device=rc.device).to_sparse_csr()

    def getM(self):
        return sparse_coo_tensor(self.indices, self.values,
                                 device=rc.device)#.to_sparse_csr()

    def vecTimesMatrix(self, vec_mine: torch.Tensor, grid: Grid = None) -> torch.Tensor:
        if self.n_procs == 1:
            return vec_mine @ self.M_mine
        #assert len(vec_mine.shape) == 1, "Need to pass 1D vector to vecTimesMatrix"
        if grid is None:
            vec = vec_mine # vector is not parallelized
        else:
            """counts = np.diff(grid.split0.n_prev)*np.prod(grid.shape[1:])
            disps = np.array(grid.split0.n_prev[:-1]*np.prod(grid.shape[1:]))

            vec = torch.empty(self.size[0], dtype=vec_mine.dtype, device=rc.device)
            print("vec sizes:", vec_mine.shape, vec.shape)
            print(counts, disps)

            mpi_type = rc.mpi_type[vec_mine.dtype]
            rc.current_stream_synchronize()
            grid.comm.Allgatherv(
                    (BufferView(vec_mine.contiguous()), vec_mine.shape[0],0, mpi_type),
                    (BufferView(vec), counts, disps, mpi_type))
            ### CODE HANGS HERE ###"""
            print(vec_mine.shape)
            print(grid.split0.n_mine, grid.split0.n_tot)
            print("hi...")
            vec = gather(vec_mine, grid.split0, grid.comm, 0).reshape(-1)
            print('Vector successfully gathered.')

        result_mine = vec @ self.M_mine

        mpi_type = rc.mpi_type[self.M_mine.dtype]
        result = torch.empty(self.size[1], dtype=self.M_mine.dtype, device=rc.device)
        # casting self.split.n_prev[:-1] to np.array below was necessary on my laptop
        self.comm.Allgatherv(
                (BufferView(result_mine), result_mine.shape[0], mpi_type),
                (BufferView(result), np.diff(self.split.n_prev), np.array(self.split.n_prev[:-1]), mpi_type))
        return result

