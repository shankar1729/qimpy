import qimpy as qp
import numpy as np
import torch
from functools import lru_cache
from qimpy.rc import MPI


@lru_cache
def is_async_reduce_supported(is_cuda: bool) -> bool:
    """Determine whether async reduce is supported."""
    if is_cuda:
        # OpenMPI does not support Ireduce, Iallreduce etc. on GPU buffers
        # TODO: add similar checks for other MPI implementations
        return "Open MPI" not in MPI.Get_library_version()
    else:
        return True  # cpu MPI likely always supports it


class Iallreduce_in_place:
    """Perform async Iallreduce in-place on `buf`.
    Provides async semantics and completes on wait() of the return value,
    even if the MPI implementation does not support Iallreduce.
    This is true of some MPI implementations on GPU buffers eg. OpenMPI."""

    def __init__(self, comm: MPI.Comm, buf: torch.Tensor, op: MPI.Op) -> None:
        # Check if real async supported (or need to fake it):
        self.async_supported = is_async_reduce_supported(buf.is_cuda)
        self.local_reduce = op is MPI.SUM  # could optimize other Ops when needed
        self.local_reduce_op = torch.sum
        self.buf = buf
        if self.async_supported:
            # Initiate the MPI async operation:
            self.request = comm.Iallreduce(MPI.IN_PLACE, qp.utils.BufferView(buf), op)
        elif self.local_reduce:
            # Initiate an MPI transpose for subsequent local reduction:
            # Determine division for MPI transpose:
            n_procs = comm.Get_size()
            division = qp.utils.TaskDivision(
                n_tot=np.prod(buf.shape), n_procs=n_procs, i_proc=comm.Get_rank()
            )
            send_counts = np.diff(division.n_prev)
            send_offset = division.n_prev[:-1]
            recv_counts = division.n_mine
            recv_offset = np.arange(n_procs) * recv_counts
            mpi_type = qp.rc.mpi_type[buf.dtype]
            # Initiate MPI transpose:
            self.buf_t = torch.empty(
                (n_procs, division.n_mine), dtype=buf.dtype, device=buf.device
            )
            self.buf_view = (
                qp.utils.BufferView(buf),
                send_counts,
                send_offset,
                mpi_type,
            )
            self.request = comm.Ialltoallv(
                self.buf_view,
                (qp.utils.BufferView(self.buf_t), recv_counts, recv_offset, mpi_type),
            )
            self.n_mine = division.n_mine
            self.mpi_type = mpi_type
            self.comm = comm
        else:
            # Remember inputs and return:
            self.comm = comm
            self.op = op

    def wait(self) -> torch.Tensor:
        if self.async_supported:
            # Complete the MPI async operation:
            self.request.Wait()
        elif self.local_reduce:
            # Complete MPI transpose:
            self.request.Wait()
            # Local reduction:
            result = self.local_reduce_op(self.buf_t, dim=0)
            # Gather results:
            self.comm.Allgatherv(
                (qp.utils.BufferView(result), self.n_mine, 0, self.mpi_type),
                self.buf_view,  # back in original buffer
            )
        else:
            # Perform the blocking MPI operation now:
            self.comm.Allreduce(MPI.IN_PLACE, qp.utils.BufferView(self.buf), self.op)
        return self.buf
