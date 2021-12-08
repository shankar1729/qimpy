import qimpy as qp
import torch
from functools import lru_cache


@lru_cache
def is_async_reduce_supported(is_cuda: bool) -> bool:
    """Determine whether async reduce is supported."""
    if is_cuda:
        # OpenMPI does not support Ireduce, Iallreduce etc. on GPU buffers
        # TODO: add similar checks for other MPI implementations
        return "Open MPI" not in qp.MPI.Get_library_version()
    else:
        return True  # cpu MPI likely always supports it


class Iallreduce_in_place:
    """Perform async Iallreduce in-place on `buf`.
    Provides async semantics and completes on wait() of the return value,
    even if the MPI implementation does not support Iallreduce.
    This is true of some MPI implementations on GPU buffers eg. OpenMPI."""

    def __init__(self, comm: qp.MPI.Comm, buf: torch.Tensor, op: qp.MPI.Op) -> None:
        # Check if real async supported (or need to fake it):
        self.async_supported = is_async_reduce_supported(buf.is_cuda)
        self.buf = buf
        if self.async_supported:
            # Initiate the MPI async operation:
            self.request = comm.Iallreduce(
                qp.MPI.IN_PLACE, qp.utils.BufferView(buf), op
            )
        else:
            # Remember inputs and return:
            self.comm = comm
            self.op = op

    def wait(self) -> torch.Tensor:
        if self.async_supported:
            # Complete the MPI async operation:
            self.request.Wait()
        else:
            # Perform the blocking MPI operation now:
            self.comm.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(self.buf), self.op)
        return self.buf
