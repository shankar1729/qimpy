import h5py
from mpi4py import MPI
import qimpy as qp
import numpy as np
import torch
from typing import Sequence, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from ._runconfig import RunConfig


class Checkpoint(h5py.File):
    """Helper for checkpoint load/save from HDF5 files."""
    __slots__ = ('rc', 'writable')
    rc: 'RunConfig'  #: Current run configuration
    writable: bool  #: Whether file has been opened for writing

    def __init__(self, filename: str, *, rc: 'RunConfig',
                 mode: str = 'r') -> None:
        super().__init__(filename, mode, driver='mpio', comm=rc.comm)
        self.rc = rc
        self.writable = (not mode.startswith('r'))
        mode_name = 'writing:' if self.writable else 'reading'
        qp.log.info(f"Opened checkpoint file '{filename}' for {mode_name}")

    def write_slice(self, dset: Any, offset: Sequence[int],
                    data: torch.Tensor) -> None:
        """Write a slice of data to dataset `dset` at offset `offset`
        from `data` (taking care of transfer to CPU if needed).
        Note that all of `data` is written, so pass in the slice to be
        written from current process.
        This may be called from any subset of MPI processes independently,
        as no metadata modification such as dataset creation is done here.
        """
        assert self.writable
        assert len(offset) == len(data.shape)
        assert len(offset) == len(dset.shape)
        index = tuple(slice(offset[i], offset[i] + s_i)
                      for i, s_i in enumerate(data.shape))
        dset[index] = data.to(self.rc.cpu).numpy()

    def read_slice(self, dset_name: str, offset: Sequence[int],
                   size: Sequence[int]) -> torch.Tensor:
        """Read a slice of data from data set named `dset_name` in file,
        starting at `offset` and of length `size` in each dimension.
        Returns data on CPU or GPU as specified by `rc.device`.
        """
        dset = self.f[dset_name]
        assert len(offset) == len(dset.shape)
        assert len(offset) == len(size)
        index = tuple(slice(offset[i], offset[i] + size[i])
                      for i, s_i in enumerate(dset.shape))
        return torch.from_numpy(dset[index]).to(self.rc.device)
