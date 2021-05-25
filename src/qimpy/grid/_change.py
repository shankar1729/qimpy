import qimpy as qp
import numpy as np
import torch
from qimpy.utils import BufferView
from typing import Optional, TypeVar, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import TaskDivision, RunConfig
    from ._grid import Grid
    from ._field import FieldR, FieldC


def scatter(v: torch.Tensor, split_out: 'TaskDivision',
            dim: int) -> torch.Tensor:
    '''Return the contents of v, changed from not-split (i.e. fully available
     on all processes) to split based on split_out along dimension dim.'''
    dim = dim % len(v.shape)  # handle negative dim correctly
    return v[((slice(None),) * dim
             + (slice(split_out.i_start, split_out.i_stop),))]


def gather(v: torch.Tensor, split_in: 'TaskDivision', comm: qp.MPI.Comm,
           rc: 'RunConfig', dim: int) -> torch.Tensor:
    '''Return the contents of v, changed from split based on split_in on
     communicator comm and dimension dim, to not-split i.e. fully available
     on all processes.'''
    # Bring split dimension to outermost (if necessary):
    sendbuf = (v.swapaxes(0, dim) if dim else v).contiguous()
    prod_rest = np.prod(sendbuf.shape[1:])  # number in all but the split dim
    mpi_type = rc.mpi_type[v.dtype]
    # Gather pieces from each process to all:
    recvbuf = torch.empty((split_in.n_tot,) + sendbuf.shape[1:],
                          dtype=v.dtype, device=v.device)
    recv_prev = split_in.n_prev * prod_rest
    comm.Allgatherv(
        (BufferView(sendbuf), split_in.n_mine * prod_rest, 0, mpi_type),
        (BufferView(recvbuf), np.diff(recv_prev), recv_prev[:-1], mpi_type))
    # Restore dimension order of output (if necessary):
    return (recvbuf.swapaxes(0, dim) if dim else recvbuf)


def redistribute(v: torch.Tensor, split_in: 'TaskDivision',
                 split_out: 'TaskDivision', comm: qp.MPI.Comm,
                 rc: 'RunConfig', dim: int) -> torch.Tensor:
    '''Return the contents of v, changed from split based on split_in to split
    based on split_out, on the same communicator comm and dimension dim.'''
    # Bring split dimension to outermost (if necessary):
    sendbuf = (v.swapaxes(0, dim) if dim else v).contiguous()
    prod_rest = np.prod(sendbuf.shape[1:])  # number in all but the split dim
    mpi_type = rc.mpi_type[v.dtype]
    # Determine destinations of my input pieces:
    send_prev = (np.maximum(np.minimum(split_out.n_prev, split_in.i_stop),
                            split_in.i_start) - split_in.i_start) * prod_rest
    # Determine sources of my output pieces:
    recv_prev = (np.maximum(np.minimum(split_in.n_prev, split_out.i_stop),
                            split_out.i_start) - split_out.i_start) * prod_rest
    # Redistribute:
    recvbuf = torch.empty((split_out.n_mine,) + sendbuf.shape[1:],
                          dtype=v.dtype, device=v.device)
    comm.Alltoallv(
        (BufferView(sendbuf), np.diff(send_prev), send_prev[:-1], mpi_type),
        (BufferView(recvbuf), np.diff(recv_prev), recv_prev[:-1], mpi_type))
    # Restore dimension order of output (if necessary):
    return (recvbuf.swapaxes(0, dim) if dim else recvbuf)


def fix_split(v: torch.Tensor,
              split_in: 'TaskDivision', comm_in: Optional[qp.MPI.Comm],
              split_out: 'TaskDivision', comm_out: Optional[qp.MPI.Comm],
              rc: 'RunConfig', dim: int) -> torch.Tensor:
    '''Fix how v is split along dimension dim, from split_in on comm_in
    at input to split_out on comm_out at output. One or more communicators
    could be None, corresponding to no split in the data at input and/or
    output. If data is split both before and after, comm_in and comm_out
    must be the same communicator.'''
    if comm_in is None:
        if comm_out is None:
            return v  # No split before or after
        else:
            return scatter(v, split_out, dim)
    else:
        if comm_out is None:
            return gather(v, split_in, comm_in, rc, dim)
        else:
            assert comm_in is comm_out
            return redistribute(v, split_in, split_out, comm_in, rc, dim)


FieldTypeReal = TypeVar('FieldTypeReal', 'FieldR', 'FieldC')


def _change_real(v: FieldTypeReal, grid_out: 'Grid') -> FieldTypeReal:
    'Switch real-space field to grid_out'
    grid_in = v.grid
    assert grid_in.shape == grid_out.shape
    data_out = fix_split(v.data, grid_in.split0, grid_in.comm,
                         grid_out.split0, grid_out.comm, grid_in.rc, -3)
    return v.__class__(grid_out, data=data_out)


if __name__ == "__main__":
    qp.utils.log_config()
    qp.log.info('*'*15 + ' QimPy ' + qp.__version__ + ' ' + '*'*15)
    rc = qp.utils.RunConfig()
    # Prepare a grid for testing:
    lattice = qp.lattice.Lattice(
        rc=rc, system='triclinic', a=2.1, b=2.2, c=2.3,
        alpha=75, beta=80, gamma=85)  # pick one with no symmetries
    ions = qp.ions.Ions(rc=rc, pseudopotentials=[], coordinates=[])
    symmetries = qp.symmetries.Symmetries(rc=rc, lattice=lattice, ions=ions)
    shape0 = (96, 108, 112)
    grid1 = qp.grid.Grid(rc=rc, lattice=lattice, symmetries=symmetries,
                         shape=shape0, comm=rc.comm)  # distributed
    grid2 = qp.grid.Grid(rc=rc, lattice=lattice, symmetries=symmetries,
                         shape=shape0, comm=None)  # sequential

    # Test grid changes:
    def get_ref_field(cls, grid):
        shape_batch = (2, 3)
        result = cls(grid, shape_batch=shape_batch)  # all zeroes
        shape_mine = result.data.shape
        shape_full = shape_batch + grid.shape  # without split or H symmetry
        offsets = [0] * len(shape_full)
        if cls is qp.grid.FieldH:
            offsets[-1] = grid.split2H.i_start
        elif cls is qp.grid.FieldG:
            offsets[-1] = grid.split2.i_start
        else:
            offsets[-3] = grid.split0.i_start
        # Make each entry unique:
        for i_dim, offset in enumerate(offsets):
            cur_offset = torch.arange(shape_mine[i_dim],
                                      device=grid.rc.device) + offset
            stride = np.prod(shape_full[i_dim+1:])
            bcast_shape = [1] * len(shape_full)
            bcast_shape[i_dim] = -1
            result.data += stride * cur_offset.view(bcast_shape)
        return result
    for cls in (qp.grid.FieldR, qp.grid.FieldC):
        name = cls.__qualname__
        v1 = get_ref_field(cls, grid1)
        v2 = get_ref_field(cls, grid2)
        qp.log.info(f'{name} scatter err: {(v1 - v2.to(grid1)).norm():e}')
        qp.log.info(f'{name} gather err: {(v2 - v1.to(grid2)).norm():e}')
    qp.utils.StopWatch.print_stats()
