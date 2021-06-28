import qimpy as qp
import numpy as np
import torch
from qimpy.utils import BufferView
from typing import Optional, TypeVar, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import TaskDivision, RunConfig
    from ._grid import Grid
    from ._field import FieldR, FieldC, FieldH, FieldG


def scatter(v: torch.Tensor, split_out: 'TaskDivision',
            dim: int) -> torch.Tensor:
    """Return the contents of v, changed from not-split (i.e. fully available
     on all processes) to split based on split_out along dimension dim."""
    dim = dim % len(v.shape)  # handle negative dim correctly
    return v[((slice(None),) * dim
             + (slice(split_out.i_start, split_out.i_stop),))]


def gather(v: torch.Tensor, split_in: 'TaskDivision', comm: qp.MPI.Comm,
           rc: 'RunConfig', dim: int) -> torch.Tensor:
    """Return the contents of v, changed from split based on split_in on
     communicator comm and dimension dim, to not-split i.e. fully available
     on all processes."""
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
    """Return the contents of v, changed from split based on split_in to split
    based on split_out, on the same communicator comm and dimension dim."""
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
    """Fix how v is split along dimension dim, from split_in on comm_in
    at input to split_out on comm_out at output. One or more communicators
    could be None, corresponding to no split in the data at input and/or
    output. If data is split both before and after, comm_in and comm_out
    must be the same communicator."""
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
FieldTypeRecip = TypeVar('FieldTypeRecip', 'FieldH', 'FieldG')


def _change_real(v: FieldTypeReal, grid_out: 'Grid') -> FieldTypeReal:
    """Switch real-space field to grid_out"""
    grid_in = v.grid
    assert grid_in.shape == grid_out.shape
    data_out = fix_split(v.data, grid_in.split0, grid_in.comm,
                         grid_out.split0, grid_out.comm, grid_in.rc, -3)
    return v.__class__(grid_out, data=data_out)


def _change_recip(v: FieldTypeRecip, grid_out: 'Grid') -> FieldTypeRecip:
    """Switch reciprocal-space field to grid_out"""
    grid_in = v.grid
    is_half = (v.__class__ is qp.grid.FieldH)
    if is_half:
        split_in = grid_in.split2H
        split_out = grid_out.split2H
        shape_grid_out = grid_out.shapeH_mine
    else:
        split_in = grid_in.split2
        split_out = grid_out.split2
        shape_grid_out = grid_out.shapeG_mine

    # Change local dimenions with appropriate slicing:
    data = v.data
    for dim in (-3, -2):
        S_in = data.shape[dim]
        S_out = shape_grid_out[dim]
        if S_in == S_out:
            continue
        # Pick up the positive and negative frequency halves:
        S_min = min(S_in, S_out)
        S_hlf = S_min // 2
        slice_prev = (slice(None),) * (dim % len(data.shape))
        slice_pos = slice_prev + (slice(0, S_hlf+1),)
        slice_neg = slice_prev + (slice(-S_hlf, None),)
        # Put them together in the output size:
        out_shape = list(data.shape)
        out_shape[dim] = S_out
        data_out = torch.zeros(out_shape, dtype=data.dtype, device=data.device)
        data_out[slice_pos] += data[slice_pos]
        data_out[slice_neg] += data[slice_neg]
        # Handle double-counts of Nyquist components:
        if S_hlf*2 == S_min:  # i.e. S_min is even
            i_nyq = (S_hlf,) if (S_min == S_out) else (S_hlf, -S_hlf)
            data_out[slice_prev + (i_nyq,)] *= 0.5
        data = data_out

    # Change final dimension (which could be distributed):
    if split_in.n_tot != split_out.n_tot:
        S_in = grid_in.shape[-1]  # full global shape (regardless of H)
        S_out = grid_out.shape[-1]  # full global shape (regardless of H)
        S_min = min(S_in, S_out)
        S_hlf = S_min // 2
        is_even = (S_hlf * 2 == S_min)  # whether S_min is even

        # Adjust local piece for change in S:
        if S_in < S_out:
            # Insert zeros after index S_in//2 to pad:
            if split_in.is_mine(S_hlf):
                # Determine local insert location:
                i_insert = S_hlf - split_in.i_start + 1
                n_after = split_in.n_mine - i_insert
                # Insert within larger array:
                n_mine_new = split_in.n_mine + split_out.n_tot - split_in.n_tot
                data_out = torch.zeros(data.shape[:-1] + (n_mine_new,),
                                       dtype=data.dtype, device=data.device)
                data_out[..., :i_insert] = data[..., :i_insert]
                if n_after:
                    data_out[..., -n_after:] = data[..., -n_after:]
                # Handle double-counts of Nyquist components:
                if is_even:  # i.e. S_min is even
                    data_out[..., i_insert - 1] *= 0.5
                    if not is_half:
                        data_out[..., -n_after-1] = data_out[..., i_insert - 1]
            else:
                data_out = data
                n_mine_new = split_in.n_mine  # no local change
        else:
            # Drop high wave vectors:
            i_mine = torch.arange(split_in.i_start, split_in.i_stop,
                                  device=data.device)
            neg_start = S_in - S_hlf
            if is_even:
                # Which processes hold the + and - Nyquist slices:
                whose_pos = split_in.whose(S_hlf)
                whose_neg = split_in.whose(neg_start)
                # Prepare negative Nyquist slice for symmetrization:
                if (not is_half):
                    if (whose_neg == split_in.i_proc):
                        neg_slice = data[..., neg_start - split_in.i_start]
                        if whose_pos != split_in.i_proc:
                            assert (grid_in.comm is not None)
                            neg_slice = neg_slice.contiguous()
                            grid_in.comm.Send(BufferView(neg_slice), whose_pos)
                    elif (whose_pos == split_in.i_proc):
                        neg_slice = torch.zeros(data.shape[:-1],
                                                dtype=data.dtype,
                                                device=data.device)
                        assert (grid_in.comm is not None)
                        grid_in.comm.Recv(BufferView(neg_slice), whose_neg)
                    # Negative Nyquist slice is now on proc with positive one
                neg_start += 1  # can only have + Nyquist freq in output
            sel = torch.where(torch.logical_or(i_mine <= S_hlf,
                                               i_mine >= neg_start))[0]
            data_out = data[..., sel]
            n_mine_new = len(sel)
            # Handle Nyquist frequency symmetrization:
            if is_even and (whose_pos == split_in.i_proc):
                i_pos = S_hlf - split_in.i_start
                if is_half:
                    # Negative slice is implicit (conjugate & G-inverted):
                    neg_slice = data_out[..., i_pos].conj().flip(
                        dims=(-2, -1)).roll((1, 1), dims=(-2, -1))
                data_out[..., i_pos] += neg_slice
                data_out[..., i_pos] *= 0.5

        # Revised split_in to match updated data (for MPI rearrangement below):
        split_in = qp.utils.TaskDivisionCustom(n_mine_new, grid_in.comm)
    else:
        data_out = data

        # Rearrange data as needed:
    data_out = fix_split(data_out, split_in, grid_in.comm,
                         split_out, grid_out.comm, grid_in.rc, -1)
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

    # MPI-reproducible grid for testing:
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

    # Make parallel and sequential grids of given shape:
    def make_grid(shape, comm):
        return qp.grid.Grid(rc=rc, lattice=lattice, symmetries=symmetries,
                            shape=shape, comm=comm)

    def make_grids(shape):
        grid_p = make_grid(shape, rc.comm)  # parallel
        grid_s = make_grid(shape, None)  # sequential
        return grid_p, grid_s

    def norm_str(v: 'Field') -> str:
        """Report norm of field averaged over batch dimensions"""
        return f'{v.norm().mean().item():.3e}'
    qp.log.info('\n--- Tests with same shape ---')
    grid0p, grid0s = make_grids((96, 108, 112))
    for cls in (qp.grid.FieldR, qp.grid.FieldC,
                qp.grid.FieldH, qp.grid.FieldG):
        name = cls.__qualname__
        v0p, v0s = (get_ref_field(cls, grid) for grid in (grid0p, grid0s))
        qp.log.info(f'{name} scatter err: {norm_str(v0p - v0s.to(grid0p))}')
        qp.log.info(f'{name} gather err: {norm_str(v0s - v0p.to(grid0s))}')
    qp.utils.StopWatch.print_stats()

    qp.log.info('\n--- Tests with shape change ---')
    grid1p, grid1s = make_grids((2, 4, 6))
    grid2p, grid2s = make_grids((3, 6, 8))

    def summary(v):
        return rc.fmt(v.data.real[0, 0])

    for cls in (qp.grid.FieldH, qp.grid.FieldG):
        name = cls.__qualname__
        v1p, v1s, v2p, v2s = (get_ref_field(cls, grid)
                              for grid in (grid1p, grid1s, grid2p, grid2s))
        # Inspect reference sequential conversions manually:
        v12s = v1s.to(grid2s)
        v21s = v2s.to(grid1s)
        qp.log.info(f'\n{name} v1 on grid1:\n' + summary(v1s))
        qp.log.info(f'\n{name} v1 on grid2:\n' + summary(v12s))
        qp.log.info(f'\n{name} v2 on grid2:\n' + summary(v2s))
        qp.log.info(f'\n{name} v2 on grid1:\n' + summary(v21s))
        # Report errors of parallel cases against sequential:
        v12p = v12s.to(grid2p)
        v21p = v21s.to(grid1p)
        qp.log.info('')
        qp.log.info(f'{name} 1s-2p err: {norm_str(v12p - v1s.to(grid2p))}')
        qp.log.info(f'{name} 1p-2s err: {norm_str(v12s - v1p.to(grid2s))}')
        qp.log.info(f'{name} 1p-2p err: {norm_str(v12p - v1p.to(grid2p))}')
        qp.log.info(f'{name} 2s-1p err: {norm_str(v21p - v2s.to(grid1p))}')
        qp.log.info(f'{name} 2p-1s err: {norm_str(v21s - v2p.to(grid1s))}')
        qp.log.info(f'{name} 2p-1p err: {norm_str(v21p - v2p.to(grid1p))}')

    qp.log.info('\n--- Visual inspection of Fourier resampling ---')
    # Do this sequentially, as MPI equivalence tested above already
    if rc.is_head:
        import matplotlib.pyplot as plt
        grid1 = make_grid((36, 40, 48), None)
        grid2 = make_grid((40, 48, 64), None)

        def get_test_field(grid: 'Grid'):
            """A highly oscillatory and non-trivial function to test resampling
            """
            x = grid.get_mesh('R') / torch.tensor(grid.shape,
                                                  device=rc.device)
            k1 = (2 * np.pi) * torch.tensor([2, 4, 5], device=rc.device)
            k2 = (2 * np.pi) * torch.tensor([6, 1, 3], device=rc.device)
            x_plot = x[0, 0, :, 2].to(rc.cpu)
            return x_plot, qp.grid.FieldR(
                grid, data=torch.exp(torch.cos(x @ k1) + torch.sin(x @ k2)))

        x1, v1 = get_test_field(grid1)
        x2, v2 = get_test_field(grid2)
        v12 = ~((~v1).to(grid2))
        v21 = ~((~v2).to(grid1))

        def get_plot_slice(v):
            return v.data[0, 0].to(rc.cpu)

        plt.plot(x1, get_plot_slice(v1), 'r', label='Created on 1')
        plt.plot(x1, get_plot_slice(v21), 'r+', label=r'Sampled 2$\to$1')
        plt.plot(x2, get_plot_slice(v2), 'b', label='Created on 2')
        plt.plot(x2, get_plot_slice(v12), 'b+', label=r'Sampled 1$\to$2')
        plt.legend()
        plt.show()
    qp.utils.StopWatch.print_stats()
