from __future__ import annotations
from typing import Optional, TypeVar

import numpy as np
import torch
from mpi4py import MPI

from qimpy import rc, grid
from qimpy.utils import TaskDivision, TaskDivisionCustom, BufferView
from . import Grid


def scatter(v: torch.Tensor, split_out: TaskDivision, dim: int) -> torch.Tensor:
    """Return the contents of v, changed from not-split (i.e. fully available
    on all processes) to split based on split_out along dimension dim."""
    dim = dim % len(v.shape)  # handle negative dim correctly
    return v[((slice(None),) * dim + (slice(split_out.i_start, split_out.i_stop),))]


def gather(
    v: torch.Tensor,
    split_in: TaskDivision,
    comm: MPI.Comm,
    dim: int,
) -> torch.Tensor:
    """Return the contents of v, changed from split based on split_in on
    communicator comm and dimension dim, to not-split i.e. fully available
    on all processes."""
    # Bring split dimension to outermost (if necessary):
    sendbuf = (v.swapaxes(0, dim) if dim else v).contiguous()
    prod_rest = np.prod(sendbuf.shape[1:])  # number in all but the split dim
    mpi_type = rc.mpi_type[v.dtype]
    # Gather pieces from each process to all:
    recvbuf = torch.empty(
        (split_in.n_tot,) + sendbuf.shape[1:], dtype=v.dtype, device=v.device
    )
    recv_prev = split_in.n_prev * prod_rest
    rc.current_stream_synchronize()
    comm.Allgatherv(
        (BufferView(sendbuf), split_in.n_mine * prod_rest, 0, mpi_type),
        (BufferView(recvbuf), np.diff(recv_prev), recv_prev[:-1], mpi_type),
    )
    # Restore dimension order of output (if necessary):
    return recvbuf.swapaxes(0, dim) if dim else recvbuf


def redistribute(
    v: torch.Tensor,
    split_in: TaskDivision,
    split_out: TaskDivision,
    comm: MPI.Comm,
    dim: int,
) -> torch.Tensor:
    """Return the contents of v, changed from split based on split_in to split
    based on split_out, on the same communicator comm and dimension dim."""
    # Bring split dimension to outermost (if necessary):
    sendbuf = (v.swapaxes(0, dim) if dim else v).contiguous()
    prod_rest = np.prod(sendbuf.shape[1:])  # number in all but the split dim
    mpi_type = rc.mpi_type[v.dtype]
    # Determine destinations of my input pieces:
    send_prev = (
        np.maximum(np.minimum(split_out.n_prev, split_in.i_stop), split_in.i_start)
        - split_in.i_start
    ) * prod_rest
    # Determine sources of my output pieces:
    recv_prev = (
        np.maximum(np.minimum(split_in.n_prev, split_out.i_stop), split_out.i_start)
        - split_out.i_start
    ) * prod_rest
    # Redistribute:
    recvbuf = torch.empty(
        (split_out.n_mine,) + sendbuf.shape[1:], dtype=v.dtype, device=v.device
    )
    rc.current_stream_synchronize()
    comm.Alltoallv(
        (BufferView(sendbuf), np.diff(send_prev), send_prev[:-1], mpi_type),
        (BufferView(recvbuf), np.diff(recv_prev), recv_prev[:-1], mpi_type),
    )
    # Restore dimension order of output (if necessary):
    return recvbuf.swapaxes(0, dim) if dim else recvbuf


def fix_split(
    v: torch.Tensor,
    split_in: TaskDivision,
    comm_in: Optional[MPI.Comm],
    split_out: TaskDivision,
    comm_out: Optional[MPI.Comm],
    dim: int,
) -> torch.Tensor:
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
            return gather(v, split_in, comm_in, dim)
        else:
            assert comm_in is comm_out
            return redistribute(v, split_in, split_out, comm_in, dim)


FieldTypeReal = TypeVar("FieldTypeReal", "grid.FieldR", "grid.FieldC")
FieldTypeRecip = TypeVar("FieldTypeRecip", "grid.FieldH", "grid.FieldG")


def _change_real(v: FieldTypeReal, grid_out: Grid) -> FieldTypeReal:
    """Switch real-space field to grid_out."""
    grid_in = v.grid
    assert grid_in.shape == grid_out.shape
    data_out = fix_split(
        v.data, grid_in.split0, grid_in.comm, grid_out.split0, grid_out.comm, -3
    )
    return v.__class__(grid_out, data=data_out)


def _change_recip(v: FieldTypeRecip, grid_out: Grid) -> FieldTypeRecip:
    """Switch reciprocal-space field to grid_out."""
    grid_in = v.grid
    is_half = v.__class__ is grid.FieldH
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
        slice_pos = slice_prev + (slice(0, S_hlf + 1),)
        slice_neg = slice_prev + (slice(-S_hlf, None),)
        # Put them together in the output size:
        out_shape = list(data.shape)
        out_shape[dim] = S_out
        data_out = torch.zeros(out_shape, dtype=data.dtype, device=data.device)
        data_out[slice_pos] += data[slice_pos]
        data_out[slice_neg] += data[slice_neg]
        # Handle double-counts of Nyquist components:
        if S_hlf * 2 == S_min:  # i.e. S_min is even
            i_nyq = (S_hlf,) if (S_min == S_out) else (S_hlf, -S_hlf)
            data_out[slice_prev + (i_nyq,)] *= 0.5
        data = data_out

    # Change final dimension (which could be distributed):
    if split_in.n_tot != split_out.n_tot:
        S_in = grid_in.shape[-1]  # full global shape (regardless of H)
        S_out = grid_out.shape[-1]  # full global shape (regardless of H)
        S_min = min(S_in, S_out)
        S_hlf = S_min // 2
        is_even = S_hlf * 2 == S_min  # whether S_min is even

        # Adjust local piece for change in S:
        if S_in < S_out:
            # Insert zeros after index S_in//2 to pad:
            if split_in.is_mine(S_hlf):
                # Determine local insert location:
                i_insert = S_hlf - split_in.i_start + 1
                n_after = split_in.n_mine - i_insert
                # Insert within larger array:
                n_mine_new = split_in.n_mine + split_out.n_tot - split_in.n_tot
                data_out = torch.zeros(
                    data.shape[:-1] + (n_mine_new,),
                    dtype=data.dtype,
                    device=data.device,
                )
                data_out[..., :i_insert] = data[..., :i_insert]
                if n_after:
                    data_out[..., -n_after:] = data[..., -n_after:]
                # Handle double-counts of Nyquist components:
                if is_even:  # i.e. S_min is even
                    data_out[..., i_insert - 1] *= 0.5
                    if not is_half:
                        data_out[..., -n_after - 1] = data_out[..., i_insert - 1]
            else:
                data_out = data
                n_mine_new = split_in.n_mine  # no local change
        else:
            # Drop high wave vectors:
            i_mine = torch.arange(split_in.i_start, split_in.i_stop, device=data.device)
            neg_start = S_in - S_hlf
            if is_even:
                # Which processes hold the + and - Nyquist slices:
                whose_pos = split_in.whose(S_hlf)
                whose_neg = split_in.whose(neg_start)
                # Prepare negative Nyquist slice for symmetrization:
                if not is_half:
                    if whose_neg == split_in.i_proc:
                        neg_slice = data[..., neg_start - split_in.i_start]
                        if whose_pos != split_in.i_proc:
                            assert grid_in.comm is not None
                            neg_slice = neg_slice.contiguous()
                            rc.current_stream_synchronize()
                            grid_in.comm.Send(BufferView(neg_slice), whose_pos)
                    elif whose_pos == split_in.i_proc:
                        neg_slice = torch.zeros(
                            data.shape[:-1], dtype=data.dtype, device=data.device
                        )
                        assert grid_in.comm is not None
                        rc.current_stream_synchronize()
                        grid_in.comm.Recv(BufferView(neg_slice), whose_neg)
                    # Negative Nyquist slice is now on proc with positive one
                neg_start += 1  # can only have + Nyquist freq in output
            sel = torch.where(torch.logical_or(i_mine <= S_hlf, i_mine >= neg_start))[0]
            data_out = data[..., sel]
            n_mine_new = len(sel)
            # Handle Nyquist frequency symmetrization:
            if is_even and (whose_pos == split_in.i_proc):
                i_pos = S_hlf - split_in.i_start
                if is_half:
                    # Negative slice is implicit (conjugate & G-inverted):
                    neg_slice = (
                        data_out[..., i_pos]
                        .conj()
                        .flip(dims=(-2, -1))
                        .roll((1, 1), dims=(-2, -1))
                    )
                data_out[..., i_pos] += neg_slice
                data_out[..., i_pos] *= 0.5

        # Revised split_in to match updated data (for MPI rearrangement below):
        split_in = TaskDivisionCustom(n_mine=n_mine_new, comm=grid_in.comm)
    else:
        data_out = data

    # Rearrange data as needed:
    data_out = fix_split(data_out, split_in, grid_in.comm, split_out, grid_out.comm, -1)
    return v.__class__(grid_out, data=data_out)
