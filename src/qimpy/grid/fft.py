from __future__ import annotations
from typing import Callable

import numpy as np
import torch
from mpi4py import MPI

from qimpy import log, rc, grid
from qimpy.mpi import TaskDivision, BufferView


IndicesType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
FunctionFFT = Callable[[torch.Tensor, str], torch.Tensor]


class FFT(torch.autograd.Function):
    """Differentiable interface to :func:`_fft`."""

    @staticmethod
    def forward(ctx, grid: grid.Grid, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        ctx.grid = grid
        return fft(grid, input, norm="forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:  # type: ignore
        return None, ifft(ctx.grid, grad_output, norm="backward")


class IFFT(torch.autograd.Function):
    """Differentiable interface to :func:`_ifft`."""

    @staticmethod
    def forward(ctx, grid: grid.Grid, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        ctx.grid = grid
        return ifft(grid, input, norm="forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:  # type: ignore
        return None, fft(ctx.grid, grad_output, norm="backward")


def init_grid_fft(self: grid.Grid) -> None:
    """Initialize local or parallel FFTs for class Grid."""
    # Half-reciprocal space global dimensions (for rfft/irfft):
    self.shapeH = (self.shape[0], self.shape[1], 1 + self.shape[2] // 2)
    log.info(f"real-fft shape: {self.shapeH}")
    # MPI division:
    self.split0 = TaskDivision(
        n_tot=self.shape[0], n_procs=self.n_procs, i_proc=self.i_proc
    )
    self.split2 = TaskDivision(
        n_tot=self.shape[2], n_procs=self.n_procs, i_proc=self.i_proc
    )
    self.split2H = TaskDivision(
        n_tot=self.shapeH[2], n_procs=self.n_procs, i_proc=self.i_proc
    )
    # Overall local grid dimensions:
    self.shapeR_mine = (self.split0.n_mine, self.shape[1], self.shape[2])
    self.shapeG_mine = (self.shape[0], self.shape[1], self.split2.n_mine)
    self.shapeH_mine = (self.shape[0], self.shape[1], self.split2H.n_mine)
    if self.n_procs > 1:
        log.info(f"split over {self.n_procs} processes:")
        log.info(f"  local selected shape: {self.shapeR_mine}")
        log.info(f"  local full-fft shape: {self.shapeG_mine}")
        log.info(f"  local real-fft shape: {self.shapeH_mine}")
    # Create 1D grids for real and reciprocal spaces:
    # --- global versions first
    iv1D = tuple(torch.arange(s, device=rc.device) for s in self.shape)
    iG1D = tuple(
        torch.where(iv <= self.shape[dim] // 2, iv, iv - self.shape[dim])
        for dim, iv in enumerate(iv1D)
    )
    # --- slice parts of Real, G-space and Half G-space for `get_mesh`:
    self._mesh1D = {  # Global versions:
        "R": iv1D,
        "G": iG1D,
        "H": iG1D[:2] + (iG1D[2][: self.shapeH[2]],),
    }
    self._mesh1D_mine = {  # Local versions:
        "R": (iv1D[0][self.split0.i_start : self.split0.i_stop],) + iv1D[1:],
        "G": iG1D[:2] + (iG1D[2][self.split2.i_start : self.split2.i_stop],),
        "H": iG1D[:2] + (iG1D[2][self.split2H.i_start : self.split2H.i_stop],),
    }

    def get_indices(in_prev: np.ndarray, n_out_mine: int) -> IndicesType:
        """Get index arrays for unscrambling data after MPI rearrangement.

        A common operation below is taking an array split along axis
        'in' and doing an MPI all-to-all to split it along axis 'out'.
        Before the MPI transfer, the array must be rearranged to bring
        the out axis as dimension 0. After doing this, the array will
        have dimensions n_out x (batch-dims) x n_inMine x S[1]. Note
        that the middle spatial dimension S[1] is never split.

        The differing chunk-size in all-to-all scrambles the result,
        and this routine provides indices that put the data in the right
        order to then view as (batch-dims) x n_out_mine x S[1] x n_in.
        The results of this function should be linearly combined with 1,
        i_batch and n_batch to get net indexes for a given batch size.

        Parameters
        ----------
        in_prev : numpy.array of ints
            Cumulative counts of dimension split at input
        n_out_mine : int
            Local length of dimension split at output

        Returns
        -------
        index_1 : torch.Tensor of ints
            Coefficient of 1 in final index
        index_i_batch : torch.Tensor of ints
            Coefficient of i_batch in final index
        index_n_batch : torch.Tensor of ints
            Coefficient of n_batch in final index
        """
        i_out_mine = np.arange(n_out_mine)  # 1D index on out-split array
        i_in = np.arange(in_prev[-1])  # 1D index on out-unified array
        in_each = in_prev[1] - in_prev[0]  # block size of input split
        in_counts = np.diff(in_prev)  # actual n_in on each process
        src_proc = i_in // in_each  # index of source process by output entry
        # Return index as a linear combination with three terms:
        # (This allows handling all batch combinations with the same arrays)
        S1 = self.shape[1]  # length of middle spatial dimension (never split)
        i1 = np.arange(S1)  # index over middle spatial dimension
        # --- coefficient of 1
        index_1 = torch.tensor(
            S1 * (i_in - in_prev[src_proc])[None, None, None, :]
            + i1[None, None, :, None]
        ).to(rc.device)
        # --- coefficient of i_batch
        index_i_batch = torch.tensor(S1 * in_counts[None, None, None, src_proc]).to(
            rc.device
        )
        # --- coefficient of n_batch
        index_n_batch = torch.tensor(
            n_out_mine * S1 * in_prev[None, None, None, src_proc]
            + (
                i_out_mine[None, :, None, None]
                * S1
                * in_counts[None, None, None, src_proc]
            )
        ).to(rc.device)
        return index_1, index_i_batch, index_n_batch

    # Pre-calculate these arrays for each of the transforms:
    self._indices_fft = get_indices(self.split0.n_prev, self.split2.n_mine)
    self._indices_ifft = get_indices(self.split2.n_prev, self.split0.n_mine)
    self._indices_rfft = get_indices(self.split0.n_prev, self.split2H.n_mine)
    self._indices_irfft = get_indices(self.split2H.n_prev, self.split0.n_mine)


def parallel_transform(
    comm: MPI.Comm,
    v: torch.Tensor,
    norm: str,
    shape_in: tuple[int, ...],
    shape_out: tuple[int, ...],
    fft_before: FunctionFFT,
    fft_after: FunctionFFT,
    in_prev: np.ndarray,
    out_prev: np.ndarray,
    index_1: torch.Tensor,
    index_i_batch: torch.Tensor,
    index_n_batch: torch.Tensor,
) -> torch.Tensor:
    """Helper function that performs the work of all the parallel
    FFT functions in class qimpy.grid.Grid. This function should
    be called only for n_procs > 1 i.e. when actually parallelized.
    Also note that this function exchanges order of first and last
    trasnformed dimensions; the driver routine needs to locally
    transpose these dimensions of the input arrays for forward
    transforms, and for the output arrays for inverse transforms.

    Parameters
    ----------
    comm
        Communicator that this transform is split on
    v
        Input tensor, 3D, real for rfft and complex for all else
    norm
        Normalization mode (see `torch.fft`) for `fft_before` and `fft_after`
    shape_in
        local spatial dimensions before MPI exchange
    shape_out
        local spatial dimensions after MPI exchange
    fft_before
        corresponding 2D/1D torch FFT routine before MPI exchange
    fft_after
        corresponding 1D/2D torch FFT routine after MPI exchange
    in_prev
        TaskDivision.n_prev of the dimension together at input, that splits
    out_prev
        TaskDivision.n_prev of the dimension initially split, joined at output
    index_1
        relevant unscramble index (coefficient of 1) from _init_grid_fft
    index_i_batch
        relevant unscramble index (coefficient of i_batch) from _init_grid_fft
    index_n_batch
        relevant unscramble index (coefficient of n_batch) from _init_grid_fft
    """
    assert v.shape[-3:] == shape_in
    n_batch = int(np.prod(v.shape[:-3]))
    v_tilde = fft_before(v, norm)  # Transform 2 or 1 dims here
    v_tilde = v_tilde.flatten(0, -2).T.contiguous()  # bring last dim to front

    # MPI rearrangement:
    send_prev = in_prev * v_tilde.shape[1]
    recv_prev = out_prev * (np.prod(shape_out[:2]) * n_batch)
    v_tmp = torch.zeros(recv_prev[-1], dtype=v_tilde.dtype, device=v_tilde.device)
    mpi_type = rc.mpi_type[v_tilde.dtype]
    rc.current_stream_synchronize()
    comm.Alltoallv(
        (BufferView(v_tilde), np.diff(send_prev), send_prev[:-1], mpi_type),
        (BufferView(v_tmp), np.diff(recv_prev), recv_prev[:-1], mpi_type),
    )

    # Unscramble:
    if n_batch == 1:
        index = index_1 + index_n_batch
    else:
        i_batch = torch.arange(n_batch, device=index_1.device).view(n_batch, 1, 1, 1)
        index = index_1 + index_i_batch * i_batch + index_n_batch * n_batch
    v_tilde = v_tmp[index].view(v.shape[:-3] + shape_out)
    del v_tmp
    return fft_after(v_tilde, norm)  # Transform 1 or 2 dims here


def fft(self: grid.Grid, v: torch.Tensor, norm: str) -> torch.Tensor:
    """
    Underlying implementation of :meth:`qimpy.grid.Grid.fft`.
    Additional argument `norm` matches torch.fft routines and is used internally
    by the differentiable interface in :class:`_FFT` and :class:`_IFFT`.
    """
    if v.dtype.is_complex:
        # Complex to complex forward transform:
        if v.shape[:-3].count(0):  # zero-sized batches
            return v
        if self.n_procs == 1:
            return torch.fft.fftn(v, s=self.shape, norm=norm)
        assert self.comm is not None
        return parallel_transform(
            self.comm,
            v,
            norm,
            self.shapeR_mine,
            self.shapeG_mine[::-1],
            safe_fft2,
            safe_fft,
            self.split2.n_prev,
            self.split0.n_prev,
            *self._indices_fft,
        ).swapaxes(-1, -3)
    else:
        # Real to complex forward transform:
        assert v.dtype.is_floating_point
        if v.shape[:-3].count(0):  # zero-sized batches
            return torch.zeros(
                v.shape[:-3] + self.shapeH_mine,
                dtype=COMPLEX_TYPE[v.dtype],
                device=v.device,
            )
        if self.n_procs == 1:
            return torch.fft.rfftn(v, s=self.shape, norm=norm)
        assert v.dtype.is_floating_point
        assert self.comm is not None
        return parallel_transform(
            self.comm,
            v,
            norm,
            self.shapeR_mine,
            self.shapeH_mine[::-1],
            safe_rfft,
            safe_fft2,
            self.split2H.n_prev,
            self.split0.n_prev,
            *self._indices_rfft,
        ).swapaxes(-1, -3)


def ifft(self: grid.Grid, v: torch.Tensor, norm: str) -> torch.Tensor:
    """
    Underlying implementation of :meth:`qimpy.grid.Grid.ifft`.
    Additional argument `norm` matches torch.fft routines and is used internally
    by the differentiable interface in :class:`_FFT` and :class:`_IFFT`.
    """
    # Get total size of last dimension to dispatch complex vs real:
    assert v.dtype.is_complex
    shape2 = v.shape[-1]
    if self.n_procs > 1:
        assert self.comm is not None
        shape2 = self.comm.allreduce(shape2, MPI.SUM)

    if shape2 == self.shape[2]:
        # Complex to complex inverse transform:
        if v.shape[:-3].count(0):  # zero-sized batches
            return v
        if self.n_procs == 1:
            return torch.fft.ifftn(v, s=self.shape, norm=norm)
        assert self.comm is not None
        return parallel_transform(
            self.comm,
            v.swapaxes(-1, -3),
            norm,
            self.shapeG_mine[::-1],
            self.shapeR_mine,
            safe_ifft,
            safe_ifft2,
            self.split0.n_prev,
            self.split2.n_prev,
            *self._indices_ifft,
        )
    else:
        # Complex to real inverse transform:
        assert shape2 == self.shapeH[2]
        if v.shape[:-3].count(0):  # zero-sized batches
            return torch.zeros(
                v.shape[:-3] + self.shapeR_mine,
                dtype=REAL_TYPE[v.dtype],
                device=v.device,
            )
        if self.n_procs == 1:
            return torch.fft.irfftn(v, s=self.shape, norm=norm)
        assert v.dtype.is_complex
        assert self.comm is not None
        shapeR_mine_complex = (self.split0.n_mine, self.shape[1], self.shapeH[2])
        return parallel_transform(
            self.comm,
            v.swapaxes(-1, -3),
            norm,
            self.shapeH_mine[::-1],
            shapeR_mine_complex,
            safe_ifft2,
            safe_irfft,
            self.split0.n_prev,
            self.split2H.n_prev,
            *self._indices_irfft,
        )


# Maps between corresponding real and complex types:
REAL_TYPE = {torch.complex128: torch.double, torch.complex64: torch.float}
COMPLEX_TYPE = {torch.double: torch.complex128, torch.float: torch.complex64}


# --- Wrappers to torch FFTs that are safe for zero sizes ---
def safe_fft(v: torch.Tensor, norm: str) -> torch.Tensor:
    return torch.fft.fft(v, norm=norm) if v.numel() else v


def safe_fft2(v: torch.Tensor, norm: str) -> torch.Tensor:
    return torch.fft.fft2(v, norm=norm) if v.numel() else v


def safe_ifft(v: torch.Tensor, norm: str) -> torch.Tensor:
    return torch.fft.ifft(v, norm=norm) if v.numel() else v


def safe_ifft2(v: torch.Tensor, norm: str) -> torch.Tensor:
    return torch.fft.ifft2(v, norm=norm) if v.numel() else v


def safe_rfft(v: torch.Tensor, norm: str) -> torch.Tensor:
    if v.numel():
        return torch.fft.rfft(v, norm=norm)
    else:
        return torch.zeros(
            v.shape[:-1] + (v.shape[-1] // 2 + 1,),
            dtype=COMPLEX_TYPE[v.dtype],
            device=v.device,
        )


def safe_irfft(v: torch.Tensor, norm: str) -> torch.Tensor:
    if v.numel():
        return torch.fft.irfft(v, norm=norm)
    else:
        return torch.zeros(
            v.shape[:-1] + (2 * (v.shape[-1] - 1),),
            dtype=REAL_TYPE[v.dtype],
            device=v.device,
        )
