import qimpy as qp
import numpy as np
import torch
from qimpy.utils import TaskDivision, BufferView
from typing import Tuple, Sequence, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from ._grid import Grid
    from ..utils import RunConfig


IndicesType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MethodFFT = Callable[['Grid', torch.Tensor, str], torch.Tensor]
FunctionFFT = Callable[[torch.Tensor, str], torch.Tensor]


def _init_grid_fft(self: 'Grid') -> None:
    'Initialize local or parallel FFTs for class Grid'
    # Half-reciprocal space global dimensions (for rfft/irfft):
    self.shapeH = (self.shape[0], self.shape[1], 1+self.shape[2]//2)
    qp.log.info(f'real-fft shape: {self.shapeH}')
    # MPI division:
    self.split0 = TaskDivision(self.shape[0], self.n_procs, self.i_proc)
    self.split2 = TaskDivision(self.shape[2], self.n_procs, self.i_proc)
    self.split2H = TaskDivision(self.shapeH[2], self.n_procs, self.i_proc)
    # Overall local grid dimensions:
    self.shapeR_mine = (self.split0.n_mine, self.shape[1], self.shape[2])
    self.shapeG_mine = (self.shape[0], self.shape[1], self.split2.n_mine)
    self.shapeH_mine = (self.shape[0], self.shape[1], self.split2H.n_mine)
    if self.n_procs > 1:
        qp.log.info(f'split over {self.n_procs} processes:')
        qp.log.info(f'  local selected shape: {self.shapeR_mine}')
        qp.log.info(f'  local full-fft shape: {self.shapeG_mine}')
        qp.log.info(f'  local real-fft shape: {self.shapeH_mine}')
    # Create 1D grids for real and reciprocal spaces:
    # --- global versions first
    iv1D = tuple(torch.arange(s, device=self.rc.device) for s in self.shape)
    iG1D = tuple(torch.where(iv <= self.shape[dim]//2, iv, iv-self.shape[dim])
                 for dim, iv in enumerate(iv1D))
    # --- slice local parts of Real, G-space and Half g-space for self.mesh():
    self.mesh1D = {
        'R': (iv1D[0][self.split0.i_start:self.split0.i_stop],) + iv1D[1:],
        'G': iG1D[:2] + (iG1D[2][self.split2.i_start:self.split2.i_stop],),
        'H': iG1D[:2] + (iG1D[2][self.split2H.i_start:self.split2H.i_stop],)}

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
            + i1[None, None, :, None]).to(self.rc.device)
        # --- coefficient of i_batch
        index_i_batch = torch.tensor(
            S1 * in_counts[None, None, None, src_proc]).to(self.rc.device)
        # --- coefficient of n_batch
        index_n_batch = torch.tensor(
            n_out_mine * S1 * in_prev[None, None, None, src_proc]
            + (i_out_mine[None, :, None, None] * S1
               * in_counts[None, None, None, src_proc])).to(self.rc.device)
        return index_1, index_i_batch, index_n_batch

    # Pre-calculate these arrays for each of the transforms:
    self.indices_fft = get_indices(self.split0.n_prev, self.split2.n_mine)
    self.indices_ifft = get_indices(self.split2.n_prev, self.split0.n_mine)
    self.indices_rfft = get_indices(self.split0.n_prev, self.split2H.n_mine)
    self.indices_irfft = get_indices(self.split2H.n_prev, self.split0.n_mine)


def parallel_transform(
        rc: 'RunConfig', comm: qp.MPI.Comm, v: torch.Tensor, norm: str,
        shape_in: Tuple[int, ...], shape_out: Tuple[int, ...],
        fft_before: FunctionFFT, fft_after: FunctionFFT,
        in_prev: np.ndarray, out_prev: np.ndarray,
        index_1: torch.Tensor, index_i_batch: torch.Tensor,
        index_n_batch: torch.Tensor) -> torch.Tensor:
    """Helper function that performs the work of all the parallel
    FFT functions in class qimpy.grid.Grid. This function should
    be called only for n_procs > 1 i.e. when actually parallelized.
    Also note that this function exchanges order of first and last
    trasnformed dimensions; the driver routine needs to locally
    transpose these dimensions of the input arrays for forward
    transforms, and for the output arrays for inverse transforms.

    Parameters
    ----------
    rc
        Current run configuration
    comm
        Communicator that this transform is split on
    v
        Input tensor, 3D, real for rfft and complex for all else
    norm :  {'backward', 'forward', 'ortho'}
        Which direction of the transform is the normalization applied to
        (same as norm in the torch.fft routines)
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
    assert(v.shape[-3:] == shape_in)
    n_batch = int(np.prod(v.shape[:-3]))
    v_tilde = fft_before(v, norm)  # Transform 2 or 1 dims here
    v_tilde = v_tilde.flatten(0, -2).T.contiguous()  # bring last dim to front

    # MPI rearrangement:
    send_prev = in_prev * v_tilde.shape[1]
    recv_prev = out_prev * (np.prod(shape_out[:2]) * n_batch)
    v_tmp = torch.zeros(recv_prev[-1], dtype=v_tilde.dtype,
                        device=v_tilde.device)
    mpi_type = rc.mpi_type[v_tilde.dtype]
    comm.Alltoallv(
        (BufferView(v_tilde), np.diff(send_prev), send_prev[:-1], mpi_type),
        (BufferView(v_tmp), np.diff(recv_prev), recv_prev[:-1], mpi_type))

    # Unscramble:
    if n_batch == 1:
        index = index_1 + index_n_batch
    else:
        i_batch = torch.arange(n_batch, device=index_1.device).view(n_batch,
                                                                    1, 1, 1)
        index = index_1 + index_i_batch * i_batch + index_n_batch * n_batch
    v_tilde = v_tmp[index].view(v.shape[:-3] + shape_out)
    del v_tmp
    return fft_after(v_tilde, norm)  # Transform 1 or 2 dims here


def _fft(self: 'Grid', v: torch.Tensor,
         norm: str = 'forward') -> torch.Tensor:
    """Complex to complex forward transform

    Parameters
    ----------
    v : torch.Tensor (complex)
        Last 3 dimensions must match shapeR_mine,
        and any preceding dimensions are batched over
    norm :  {'backward', 'forward', 'ortho'}, default: 'forward'
        Which direction of the transform is the normalization applied to.
        This is same as norm in the torch.fft routines, but the default
        is to apply to forward transforms. This makes the G=0 components
        in reciprocal space correspond to the mean value of the real
        space version (instead of sum for norm='backward')

    Returns
    -------
    torch.Tensor (complex)
        Last 3 dimensions will be shapeG_mine,
        preceded by any batch dimensions in the input.
    """
    if self.n_procs == 1:
        return torch.fft.fftn(v, s=self.shape, norm=norm)
    assert(v.dtype.is_complex)
    return parallel_transform(
        self.rc, self.comm, v, norm,
        self.shapeR_mine, self.shapeG_mine[::-1],
        safe_fft2, safe_fft, self.split2.n_prev, self.split0.n_prev,
        *self.indices_fft).swapaxes(-1, -3)


def _ifft(self: 'Grid', v: torch.Tensor,
          norm: str = 'forward') -> torch.Tensor:
    """Complex to complex inverse transform

    Parameters
    ----------
    v : torch.Tensor (complex)
        Last 3 dimensions must match shapeG_mine,
        and any preceding dimensions are batched over
    norm :  {'backward', 'forward', 'ortho'}, default: 'forward'
        Same as in torch.fft routines, but with default normalization
        to forward transforms (see :meth:`qimpy.grid.Grid.fft`)

    Returns
    -------
    torch.Tensor (complex)
        Last 3 dimensions will be shapeR_mine,
        preceded by any batch dimensions in the input.
    """
    if self.n_procs == 1:
        return torch.fft.ifftn(v, s=self.shape, norm=norm)
    assert(v.dtype.is_complex)
    return parallel_transform(
        self.rc, self.comm, v.swapaxes(-1, -3), norm,
        self.shapeG_mine[::-1], self.shapeR_mine,
        safe_ifft, safe_ifft2, self.split0.n_prev, self.split2.n_prev,
        *self.indices_ifft)


def _rfft(self: 'Grid', v: torch.Tensor,
          norm: str = 'forward') -> torch.Tensor:
    """Real to complex forward transform

    Parameters
    ----------
    v : torch.Tensor (float)
        Last 3 dimensions must match shapeR_mine,
        and any preceding dimensions are batched over
    norm :  {'backward', 'forward', 'ortho'}, default: 'forward'
        Same as in torch.fft routines, but with default normalization
        to forward transforms (see :meth:`qimpy.grid.Grid.fft`)

    Returns
    -------
    torch.Tensor (complex)
        Last 3 dimensions will be shapeH_mine,
        preceded by any batch dimensions in the input.
    """
    if self.n_procs == 1:
        return torch.fft.rfftn(v, s=self.shape, norm=norm)
    assert(v.dtype.is_floating_point)
    return parallel_transform(
        self.rc, self.comm, v, norm,
        self.shapeR_mine, self.shapeH_mine[::-1],
        safe_rfft, safe_fft2, self.split2H.n_prev, self.split0.n_prev,
        *self.indices_rfft).swapaxes(-1, -3)


def _irfft(self: 'Grid', v: torch.Tensor,
           norm: str = 'forward') -> torch.Tensor:
    """Complex to real inverse transform

    Parameters
    ----------
    v : torch.Tensor (float)
        Last 3 dimensions must match shapeH_mine,
        and any preceding dimensions are batched over
    norm :  {'backward', 'forward', 'ortho'}, default: 'forward'
        Same as in torch.fft routines, but with default normalization
        to forward transforms (see :meth:`qimpy.grid.Grid.fft`)

    Returns
    -------
    torch.Tensor (complex)
        Last 3 dimensions will be shapeR_mine,
        preceded by any batch dimensions in the input.
    """
    if self.n_procs == 1:
        return torch.fft.irfftn(v, s=self.shape, norm=norm)
    assert(v.dtype.is_complex)
    shapeR_mine_complex = (self.split0.n_mine, self.shape[1], self.shapeH[2])
    return parallel_transform(
        self.rc, self.comm, v.swapaxes(-1, -3), norm,
        self.shapeH_mine[::-1], shapeR_mine_complex,
        safe_ifft2, safe_irfft, self.split0.n_prev, self.split2H.n_prev,
        *self.indices_irfft)


# Maps between corresponding real and complex types:
realType = {
    torch.complex128: torch.double,
    torch.complex64: torch.float}
complexType = {
    torch.double: torch.complex128,
    torch.float: torch.complex64}


# --- Wrappers to torch FFTs that are safe for zero sizes ---
def safe_fft(v: torch.Tensor, norm: str) -> torch.Tensor:
    return v if v.shape.count(0) else torch.fft.fft(v, norm=norm)


def safe_fft2(v: torch.Tensor, norm: str) -> torch.Tensor:
    return v if v.shape.count(0) else torch.fft.fft2(v, norm=norm)


def safe_ifft(v: torch.Tensor, norm: str) -> torch.Tensor:
    return v if v.shape.count(0) else torch.fft.ifft(v, norm=norm)


def safe_ifft2(v: torch.Tensor, norm: str) -> torch.Tensor:
    return v if v.shape.count(0) else torch.fft.ifft2(v, norm=norm)


def safe_rfft(v: torch.Tensor, norm: str) -> torch.Tensor:
    if v.shape.count(0):
        return torch.zeros(v.shape[:-1] + (v.shape[-1]//2+1,),
                           dtype=complexType[v.dtype], device=v.device)
    else:
        return torch.fft.rfft(v, norm=norm)


def safe_irfft(v: torch.Tensor, norm: str) -> torch.Tensor:
    if v.shape.count(0):
        return torch.zeros(v.shape[:-1] + (2*(v.shape[-1]-1),),
                           dtype=realType[v.dtype], device=v.device)
    else:
        return torch.fft.irfft(v, norm=norm)


# Test / benchmark parallelization of FFTs:
if __name__ == "__main__":
    qp.utils.log_config()
    qp.log.info('*'*15 + ' QimPy ' + qp.__version__ + ' ' + '*'*15)
    rc = qp.utils.RunConfig()

    # Get dimensions from input:
    import sys
    if len(sys.argv) < 4:
        qp.log.info(
            'Usage: _fft.py [<n_batch1> ...] <shape_0> <shape_1> <shape_2>')
        exit(1)
    shape = tuple(int(arg) for arg in sys.argv[-3:])
    n_batch = tuple(int(arg) for arg in sys.argv[1:-3])

    # Prerequisites for creating grid:
    lattice = qp.lattice.Lattice(
        rc=rc, system='triclinic', a=2.1, b=2.2, c=2.3,
        alpha=75, beta=80, gamma=85)  # pick one with no symmetries
    ions = qp.ions.Ions(rc=rc, pseudopotentials=[], coordinates=[])
    symmetries = qp.symmetries.Symmetries(rc=rc, lattice=lattice, ions=ions)

    # Create grids with and without parallelization:
    grid_par = qp.grid.Grid(rc=rc, lattice=lattice, symmetries=symmetries,
                            shape=shape, comm=rc.comm)  # parallel version
    grid_seq = qp.grid.Grid(rc=rc, lattice=lattice, symmetries=symmetries,
                            shape=shape, comm=None)  # sequential version

    def test(name, dtype_in, seq_func, par_func,
             in_start, in_stop, out_start, out_stop,
             shape_in, inverse):
        '''Helper function to test parallel and sequential versions
        of each Grid.fft routine against each other'''
        # Create test data:
        v_ref = torch.randn(n_batch + shape_in, dtype=dtype_in,
                            device=rc.device)
        rc.comm.Bcast(BufferView(v_ref), 0)
        n_repeats = 2 + int(1E8/np.prod(n_batch + shape))
        for i_repeat in range(n_repeats):
            # --- transform locally:
            if i_repeat:
                watch = qp.utils.StopWatch(name + '(seq)', rc)
            v_tld = seq_func(v_ref)
            if i_repeat:
                watch.stop()
            # --- extract MPI split piece of input and output:
            if inverse:
                v = v_ref[..., in_start:in_stop].contiguous()
                v_tld_ref = v_tld[..., out_start:out_stop, :, :].contiguous()
            else:
                v = v_ref[..., in_start:in_stop, :, :].contiguous()
                v_tld_ref = v_tld[..., out_start:out_stop].contiguous()
            # --- transform with MPI version:
            rc.comm.Barrier()  # for accurate timing
            if i_repeat:
                watch = qp.utils.StopWatch(name+'(par)', rc)
            v_tld = par_func(v)
            if i_repeat:
                watch.stop()
            # --- check accuracy:
            if not i_repeat:
                errors = np.array([
                    (torch.abs(v_tld - v_tld_ref)**2).sum().item(),
                    (torch.abs(v_tld_ref)**2).sum().item()])
                rc.comm.Allreduce(qp.MPI.IN_PLACE, errors)
                rmse = np.sqrt(errors[0]/errors[1])
                qp.log.info(f'{name} RMSE: {rmse:.2e}')

    # Run tests for all four transform types:
    test('fft', torch.complex128, grid_seq.fft, grid_par.fft,
         grid_par.split0.i_start, grid_par.split0.i_stop,
         grid_par.split2.i_start, grid_par.split2.i_stop,
         grid_par.shape, False)
    test('ifft', torch.complex128, grid_seq.ifft, grid_par.ifft,
         grid_par.split2.i_start, grid_par.split2.i_stop,
         grid_par.split0.i_start, grid_par.split0.i_stop,
         grid_par.shape, True)
    test('rfft', torch.double, grid_seq.rfft, grid_par.rfft,
         grid_par.split0.i_start, grid_par.split0.i_stop,
         grid_par.split2H.i_start, grid_par.split2H.i_stop,
         grid_par.shape, False)
    test('irfft', torch.complex128, grid_seq.irfft, grid_par.irfft,
         grid_par.split2H.i_start, grid_par.split2H.i_stop,
         grid_par.split0.i_start, grid_par.split0.i_stop,
         grid_par.shapeH, True)
    qp.utils.StopWatch.print_stats()
