import qimpy as qp
import numpy as np
import torch
from ._fft import _init_grid_fft, _fft, _ifft, IndicesType, MethodFFT
from typing import Optional, Sequence, Tuple, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import RunConfig, TaskDivision
    from ..lattice import Lattice
    from ..symmetries import Symmetries


class Grid(qp.Constructable):
    """Real and reciprocal space grids for a unit cell.
    The grid could either be local or distributed over an MPI communicator,
    and this class provides FFT routines to switch fields on these grids,
    and routines to convert fields between grids.
    """
    __slots__ = [
        'rc', 'lattice', 'comm', 'n_procs', 'i_proc', 'is_split', 'ke_cutoff',
        'dV', 'shape', 'shapeH', 'shapeR_mine', 'shapeG_mine', 'shapeH_mine',
        'split0', 'split2', 'split2H', 'mesh1D',
        '_indices_fft', '_indices_ifft', '_indices_rfft', '_indices_irfft']
    rc: 'RunConfig'
    lattice: 'Lattice'
    comm: Optional[qp.MPI.Comm]  #: Communicator to split grid and FFTs over
    n_procs: int  #: Size of comm
    i_proc: int  #: Rank within comm
    is_split: bool  #: Whether the grid is split over MPI
    ke_cutoff: float  #: Kinetic energy of Nyquist-frequency plane-waves
    dV: float  #: Volume per grid point (real-space integration weight)
    shape: Tuple[int, ...]  #: Global real-space grid dimensions
    shapeH: Tuple[int, ...]  #: Global half-reciprocal-space grid dimensions
    shapeR_mine: Tuple[int, ...]  #: Local real grid dimensions
    shapeG_mine: Tuple[int, ...]  #: Local reciprocal grid dimensions
    shapeH_mine: Tuple[int, ...]  #: Local half-reciprocal grid dimensions
    split0: 'TaskDivision'  #: MPI division of real-space dimension 0
    split2: 'TaskDivision'  #: MPI division of reciprocal dimension 2
    split2H: 'TaskDivision'  #: MPI division of half-reciprocal dimension 2
    mesh1D: Dict[str, Tuple[torch.Tensor, ...]]  #: 1D meshes for `get_mesh`
    _indices_fft: IndicesType  #: All-to-all unscramble indices for `fft`
    _indices_ifft: IndicesType  #: All-to-all unscramble indices for `ifft`
    _indices_rfft: IndicesType  #: All-to-all unscramble indices for `rfft`
    _indices_irfft: IndicesType  #: All-to-all unscramble indices for `irfft`

    fft: MethodFFT = _fft
    ifft: MethodFFT = _ifft

    def __init__(self, *, rc: 'RunConfig', lattice: 'Lattice',
                 symmetries: 'Symmetries', comm: Optional[qp.MPI.Comm],
                 ke_cutoff_wavefunction: Optional[float] = None,
                 ke_cutoff: Optional[float] = None,
                 shape: Optional[Sequence[int]] = None) -> None:
        """Create local or distributed grid for `lattice`.

        Parameters
        ----------
        lattice
            Lattice whose reciprocal lattice vectors define plane-wave basis
        symmetries
            Symmetries with which grid dimensions will be made commensurate,
            or checked if specified explicitly by shape below.
        comm
            Communicator to split grid (and its FFTs) over, if provided.
        ke_cutoff_wavefunction
            Plane-wave kinetic-energy cutoff in :math:`E_h` for any electronic
            wavefunctions to be used with this grid. This is an internally set
            parameter (should not be specified in dict / YAML input) that
            effectively sets the default for ke_cutoff.
        ke_cutoff
            Plane-wave kinetic-energy cutoff in :math:`E_h` for the grid
            (i.e. the charge-density cutoff). This supercedes the default
            of `4 * ke_cutoff_wavefunction` (if specified), but may be
            superceded by explicitly specified shape
        shape
            Explicit grid dimensions. Highest precedence, and if specified,
            will supercede ke_cutoff
        """
        super().__init__()
        self.rc = rc
        self.lattice = lattice

        # MPI settings (identify local or split):
        self.comm = comm
        self.n_procs, self.i_proc = ((1, 0) if (comm is None)
                                     else (comm.Get_size(), comm.Get_rank()))
        self.is_split = (self.n_procs == 1)

        # Select the relevant ke-cutoff:
        self.ke_cutoff = (ke_cutoff if ke_cutoff else 0.)
        if ke_cutoff_wavefunction:
            if not ke_cutoff:  # note that ke_cutoff takes precedence
                self.ke_cutoff = 4*ke_cutoff_wavefunction
            # Make sure cutoff is sufficient to resolve wavefunctions:
            if self.ke_cutoff < ke_cutoff_wavefunction:
                raise ValueError(
                    f'ke_cutoff (={self.ke_cutoff:g}) must be >= '
                    f'ke_cutoff_wavefunction (={ke_cutoff_wavefunction:g})')
            elif self.ke_cutoff < 4*ke_cutoff_wavefunction:
                qp.log.info(
                    f'Note: ke_cutoff (={self.ke_cutoff:g}) < 4'
                    f'*ke_cutoff_wavefunction (={4*ke_cutoff_wavefunction:g})'
                    ' truncates high wave vectors in density calculation')

        # Compute minimum grid dimensions for cutoff:
        shape_min = None
        if self.ke_cutoff:
            Gmax = np.sqrt(2.*self.ke_cutoff)  # G-sphere radius at cutoff
            # This sphere should be within shape_min/2 in each direction x
            # corresponding spacing between reciprocal lattice planes (2pi/R).
            # Therefore shape_min >= 2 * Gmax / (2pi/R)
            shape_min = (lattice.Rbasis.norm(dim=0) * (Gmax / np.pi)).tolist()
            qp.log.info(f'minimum shape for ke-cutoff: ({shape_min[0]:.2f},'
                        f' {shape_min[1]:.2f}, {shape_min[2]:.2f})')
            # Align to multiple of 4 for FFT efficiency:
            shape_min = 4 * np.ceil(np.array(shape_min) / 4).astype(int)
            qp.log.info(f'minimum multiple-of-4 shape: {tuple(shape_min)}')

        if shape:
            self.shape = tuple(shape)
            # Check symmetries and cutoff of specified shape:
            symmetries.check_grid_shape(self.shape)
            if ((shape_min is not None)
                    and np.any(np.array(self.shape) < shape_min)):
                raise ValueError(
                    f'Specified shape {self.shape} < minimum shape')
        else:
            if shape_min is None:
                raise KeyError('At least one of ke-cutoff-wavefunction, '
                               'ke-cutoff or shape must be specified')
            self.shape = tuple(symmetries.get_grid_shape(shape_min))
        qp.log.info(f'selected shape: {self.shape}')
        self.dV = self.lattice.volume / float(np.prod(self.shape))
        _init_grid_fft(self)

    def get_mesh(self, space: str) -> torch.Tensor:
        """Get mesh integer coordinates for real or reciprocal space

        Parameters
        ----------
        space : {'R', 'G', 'H'}
            Which space to compute mesh coordinates for: 'R' = real space,
            'G' = full reciprocal space and 'H' = half or Hermitian-symmetric
            recipocal space resulting from FFT of real functions on grid.

        Returns
        -------
        Tensor
            Integer tensor with dimensions shape_mine + (3,), where
            shape_mine is the relevant local dimensions of requested space
        """
        return torch.stack(
            torch.meshgrid(*self.mesh1D[space])).permute(1, 2, 3, 0)

    def get_gradient_operator(self, space: str) -> torch.Tensor:
        """Get gradient operator in reciprocal space.

        Parameters
        ----------
        space : {'G', 'H'}
            Which space to compute mesh coordinates for: 'G' = full reciprocal
            space and 'H' = half or Hermitian-symmetric recipocal space.

        Returns
        -------
        Tensor
            Tensor with dimensions (3,) + shape_mine, where
            shape_mine is the relevant local dimensions of requested space
        """
        iG = torch.stack(torch.meshgrid(*self.mesh1D[space])).to(torch.double)
        return 1j * torch.tensordot(self.lattice.Gbasis, iG, dims=1)

    def get_Gmax(self) -> float:
        """Get maximum wave-vector magnitude of the FFT grid."""
        iG_box = torch.tensor(np.array([
                [+1, +1, +1],
                [+1, +1, -1],
                [+1, -1, +1],
                [+1, -1, -1]]) * (np.array(self.shape) // 2)[None, :],
            device=self.rc.device, dtype=torch.double)
        return (iG_box @ self.lattice.Gbasis.T).norm(dim=1).max().item()
