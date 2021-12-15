import qimpy as qp
import numpy as np
import torch
from ._fft import _init_grid_fft, _fft, _ifft, IndicesType, MethodFFT
from typing import Optional, Sequence, Tuple, Dict


class Grid(qp.TreeNode):
    """Real and reciprocal space grids for a unit cell.
    The grid could either be local or distributed over an MPI communicator,
    and this class provides FFT routines to switch fields on these grids,
    and routines to convert fields between grids.
    """

    __slots__ = (
        "lattice",
        "symmetries",
        "_field_symmetrizer",
        "comm",
        "n_procs",
        "i_proc",
        "is_split",
        "ke_cutoff",
        "dV",
        "shape",
        "shapeH",
        "shapeR_mine",
        "shapeG_mine",
        "shapeH_mine",
        "split0",
        "split2",
        "split2H",
        "_mesh1D",
        "_mesh1D_mine",
        "_indices_fft",
        "_indices_ifft",
        "_indices_rfft",
        "_indices_irfft",
    )
    lattice: qp.lattice.Lattice
    symmetries: qp.symmetries.Symmetries
    _field_symmetrizer: Optional[qp.symmetries.FieldSymmetrizer]
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
    split0: qp.utils.TaskDivision  #: MPI split of real-space dimension 0
    split2: qp.utils.TaskDivision  #: MPI split of reciprocal dimension 2
    split2H: qp.utils.TaskDivision  #: MPI split of half-reciprocal dimension 2
    _mesh1D: Dict[str, Tuple[torch.Tensor, ...]]  # Global 1D meshes
    _mesh1D_mine: Dict[str, Tuple[torch.Tensor, ...]]  # Local 1D meshes
    _indices_fft: IndicesType  #: All-to-all unscramble indices for `fft`
    _indices_ifft: IndicesType  #: All-to-all unscramble indices for `ifft`
    _indices_rfft: IndicesType  #: All-to-all unscramble indices for `rfft`
    _indices_irfft: IndicesType  #: All-to-all unscramble indices for `irfft`

    fft: MethodFFT = _fft
    ifft: MethodFFT = _ifft

    def __init__(
        self,
        *,
        lattice: qp.lattice.Lattice,
        symmetries: qp.symmetries.Symmetries,
        comm: Optional[qp.MPI.Comm],
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        ke_cutoff_wavefunction: Optional[float] = None,
        ke_cutoff: Optional[float] = None,
        shape: Optional[Sequence[int]] = None,
    ) -> None:
        """Create local or distributed grid for `lattice`.

        Parameters
        ----------
        lattice
            Lattice whose reciprocal lattice vectors define plane-wave basis
        symmetries
            Symmetries with which grid dimensions will be made commensurate,
            checked if specified explicitly by shape below and used for
            symmetrization of :class:`Field`'s associated with this grid.
        comm
            Communicator to split grid (and its FFTs) over, if provided.
        ke_cutoff_wavefunction
            Plane-wave kinetic-energy cutoff in :math:`E_h` for any electronic
            wavefunctions to be used with this grid. This is an internally set
            parameter that effectively sets the default for `ke_cutoff` to
            `4 * ke_cutoff_wavefunction`, and has no effect if either
            `ke_cutoff` or `shape` is specified explicitly.
        ke_cutoff
            :yaml:`Kinetic-energy cutoff for grid in Hartrees.`
            If unspecified, this is taken to be its minimum value:
            4x the wavefunction kinetic energy cutoff.
            This has no effect if `shape` is specified explicitly.
        shape
            :yaml:`Explicit grid dimensions [Nx, Ny, Nz].`
            Highest precedence, and if specified, will supercede `ke_cutoff`.
        """
        super().__init__()
        self.lattice = lattice
        self.symmetries = symmetries
        self._field_symmetrizer = None

        # MPI settings (identify local or split):
        self.comm = comm
        self.n_procs, self.i_proc = (
            (1, 0) if (comm is None) else (comm.Get_size(), comm.Get_rank())
        )
        self.is_split = self.n_procs == 1

        # Select the relevant ke-cutoff:
        self.ke_cutoff = ke_cutoff if ke_cutoff else 0.0
        if ke_cutoff_wavefunction:
            if not ke_cutoff:  # note that ke_cutoff takes precedence
                self.ke_cutoff = 4 * ke_cutoff_wavefunction
            # Make sure cutoff is sufficient to resolve wavefunctions:
            if self.ke_cutoff < ke_cutoff_wavefunction:
                raise ValueError(
                    f"ke_cutoff (={self.ke_cutoff:g}) must be >= "
                    f"ke_cutoff_wavefunction (={ke_cutoff_wavefunction:g})"
                )
            elif self.ke_cutoff < 4 * ke_cutoff_wavefunction:
                qp.log.info(
                    f"Note: ke_cutoff (={self.ke_cutoff:g}) < 4"
                    f"*ke_cutoff_wavefunction (={4*ke_cutoff_wavefunction:g})"
                    " truncates high wave vectors in density calculation"
                )

        # Compute minimum grid dimensions for cutoff:
        shape_min = None
        if self.ke_cutoff:
            Gmax = np.sqrt(2.0 * self.ke_cutoff)  # G-sphere radius at cutoff
            # This sphere should be within shape_min/2 in each direction x
            # corresponding spacing between reciprocal lattice planes (2pi/R).
            # Therefore shape_min >= 2 * Gmax / (2pi/R)
            shape_min = (lattice.Rbasis.norm(dim=0) * (Gmax / np.pi)).tolist()
            qp.log.info(
                f"minimum shape for ke-cutoff: ({shape_min[0]:.2f},"
                f" {shape_min[1]:.2f}, {shape_min[2]:.2f})"
            )
            # Align to multiple of 4 for FFT efficiency:
            shape_min = 4 * np.ceil(np.array(shape_min) / 4).astype(int)
            qp.log.info(f"minimum multiple-of-4 shape: {tuple(shape_min)}")

        if shape:
            self.shape = tuple(shape)
            # Check symmetries and cutoff of specified shape:
            symmetries.check_grid_shape(self.shape)
            if (shape_min is not None) and np.any(np.array(self.shape) < shape_min):
                raise ValueError(f"Specified shape {self.shape} < minimum shape")
        else:
            if shape_min is None:
                raise KeyError(
                    "At least one of ke-cutoff-wavefunction, "
                    "ke-cutoff or shape must be specified"
                )
            self.shape = tuple(symmetries.get_grid_shape(shape_min))
        qp.log.info(f"selected shape: {self.shape}")
        self.dV = self.lattice.volume / float(np.prod(self.shape))
        _init_grid_fft(self)

    def get_mesh(self, space: str, mine: bool = True) -> torch.Tensor:
        """Get mesh integer coordinates for real or reciprocal space

        Parameters
        ----------
        space : {'R', 'G', 'H'}
            Which space to compute mesh coordinates for: 'R' = real space,
            'G' = full reciprocal space and 'H' = half or Hermitian-symmetric
            recipocal space resulting from FFT of real functions on grid.

        mine
            Only get local portion of mesh if True (default).
            Otherwise, get the entire mesh, even if grid is split over MPI.

        Returns
        -------
        Tensor
            Integer tensor with dimensions shape_mine + (3,), where
            shape_mine is the relevant local dimensions of requested space
            if mine is True (and with shape, instead if mine is False).
        """
        mesh1D = self._mesh1D_mine[space] if mine else self._mesh1D[space]
        return torch.stack(torch.meshgrid(*mesh1D, indexing="ij")).permute(1, 2, 3, 0)

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
        mesh1D = self._mesh1D_mine[space]
        iG = torch.stack(torch.meshgrid(*mesh1D)).to(torch.double)
        return 1j * torch.tensordot(self.lattice.Gbasis, iG, dims=1)

    def get_Gmax(self) -> float:
        """Get maximum wave-vector magnitude of the FFT grid."""
        iG_box = torch.tensor(
            np.array([[+1, +1, +1], [+1, +1, -1], [+1, -1, +1], [+1, -1, -1]])
            * (np.array(self.shape) // 2)[None, :],
            device=qp.rc.device,
            dtype=torch.double,
        )
        return (iG_box @ self.lattice.Gbasis.T).norm(dim=1).max().item()

    @property
    def field_symmetrizer(self) -> qp.symmetries.FieldSymmetrizer:
        """Symmetrizer for fields on this grid (initialized on first use)."""
        if self._field_symmetrizer is None:
            self._field_symmetrizer = qp.symmetries.FieldSymmetrizer(self)
        return self._field_symmetrizer
