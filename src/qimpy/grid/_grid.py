from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import torch

from qimpy import rc, log, TreeNode, grid, MPI
from qimpy.mpi import TaskDivision
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.lattice import Lattice
from qimpy.symmetries import Symmetries
from ._fft import init_grid_fft, FFT, IFFT, IndicesType


class Grid(TreeNode):
    """Real and reciprocal space grids for a unit cell.
    The grid could either be local or distributed over an MPI communicator,
    and this class provides FFT routines to switch fields on these grids,
    and routines to convert fields between grids.
    """

    lattice: Lattice
    symmetries: Symmetries
    _field_symmetrizer: Optional[grid.FieldSymmetrizer]
    comm: Optional[MPI.Comm]  #: Communicator to split grid and FFTs over
    n_procs: int  #: Size of comm
    i_proc: int  #: Rank within comm
    is_split: bool  #: Whether the grid is split over MPI
    ke_cutoff: float  #: Kinetic energy of Nyquist-frequency plane-waves
    shape: tuple[int, ...]  #: Global real-space grid dimensions
    shapeH: tuple[int, ...]  #: Global half-reciprocal-space grid dimensions
    shapeR_mine: tuple[int, ...]  #: Local real grid dimensions
    shapeG_mine: tuple[int, ...]  #: Local reciprocal grid dimensions
    shapeH_mine: tuple[int, ...]  #: Local half-reciprocal grid dimensions
    split0: TaskDivision  #: MPI split of real-space dimension 0
    split2: TaskDivision  #: MPI split of reciprocal dimension 2
    split2H: TaskDivision  #: MPI split of half-reciprocal dimension 2
    _mesh1D: dict[str, tuple[torch.Tensor, ...]]  # Global 1D meshes
    _mesh1D_mine: dict[str, tuple[torch.Tensor, ...]]  # Local 1D meshes
    _indices_fft: IndicesType  #: All-to-all unscramble indices for `fft`
    _indices_ifft: IndicesType  #: All-to-all unscramble indices for `ifft`
    _indices_rfft: IndicesType  #: All-to-all unscramble indices for `rfft`
    _indices_irfft: IndicesType  #: All-to-all unscramble indices for `irfft`

    def __init__(
        self,
        *,
        lattice: Lattice,
        symmetries: Symmetries,
        comm: Optional[MPI.Comm],
        checkpoint_in: CheckpointPath = CheckpointPath(),
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
                log.info(
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
            log.info(
                f"minimum shape for ke-cutoff: ({shape_min[0]:.2f},"
                f" {shape_min[1]:.2f}, {shape_min[2]:.2f})"
            )
            # Align to multiple of 4 for FFT efficiency:
            shape_min = 4 * np.ceil(np.array(shape_min) / 4).astype(int)
            log.info(f"minimum multiple-of-4 shape: {tuple(shape_min)}")

        if shape is not None:
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
        log.info(f"selected shape: {self.shape}")
        init_grid_fft(self)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["shape"] = self.shape
        attrs["ke_cutoff"] = self.ke_cutoff
        return list(attrs.keys())

    @property
    def dV(self) -> float:
        """Volume per grid point (real-space integration weight)."""
        return self.lattice.volume / float(np.prod(self.shape))

    @property
    def weight2H(self) -> torch.Tensor:
        """Hermitian-symmetry weights for reduced reciprocal space."""
        split2H = self.split2H
        iz_start = max(1, split2H.i_start) - split2H.i_start
        iz_stop = min(split2H.n_tot - 1, split2H.i_stop) - split2H.i_start
        result = torch.ones(split2H.n_mine, device=rc.device)
        result[iz_start:iz_stop] *= 2.0
        return result

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
        iG = torch.stack(torch.meshgrid(*mesh1D, indexing="ij")).to(torch.double)
        return 1j * torch.tensordot(self.lattice.Gbasis, iG, dims=1)

    def get_Gmax(self) -> float:
        """Get maximum wave-vector magnitude of the FFT grid."""
        iG_box = torch.tensor(
            np.array([[+1, +1, +1], [+1, +1, -1], [+1, -1, +1], [+1, -1, -1]])
            * (np.array(self.shape) // 2)[None, :],
            device=rc.device,
            dtype=torch.double,
        )
        return (iG_box @ self.lattice.Gbasis.T).norm(dim=1).max().item()

    @property
    def field_symmetrizer(self) -> grid.FieldSymmetrizer:
        """Symmetrizer for fields on this grid (initialized on first use)."""
        if self._field_symmetrizer is None:
            self._field_symmetrizer = grid.FieldSymmetrizer(self)
        return self._field_symmetrizer

    def fft(self, v: torch.Tensor) -> torch.Tensor:
        """
        Forward Fast Fourier Transform.
        This method dispatches to complex-to-complex or real-to-complex
        transforms depending on whether the input `v` is complex or real.
        Note that QimPy applies normalization in forward transforms,
        corresponding to norm='forward' in the torch.fft routines.
        This makes the G=0 components in reciprocal space correspond
        to the mean value of the real space version.

        Parameters
        ----------
        v : torch.Tensor (complex or real)
            Last 3 dimensions must match `shapeR_mine`,
            and any preceding dimensions are batched over.

        Returns
        -------
        torch.Tensor (complex)
            Last 3 dimensions will be `shapeG_mine` or `shapeH_mine`,
            depending on whether `v` is complex or real respectively,
            preceded by any batch dimensions in the input.
        """
        return FFT.apply(self, v)

    def ifft(self, v: torch.Tensor) -> torch.Tensor:
        """
        Inverse Fast Fourier Transform.
        This method dispatches to complex-to-complex or complex-to-real
        transforms depending on whether the last three dimensions of `v`
        match `shapeG_mine` or `shapeH_mine` respectively.
        Note that QimPy applies normalization in forward transforms
        (see :meth:`qimpy.grid.Grid.fft`).

        Parameters
        ----------
        v : torch.Tensor (complex)
            Last 3 dimensions must match shapeG_mine,
            and any preceding dimensions are batched over

        Returns
        -------
        torch.Tensor (complex or real)
            Last 3 dimensions will be shapeR_mine,
            preceded by any batch dimensions in the input.
            The result will be complex or real, depending on whether the last
            three dimensions of `v` match `shapeG_mine` or `shapeH_mine`.
        """
        return IFFT.apply(self, v)
