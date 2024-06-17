from __future__ import annotations
from typing import Optional, Union
import math

import numpy as np
import torch

from qimpy import rc, log, TreeNode, MPI
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import TaskDivision, ProcessGrid, BufferView, globalreduce
from qimpy.math import ceildiv
from qimpy.lattice import Lattice, Kpoints
from qimpy.symmetries import Symmetries
from qimpy.grid import Grid
from qimpy.dft.ions import Ions
from ._basis_ops import _apply_gradient, _apply_ke, _apply_potential, _collect_density
from ._basis_real import BasisReal


class Basis(TreeNode):
    """Plane-wave basis for electronic wavefunctions. The underlying
    :class:`qimpy.mpi.TaskDivision` splits plane waves over `rc.comm_b`"""

    comm: MPI.Comm  #: Basis/bands communicator
    comm_kb: MPI.Comm  #: Overall k-points and basis/bands communicator
    lattice: Lattice  #: Lattice vectors of unit cell
    ions: Ions  #: Ionic system: implicit in basis for ultrasoft / PAW
    kpoints: Kpoints  #: Corresponding k-point set
    n_spins: int  #: Default number of spin channels
    n_spinor: int  #: Default number of spinorial components
    k: torch.Tensor  #: Subset of k handled by this basis (due to MPI division)
    wk: torch.Tensor  #: Subset of weights corresponding to `k`
    w_sk: torch.Tensor  #: Combined spin and k-point weights
    real_wavefunctions: bool  #: Whether wavefunctions are real
    ke_cutoff: float  #: Kinetic energy cutoff
    grid: Grid  #: Wavefunction grid (always process-local)
    iG: torch.Tensor  #: Plane waves in reciprocal lattice coordinates
    n: torch.Tensor  #: Number of plane waves for each `k`
    n_min: int  #: Minimum of `n` across all `k` (including on other processes)
    n_max: int  #: Maximum of `n` across all `k` (including on other processes)
    n_avg: float  #: Average `n` across all `k` (weighted by `wk`)
    n_tot: int  #: Actual common `n` stored for each `k` including padding
    fft_index: torch.Tensor  #: Index of each plane wave in reciprocal grid
    fft_block_size: int  #: Number of bands to FFT together
    mpi_block_size: int  #: Number of bands to MPI transfer together
    PadIndex = tuple[
        slice, torch.Tensor, slice, slice, torch.Tensor
    ]  #: Indexing datatype for `pad_index` and `pad_index_mine`
    pad_index: PadIndex  #: Which basis entries are padding (beyond `n`)
    pad_index_mine: PadIndex  #: Subset of `pad_index` on this process
    division: TaskDivision  #: Division of basis across `rc.comm_b`
    mine: slice  #: Slice of basis entries local to this process
    real: BasisReal  #: Extra indices for real wavefunctions

    apply_gradient = _apply_gradient
    apply_ke = _apply_ke
    apply_potential = _apply_potential
    collect_density = _collect_density

    def __init__(
        self,
        *,
        process_grid: ProcessGrid,
        lattice: Lattice,
        ions: Ions,
        symmetries: Symmetries,
        kpoints: Kpoints,
        n_spins: int,
        n_spinor: int,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        ke_cutoff: float = 20.0,
        real_wavefunctions: bool = False,
        grid: Optional[Union[Grid, dict]] = None,
        fft_block_size: int = 0,
        mpi_block_size: int = 0,
    ) -> None:
        """Initialize plane-wave basis with `ke_cutoff`.

        Parameters
        ----------
        lattice
            Lattice whose reciprocal lattice vectors define plane-wave basis.
        ions
            Ions that specify the pseudopotential portion of the basis;
            the basis implicitly depends on the ion positions for ultrasoft
            or PAW due to the augmentation of all operators at each ion.
        symmetries
            Symmetries with which the wavefunction grid should be commensurate.
        kpoints
            Set of k-points to initialize basis for. Note that the basis is
            only initialized for k-points to be operated on by current process
            i.e. for k = kpoints.k[kpoints.i_start : kpoints.i_stop].
        n_spins
            Default number of spin channels for wavefunctions in this basis.
        n_spinor
            Default number of spinor components for wavefunctions in this
            basis. Also used only to check support for real_wavefunctions.
        ke_cutoff
            :yaml:`Wavefunction kinetic energy cutoff in Hartrees.`
        real_wavefunctions
            :yaml:`Whether to use real wavefunctions (instead of complex).`
            This is only supported for non-spinorial, Gamma-point-only
            calculations, where conjugate symmetry allows real wavefunctions.
        grid
            :yaml:`Override parameters of grid for wavefunction operations.`
        fft_block_size
            :yaml:`Number of wavefunction bands to FFT simultaneously.`
            Higher numbers require more memory, but can achieve
            better occupancy of GPUs or high-core-count CPUs.
            The default of 0 auto-selects the block size based on the number
            of bands and k-points being processed by each process.
        mpi_block_size
            :yaml:`Number of wavefunction bands to MPI transfer simultaneously.`
            Lower numbers may allow better overlap between computation and transfers,
            which is beneficial if MPI implementation supports asynchronous progress
            and/or CUDA streams are used to compute asynrchronously.
            Higher numbers mitigate MPI latency, but may require more memory.
            This number is automatically rounded up to nearest multiple of
            `fft_block_size * comm.size`. The default of 0 selects the block size
            based on the number of bands and k-points being processed by each process.
        """
        super().__init__()
        self.comm = process_grid.get_comm("b")
        self.comm_kb = process_grid.get_comm("kb")
        self.lattice = lattice
        self.ions = ions
        self.kpoints = kpoints
        self.n_spins = n_spins
        self.n_spinor = n_spinor
        self.fft_block_size = int(fft_block_size)
        self.mpi_block_size = int(mpi_block_size)

        # Select subset of k-points relevant on this process:
        k_mine = slice(kpoints.division.i_start, kpoints.division.i_stop)
        self.k = kpoints.k[k_mine]
        self.wk = kpoints.wk[k_mine]
        w_spin = 2 // (self.n_spins * self.n_spinor)  # spin weight
        self.w_sk = w_spin * self.wk.view(1, -1, 1)  # combined k and spin

        # Check real wavefunction support:
        self.real_wavefunctions = real_wavefunctions
        if self.real_wavefunctions:
            if n_spinor == 2:
                raise ValueError(
                    "real-wavefunctions not compatible" " with spinorial calculations"
                )
            if kpoints.k.norm().item():  # i.e. not all k = 0 (Gamma)
                raise ValueError(
                    "real-wavefunctions only compatible with"
                    " Gamma-point-only calculations"
                )

        # Initialize grid to match cutoff:
        self.ke_cutoff = float(ke_cutoff)
        log.info("\nInitializing wavefunction grid:")
        self.add_child(
            "grid",
            Grid,
            grid,
            checkpoint_in,
            lattice=lattice,
            symmetries=symmetries,
            comm=None,  # Never parallel
            ke_cutoff_wavefunction=self.ke_cutoff,
        )

        # Initialize or read indices, along with process-grid-dependent padding:
        # Both paths set iG, fft_index, n and its statistics (min, max, avg and tot)
        if checkpoint_in:
            self._read_indices(checkpoint_in)
        else:
            self._initialize_indices()
        self.pad_index = self._get_pad_index(0, self.n_tot)

        # Divide basis on comm_b:
        div = TaskDivision(
            n_tot=self.n_tot,
            n_procs=self.comm.size,
            i_proc=self.comm.rank,
            name="padded basis",
        )
        self.division = div
        self.mine = slice(div.i_start, div.i_stop)
        self.pad_index_mine = self._get_pad_index(div.i_start, div.i_stop)

        if self.real_wavefunctions and kpoints.division.n_mine:
            self.real = BasisReal(self)

    @property
    def n_avg_weighted(self) -> float:
        """`n_avg` accounting for real-basis weights if any."""
        return self.real.Gweight_tot if self.real_wavefunctions else self.n_avg

    @property
    def n_ideal(self) -> float:
        """Ideal `n_avg_weighted` based on `ke_cutoff` G-sphere volume."""
        return ((2.0 * self.ke_cutoff) ** 1.5) * self.lattice.volume / (6 * np.pi**2)

    def get_gradient(self, basis_slice: slice = slice(None)) -> torch.Tensor:
        """Kinetic energy (KE) of each plane wave in basis in :math:`E_h`

        Parameters
        ----------
        basis_slice
            Selection of basis functions to get KE for (default: full basis)

        Returns
        -------
        torch.Tensor
            TODO, dimensions: `nk_mine` x len(`basis_slice`)
        """
        G = (self.iG[:, basis_slice] + self.k[:, None, :]) @ self.lattice.Gbasis.T
        return 1.0j * G

    def get_ke(self, basis_slice: slice = slice(None)) -> torch.Tensor:
        """Kinetic energy (KE) of each plane wave in basis in :math:`E_h`

        Parameters
        ----------
        basis_slice
            Selection of basis functions to get KE for (default: full basis)

        Returns
        -------
        torch.Tensor
            KE for each plane-wave, dimensions: `nk_mine` x len(`basis_slice`)
        """
        G = (self.iG[:, basis_slice] + self.k[:, None, :]) @ self.lattice.Gbasis.T
        return 0.5 * G.square().sum(dim=-1)

    def get_ke_stress(self, basis_slice: slice = slice(None)) -> torch.Tensor:
        """Kinetic energy (KE) stress tensor of each plane wave in basis in :math:`E_h`

        Parameters
        ----------
        basis_slice
            Selection of basis functions to get KE for (default: full basis)

        Returns
        -------
        torch.Tensor
            KE tensor for each plane-wave,
             dimensions: `nk_mine` x len(`basis_slice`) x 3 x 3
        """
        G = (self.iG[:, basis_slice] + self.k[:, None, :]) @ self.lattice.Gbasis.T
        return -G.unsqueeze(2) * G.unsqueeze(3)  # 3 x 3 outer product

    def get_fft_block_size(self, n_batch: int, n_bands: int) -> int:
        """Number of FFTs to perform together. Equals `fft_block_size`, if that is
        non-zero, and uses a heuristic based on batch dimension and number of bands."""
        if self.fft_block_size:
            block_size = self.fft_block_size
        else:
            if not (n_batch and n_bands):
                return 1  # Irrelevant since no FFTs to perform anyway
            # TODO: better heuristics on how much data to FFT at once
            min_data = 16_000_000 if rc.use_cuda else 100_000
            min_block = ceildiv(min_data, n_batch * math.prod(self.grid.shape))
            max_block = ceildiv(n_bands, 16)  # based on memory limit
            block_size = min(min_block, max_block)
        # Report selected block-size once:
        if not Basis._fft_block_size_reported:
            log.info(f"Selected block-size for band FFT: {block_size}")
            Basis._fft_block_size_reported = True
        return block_size

    def get_mpi_block_size(
        self, n_batch: int, n_bands: int, fft_block_size: int
    ) -> int:
        """Number of bands to MPI transfer together for `collect_density` and
        `apply_potential`. Uses `mpi_block_size`, if that is non-zero, and uses a
        heuristic based on batch dimension and number of bands. The final number is
        coerced to a multiple of `fft_block_size * comm.size` or rounded up to
        `n_bands`, if it is already close to that limit."""
        if self.mpi_block_size:
            mpi_block_size = self.mpi_block_size
        else:
            if not (n_batch and n_bands):
                return 1  # Irrelevant since nothing to transfer anyway
            # TODO: better heuristics on how much data to MPI-transfer at once
            min_data = 2_000_000  # TODO: incorporate MPI latency info somehow
            mpi_block_size = ceildiv(min_data, n_batch * self.n_tot)
        # Enforce multiple of fft_block_size * comm.size:
        divisor = fft_block_size * self.division.n_procs
        mpi_block_size = ceildiv(mpi_block_size, divisor) * divisor
        # Round up to n_bands if not enough blocks:
        if mpi_block_size * 2 > n_bands:
            mpi_block_size = n_bands  # no gain in working with <= 2 blocks
        # Report selected block-size once for each mode:
        if not Basis._mpi_block_size_reported:
            log.info(f"Selected block-size for band MPI: {mpi_block_size}")
            Basis._mpi_block_size_reported = True
        return mpi_block_size

    _fft_block_size_reported = False  #: Make sure FFT block size reported once
    _mpi_block_size_reported = False  #: Make sure MPI block size reported once

    def allreduce_in_place(self, x: torch.Tensor, op: MPI.Op = MPI.SUM) -> None:
        """Allreduce `x` in place using `op` over `self.comm`.
        Convenient wrapper used in many basis operations."""
        if self.division.n_procs > 1:
            rc.current_stream_synchronize()
            self.comm.Allreduce(MPI.IN_PLACE, BufferView(x), op)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["ke_cutoff"] = self.ke_cutoff
        attrs["real_wavefunctions"] = self.real_wavefunctions
        attrs["fft_block_size"] = self.fft_block_size
        attrs["mpi_block_size"] = self.mpi_block_size

        # Write basis count:
        checkpoint, path = cp_path
        kdiv = self.kpoints.division
        dset = checkpoint.create_dataset_real(f"{path}/n", (kdiv.n_tot,), self.n.dtype)
        checkpoint.write_slice(dset, (kdiv.i_start,), self.n)

        # Write fft index:
        dset = checkpoint.create_dataset_real(
            f"{path}/fft_index", (kdiv.n_tot, self.n_max), self.fft_index.dtype
        )
        checkpoint.write_slice(dset, (kdiv.i_start, 0), self.fft_index[:, : self.n_max])
        return [*attrs.keys(), "n", "fft_index"]

    def _set_n(self, n: torch.Tensor) -> None:
        """Set n and its stats (min, max, avg and tot)."""
        comm = self.kpoints.comm
        self.n = n
        self.n_min = globalreduce.min(n, comm)
        self.n_max = globalreduce.max(n, comm)
        n_procs_b = self.comm.size
        self.n_tot = ceildiv(self.n_max, n_procs_b) * n_procs_b
        self.n_avg = globalreduce.sum(self.n * self.wk, comm)
        log.info(
            f"n_basis:  min: {self.n_min}  max: {self.n_max}"
            f"  avg: {self.n_avg:.3f}  ideal: {self.n_ideal:.3f}"
        )

    def _initialize_indices(self) -> None:
        """Initialize iG, fft_index, n and its stats based on ke_cutoff."""
        iG = self.grid.get_mesh("H" if self.real_wavefunctions else "G").view(-1, 3)
        self.iG = iG  # used by get_ke below
        within_cutoff = self.get_ke() < self.ke_cutoff  # mask of which iG to keep
        self._set_n(within_cutoff.count_nonzero(dim=1))
        # Set indices from basis set to FFT grid, including padding up to n_tot:
        n_fft = iG.shape[0]  # number of points on FFT grid
        assert self.n_tot <= n_fft  # make sure padding doesn't exceed grid
        fft_range = torch.arange(n_fft, device=rc.device)
        priority = torch.where(within_cutoff, 0, n_fft) + fft_range[None, :]
        self.fft_index = priority.argsort(dim=1)[:, : self.n_tot]  # ke<cutoff to front
        self.iG = iG[self.fft_index]

    def _read_indices(self, checkpoint_in: CheckpointPath) -> None:
        """Read fft_index and n from checkpoint, and then set iG and n stats."""
        # Read portion of n and fft_index belonging to this process
        kdiv = self.kpoints.division
        cp, path = checkpoint_in
        n = cp.read_slice(cp[f"{path}/n"], (kdiv.i_start,), (kdiv.n_mine,))
        self._set_n(n)
        fft_index = cp.read_slice(
            cp[f"{path}/fft_index"], (kdiv.i_start, 0), (kdiv.n_mine, self.n_max)
        )
        # Introduce padding from n_max to n_tot (which depends on parallelization)
        iG = self.grid.get_mesh("H" if self.real_wavefunctions else "G").view(-1, 3)
        n_fft = iG.shape[0]  # number of points on FFT grid
        assert self.n_tot <= n_fft  # make sure padding doesn't exceed grid
        # --- start out with high values in priority for all indices
        # --- then replace the ones in fft_index such that they appear in order
        priority = n_fft + torch.arange(n_fft, device=rc.device).repeat(kdiv.n_mine, 1)
        ik = torch.arange(kdiv.i_start, kdiv.i_stop, device=rc.device)[:, None]
        priority[ik, fft_index] = torch.arange(self.n_max, device=rc.device)
        self.fft_index = priority.argsort(dim=1)[:, : self.n_tot]
        self.iG = iG[self.fft_index]

    def _get_pad_index(self, index_start: int, index_stop: int) -> PadIndex:
        """Get tuple for indexing padded entries in wavefunctions,
        for range of basis indices, index_start to index_stop."""
        indices = torch.arange(index_start, index_stop, device=rc.device)
        pad_k, pad_G = torch.where(indices >= self.n[:, None])
        return slice(None), pad_k, slice(None), slice(None), pad_G  # add s, b, spinor
