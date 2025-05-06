from __future__ import annotations
from typing import Callable
from abc import abstractmethod

import torch
import numpy as np

from qimpy import TreeNode, MPI, rc
from qimpy.mpi import ProcessGrid, TaskDivision, BufferView
from qimpy.io import CheckpointPath


class Material(TreeNode):
    """Base class / interface for material specifications."""

    comm: MPI.Comm  #: Communicator for reciprocal-space split over k
    k_division: TaskDivision  #: Division of k-points over MPI
    k_mine: slice  #: slice of k on current process
    nk_mine: int  #: number of k-points on current process
    n_bands: int  #: number of bands at each k
    n_dim: int  #: dimensionality of material (2 or 3)
    wk: float  #: Brillouin zone integration weight
    k: torch.Tensor  #: nk_mine x n_dim wave vectors
    E: torch.Tensor  #: nk_mine x n_bands energies
    v: torch.Tensor  #: nk_mine x n_bands x n_dim velocities in plane
    rho0: torch.Tensor  #: nk_mine x n_bands x n_bands initial density matrix
    dt_max: float  #: maximum stable time-step (set to inf if not available)

    def __init__(
        self,
        *,
        wk: float,
        nk: int,
        n_bands: int,
        n_dim: int,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """Initialize material parameters, typically used from a derived class."""
        super().__init__()
        self.comm = process_grid.get_comm("k")
        self.k_division = TaskDivision(
            n_tot=nk, i_proc=self.comm.rank, n_procs=self.comm.size
        )
        self.nk_mine = self.k_division.n_mine
        self.k_mine = slice(self.k_division.i_start, self.k_division.i_stop)
        self.n_bands = n_bands
        self.n_dim = n_dim
        self.wk = wk
        self.k = torch.zeros((self.nk_mine, n_dim), device=rc.device)
        self.E = torch.zeros((self.nk_mine, n_bands), device=rc.device)
        self.v = torch.zeros((self.nk_mine, n_bands, n_dim), device=rc.device)
        self.rho0 = torch.zeros((self.nk_mine, n_bands, n_bands), device=rc.device)
        self.dt_max = np.inf

    @abstractmethod
    def initialize_fields(
        self, rho: torch.Tensor, params: dict[str, torch.Tensor], patch_id: int
    ) -> None:
        """Initialize spatially-dependent / parameter sweep values.
        Using named properties `params`, containing tensors that should broadcast
        with the grid dimensions, update the density matrix `rho` to be spatially
        varying if necessary and calculate any named fields for use during dynamics.
        Note that the cached quantities must be associated with `patch_id`, as there
        may be multiple spatial domains (patches) sharing the same material."""

    @property
    def transport_velocity(self) -> torch.Tensor:
        """Effective velocity for each density-matrix component.
        This always has dimensions (nk_mine * (n_bands**2)) x 2."""
        v_plane = self.v[..., :2]  # ignore out-of-plane component if present
        v_dm = 0.5 * (v_plane[:, :, None] + v_plane[:, None])  # for density matrix
        return v_dm.flatten(0, 2)  # flatten k and both band dimensions

    @abstractmethod
    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a function (or callable object) to calculate reflections for a
        sequence of surface points with unit normals (Nsurf x 2). This function will
        be called with a Nghost x Nsurf x Nkbb_mine tensor, and the reflection should
        be calculated pointwise in real-space with output of the same dimensions."""

    @abstractmethod
    def get_contactor(
        self, n: torch.Tensor, **kwargs
    ) -> Callable[[float], torch.Tensor]:
        """Return a function (or callable object) to calculate the distribution
        function at a contact with orientation `n` and specified keyword arguments.
        For an Nsurf x 2 tensor n, the function should take time `t` as an input
        and return the corresponding Nsurf x Nkbb_mine distribution function."""

    @abstractmethod
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        """Return material contribution to drho/dt.
        This should include scattering and any coherent evolution in band space."""

    @abstractmethod
    def get_observable_names(self) -> list[str]:
        """Return list of observable names, specific to each material."""

    @abstractmethod
    def get_observables(self, t: float) -> torch.Tensor:
        """Return tensor of complex conjugates of all observables specific to each
        material. (No x Nkbb_mine) where No is number of observables."""

    def measure_observables(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        """Return expectation value of observables, (Nx x Ny x No)."""
        result = self.wk * torch.einsum("xya, oa -> xyo", rho, self.get_observables(t))
        if self.comm.size > 1:
            self.comm.Allreduce(MPI.IN_PLACE, BufferView(result))
        return result


def fermi(E, mu, T):
    return torch.special.expit((mu - E) / T)


def bose(omegaPh, T):
    return 1 / torch.expm1(omegaPh / T)
