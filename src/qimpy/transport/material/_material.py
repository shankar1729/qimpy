from __future__ import annotations
from typing import Callable
from abc import abstractmethod

import torch

from qimpy import TreeNode, MPI, rc
from qimpy.mpi import ProcessGrid, TaskDivision, BufferView
from qimpy.io import CheckpointPath


class Material(TreeNode):
    """Base class / interface for material specifications."""

    comm: MPI.Comm  #: Communicator for reciprocal-space split over k
    k_division: TaskDivision  #: Division of k-points over MPI
    k_mine: slice  #: slice of k on current process
    n_bands: int  #: number of bands at each k
    n_dim: int  #: dimensionality of material (2 or 3)
    wk: float  #: Brillouin zone integration weight
    k: torch.Tensor  #: nk x n_dim wave vectors
    E: torch.Tensor  #: nk x n_bands energies
    v: torch.Tensor  #: nk x n_bands x n_dim velocities in plane

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
        self.k_mine = slice(self.k_division.i_start, self.k_division.i_stop)
        self.n_bands = n_bands
        self.n_dim = n_dim
        self.wk = wk
        self.nk = nk
        self.k = torch.zeros((nk, n_dim), device=rc.device)
        self.E = torch.zeros((nk, n_bands), device=rc.device)
        self.v = torch.zeros((nk, n_bands, n_dim), device=rc.device)

    @property
    def transport_velocity(self) -> torch.Tensor:
        """Effective velocity for each density-matrix component.
        This always has dimensions (nk_mine * (n_bands**2)) x 2."""
        v_plane = self.v[self.k_mine, :, :2]  # ignore out-of-plane component if present
        v_dm = 0.5 * (v_plane[:, :, None] + v_plane[:, None])  # for density matrix
        return v_dm.flatten(0, 2)  # flatten k and both band dimensions

    @abstractmethod
    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a function (or callable object) to calculate reflections for a
        sequence of surface points with unit normals (Nsurf x 2). This function will
        be called with a Nghost x Nsurf x Nkbb_mine tensor, and the reflection should
        be calculated pointwise in real-space with output of the same dimensions."""

    @abstractmethod
    def get_contact_distribution(self, n: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return the distribution function at a contact with orientation `n`
        and specified keyword arguments. For an Nsurf x 2 tensor n, the output
        should be the corresponding Nsurf x Nkbb_mine distribution function."""

    @abstractmethod
    def rho_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        """Return material contribution to drho/dt.
        This should include scattering and any coherent evolution in band space."""

    @abstractmethod
    def get_observable_names(self) -> list[str]:
        """Return string of observables, comma seperated, specific to each material."""

    @abstractmethod
    def get_observables(self, Nkbb: int, t: float) -> torch.Tensor:
        """Return tensor of complex conjugates of all observables specific to each
        material. (No x Nkbb_mine) where No is number of observables."""

    def measure_observables(
        self, rho: torch.Tensor, t: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrun density and flux of observables, (Nx x Ny x No) for density and
        (Nx x Ny x No x 2) for flux."""
        Nkbb = rho.shape[-1]
        obs = self.get_observables(Nkbb, t)
        density = self.wk * torch.einsum("xya, oa -> xyo", rho, obs)
        flux = self.wk * torch.einsum(
            "xya, av, oa -> xyov", rho, self.transport_velocity, obs
        )
        if self.comm.size > 1:
            self.comm.Allreduce(MPI.IN_PLACE, BufferView(density))
            self.comm.Allreduce(MPI.IN_PLACE, BufferView(flux))
        return density, flux
