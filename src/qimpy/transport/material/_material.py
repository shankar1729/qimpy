from __future__ import annotations
from typing import Callable
from abc import abstractmethod

import torch
import torch.distributed as dist
import numpy as np

from qimpy import TreeNode, rc
from qimpy.mpi import ProcessGrid, TaskDivision
from qimpy.io import CheckpointPath


class Material(TreeNode):
    """Base class / interface for material specifications."""

    group: dist.ProcessGroup  #: Process group for reciprocal-space split over k
    k_division: TaskDivision  #: Division of k-points over MPI
    k_mine: slice  #: slice of k on current process
    nk_mine: int  #: number of k-points on current process
    n_bands: int  #: number of bands at each k
    n_dim: int  #: dimensionality of material (2 or 3)
    wk: torch.Tensor  #: Brillouin zone integration weight
    k: torch.Tensor  #: nk_mine x n_dim wave vectors
    E: torch.Tensor  #: nk_mine x n_bands energies
    v: torch.Tensor  #: nk_mine x n_bands x n_dim velocities in plane
    rho0: torch.Tensor  #: nk_mine x n_bands x n_bands initial density matrix
    dt_max: float  #: maximum stable time-step (set to inf if not available)

    def __init__(self, *, checkpoint_in: CheckpointPath = CheckpointPath()) -> None:
        super().__init__()

    def initialize(
        self,
        *,
        wk: torch.Tensor,
        nk: int,
        n_bands: int,
        n_dim: int,
        process_grid: ProcessGrid,
    ) -> None:
        """Initialize shared material parameters (call from a derived class.)"""
        self.group = process_grid.get_group("k")
        self.k_division = TaskDivision(
            n_tot=nk, i_proc=self.group.rank(), n_procs=self.group.size()
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
        be called with a Nsurf x Nkbb_mine tensor, and the reflection should be
        calculated pointwise in real-space with output of the same dimensions."""

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

    @abstractmethod
    def get_flux_indices(self) -> list[int]:
        """Return list of flux names and observable indices, specific to each material.
        Fluxes are computed on edges (faces) of the mesh, rather than on the cell
        centers, as done for the observables. These are useful for plotting accurate
        streamlines of fluxes, and are most useful for conserved quantities such
        as particle number. Hence, specify indices to a subset of observables from
        `get_observables` to compute fluxes for."""

    def constrain(self, rho: torch.Tensor) -> None:
        """Override to apply any per-material constraints on `rho`
        that may be violated during spatial transport or dynamics."""

    def measure_observables(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        """Return expectation value of observables, (Nx x Ny x No).
        Optionally, override to add/replace observables that cannot be expressed
        as a linear operator `get_observables` on the occupations / density-matrices.
        Include the names of these added/replaced observables in `get_observable_names`.
        """
        result = torch.einsum("a, ra, oa -> ro", self.wk, rho, self.get_observables(t))
        if self.group.size() > 1:
            result = result.contiguous()
            dist.all_reduce(result, group=self.group)
        return result

    def update_stash(self, rho: torch.Tensor, r_division: TaskDivision) -> None:
        """Override to stash extra objects to output with the next checkpoint.
        Use real-space division `r_division` to coordinate slices in the checkpoint.
        `Geometry` already handles this for observables, so this is only for
        any extra diagnostics not captured in the obersvables list above."""


def fermi(E, mu, T):
    return torch.special.expit((mu - E) / T)


def bose(omegaPh, T):
    return 1 / torch.expm1(omegaPh / T)
