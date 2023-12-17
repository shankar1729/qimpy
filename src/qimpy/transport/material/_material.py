from __future__ import annotations

import torch

from qimpy import TreeNode, MPI
from qimpy.mpi import ProcessGrid
from qimpy.io import CheckpointPath


class Material(TreeNode):
    """Base class / interface for material specifications."""

    comm: MPI.Comm  #: Communicator for reciprocal-space split over k (TODO)
    n_bands: int  #: number of bands at each k
    wk: float  #: Brillouin zone integration weight
    k: torch.Tensor  #: Nk x (2 or 3) wave vectors
    E: torch.Tensor  #: Nk x n_bands energies
    v: torch.Tensor  #: Nk x n_bands x (2 or 3) velocities in plane

    def __init__(
        self,
        *,
        wk: float,
        k: torch.Tensor,
        E: torch.Tensor,
        v: torch.Tensor,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """Initialize material parameters, typically used from a derived class."""
        super().__init__()
        self.comm = process_grid.get_comm("k")
        self.n_bands = E.shape[-1]
        self.wk = wk
        self.k = k
        self.E = E
        self.v = v

    @property
    def transport_velocity(self) -> torch.Tensor:
        """Effective velocity for each density-matrix component.
        This always has dimensions (n_k * (n_bands**2)) x 2."""
        v_plane = self.v[..., :2]  # ignore out-of-plane component if present
        v_dm = 0.5 * (v_plane[:, :, None] + v_plane[:, None])  # for density matrix
        return v_dm.flatten(0, 2)  # flatten k and both band dimensions
