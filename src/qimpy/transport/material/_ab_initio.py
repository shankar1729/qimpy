from __future__ import annotations
from typing import Sequence, Callable

import torch

from qimpy.io import CheckpointPath
from qimpy.mpi import ProcessGrid
from . import Material


class AbInitio(Material):
    """Ab initio material specification."""

    def __init__(
        self,
        *,
        fname: str,
        rotation: Sequence[Sequence[float]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize ab initio material.

        Parameters
        ----------
        fname
            :yaml:`File name to load materials data from.`
        rotation
            :yaml:`3 x 3 rotation matrix from material to simulation frame.`
        """
        # TODO: read data from FeynWann file
        nk_tot = 100
        nk = 1
        n_bands = 1
        super().__init__(
            wk=1.0 / nk_tot,
            nk=nk,
            n_bands=n_bands,
            n_dim=3,
            checkpoint_in=checkpoint_in,
            process_grid=process_grid,
        )

    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        return NotImplemented

    def get_contact_distribution(self, n: torch.Tensor, **kwargs) -> torch.Tensor:
        return NotImplemented

    def rho_dot(self, rho: torch.Tensor) -> torch.Tensor:
        return NotImplemented
