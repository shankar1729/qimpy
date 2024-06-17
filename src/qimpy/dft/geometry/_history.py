from __future__ import annotations
from typing import Union

import torch
import numpy as np


from qimpy import rc, TreeNode, MPI
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import TaskDivision


class History(TreeNode):
    """Helper to save history along trajectory."""

    comm: MPI.Comm
    iter_division: TaskDivision  # Division of iterations over MPI
    i_iter: int  # Current iteration / last iteration for which history available
    save_map: dict[str, np.ndarray]  # Names and data for quantities to save

    def __init__(
        self,
        *,
        comm: MPI.Comm,
        n_max: int,
        i_iter: int = 0,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        super().__init__()
        self.comm = comm
        self.iter_division = TaskDivision(
            n_tot=n_max, n_procs=comm.size, i_proc=comm.rank
        )
        self.i_iter = i_iter
        self.save_map = {}

        if checkpoint_in:
            checkpoint, path = checkpoint_in
            assert checkpoint is not None
            group = checkpoint[path]
            i_start = self.iter_division.i_start
            i_stop = min(self.iter_division.i_stop, self.i_iter + 1)
            n_in = i_stop - i_start  # number of iterations to be read at this process
            for name in group.keys():
                dset = group[name]  # version in file
                data = np.empty(
                    (self.iter_division.n_mine,) + dset.shape[1:], dtype=dset.dtype
                )  # for version in memory, split over MPI
                if n_in > 0:
                    data[:n_in] = dset[i_start:i_stop]
                self.save_map[name] = data

    def add(self, name: str, value: Union[float, torch.Tensor]) -> None:
        """Add current `value` for variable `name` to history."""
        data = np.array(value) if isinstance(value, float) else value.to(rc.cpu).numpy()
        if name not in self.save_map:
            assert self.i_iter == 0  # if not, previous history must have been read in
            self.save_map[name] = np.empty(
                (self.iter_division.n_mine,) + data.shape, dtype=data.dtype
            )
        if self.iter_division.is_mine(self.i_iter):
            i_out = self.i_iter - self.iter_division.i_start  # local index
            self.save_map[name][i_out] = data

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        cp_path.attrs["i_iter"] = self.i_iter
        saved_list = []
        for name, data in self.save_map.items():
            self._save(cp_path.relative(name), data)
            saved_list.append(name)
        return saved_list

    def _save(self, cp_path: CheckpointPath, data: np.ndarray) -> None:
        """Save history `data` up to `i_iter`'th iteration to `cp_path`."""
        checkpoint, path = cp_path
        assert checkpoint is not None
        dset = checkpoint.create_dataset(
            path, shape=(self.i_iter + 1,) + data.shape[1:], dtype=data.dtype
        )
        i_start = self.iter_division.i_start
        i_stop = min(self.iter_division.i_stop, self.i_iter + 1)
        n_out = i_stop - i_start
        if n_out > 0:
            dset[i_start:i_stop] = data[:n_out]
