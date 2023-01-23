import numpy as np
import qimpy as qp
import torch
from typing import Optional
from qimpy.rc import MPI


class TaskDivision:
    """Division of a number of tasks over MPI."""

    n_tot: int  #: Total number of tasks over all processes
    n_procs: int  #: Number of processes to split over
    i_proc: int  #: Rank of current process
    n_each: int  #: Number of tasks on each process (till we run out)
    n_prev: np.ndarray  #: Cumulative task counts (n_procs+1 ints)
    i_start: int  #: Task start index on current process
    i_stop: int  #: Task stop index on current process
    n_mine: int  #: Number of tasks on current process

    def __init__(
        self, *, n_tot: int, n_procs: int, i_proc: int, name: Optional[str] = None
    ) -> None:
        """Divide `n_tot` tasks among `n_procs` processes.
        Report division and load balance if `name` is not None."""
        # Store inputs:
        self.n_tot = n_tot
        self.n_procs = n_procs
        self.i_proc = i_proc
        # Compute remaining attributes:
        self.n_each = qp.utils.ceildiv(n_tot, n_procs)
        self.n_prev = np.minimum(n_tot, self.n_each * np.arange(n_procs + 1))
        self.i_start = self.n_prev[i_proc]
        self.i_stop = self.n_prev[i_proc + 1]
        self.n_mine = self.i_stop - self.i_start
        # Optionally report counts and imbalance:
        if name:
            imbalance = 100.0 * (1.0 - n_tot / (self.n_each * n_procs))
            qp.log.info(
                f"{name} division:  n_tot: {n_tot}  "
                f"n_each: {self.n_each}  imbalance: {imbalance:.0f}%"
            )

    def whose(self, i: int) -> int:
        """Return process index `i_proc` responsible for task `i`"""
        return i // self.n_each

    def whose_each(self, i: torch.Tensor) -> torch.Tensor:
        """Return process index `i_proc` responsible for each task in `i`"""
        return torch.div(i, self.n_each, rounding_mode="floor")

    def is_mine(self, i: int) -> bool:
        """Return whether current process is responsible for task i"""
        return self.i_start <= i < self.i_stop


class TaskDivisionCustom(TaskDivision):
    """Customized division of a number of tasks over MPI."""

    n_each_custom: np.ndarray  #: Custom number of tasks on each process

    def __init__(self, *, n_mine: int, comm: Optional[MPI.Comm]) -> None:
        """Initialize given local number of tasks on each processes."""
        # Collect n_mine on each process and initialize process parameters:
        if comm is None:
            self.n_each_custom = np.full(1, n_mine)
            super().__init__(n_tot=0, n_procs=1, i_proc=0)
        else:
            self.n_each_custom = np.array(comm.allgather(n_mine))
            super().__init__(n_tot=0, n_procs=comm.Get_size(), i_proc=comm.Get_rank())
        # Override base-class settings:
        self.n_mine = n_mine
        self.n_each = 0  # not applicable for custom division
        # Compute remaining attributes:
        self.n_prev = np.concatenate((np.zeros(1), self.n_each_custom.cumsum())).astype(
            int
        )
        self.n_tot = self.n_prev[-1]
        self.i_start = self.n_prev[self.i_proc]
        self.i_stop = self.n_prev[self.i_proc + 1]

    def whose(self, i: int) -> int:
        """Return process index i_proc responsible for task i"""
        return int(np.searchsorted(self.n_prev, i, side="right"))


def get_block_slices(n_tot: int, block_size: int) -> list[slice]:
    """Split `n_tot` tasks into blocks of size `block_size`.
    Returns a list of slices for each block.
    All blocks will have equal size (equal to `block_size`),
    except the last one that may be smaller."""
    if n_tot:
        starts = np.arange(0, n_tot, block_size)
        slices = [slice(start, stop) for start, stop in zip(starts[:-1], starts[1:])]
        slices.append(slice(starts[-1], n_tot))  # add final block (possibly smaller)
        return slices
    else:
        return []
