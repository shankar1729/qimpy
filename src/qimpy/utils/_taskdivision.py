import numpy as np
import qimpy as qp
from typing import Optional


class TaskDivision:
    """Division of a number of tasks over MPI."""
    __slots__ = ('n_tot', 'n_procs', 'i_proc', 'n_each', 'n_prev',
                 'i_start', 'i_stop', 'n_mine')

    n_tot: int  #: Total number of tasks over all processes
    n_procs: int  #: Number of processes to split over
    i_proc: int  #: Rank of current process
    n_each: int  #: Number of tasks on each process (till we run out)
    n_prev: np.ndarray  #: Cumulative task counts (n_procs+1 ints)
    i_start: int  #: Task start index on current process
    i_stop: int  #: Task stop index on current process
    n_mine: int  #: Number of tasks on current process

    def __init__(self, n_tot: int, n_procs: int, i_proc: int,
                 name: Optional[str] = None):
        '''Divide `n_tot` tasks among `n_procs` processes'''
        # Store inputs:
        self.n_tot = n_tot
        self.n_procs = n_procs
        self.i_proc = i_proc
        # Compute remaining attributes:
        self.n_each = qp.utils.ceildiv(n_tot, n_procs)
        self.n_prev = np.minimum(n_tot, self.n_each * np.arange(n_procs+1))
        self.i_start = self.n_prev[i_proc]
        self.i_stop = self.n_prev[i_proc+1]
        self.n_mine = self.i_stop - self.i_start
        # Optionally report counts and imbalance:
        if name:
            imbalance = 100.*(1. - n_tot / (self.n_each * n_procs))
            qp.log.info(f'{name} division:  n_tot: {n_tot}  '
                        f'n_each: {self.n_each}  imbalance: {imbalance:.0f}%')

    def whose(self, i: int) -> int:
        'Return process index i_proc responsible for task i'
        return i // self.n_each

    def is_mine(self, i: int) -> bool:
        'Return whether current process is responsible for task i'
        return (i // self.n_each == self.i_proc)
