import numpy as np
import qimpy as qp
from typing import Optional


class TaskDivision(qp.Constructable):
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

    def __init__(self, *, n_tot: int = 0, n_procs: int = 1, i_proc: int = 0,
                 name: Optional[str] = None,
                 co: qp.ConstructOptions = qp.ConstructOptions()) -> None:
        """Divide `n_tot` tasks among `n_procs` processes.
        Default corresponds to no tasks, divided on a single process.
        Use :meth:`update_division` to change this later if `n_tot`
        not known initially. """
        super().__init__(co=co)
        self.update_division(n_tot, n_procs, i_proc, name)

    def update_division(self, n_tot: int, n_procs: int, i_proc: int,
                        name: Optional[str] = None) -> None:
        """Update division to `n_tot` tasks among `n_procs` processes.
        Report division and load balance if `name` is not None."""
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
        """Return process index i_proc responsible for task i"""
        return i // self.n_each

    def is_mine(self, i: int) -> bool:
        """Return whether current process is responsible for task i"""
        return self.i_start <= i < self.i_stop


class TaskDivisionCustom(TaskDivision):
    """Customized division of a number of tasks over MPI."""
    __slots__ = ('n_each_custom',)
    n_each_custom: np.ndarray  #: Custom number of tasks on each process

    def __init__(self, *, n_mine: int, comm: Optional[qp.MPI.Comm],
                 co: qp.ConstructOptions = qp.ConstructOptions()) -> None:
        """Initialize given local number of tasks on each processes."""
        super().__init__(co=co)
        # Collect n_mine on each process and store inputs:
        if comm is None:
            self.n_each_custom = np.full(1, n_mine)
            self.n_procs = 1
            self.i_proc = 0
        else:
            self.n_each_custom = np.array(comm.allgather(n_mine))
            self.n_procs = comm.Get_size()
            self.i_proc = comm.Get_rank()
        self.n_mine = n_mine
        self.n_each = 0
        # Compute remaining attributes:
        self.n_prev = np.concatenate(([0], self.n_each_custom.cumsum()))
        self.n_tot = self.n_prev[-1]
        self.i_start = self.n_prev[self.i_proc]
        self.i_stop = self.n_prev[self.i_proc+1]

    def whose(self, i: int) -> int:
        """Return process index i_proc responsible for task i"""
        return int(np.searchsorted(self.n_prev, i, side='right'))
