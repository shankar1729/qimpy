import numpy as np
import qimpy as qp


class TaskDivision:
    """Division of a number of tasks over MPI.

    Attributes
    ----------
    n_tot : int
        total number of tasks over all processes (constructor input)
    n_procs : int
        number of processes to split over (constructor input)
    i_proc : int
        rank of current process (constructor input)
    n_each : int
        number of tasks on each process, till we run out on last few processes
    n_prev : np.array of n_procs+1 ints
        cumulative task counts on all previous processes
    i_start : int
        task start index on current process
    i_stop : int
        task stop index on current process
    n_mine : int
        number of tasks on current process"""

    def __init__(self, n_tot, n_procs, i_proc, name=None):
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
            qp.log.info('{:s} division:  n_tot: {:d}  n_each: {:d}  imbalance:'
                        ' {:.0f}%'.format(name, n_tot, self.n_each, imbalance))

    def whose(self, i):
        'Return process index i_proc responsible for task i'
        return i // self.n_each

    def is_mine(self, i):
        'Return True if current process is responsible for task i, else False'
        return (i // self.n_each == self.i_proc)
