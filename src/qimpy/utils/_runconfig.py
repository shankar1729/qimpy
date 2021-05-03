import os
import time
import datetime
import torch
import qimpy as qp
import numpy as np
from mpi4py import MPI
from psutil import cpu_count


class RunConfig:
    """Run configuration: hardware resources including CPU cores or GPU,
    and MPI communicators to be used by current QimPy instance.

    RunConfig must be created before any torch CUDA calls,
    so that a single CUDA context is associated with this process.
    Otherwise, on multi-GPU systems, CUDA MPI will subsequently fail."""

    def __init__(self, *, comm=None, cores=None, process_grid=(1, -1, -1)):
        """
        Parameters
        ----------
        comm : mpi4py.MPI.COMM or None, optional
            Top-level MPI communicator.
            Default (None) corresponds to mpi4py.MPI.COMM_WORLD.
        cores : int or None, optional
            Number of CPU cores (and hence torch threads) to use per process.
            Default (None) will divide up available physical cores equally
            between processes on each node.
            Note that the environment variable SLURM_CPUS_PER_TASK (typically
            set by SLURM) will override cores, if set.
        process_grid : tuple of 3 ints, optional
            Division of processes into a grid, where the dimensions in order
            are replicas (eg. in NEB or phonon perturbations), k-points and
            bands/basis (split over basis usually, and over bands for FFTs).
            The product should equal the total number of processes.
            One or more dimensions can be set to -1; these are then set to
            most equally divide the remaining factor in the process count.
            Default: (1, -1, -1) implies do not split over replicas, and
            split over k-points and band/basis as equally as possible."""

        # Set and report start time:
        self.t_start = time.time()  # start time used by clock()
        qp.log.info('Start time: ' + time.ctime(self.t_start))

        # MPI initialization:
        self.comm = MPI.COMM_WORLD if (comm is None) else comm
        self.i_proc = self.comm.Get_rank()
        self.n_procs = self.comm.Get_size()
        self.is_head = (self.i_proc == 0)
        self.mpi_type = {  # Map from relevant torch to MPI datatypes
            torch.int32: MPI.INT,
            torch.int64: MPI.LONG,
            torch.float32: MPI.FLOAT,
            torch.float64: MPI.DOUBLE,
            torch.complex64: MPI.COMPLEX,
            torch.complex128: MPI.DOUBLE_COMPLEX
        }

        # Select GPU before initializing torch:
        self.comm_node = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
        i_proc_node = self.comm_node.Get_rank()
        n_procs_node = self.comm_node.Get_size()
        cuda_devs = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_devs:
            # Select one GPU and make sure it's only one visible to torch:
            cuda_devs = [int(s) for s in cuda_devs.split(',')]
            cuda_dev_selected = cuda_devs[i_proc_node % len(cuda_devs)]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_dev_selected)
            n_gpus = min(1., len(cuda_devs) / n_procs_node)
        else:
            # Disable GPUs unless explicitly requested:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            n_gpus = 0.

        # Initialize torch:
        self.cpu = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.use_cuda = True
            torch.cuda.device(self.device)
        else:
            self.device = self.cpu
            self.use_cuda = False
            n_gpus = 0.
        torch.set_default_tensor_type(torch.DoubleTensor)

        # Threads:
        slurm_threads = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurm_threads:
            self.n_threads = int(slurm_threads)
        elif cores is None:
            # Divide up threads available on node:
            n_cores = cpu_count(logical=False)
            core_start = (i_proc_node * n_cores) // n_procs_node
            core_stop = ((i_proc_node + 1) * n_cores) // n_procs_node
            self.n_threads = core_stop - core_start
        else:
            self.n_threads = cores
        assert(self.n_threads >= 1)
        torch.set_num_threads(self.n_threads)

        # Report total resources:
        run_totals = np.array([self.n_threads,  n_gpus])
        self.comm.Allreduce(MPI.IN_PLACE, run_totals, op=MPI.SUM)
        qp.log.info(
            'Run totals: {:g} processes, {:d} threads, {:d} GPUs'.format(
                self.n_procs, int(run_totals[0]), int(run_totals[1])))

        # Process grid:
        process_grid = np.array(process_grid, dtype=int)
        assert(process_grid.shape == (3,))
        try:
            distribute_product_evenly(self.n_procs, process_grid)
        except AssertionError:
            raise ValueError(('Cannot distribute {:d} processes to'
                              + ' {:d} x {:d} x {:d} grid').format(
                                  self.n_procs, *tuple(process_grid)))
        qp.log.info(('Process grid: {:d} replicas x {:d} k-points x {:d} '
                     + 'bands/basis').format(*tuple(process_grid)))

    def clock(self):
        "Time in seconds since start of this run."
        return time.time() - self.t_start

    def report_end(self):
        "Report end time and duration."
        t_stop = time.time()
        duration = datetime.timedelta(seconds=(t_stop - self.t_start))
        qp.log.info('\nEnd time: {:s} (Duration: {:s})'.format(
            time.ctime(t_stop), str(duration)))

    def fmt(self, tensor):
        'Standardized conversion of torch tensors for log'
        return np.array2string(
            tensor.to(self.cpu).numpy(),
            precision=8, suppress_small=True, separator=', ')


def distribute_product_evenly(product, dims):
    'Distribute product over any -1 in dims (np.array) as evenly as possible'
    # Split known and unknown pieces:
    i_unknown = np.where(dims == -1)[0]
    i_known = np.where(dims != -1)[0]
    product_known = np.prod(dims[i_known])
    product_unknown = product // product_known
    assert(product % product_known == 0)
    # Trivial if 0 or 1 unknown:
    if len(i_unknown) == 0:
        assert(product_unknown == 1)
        return dims
    if len(i_unknown) == 1:
        dims[i_unknown] = product_unknown
        return dims
    # Distribute product_unknown evenly between unknown factors:
    # --- simple split over prime factors in order to get close to even
    # --- global optimum split is not required (and not an easy problem)
    n_unknown = len(i_unknown)
    factors = [1] + prime_factorization(product_unknown)
    cumulative_product = np.cumprod(factors)
    cut_ideal = product_unknown ** (np.arange(n_unknown+1) / n_unknown)
    i_cut = np.abs(np.log(cumulative_product[None, :]
                          / cut_ideal[:, None])).argmin(axis=-1)
    cut = cumulative_product[i_cut]
    dims[i_unknown] = (cut[1:] // cut[:-1])[::-1]
    return dims  # note smallest prime factors put at end


def prime_factorization(N):
    'Return prime factorization of a number'
    factors = []
    p = 2
    while p*p <= N:
        while N % p == 0:
            factors.append(p)
            N //= p
        p += 1
    if N > 1:  # any left-over factor must be prime itself
        factors.append(N)
    return factors
