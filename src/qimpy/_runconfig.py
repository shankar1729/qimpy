import os
import torch
import qimpy as qp
import numpy as np
from mpi4py import MPI
from psutil import cpu_count


class RunConfig:
    """Run configuration: hardware resources including CPU cores or GPU,
    and top-level MPI communicator to be used by current QimPy instance.

    RunConfig must be created before any torch CUDA calls,
    so that a single CUDA context is associated with this process.
    Otherwise, on multi-GPU systems, CUDA MPI will subsequently fail."""

    def __init__(self, comm=None):
        """Initialize run configuration

        Parameters
        ----------
        comm : mpi4py.MPI.COMM or None, optional
            If provided, use as the top-level MPI communicator.
            Default (None) corresponds to mpi4py.MPI.COMM_WORLD."""

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
        else:
            # Divide up threads available on node:
            n_cores = cpu_count(logical=False)
            core_start = (i_proc_node * n_cores) // n_procs_node
            core_stop = ((i_proc_node + 1) * n_cores) // n_procs_node
            self.n_threads = core_stop - core_start
        torch.set_num_threads(self.n_threads)

        # Report total resources:
        run_totals = np.array([self.n_threads,  n_gpus])
        self.comm.Allreduce(MPI.IN_PLACE, run_totals, op=MPI.SUM)
        qp.log.info(
            '\nRun totals: {:g} processes, {:d} threads, {:d} GPUs'.format(
                self.n_procs, int(run_totals[0]), int(run_totals[1])))
