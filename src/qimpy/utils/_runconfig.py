import os
import time
import datetime
import torch
import qimpy as qp
import numpy as np
from psutil import cpu_count
from typing import Optional


class RunConfig:
    """Run configuration / hardware resources. This includes CPU cores or GPU,
    and MPI communicators to be used by current QimPy instance.

    RunConfig must be created before any torch CUDA calls,
    so that a single CUDA context is associated with this process.
    Otherwise, on multi-GPU systems, CUDA MPI will subsequently fail."""

    comm: qp.MPI.Comm  #: Global communicator for QimPy
    i_proc: int  #: Rank within `comm`
    n_procs: int  #: Size of `comm`
    is_head: bool  #: Whether head of `comm`

    cpu: torch.device  #: CPU torch device
    device: torch.device  #: Preferred torch device for calculation (CPU / GPU)
    use_cuda: bool  #: Whether `device` is a CUDA GPU
    compute_stream: Optional[torch.cuda.Stream]  #: Asynchronous CUDA compute stream

    def __init__(
        self,
        *,
        comm: Optional[qp.MPI.Comm] = None,
        cores: Optional[int] = None,
    ):
        """Initialize hardware resources, process grid and communicators.

        Parameters
        ----------
        comm
            Top-level MPI communicator.
            Default (None) corresponds to mpi4py.MPI.COMM_WORLD.
        cores
            Number of CPU cores (and hence torch threads) to use per process.
            Default (None) will divide up available physical cores equally
            between processes on each node.
            Note that the environment variable SLURM_CPUS_PER_TASK (typically
            set by SLURM) will override cores, if set.
        """

        # Set and report start time:
        self.t_start: float = time.time()  #: start time used by :meth:`clock`
        qp.log.info("Start time: " + time.ctime(self.t_start))

        # MPI initialization:
        self.comm = qp.MPI.COMM_WORLD if (comm is None) else comm
        self.i_proc = self.comm.Get_rank()
        self.n_procs = self.comm.Get_size()
        self.is_head = self.i_proc == 0
        self.mpi_type: dict = {
            torch.int32: qp.MPI.INT,
            torch.int64: qp.MPI.LONG,
            torch.float32: qp.MPI.FLOAT,
            torch.float64: qp.MPI.DOUBLE,
            torch.complex64: qp.MPI.COMPLEX,
            torch.complex128: qp.MPI.DOUBLE_COMPLEX,
        }  #: Mapping from torch dtypes to MPI datatypes
        self.np_type: dict = {
            torch.bool: np.bool8,
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.complex64: np.complex64,
            torch.complex128: np.complex128,
        }  #: Mapping from torch dtypes to numpy datatypes

        # Select GPU before initializing torch:
        self.comm_node: qp.MPI.Comm = self.comm.Split_type(
            qp.MPI.COMM_TYPE_SHARED
        )  #: Communicator for processes on same shared-memory node
        i_proc_node = self.comm_node.Get_rank()
        n_procs_node = self.comm_node.Get_size()
        cuda_dev_str = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_dev_str:
            # Select one GPU and make sure it's only one visible to torch:
            cuda_devs = [int(s) for s in cuda_dev_str.split(",")]
            cuda_dev_selected = cuda_devs[i_proc_node % len(cuda_devs)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev_selected)
            n_gpus = min(1.0, len(cuda_devs) / n_procs_node)
        else:
            # Disable GPUs unless explicitly requested:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            n_gpus = 0.0

        # Initialize torch:
        self.compute_stream = None
        self.cpu = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.use_cuda = True
            torch.cuda.device(self.device)
            if os.environ.get("QIMPY_COMPUTE_STREAM", "1") in {"1", "yes"}:
                self.compute_stream = torch.cuda.Stream(device=self.device)
                qp.log.info("Async compute stream enabled for GPU operations.")
            else:
                qp.log.info("Async compute stream disabled for GPU operations.")
        else:
            self.device = self.cpu
            self.use_cuda = False
            n_gpus = 0.0
        torch.set_default_tensor_type(torch.DoubleTensor)

        # Threads:
        slurm_threads = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_threads:
            self.n_threads: int = int(
                slurm_threads
            )  #: number of threads to use on each process
        elif cores is None:
            # Divide up threads available on node:
            n_cores = cpu_count(logical=False)
            core_start = (i_proc_node * n_cores) // n_procs_node
            core_stop = ((i_proc_node + 1) * n_cores) // n_procs_node
            self.n_threads = core_stop - core_start
        else:
            self.n_threads = cores
        assert self.n_threads >= 1
        torch.set_num_threads(self.n_threads)
        self.no_grad = torch.no_grad()

        # Report total resources:
        run_totals = np.array([self.n_threads, n_gpus])
        self.comm.Allreduce(qp.MPI.IN_PLACE, run_totals, op=qp.MPI.SUM)
        qp.log.info(
            f"Run totals: {self.n_procs:g} processes, "
            f"{int(run_totals[0])} threads, {int(run_totals[1])} GPUs"
        )

    def compute_stream_wait_current(self):
        """Make `compute_stream` (if used) wait on current stream."""
        if self.compute_stream is not None:
            self.compute_stream.wait_stream(torch.cuda.current_stream())

    def current_stream_wait_compute(self):
        """Make current stream wait on `compute_stream` (if used)."""
        if self.compute_stream is not None:
            torch.cuda.current_stream().wait_stream(self.compute_stream)

    def current_stream_synchronize(self):
        """Wait for all tasks in current CUDA stream to complete."""
        if self.use_cuda:
            torch.cuda.current_stream().synchronize()

    def clock(self):
        """Time in seconds since start of this run."""
        return time.time() - self.t_start

    def report_end(self):
        """Report end time and duration."""
        t_stop = time.time()
        duration = datetime.timedelta(seconds=(t_stop - self.t_start))
        qp.log.info(f"\nEnd time: {time.ctime(t_stop)} (Duration: {duration})")

    def fmt(self, tensor: torch.Tensor, **kwargs) -> str:
        """Standardized conversion of torch tensors for logging.
        Keyword arguments are forwarded to numpy.array2string."""
        # Set some defaults in formatter:
        kwargs.setdefault("precision", 8)
        kwargs.setdefault("suppress_small", True)
        kwargs.setdefault("separator", ", ")
        return np.array2string(tensor.to(self.cpu).numpy(), **kwargs)
