"""Run configuration / hardware resources. This includes CPU cores or GPU, and MPI
communicators to be used by the current QimPy instance. The import-time configuration
selects a single CPU core for each MPI process in `mpi4py.MPI.COMM_WORLD`.

Call `init` to select the number of cores or a GPU device, as available and based
on environment variables including SLURM_CPUS_PER_TASK and CUDA_VISIBLE_DEVICES.
Note that `init` must be called before any torch CUDA calls, so that a single CUDA
context is associated with this process. Otherwise, on multi-GPU systems,
any CUDA MPI will subsequently fail.
"""

import os
import time
import datetime
import torch
import qimpy as qp
import numpy as np
from psutil import cpu_count
from typing import Optional, Dict

# List exported symbols for doc generation
__all__ = (
    "comm",
    "i_proc",
    "n_procs",
    "is_head",
    "cpu",
    "device",
    "use_cuda",
    "compute_stream",
    "compute_stream_wait_current",
    "current_stream_wait_compute",
    "current_stream_synchronize",
    "clock",
    "report_end",
)

comm: qp.MPI.Comm = qp.MPI.COMM_WORLD  #: Global communicator for QimPy
i_proc: int = comm.rank  #: Rank within `comm`
n_procs: int = comm.size  #: Size of `comm`
is_head: bool = i_proc == 0  #: Whether head of `comm`
cpu: torch.device = torch.device("cpu")  #: CPU torch device
device: torch.device = cpu  #: Preferred torch device for calculation (CPU / GPU)
use_cuda: bool = False  #: Whether `device` is a CUDA GPU
compute_stream: Optional[torch.cuda.Stream] = None  #: Asynchronous CUDA compute stream
t_start: float = time.time()  #: Start time used for `clock` (set by `init`)

# Set reasonable pre-init defaults for torch:
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_num_threads(1)  # to prevent overcommit between MPI processes

# Declare type mappings from torch to MPI and numpy:
mpi_type: Dict[torch.dtype, qp.MPI.Datatype] = {
    torch.int32: qp.MPI.INT,
    torch.int64: qp.MPI.LONG,
    torch.float32: qp.MPI.FLOAT,
    torch.float64: qp.MPI.DOUBLE,
    torch.complex64: qp.MPI.COMPLEX,
    torch.complex128: qp.MPI.DOUBLE_COMPLEX,
}  #: Mapping from torch to MPI datatypes

np_type: Dict[torch.dtype, type] = {
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
}  #: Mapping from torch to numpy datatypes


def init(
    *, comm_override: Optional[qp.MPI.Comm] = None, cores_override: Optional[int] = None
) -> None:
    """Initialize overall hardware resources to be used by QimPy.

    Parameters
    ----------
    comm_override
        If specified, override `qimpy.rc.comm` (defaults to `mpi4py.MPI.COMM_WORLD`).
    cores_override
        If specified, override number of CPU cores (torch threads) to use per process.
        Before `init`, only a single core will be used by each process.
        If `cores_override` is not specified, `init` will set the thread count based
        on environment variable SLURM_CPUS_PER_TASK (set by slurm) if available, and if
        not, it will divide physical cores equally between processes on each node."""

    # Reset and report start time:
    global t_start
    t_start = time.time()
    qp.log.info("Start time: " + time.ctime(t_start))

    # Change MPI communicator if needed:
    if comm_override:
        global comm, i_proc, n_procs, is_head
        comm = comm_override
        i_proc = comm.rank
        n_procs = comm.size
        is_head = i_proc == 0

    # Select GPU before initializing torch:
    comm_node = comm.Split_type(qp.MPI.COMM_TYPE_SHARED)  # on-node communicator
    i_proc_node = comm_node.Get_rank()
    n_procs_node = comm_node.Get_size()
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
    global device, use_cuda, compute_stream
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        use_cuda = True
        torch.cuda.device(device)  # set as defeault CUDA device
        # Enable compute stream based on environment (default on):
        if os.environ.get("QIMPY_COMPUTE_STREAM", "1") in {"1", "yes"}:
            compute_stream = torch.cuda.Stream(device=device)
            qp.log.info("Async compute stream enabled for GPU operations.")
        else:
            qp.log.info("Async compute stream disabled for GPU operations.")
    else:
        n_gpus = 0.0

    # Threads:
    # --- First priority: override argument
    n_threads = cores_override if cores_override else 0
    # --- Second priority: SLURM environment
    if not n_threads:
        slurm_threads = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_threads:
            n_threads = int(slurm_threads)  #: number of threads to use on each process
    # --- Lowest priority: physical core division
    if not n_threads:
        # Divide up threads available on node:
        n_cores = cpu_count(logical=False)
        core_start = (i_proc_node * n_cores) // n_procs_node
        core_stop = ((i_proc_node + 1) * n_cores) // n_procs_node
        n_threads = core_stop - core_start
    assert n_threads >= 1
    torch.set_num_threads(n_threads)

    # Report total resources:
    run_totals = np.array([n_threads, n_gpus])
    comm.Allreduce(qp.MPI.IN_PLACE, run_totals, op=qp.MPI.SUM)
    qp.log.info(
        f"Run totals: {n_procs:g} processes, "
        f"{int(run_totals[0])} threads, {int(run_totals[1])} GPUs"
    )


def compute_stream_wait_current():
    """Make `compute_stream` (if used) wait on current stream."""
    if compute_stream is not None:
        compute_stream.wait_stream(torch.cuda.current_stream())


def current_stream_wait_compute():
    """Make current stream wait on `compute_stream` (if used)."""
    if compute_stream is not None:
        torch.cuda.current_stream().wait_stream(compute_stream)


def current_stream_synchronize():
    """Wait for all tasks in current CUDA stream to complete."""
    if use_cuda:
        torch.cuda.current_stream().synchronize()


def clock():
    """Time in seconds since start of this run."""
    return time.time() - t_start


def report_end():
    """Report end time and duration."""
    t_stop = time.time()
    duration = datetime.timedelta(seconds=(t_stop - t_start))
    qp.log.info(f"\nEnd time: {time.ctime(t_stop)} (Duration: {duration})")
