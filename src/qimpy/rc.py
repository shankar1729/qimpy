"""Run configuration / hardware resources. This includes CPU cores or GPU, and MPI
communicators to be used by the current QimPy instance. The import-time configuration
selects a single CPU core for each MPI process in `mpi4py.MPI.COMM_WORLD`.

Call `init` to select the number of cores or a GPU device, as available and based
on environment variables such as SLURM_CPUS_PER_TASK, as well as to initialize
`torch.distributed` communication controlled by BACKEND and its standard environment
variables MASTER_ADDR and MASTER_PORT. None of these environment variables are required;
`init` uses MPI to determine a consistent address/port for the backend set-up.
"""

from typing import NamedTuple
from itertools import groupby
from operator import itemgetter
import socket
import datetime
import time
import os

import torch
import numpy as np
from psutil import cpu_count
import torch.distributed as dist
from torch.distributed.elastic.utils.distributed import get_free_port

from . import log, MPI

# List exported symbols for doc generation
__all__ = (
    "comm",
    "i_proc",
    "n_procs",
    "is_head",
    "cpu",
    "device",
    "use_accelerator",
    "init",
    "free",
    "clock",
    "report_end",
)

comm: MPI.Comm = MPI.COMM_WORLD  #: Global communicator for QimPy
i_proc: int = comm.rank  #: Rank within `comm`
n_procs: int = comm.size  #: Size of `comm`
is_head: bool = i_proc == 0  #: Whether head of `comm`
cpu: torch.device = torch.device("cpu")  #: CPU torch device
device: torch.device = cpu  #: Preferred torch device for calculation (CPU / GPU)
use_accelerator: bool = False  #: Whether `device` is an accelerator (GPU-like)
t_start: float = time.time()  #: Start time used for `clock` (set by `init`)

# Set reasonable pre-init defaults for torch:
torch.set_default_dtype(torch.double)
torch.set_num_threads(1)  # to prevent overcommit between MPI processes

# Declare type mappings from torch to numpy:
np_type: dict[torch.dtype, type] = {
    torch.bool: np.bool_,
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
    *, comm_override: MPI.Comm | None = None, cores_override: int | None = None
) -> None:
    """Initialize overall hardware resources to be used by QimPy.
    Initializes GPU resources

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
    log.info("Start time: " + time.ctime(t_start))

    # Change MPI communicator if needed:
    if comm_override:
        global comm, i_proc, n_procs, is_head
        comm = comm_override
        i_proc = comm.rank
        n_procs = comm.size
        is_head = i_proc == 0

    # Determine nodes and process distribution:
    comm_node = comm.Split_type(MPI.COMM_TYPE_SHARED)  # on-node communicator
    i_proc_node = comm_node.Get_rank()
    n_procs_node = comm_node.Get_size()
    # --- collect processes running on each host at head of comm_node
    is_node_head = i_proc_node == 0
    node_proc_list = comm_node.gather(i_proc)
    # --- collect above and hostname across heads of each node
    comm_node_inter = comm.Split(i_proc_node)  # inter-node communicator
    host_proc_lists: list[HostProcessList] = []
    if is_node_head:
        host_proc_lists = comm_node_inter.allgather(
            HostProcessList(socket.gethostname(), node_proc_list)
        )
    # --- distribute to all processes and report
    host_proc_lists = comm_node.bcast(host_proc_lists)
    host_proc_str = " ".join(str(host_proc_list) for host_proc_list in host_proc_lists)
    log.info(f"Hosts(processes): {host_proc_str}")

    # Initialize torch:
    gpu_id = -1
    global device, use_accelerator
    if torch.accelerator.is_available():
        # Select GPU based on local rank:
        gpu_id = i_proc_node % torch.accelerator.device_count()
        torch.accelerator.set_device_index(gpu_id)
        device = torch.device(gpu_id)
        use_accelerator = True
    # --- count unique GPUs on node using IDs (average over processes on same node)
    gpu_ids_mine = np.array([gpu_id], dtype=int)
    gpu_ids_local = np.zeros(n_procs_node, dtype=int)
    comm_node.Allgather(gpu_ids_mine, gpu_ids_local)
    n_gpus = np.count_nonzero(np.unique(gpu_ids_local) >= 0) / n_procs_node

    # Initialize torch distributed:
    backend = os.environ.get("BACKEND", dist.get_default_backend_for_device(device))
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = host_proc_lists[0].hostname
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(comm.bcast(get_free_port() if is_head else 0))
    os.environ["LOCAL_RANK"] = str(i_proc_node)
    os.environ["RANK"] = str(i_proc)
    os.environ["WORLD_SIZE"] = str(n_procs)
    dist.init_process_group(
        backend=backend, device_id=(device if use_accelerator else None)
    )
    dist.barrier()  # Force lazy backend intialization to complete

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
    comm.Allreduce(MPI.IN_PLACE, run_totals, op=MPI.SUM)
    n_threads_tot, n_gpus_tot = run_totals.astype(int)
    log.info(
        f"Run totals: {n_procs} processes, {n_threads_tot} threads, {n_gpus_tot} GPUs"
    )


def free():
    """Cleanup any resources initialized in `init`."""
    dist.destroy_process_group()


def clock():
    """Time in seconds since start of this run."""
    return time.time() - t_start


def report_end():
    """Report end time and duration."""
    t_stop = time.time()
    duration = datetime.timedelta(seconds=(t_stop - t_start))
    log.info(f"\nEnd time: {time.ctime(t_stop)} (Duration: {duration})")


class HostProcessList(NamedTuple):
    """List of processes running on each hostname"""

    hostname: str
    process_list: list[int]

    def __str__(self) -> str:
        """Format as, e.g., `hostname(0,3-5,8-10)`."""
        proc_ranges = []
        for _, index_proc_pair in groupby(
            enumerate(self.process_list), lambda i_pair: i_pair[0] - i_pair[1]
        ):
            procs = list(map(itemgetter(1), index_proc_pair))
            if len(procs) == 1:
                proc_ranges.append(str(procs[0]))
            else:
                proc_ranges.append(f"{procs[0]}-{procs[-1]}")
        return f"{self.hostname}({','.join(proc_ranges)})"
