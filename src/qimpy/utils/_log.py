import qimpy as qp
import numpy as np
import logging
import torch
import sys
from typing import Optional


def log_config(
    *,
    output_file: Optional[str] = None,
    mpi_log: Optional[str] = None,
    mpi_comm: Optional[qp.MPI.Comm] = None,
    append: bool = True,
    verbose: bool = False
):
    """Configure logging globally for the qimpy library. It should typically
    only be necessary to call this once during start-up. Note that the default
    log configuration before calling this function is to print only warnings
    and errors will from all processes to stdout.

    For further customization, directly modify the :class:`logging.Logger`
    object :attr:`~qimpy.log`, as required.

    Parameters
    ----------
    output_file
        Output file to write from MPI rank 0.
        Default = None implies log to stdout.
    mpi_log
        Higher-rank MPI processes will log to <mpi_log>.<process> if given.
        Default = None implies log only from head (rank=0) process.
    mpi_comm
        MPI communicator whose rank determines logging behavior.
        Default = None implies use COMM_WORLD.
    append
        Whether log files should be appended or overwritten.
    verbose
        Whether to log debug information including module/line numbers of code.
    """

    # Create handler with appropriate output file and mode, if any:
    i_proc = (mpi_comm if mpi_comm else qp.MPI.COMM_WORLD).Get_rank()
    is_head = i_proc == 0
    filemode = "a" if append else "w"
    filename = None
    if is_head and output_file:
        filename = output_file
    if (not is_head) and mpi_log:
        filename = mpi_log + "." + str(i_proc)
    handler = (
        logging.FileHandler(filename, mode=filemode)
        if filename
        else logging.StreamHandler(sys.stdout)
    )

    # Set log format:
    handler.setFormatter(
        logging.Formatter(
            ("[%(module)s:%(lineno)d] " if verbose else "") + "%(message)s"
        )
    )

    # Set handler:
    qp.log.handlers.clear()
    qp.log.addHandler(handler)

    # Select log level:
    if is_head or ((not is_head) and mpi_log):
        qp.log.setLevel(logging.DEBUG if verbose else logging.INFO)
    else:
        qp.log.setLevel(logging.WARNING)


def fmt(tensor: torch.Tensor, **kwargs) -> str:
    """Standardized conversion of torch tensors for logging.
    Keyword arguments are forwarded to `numpy.array2string`."""
    # Set some defaults in formatter:
    kwargs.setdefault("precision", 8)
    kwargs.setdefault("suppress_small", True)
    kwargs.setdefault("separator", ", ")
    return np.array2string(tensor.detach().to(qp.rc.cpu).numpy(), **kwargs)
