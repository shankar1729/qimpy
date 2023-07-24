from typing import Optional, Union
import logging
import sys

import numpy as np
import torch

from qimpy import rc, log, MPI


def log_config(
    *,
    output_file: Optional[str] = None,
    mpi_log: Optional[str] = None,
    mpi_comm: Optional[MPI.Comm] = None,
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
    i_proc = (mpi_comm if mpi_comm else MPI.COMM_WORLD).Get_rank()
    is_head = i_proc == 0
    filemode = "a" if append else "w"
    filename = ""
    if is_head and output_file:
        filename = output_file
    if (not is_head) and mpi_log:
        filename = mpi_log + "." + str(i_proc)
    handler = get_handler(filename, filemode)

    # Set log format:
    handler.setFormatter(
        logging.Formatter(
            ("[%(module)s:%(lineno)d] " if verbose else "") + "%(message)s"
        )
    )

    # Set handler:
    log.handlers.clear()
    log.addHandler(handler)

    # Select log level:
    if is_head or ((not is_head) and mpi_log):
        log.setLevel(logging.DEBUG if verbose else logging.INFO)
    else:
        log.setLevel(logging.WARNING)


def fmt(tensor: Union[torch.Tensor, np.ndarray], **kwargs) -> str:
    """Standardized conversion of torch tensors and numpy arrays for logging.
    Keyword arguments are forwarded to `numpy.array2string`."""
    # Set some defaults in formatter:
    kwargs.setdefault("precision", 8)
    kwargs.setdefault("suppress_small", True)
    kwargs.setdefault("separator", ", ")
    return np.array2string(
        tensor.detach().to(rc.cpu).numpy()
        if isinstance(tensor, torch.Tensor)
        else tensor,
        **kwargs
    )


def get_handler(filename: str, filemode: str) -> logging.Handler:
    if filename:
        return logging.FileHandler(filename, mode=filemode)
    else:
        return logging.StreamHandler(sys.stdout)
