import qimpy as qp
import logging
import sys


def log_config(*, output_file=None, mpi_log=None, mpi_comm=None,
               append=True, verbose=False):
    '''Configure logging globally for the qimpy library. It should typically
    only be necessary to call this once during start-up. Note that the default
    log configuration before calling this function is to print only warnings
    and errors will from all processes to stdout.

    For further customization, directly modify the :class:`logging.Logger`
    object :attr:`~qimpy.log`, as required.

    Parameters
    ----------
    output_file : str or None, optional
        Output file to write from MPI rank 0.
        Default = None implies log to stdout.
    mpi_log : str or None, optional
        Higher-rank MPI processes will log to <mpi_log>.<process> if given.
        Default = None implies log only from head (rank=0) process.
    mpi_comm : mpi4py.MPI.Intracomm or None, optional
        MPI communicator whose rank determines logging behavior.
        Default = None implies use COMM_WORLD.
    append : bool, optional
        Whether log files should be appended or overwritten. Default: True.
    verbose : bool, optional
        Whether to log debug information including module/line numbers of code.
        Default: False.
    '''

    # Create handler with appropriate output file and mode, if any:
    i_proc = (mpi_comm if mpi_comm else qp.MPI.COMM_WORLD).Get_rank()
    is_head = (i_proc == 0)
    filemode = ('a' if append else 'w')
    filename = None
    if is_head and output_file:
        filename = output_file
    if (not is_head) and mpi_log:
        filename = mpi_log + '.' + str(i_proc)
    handler = (logging.FileHandler(filename, mode=filemode) if filename
               else logging.StreamHandler(sys.stdout))

    # Set log format:
    handler.setFormatter(logging.Formatter(
        ('[%(module)s:%(lineno)d] ' if verbose else '') + '%(message)s'))

    # Set handler:
    qp.log.handlers.clear()
    qp.log.addHandler(handler)

    # Select log level:
    if(is_head or ((not is_head) and mpi_log)):
        qp.log.setLevel(logging.DEBUG if verbose else logging.INFO)
    else:
        qp.log.setLevel(logging.WARNING)
