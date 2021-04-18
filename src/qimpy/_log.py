import qimpy as qp
import logging
import sys


log = logging.getLogger('qimpy')


def log_config(*, output_file=None, mpi_log=None, mpi_comm=None,
               append=True, verbose=False):
    '''Configure logging globally for the qimpy library. It should typically
    only be necessary to call this once during start-up. Note that the default
    log configuration before calling this function is to print only warnings
    and errors will from all processes to stdout.

    For further customization, directly modify the :class:`logging.Logger`
    object qimpy.log, as required.

    :param output_file: Output file to write from MPI rank 0,
        defaults to None i.e. log to stdout.
    :type output_file: str, optional
    '''

    # Create handler with appropriate output file and mode, if any:
    iProc = (mpi_comm if mpi_comm else qp.MPI.COMM_WORLD).Get_rank()
    isHead = (iProc == 0)
    filemode = ('a' if append else 'w')
    filename = None
    if isHead and output_file:
        filename = output_file
    if (not isHead) and mpi_log:
        filename = mpi_log + '.' + str(iProc)
    handler = (logging.FileHandler(filename, mode=filemode) if filename
               else logging.StreamHandler(sys.stdout))

    # Set log format:
    handler.setFormatter(logging.Formatter(
        ('[%(module)s:%(lineno)d] ' if verbose else '') + '%(message)s'))

    # Set handler:
    log.handlers.clear()
    log.addHandler(handler)

    # Select log level:
    if(isHead or ((not isHead) and mpi_log)):
        log.setLevel(logging.DEBUG if verbose else logging.INFO)
    else:
        log.setLevel(logging.WARNING)
