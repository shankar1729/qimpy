'''This module serves as the main executable for stand-alone QimPy calculations
from input files in the YAML format. A typical usage would look like:

:code:`mpirun [mpi-options] python -m qimpy.run -i INPUT_FILE [qimpy-options]`

Command-line parameters (obtained using :code:`python -m qimpy.run -h`):

.. code-block:: bash

    python -m qimpy.run (-h | -v | -i INPUT_FILE) [-o OUTPUT_FILE]
                        [-c CORES] [-n] [-d] [-m MPI_LOG] [-V]

Run a QimPy calculation from an input file

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         print version information and quit
  -i INPUT_FILE, --input-file INPUT_FILE
                        input file in YAML format
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        output file (stdout if unspecified)
  -c CORES, --cores CORES
                        number of cores per process (overridden by SLURM)
  -n, --dry-run         quit after initialization (to check input file)
  -d, --no-append       overwrite output file instead of appending
  -m MPI_LOG, --mpi-log MPI_LOG
                        file prefix for debug logs from other MPI processes
  -V, --verbose         print extra information in log for debugging


Note that qimpy must be installed to the python path for these to work in any
directory. For development, run `python setup.py develop --user` in the root
directory of the source repository to make the above usage possible without
instaling from pip/conda.'''

import qimpy as qp
import argparse
import logging
import yaml
import sys
import os

if __name__ == "__main__":

    # Parse the commandline arguments on main process:
    iProc = qp.MPI.COMM_WORLD.Get_rank()
    if iProc == 0:

        # Set terminal size (used by argparse) if unreasonable:
        if os.get_terminal_size().columns < 80:
            os.environ["COLUMNS"] = "80"

        # Modify ArgumentParser to not exit:
        class ArgumentParser(argparse.ArgumentParser):
            def error(self, message):
                self.print_usage(sys.stderr)
                print('{:s}: error: {:s}\n'.format(self.prog, message),
                      file=sys.stderr)
                raise ValueError(message)  # Quit after bcast'ing error
        parser = ArgumentParser(
            add_help=False,
            prog='python -m qimpy.run',
            description='Run a QimPy calculation from an input file')
        # --- mutually-exclusive group of help, version or input file
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            '-h', '--help', action='store_true',
            help='show this help message and exit')
        group.add_argument(
            '-v', '--version', action='store_true',
            help='print version information and quit')
        group.add_argument(
            '-i', '--input-file',
            help='input file in YAML format')
        # ---
        parser.add_argument(
            '-o', '--output-file',
            help='output file (stdout if unspecified)')
        parser.add_argument(
            '-c', '--cores', type=int,
            help='number of cores per process (overridden by SLURM)')
        parser.add_argument(
            '-n', '--dry-run', action='store_true',
            help='quit after initialization (to check input file)')
        parser.add_argument(
            '-d', '--no-append', action='store_true',
            help='overwrite output file instead of appending')
        parser.add_argument(
            '-m', '--mpi-log',
            help='file prefix for debug logs from other MPI processes')
        parser.add_argument(
            '-V', '--verbose', action='store_true',
            help='print extra information in log for debugging')
        try:
            args = parser.parse_args()
            setattr(args, "error_occured", False)
        except ValueError:
            args = argparse.Namespace()
            setattr(args, "error_occured", True)
    else:
        args = None

    # Make commandline arguments available on all processes:
    args = qp.MPI.COMM_WORLD.bcast(args, root=0)
    if args.error_occured:
        exit()  # exit all processes

    if args.version:
        # Print version and exit:
        if iProc == 0:
            print('QimPy', qp.__version__)
        exit()

    if args.help:
        # Print help and exit:
        if iProc == 0:
            parser.print_help()
        exit()

    # Setup logging:
    qp.utils.log_config(
        output_file=args.output_file,
        mpi_log=args.mpi_log,
        mpi_comm=qp.MPI.COMM_WORLD,
        append=(not args.no_append),
        verbose=args.verbose)

    # Print version header
    qp.log.info('*'*15 + ' QimPy ' + qp.__version__ + ' ' + '*'*15)

    # Set up run configuration
    rc = qp.utils.RunConfig(cores=args.cores)

    # Load input parameters from YAML file:
    with open(args.input_file) as f:
        inputDict = yaml.safe_load(f)

    # Initialize system with input parameters:
    system = qp.System(rc=rc, **inputDict)

    # Report timings:
    rc.report_end()
    qp.utils.StopWatch.print_stats()
