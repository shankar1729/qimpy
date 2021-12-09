"""Stand-alone QimPy calculation from YAML input file

Typical usage:

:code:`mpirun [mpi-options] python -m qimpy.run -i INPUT_FILE [qimpy-options]`

Command-line parameters (obtained using :code:`python -m qimpy.run -h`):

.. code-block:: bash

    python -m qimpy.run (-h | -v | -i FILE) [-o FILE] [-c C] [-p Pr Pk Pb]
                           [-n] [-d] [-m FILE] [-V]

Run a QimPy calculation from an input file

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         print version information and quit
  -i FILE, --input-file FILE
                        input file in YAML format
  -o FILE, --output-file FILE
                        output file (stdout if unspecified)
  -c C, --cores C       number of cores per process (overridden by SLURM)

  -p Pr Pk Pb, --process-grid Pr Pk Pb
                        dimensions of process grid: replicas x kpoints x
                        bands/basis, whose product must match process count;
                        any -1 will be set to distribute available tasks for
                        that dimension most equally. Default: -1 -1 -1 implies
                        all dimensions set automatically.

  -n, --dry-run         quit after initialization (to check input file)
  -d, --no-append       overwrite output file instead of appending
  -m FILE, --mpi-log FILE
                        file prefix for debug logs from other MPI processes
  -V, --verbose         print extra information in log for debugging


Note that qimpy must be installed to the python path for these to work in any
directory. For development, run `python setup.py develop --user` in the root
directory of the source repository to make the above usage possible without
instaling from pip/conda."""

import qimpy as qp
import argparse
import sys
import os

if __name__ == "__main__":

    # Parse the commandline arguments on main process:
    i_proc = qp.MPI.COMM_WORLD.Get_rank()
    if i_proc == 0:

        # Set terminal size (used by argparse) if unreasonable:
        columns_min = 80
        try:
            if os.get_terminal_size().columns < columns_min:
                os.environ["COLUMNS"] = str(columns_min)
        except OSError:
            os.environ["COLUMNS"] = str(columns_min)

        # Modify ArgumentParser to not exit:
        class ArgumentParser(argparse.ArgumentParser):
            def error(self, message):
                self.print_usage(sys.stderr)
                print(f"{self.prog}: error: {message}\n", file=sys.stderr)
                raise ValueError(message)  # Quit after bcast'ing error

        parser = ArgumentParser(
            add_help=False,
            prog="python -m qimpy.run",
            description="Run a QimPy calculation from an input file",
        )
        # --- mutually-exclusive group of help, version or input file
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-h", "--help", action="store_true", help="show this help message and exit"
        )
        group.add_argument(
            "-v",
            "--version",
            action="store_true",
            help="print version information and quit",
        )
        group.add_argument(
            "-i", "--input-file", metavar="FILE", help="input file in YAML format"
        )
        # ---
        parser.add_argument(
            "-o",
            "--output-file",
            metavar="FILE",
            help="output file (stdout if unspecified)",
        )
        parser.add_argument(
            "-c",
            "--cores",
            type=int,
            metavar="C",
            help="number of cores per process (overridden by SLURM)",
        )
        parser.add_argument(
            "-p",
            "--process-grid",
            type=int,
            nargs=3,
            default=[-1, -1, -1],
            metavar=("Pr", "Pk", "Pb"),
            help="dimensions of process grid: replicas x kpoints x bands/basis"
            ", whose product must match process count; any -1 will be set to "
            "distribute available tasks for that dimension most equally. "
            "Default: -1 -1 -1 implies all dimensions set automatically.",
        )
        parser.add_argument(
            "-n",
            "--dry-run",
            action="store_true",
            help="quit after initialization (to check input file)",
        )
        parser.add_argument(
            "-d",
            "--no-append",
            action="store_true",
            help="overwrite output file instead of appending",
        )
        parser.add_argument(
            "-m",
            "--mpi-log",
            metavar="FILE",
            help="file prefix for debug logs from other MPI processes",
        )
        parser.add_argument(
            "-V",
            "--verbose",
            action="store_true",
            help="print extra information in log for debugging",
        )
        try:
            args = parser.parse_args()
            setattr(args, "error_occured", False)
        except ValueError:
            args = argparse.Namespace()
            setattr(args, "error_occured", True)
    else:
        args = argparse.Namespace()

    # Make commandline arguments available on all processes:
    args = qp.MPI.COMM_WORLD.bcast(args, root=0)
    if args.error_occured:
        exit()  # exit all processes

    if args.version:
        # Print version and exit:
        if i_proc == 0:
            print("QimPy", qp.__version__)
        exit()

    if args.help:
        # Print help and exit:
        if i_proc == 0:
            parser.print_help()
        exit()

    # Setup logging:
    qp.utils.log_config(
        output_file=args.output_file,
        mpi_log=args.mpi_log,
        mpi_comm=qp.MPI.COMM_WORLD,
        append=(not args.no_append),
        verbose=args.verbose,
    )

    # Print version header
    qp.log.info("*" * 15 + " QimPy " + qp.__version__ + " " + "*" * 15)

    # Set up run configuration
    rc = qp.utils.RunConfig(cores=args.cores, process_grid=args.process_grid)

    # Load input parameters from YAML file:
    input_dict = qp.utils.dict.key_cleanup(qp.utils.yaml.load(args.input_file))
    # --- Set default checkpoint file (if not specified in input):
    input_dict.setdefault("checkpoint", os.path.splitext(args.input_file)[0] + ".h5")
    # --- Include processed input in log:
    qp.log.info(f"\n# Processed input:\n{qp.utils.yaml.dump(input_dict)}")

    # Initialize system with input parameters:
    system = qp.System(rc=rc, **input_dict)

    # Dry-run bypass:
    if args.dry_run:
        qp.log.info("Dry run initialization successful: input is valid.")
        rc.report_end()
        qp.utils.StopWatch.print_stats()
        exit()

    # Perform specified actions:
    system.run()

    # Report timings:
    rc.report_end()
    qp.utils.StopWatch.print_stats()
