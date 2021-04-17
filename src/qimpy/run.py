import qimpy as qp
import yaml


if __name__ == "__main__":

    # Parse the commandline arguments:
    import argparse
    parser = argparse.ArgumentParser(
        prog='python -m qimpy.run',
        description='Run a QimPy calculation from an input file')
    # --- mutually-exclusive group of either version or input file
    group = parser.add_mutually_exclusive_group(required=True)
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
    args = parser.parse_args()

    if args.version:
        # Print version and exit:
        print('QimPy', qp.__version__)
        exit()

    # Version header
    print('*'*15, 'QimPy', qp.__version__, '*'*15)

    # Load input parameters from YAML file:
    with open(args.input_file) as f:
        inputDict = yaml.safe_load(f)

    # Initialize system with input parameters:
    system = qp.System(**inputDict)

