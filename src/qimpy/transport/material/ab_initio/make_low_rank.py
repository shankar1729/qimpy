import argparse

import qimpy
from qimpy import log, rc, io
from qimpy.mpi import ProcessGrid
from qimpy.profiler import StopWatch
from . import AbInitio


def generate_low_rank(
    *,
    ab_initio: AbInitio | dict,
    low_rank_file: str,
) -> None:
    """Generate low-rank approximation of ab_initio.lindblad.P."""
    process_grid = ProcessGrid("rk", (1, -1))
    ab_initio = AbInitio(**ab_initio, process_grid=process_grid)
    assert not ab_initio.lindblad.low_rank_file

    # TODO: generate low rank approximation and save to `low_rank_file`


def main():
    parser = argparse.ArgumentParser(
        prog="python -m qimpy.transport.material.ab_initio.make_low_rank",
        description="Generate low-rank approximation to lindblad scattering",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="YAML input file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        metavar="FILE",
        help="output file (stdout if unspecified)",
    )

    args = parser.parse_args()
    io.log_config(output_file=args.output_file)

    # Print version header
    log.info("*" * 15 + " QimPy " + qimpy.__version__ + " " + "*" * 15)

    # Configure hardware resources
    rc.init()

    # Load input parameters from YAML file:
    input_dict = io.dict.key_cleanup(io.yaml.load(args.input_file))
    log.info(f"\n# Processed input:\n{io.yaml.dump(input_dict)}")
    input_dict = io.dict.remove_units(input_dict)  # Remove units

    # Generate low-rank approximations:
    generate_low_rank(**input_dict)

    # Cleanup and report timings:
    rc.free()
    rc.report_end()
    StopWatch.print_stats()


if __name__ == "__main__":
    main()
