import argparse
import yaml

import numpy as np
from ase.io import read

from qimpy.io import Unit


def _parse_cif(cif_file) -> tuple[dict, dict]:

    cif_data = read(cif_file, format="cif")

    latt_vec = cif_data.get_cell()[:] * Unit.MAP["Angstrom"]
    coords = cif_data.get_scaled_positions()
    species = cif_data.get_chemical_symbols()
    pbc = cif_data.get_pbc()

    lattice = {
        "vector1": latt_vec[0, :].tolist(),
        "vector2": latt_vec[1, :].tolist(),
        "vector3": latt_vec[2, :].tolist(),
        "periodic": np.array(pbc).tolist(),
    }  # list(pbc) returns something weird but putting it to an array and back doesn't

    coordinates = [
        [symbol] + position.tolist() for symbol, position in zip(species, coords)
    ]

    ions = {"ions": {"coordinates": coordinates, "fractional": True}}
    lattice = {"lattice": lattice}

    return lattice, ions


def write_yaml(cif_file: str, yaml_file: str) -> None:
    """Write YAML file from CIF file.

    Parameters
    ----------
    cif_file
        CIF file to be converted
    yaml_file
        Output file in YAML format.

    Usage
    -----
    :code:`python -m qimpy.interfaces.cif [-h] -f FILE [-y FILE]`

    Command-line parameters (obtained using :code:`python -m qimpy.interfaces.cif -h`):

    .. code-block:: bash

        python -m qimpy.interfaces.cif [-h] -f FILE [-y FILE]

    write YAML file from CIF file

    options:
      -h, --help            show this help message and exit
      -f FILE, --cif-file FILE
                            checkpoint file in HDF5 format
      -y FILE, --yaml-file FILE
                            output file in XSF format (in.yaml if unspecified)
    """

    lattice, ions = _parse_cif(cif_file)

    with open("in.yaml", "w") as f:
        yaml.dump(lattice, f, default_flow_style=None, allow_unicode=True)
        yaml.dump(ions, f, default_flow_style=None, allow_unicode=True)


def main() -> None:
    command_parser = argparse.ArgumentParser(
        prog="python -m qimpy.interfaces.cif",
        description="Parse CIF file to YAML input",
    )
    group = command_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--cif-file", metavar="FILE", help="CIF file to be parsed")
    command_parser.add_argument(
        "-y",
        "--yaml-file",
        default="in.yaml",
        metavar="FILE",
        help="output file in YAML format (in.yaml if unspecified)",
    )

    args = command_parser.parse_args()
    write_yaml(args.cif_file, args.yaml_file)


if __name__ == "__main__":
    main()
