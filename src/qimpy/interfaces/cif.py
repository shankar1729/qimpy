import argparse
import yaml

import numpy as np
from pymatgen.io.cif import CifParser

from qimpy.io import Unit


def _parse_cif(cif_file) -> (dict, dict):

    cif_parser = CifParser(cif_file)
    structure = cif_parser.parse_structures(primitive=True)[0]
    structure.remove_oxidation_states()  # remove +/- from atom names

    coords = structure.frac_coords
    species = np.array(structure.species, dtype=str).tolist()
    latt_vec = np.array(structure.lattice.as_dict()["matrix"]).T * Unit.MAP["Angstrom"]

    lattice = {
        "vector1": latt_vec[0, :].tolist(),
        "vector2": latt_vec[1, :].tolist(),
        "vector3": latt_vec[2, :].tolist(),
        "periodic": list(structure.pbc),
    }

    coordinates = [
        [symbol] + position.tolist() for symbol, position in zip(species, coords)
    ]

    ions = {"ions": {"coordinates": coordinates}}
    lattice = {"lattice": lattice}

    return lattice, ions


def write_yaml(cif_file, yaml_file) -> None:
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
