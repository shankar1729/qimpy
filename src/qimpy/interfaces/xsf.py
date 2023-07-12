from typing import Optional, TextIO
import argparse

import h5py
import numpy as np

from qimpy.io import Unit


def print_header(
    f: TextIO, animated: bool = False, animsteps: Optional[int] = None
) -> None:
    if animated:
        f.write(f"ANIMSTEPS {animsteps}\n")
    f.write("CRYSTAL\n")


def print_lattice_vecs(f: TextIO, lattice_vecs: np.ndarray, n: str) -> None:
    f.write(f"PRIMVEC {n}\n")
    for vec in lattice_vecs.T:
        f.write(f"{vec[0]:10.6f} {vec[1]:10.6f} {vec[2]:10.6f}\n")


def print_positions(
    f: TextIO, symbols: np.ndarray, positions: np.ndarray, n: str
) -> None:
    f.write(f"PRIMCOORD {n}\n")
    f.write(f"  {len(positions)} 1\n")
    for i, pos in enumerate(positions):
        f.write(f"  {symbols[i]} {pos[0]:10.6f} {pos[1]:10.6f} {pos[2]:10.6f}\n")


def print_dataset(
    f: TextIO, lattice_vecs: np.ndarray, dataset_symbol: str, dataset: np.ndarray
) -> None:
    f.write("BEGIN_BLOCK_DATAGRID_3D\n")
    f.write(f" {dataset_symbol}\n")
    f.write(f" BEGIN_DATAGRID_3D_{dataset_symbol}\n")
    f.write(f"  {dataset.shape[0]} {dataset.shape[1]} {dataset.shape[2]}\n")
    f.write("    0.000000   0.000000   0.000000\n")
    for vec in lattice_vecs.T:
        f.write(f"  {vec[0]:10.6f} {vec[1]:10.6f} {vec[2]:10.6f}\n")

    for k in range(dataset.shape[2]):
        for j in range(dataset.shape[1]):
            for i in range(dataset.shape[0]):
                f.write(f" {dataset[i, j, k]:e}")
            f.write("\n")

    f.write(" END_DATAGRID_3D\n")
    f.write("END_BLOCK_DATAGRID_3D\n")


def write_xsf(
    checkpoint: str,
    xsf_file: str,
    animated: bool = False,
    dataset_symbol: Optional[str] = None,
) -> None:
    """Write XSF file from HDF5 checkpoint file.

    Parameters
    ----------
    checkpoint
        Checkpoint file in HDF5 format.
    xsf_file
        Output file in XSF format.
    animated
        Make output an animated XSF file.
    dataset_symbol
        Add 3d data to XSF file such as electron density (dataset_symbol=n).

    Usage
    -----
    :code:`python -m qimpy.interfaces.xsf [-h] -c FILE [-x FILE] [-a] [-d SYMBOL]`

    Command-line parameters (obtained using :code:`python -m qimpy.interfaces.xsf -h`):

    .. code-block:: bash

        python -m qimpy.interfaces.xsf [-h] -c FILE [-x FILE] [-a] [-d SYMBOL]

    write XSF file from HDF5 checkpoint file

    options:
      -h, --help            show this help message and exit
      -c FILE, --checkpoint-file FILE
                            checkpoint file in HDF5 format
      -x FILE, --xsf-file FILE
                            output file in XSF format (crystal.xsf if unspecified)
      -a, --animated        make output an animated XSF file
      -d SYMBOL, --data-symbol SYMBOL
                            add 3d data to XSF file such as electron density (SYMBOL=n)"""
    h5_file = h5py.File(checkpoint, "r")
    ions = h5_file["ions"]
    types = ions["types"][:]
    symbols = np.repeat(
        np.array(ions.attrs["symbols"].split(",")),
        np.unique(types, return_counts=True)[1],
    )
    to_ang = Unit.convert(1, "Angstrom").value
    lattice = h5_file["lattice"]
    history = h5_file["geometry/action/history"]

    with open(xsf_file, "w") as f:
        if animated:
            fractional_positions = history["positions"][:]
            animsteps = fractional_positions.shape[0]
            print_header(f, animated, animsteps)

            if lattice.attrs["movable"]:
                lattice_vecs = history["Rbasis"][:] * to_ang
                positions = np.einsum(
                    "ijk,ilk->ilj", lattice_vecs, fractional_positions
                )
                for n, vec_n, pos_n in zip(range(animsteps), lattice_vecs, positions):
                    print_lattice_vecs(f, vec_n, f"{n+1}")
                    print_positions(f, symbols, pos_n, f"{n+1}")

            else:
                lattice_vecs = lattice["Rbasis"][:] * to_ang
                positions = np.einsum("ij,klj->kli", lattice_vecs, fractional_positions)
                print_lattice_vecs(f, lattice_vecs, "")

                for n, pos_n in enumerate(positions):
                    print_positions(f, symbols, pos_n, f"{n+1}")

        else:
            lattice_vecs = lattice["Rbasis"][:] * to_ang
            fractional_positions = ions["positions"][:]
            positions = (lattice_vecs @ fractional_positions.T).T
            print_header(f)
            print_lattice_vecs(f, lattice_vecs, "")
            print_positions(f, symbols, positions, "")

            if dataset_symbol is None:
                return

            dataset = h5_file[f"electrons/{dataset_symbol}"][0]
            print_dataset(f, lattice_vecs, dataset_symbol, dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m qimpy.interfaces.xsf",
        description="write XSF file from HDF5 checkpoint file",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c", "--checkpoint-file", metavar="FILE", help="checkpoint file in HDF5 format"
    )
    parser.add_argument(
        "-x",
        "--xsf-file",
        default="crystal.xsf",
        metavar="FILE",
        help="output file in XSF format (crystal.xsf if unspecified)",
    )
    parser.add_argument(
        "-a", "--animated", action="store_true", help="make output an animated XSF file"
    )
    parser.add_argument(
        "-d",
        "--data-symbol",
        default=None,
        metavar="SYMBOL",
        help="add 3d data to XSF file such as electron density (SYMBOL=n)",
    )

    args = parser.parse_args()
    write_xsf(args.checkpoint_file, args.xsf_file, args.animated, args.data_symbol)


if __name__ == "__main__":
    main()
