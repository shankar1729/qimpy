from ase.calculators.calculator import Calculator

from qimpy import rc, dft
from qimpy.utils import Unit


class QimPy(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(self, pseudopotentials: str, **kwargs) -> None:
        """Qimpy calculator for ASE

        restart: str
            Prefix for restart file.  May contain a directory. Default
            is None: don't restart.
        ignore_bad_restart_file: bool
            Deprecated, please do not use.
            Passing more than one positional argument to Calculator()
            is deprecated and will stop working in the future.
            Ignore broken or missing restart file.  By default, it is an
            error if the restart file is missing or broken.
        directory: str or PurePath
            Working directory in which to read and write files and
            perform calculations.
        label: str
            Name used for all files.  Not supported by all calculators.
            May contain a directory, but please use the directory parameter
            for that instead.
        atoms: Atoms object
            Optional Atoms object to which the calculator will be
            attached.  When restarting, atoms will get its positions and
            unit-cell updated from file.
        """
        Calculator.__init__(self, **kwargs)
        self.pseudopotentials = pseudopotentials

    def calculate(self, atoms=None, properties=["energy"], system_changes=[]) -> None:
        # Necessary units (to match ASE)
        angstrom = Unit.MAP["Angstrom"]
        eV = Unit.MAP["eV"]

        # Obtain lattice parameters and structure

        # Get lattice vectors (3x3 array):
        lattice = atoms.get_cell()[:] * angstrom

        # Get atomic positions
        positions = atoms.get_scaled_positions()

        # Get symbols
        symbols = atoms.get_chemical_symbols()

        lattice_dict = {
            "vector1": lattice[0].tolist(),
            "vector2": lattice[1].tolist(),
            "vector3": lattice[2].tolist(),
            "movable": False,
        }

        # Horrible hardcode but default pseudopotentials need to be specified...
        coordinates = [
            [symbol] + position.tolist() for symbol, position in zip(symbols, positions)
        ]
        ions = {
            "coordinates": coordinates,
            "fractional": True,
            "pseudopotentials": self.pseudopotentials,
        }

        rc.init()
        system = dft.System(lattice=lattice_dict, ions=ions)
        system.run()

        self.results = {"energy": float(system.energy) * eV}
