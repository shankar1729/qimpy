from ase.calculators.calculator import Calculator
import qimpy as qp
from qimpy.utils import Unit


class ASECalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(self, **kwargs):
        """Qimpy ASE Calculator

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

    def calculate(self, atoms=None, properties=["energy"], system_changes=[]):
        angstrom = Unit.MAP["Angstrom"]
        # Obtain lattice parameters and structure

        # Get lattice vectors (3x3 array):
        lattice = atoms.get_cell()[:] * angstrom

        # Get atomic positions
        positions = atoms.get_scaled_positions()

        # Get symbols
        symbols = atoms.get_chemical_symbols()

        input_dict = dict()

        input_dict["lattice"] = {
            "vector1": lattice[0],
            "vector2": lattice[1],
            "vector3": lattice[2],
            "movable": False,
        }

        # Horrible hardcode but default pseudopotentials need to be specified...
        input_dict["ions"] = {
            "coordinates": list(),
            "fractional": True,
            "pseudopotentials": "../../../jdftx/build/pseudopotentials/SG15/$ID_ONCV_PBE.upf",
        }

        for i in range(len(symbols)):
            input_dict["ions"]["coordinates"].append(
                [symbols[i]] + positions[i].tolist()
            )

        qp.rc.init()
        system = qp.System(**input_dict)
        system.run()

        self.results = {"energy": float(system.energy)}
