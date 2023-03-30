from ase import Atoms
from ase.calculators.calculator import Calculator

class ASECalculator(Calculator):

    implemented_properties = ['energy', 'forces']

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

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=[]):
        ...
