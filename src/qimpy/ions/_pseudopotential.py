import qimpy as qp
from ._read_upf import read_upf


class Pseudopotential:
    """TODO: document class Pseudopotential"""

    def __init__(self, filename, rc):
        """Create pseudopotential from file.

        Parameters
        ----------
        filename : str
            File to read pseudopotential from.
            Currently, only norm-conserving UPF files are supported.
        """

        assert(filename[-4:].lower() == '.upf')
        read_upf(self, filename, rc)
