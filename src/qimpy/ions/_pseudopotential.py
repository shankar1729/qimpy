import qimpy as qp
from ._read_upf import _read_upf


class Pseudopotential:
    """TODO: document class Pseudopotential"""

    read_upf = _read_upf

    def __init__(self, filename, rc):
        """
        Parameters
        ----------
        filename : str
            File to read pseudopotential from.
            Currently, only norm-conserving UPF files are supported.
        """

        assert(filename[-4:].lower() == '.upf')
        self.read_upf(filename, rc)
