import qimpy as qp
import numpy as np
from ._read_upf import _read_upf


class Pseudopotential:
    """Specification of electron-ion interactions. Contains radial functions
    for local potential, nonlocal projectors and atomic orbitals.
    Currently supports norm-conserving pseudopotentials."""

    element: str  #: Chemical symbol of element
    atomic_number: int  #: Atomic number of element
    is_paw: bool  #: Whether this is a PAW pseudopotential
    Z: float  #: Pseudo-atomic number i.e. number of valence electrons / atom
    l_max: int  #: Maximum angular momentum quantum number
    r_grid: np.ndarray  #: Radial grid
    dr_grid: np.ndarray  #: Radial grid spacings / integration weights
    rho_atom: np.ndarray  #: Atomic reference electron density radial function
    n_core: np.ndarray  #: Partial core density radial function
    Vloc: np.ndarray  #: Local potential radial function
    beta: np.ndarray  #: Nonlocal-projector radial functions
    l_beta: np.ndarray  #: Orbital angular momentum of each projector
    j_beta: np.ndarray  #: Total angular momentum of each projector
    psi: np.ndarray  #: Atomic-orbital radial functions
    l_psi: np.ndarray  #: Orbital angular momentum of each atomic orbital
    j_psi: np.ndarray  #: Total angular momentum of each atomic orbital
    eig_psi: np.ndarray  #: Energy eigenvalue of each atomic orbital
    D: np.ndarray  #: Descreened nonlocal pseudopotential matrix

    # Methods defined out of class:
    read_upf = _read_upf

    def __init__(self, filename, rc):
        """Read pseudopotential from file.

        Parameters
        ----------
        filename : str
            File to read pseudopotential from.
            Currently, only norm-conserving UPF files are supported.
        """

        assert(filename[-4:].lower() == '.upf')
        self.read_upf(filename, rc)
