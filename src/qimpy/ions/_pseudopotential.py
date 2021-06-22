import qimpy as qp
import numpy as np
import torch
from ._read_upf import _read_upf
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._radial_function import RadialFunction


class Pseudopotential:
    """Specification of electron-ion interactions. Contains
    for local potential, nonlocal projectors and atomic orbitals.
    Currently supports norm-conserving pseudopotentials."""
    __slots__ = ('element', 'atomic_number', 'is_paw', 'Z', 'l_max',
                 'r', 'dr', 'rho_atom', 'n_core', 'Vloc',
                 'beta', 'psi', 'j_beta', 'j_psi', 'eig_psi', 'D')
    element: str  #: Chemical symbol of element
    atomic_number: int  #: Atomic number of element
    is_paw: bool  #: Whether this is a PAW pseudopotential
    Z: float  #: Pseudo-atomic number i.e. number of valence electrons / atom
    l_max: int  #: Maximum angular momentum quantum number
    r: torch.Tensor  #: Radial grid
    dr: torch.Tensor  #: Radial grid spacings / integration weights
    rho_atom: 'RadialFunction'  #: Atomic reference electron density
    n_core: 'RadialFunction'  #: Partial core density
    Vloc: 'RadialFunction'  #: Local potential
    beta: 'RadialFunction'  #: Nonlocal projectors
    psi: 'RadialFunction'  #: Atomic orbitals
    j_beta: np.ndarray  #: l+s of each projector (if relativistic)
    j_psi: np.ndarray  #: l+s of each atomic orbital (if relativistic)
    eig_psi: np.ndarray  #: Energy eigenvalue of each atomic orbital
    D: torch.Tensor  #: Descreened nonlocal pseudopotential matrix

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

    def update(self, Gmax: float) -> None:
        """Update to support calculation of G upto Gmax."""
