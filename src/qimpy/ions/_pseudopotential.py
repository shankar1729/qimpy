import qimpy as qp
import numpy as np
import torch
from ._read_upf import _read_upf
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._radial_function import RadialFunction
    from ..utils import RunConfig


class Pseudopotential:
    """Specification of electron-ion interactions. Contains
    for local potential, nonlocal projectors and atomic orbitals.
    Currently supports norm-conserving pseudopotentials."""
    __slots__ = ('rc', 'element', 'atomic_number', 'is_paw', 'Z', 'l_max',
                 'r', 'dr', 'rho_atom', 'n_core', 'Vloc', 'ion_width',
                 'beta', 'psi', 'j_beta', 'j_psi', 'eig_psi', 'D', 'Gmax')
    rc: 'RunConfig'
    element: str  #: Chemical symbol of element
    atomic_number: int  #: Atomic number of element
    is_paw: bool  #: Whether this is a PAW pseudopotential
    Z: float  #: Pseudo-atomic number i.e. number of valence electrons / atom
    l_max: int  #: Maximum angular momentum quantum number
    r: torch.Tensor  #: Radial grid
    dr: torch.Tensor  #: Radial grid spacings / integration weights
    rho_atom: 'RadialFunction'  #: Atomic reference electron density
    n_core: 'RadialFunction'  #: Partial core density
    Vloc: 'RadialFunction'  #: Local potential (short-ranged part)
    ion_width: float  #: Gaussian width to extract long-ranged part of Vloc
    beta: 'RadialFunction'  #: Nonlocal projectors
    psi: 'RadialFunction'  #: Atomic orbitals
    j_beta: np.ndarray  #: l+s of each projector (if relativistic)
    j_psi: np.ndarray  #: l+s of each atomic orbital (if relativistic)
    eig_psi: np.ndarray  #: Energy eigenvalue of each atomic orbital
    D: torch.Tensor  #: Descreened nonlocal pseudopotential matrix
    Gmax: float  #: Current reciprocal space extent of radial functions

    # Methods defined out of class:
    read_upf = _read_upf

    def __init__(self, filename: str, rc: 'RunConfig') -> None:
        """Read pseudopotential from file.

        Parameters
        ----------
        filename : str
            File to read pseudopotential from.
            Currently, only norm-conserving UPF files are supported.
        """
        self.rc = rc
        assert(filename[-4:].lower() == '.upf')
        self.read_upf(filename, rc)
        self.Gmax = 0.  # reciprocal space versiosn not yet initialized

    def update(self, Gmax: float, ion_width: float) -> None:
        """Update to support calculation of G upto `Gmax`.
        Along with radial function transformations, also update the range
        separation of Vloc to be consistent with specified ion_width.
        """
        # Update ion width if necessary:
        if ion_width != self.ion_width:
            eta_self = np.sqrt(0.5) / self.ion_width  # current erf parameter
            eta = np.sqrt(0.5) / ion_width  # target erf parameter
            self.Vloc.f[0] -= self.Z * (torch.erf(eta_self * self.r)
                                        - torch.erf(eta * self.r)) / self.r
            if Gmax <= self.Gmax:
                # Not a global Gmax update, transform Vloc separately:
                qp.ions.RadialFunction.transform([self.Vloc], Gmax,
                                                 self.rc.comm_kb, self.element)
            self.ion_width = ion_width

        # Update radial transforms if necessary:
        if Gmax > self.Gmax:
            transform_list = [self.rho_atom, self.Vloc, self.beta, self.psi]
            if hasattr(self, 'n_core'):
                transform_list.append(self.n_core)
            qp.ions.RadialFunction.transform(transform_list, Gmax,
                                             self.rc.comm_kb, self.element)
            self.Gmax = Gmax
