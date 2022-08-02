from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
import pathlib
from ._read_upf import _read_upf
from typing import Optional


class Pseudopotential:
    """Specification of electron-ion interactions. Contains
    for local potential, nonlocal projectors and atomic orbitals.
    Currently supports norm-conserving pseudopotentials."""

    element: str  #: Chemical symbol of element
    atomic_number: int  #: Atomic number of element
    is_paw: bool  #: Whether this is a PAW pseudopotential
    Z: float  #: Pseudo-atomic number i.e. number of valence electrons / atom
    l_max: int  #: Maximum angular momentum quantum number
    r: torch.Tensor  #: Radial grid
    dr: torch.Tensor  #: Radial grid spacings / integration weights
    rho_atom: qp.ions.RadialFunction  #: Atomic reference electron density
    n_core: qp.ions.RadialFunction  #: Partial core density
    Vloc: qp.ions.RadialFunction  #: Local potential (short-ranged part)
    ion_width: float  #: Gaussian width to extract long-ranged part of Vloc
    beta: qp.ions.RadialFunction  #: Nonlocal projectors
    psi: qp.ions.RadialFunction  #: Atomic orbitals
    is_relativistic: bool  #: Whether this is a relativistic pseudopotential
    j_beta: Optional[torch.Tensor]  #: l+s of projectors (if relativistic)
    j_psi: Optional[torch.Tensor]  #: l+s of atomic orbitals (if relativistic)
    eig_psi: np.ndarray  #: Energy eigenvalue of each atomic orbital
    D: torch.Tensor  #: Descreened nonlocal pseudopotential matrix

    Gmax: float  #: Current reciprocal space extent of radial functions
    pqn_beta: qp.ions.PseudoQuantumNumbers  #: quantum numbers for projectors
    pqn_psi: qp.ions.PseudoQuantumNumbers  #: quantum numbers for orbitals
    pulay_data: Optional[np.ndarray]  #: Pulay correction data for finite ke cutoff

    # Methods defined out of class:
    read_upf = _read_upf

    def __init__(self, filename: str) -> None:
        """Read pseudopotential from file.

        Parameters
        ----------
        filename : str
            File to read pseudopotential from.
            Currently, only norm-conserving UPF files are supported.
        """
        assert filename[-4:].lower() == ".upf"
        self.read_upf(filename)
        self.Gmax = 0.0  # reciprocal space versions not yet initialized
        self.pqn_beta = qp.ions.PseudoQuantumNumbers(self.beta.l, self.j_beta)
        self.pqn_psi = qp.ions.PseudoQuantumNumbers(self.psi.l, self.j_psi)
        self._read_pulay(filename)

    def update(self, Gmax: float, ion_width: float, comm: qp.MPI.Comm) -> None:
        """Update to support calculation of G upto `Gmax`.
        Along with radial function transformations, also update the range
        separation of Vloc to be consistent with specified `ion_width`.
        Parallelize transformations of radial functions over `comm`.
        """
        # Update ion width if necessary:
        if ion_width != self.ion_width:
            eta_self = np.sqrt(0.5) / self.ion_width  # current erf parameter
            eta = np.sqrt(0.5) / ion_width  # target erf parameter
            self.Vloc.f[0] -= (
                self.Z
                * (torch.erf(eta_self * self.r) - torch.erf(eta * self.r))
                / self.r
            )
            if Gmax <= self.Gmax:
                # Not a global Gmax update, transform Vloc separately:
                qp.ions.RadialFunction.transform([self.Vloc], Gmax, comm, self.element)
            self.ion_width = ion_width

        # Update radial transforms if necessary:
        if Gmax > self.Gmax:
            transform_list = [self.rho_atom, self.Vloc, self.beta, self.psi]
            if hasattr(self, "n_core"):
                transform_list.append(self.n_core)
            qp.ions.RadialFunction.transform(transform_list, Gmax, comm, self.element)
            self.Gmax = Gmax

    @property
    def n_projectors(self) -> int:
        """Number of projectors per atom (including m quantum numbers)."""
        return self.pqn_beta.n_tot

    @property
    def n_orbital_projectors(self) -> int:
        """Number of projectors used to generate atomic orbitals.
        This is same as the number of atomic orbitals in non-spinorial mode,
        exactly half that for non-relativistic pseudopotentials in spinorial
        mode and smaller than the number of atomic orbitals by the number
        of l = 0 orbitals for relativistic pseudopotentials."""
        return self.pqn_psi.n_tot

    def n_atomic_orbitals(self, n_spinor: int) -> int:
        """Number of orbitals per atom."""
        if self.is_relativistic:
            if n_spinor != 2:
                raise ValueError(
                    "Relativistic pseudopotentials require spinorial calculation"
                )
            return self.pqn_psi.n_tot_s
        else:
            return self.pqn_psi.n_tot * n_spinor

    def _read_pulay(self, filename: str) -> None:
        """Initialize Pulay correction data, if available"""
        filename_parts = filename.split(".")
        filename_parts[-1] = "pulay"  # replace psp extension
        pulay_filename = ".".join(filename_parts)
        if pathlib.Path(pulay_filename).exists():
            for line in open(pulay_filename):
                if not line.startswith("#"):  # ignore comments
                    tokens = line.split()
                    if len(tokens) == 1:
                        n_cutoffs = int(tokens[0])
                        i_cutoff = 0
                        data = np.zeros((n_cutoffs, 2))
                    elif len(tokens) == 2:
                        data[i_cutoff] = [float(token) for token in tokens]
                        i_cutoff += 1
                    else:
                        raise IOError("Incorrect pulay correction file format.")
            assert n_cutoffs == i_cutoff
            self.pulay_data = data[data[:, 0].argsort()]  # sort by cutoffs
            qp.log.info(f"  Loaded Pulay corrections from {pulay_filename}")
        else:
            self.pulay_data = None

    def dE_drho_basis(self, ke_cutoff: float) -> float:
        """Get Pulay correction dE/drho_basis, if available. Here, rho_basis is
        defined as the number of plane-wave basis functions per unit cell volume.
        """
        if self.pulay_data is None:
            return 0.0
        else:
            # Check ranges:
            ke_cutoffs = self.pulay_data[:, 0]
            corrections = self.pulay_data[:, 1]
            ke_cutoff_min = ke_cutoffs[0]
            if ke_cutoff < ke_cutoff_min:
                raise ValueError(
                    f"ke_cutoff lower than {ke_cutoff_min = } Eh for {self.element}"
                    " Pulay correction"
                )
            ke_cutoff_max = ke_cutoffs[-1]
            if ke_cutoff > ke_cutoff_max:
                qp.log.warning(
                    f"ke_cutoff higher than {ke_cutoff_max = } Eh in {self.element}"
                    " Pulay correction."
                )
                return 0.0
            result = np.interp(ke_cutoff, ke_cutoffs, corrections)
            qp.log.info(
                f"Pulay dE/drho_basis = {result:.6f} Eh a0^3 for {self.element}"
            )
            return result
