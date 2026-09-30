from __future__ import annotations
import pathlib

import numpy as np
import torch
import torch.distributed as dist

from qimpy import log, rc
from qimpy.math import RadialFunction
from qimpy.dft.ions import PseudoQuantumNumbers
from qimpy.dft.ions._read_upf import read_upf
from qimpy.dft.ions._read_uspp import read_uspp


class Pseudopotential:
    """Specification of electron-ion interactions. Contains
    for local potential, nonlocal projectors and atomic orbitals.
    Supports norm-conserving and ultrasoft pseudopotentials."""

    element: str  #: Chemical symbol of element
    atomic_number: int  #: Atomic number of element
    is_paw: bool  #: Whether this is a PAW pseudopotential
    is_ultrasoft: bool  #: Whether this is an ultrasoft pseudopotential
    Z: float  #: Pseudo-atomic number i.e. number of valence electrons / atom
    l_max: int  #: Maximum angular momentum quantum number
    r: torch.Tensor  #: Radial grid
    dr: torch.Tensor  #: Radial grid spacings / integration weights
    rho_atom: RadialFunction  #: Atomic reference electron density
    n_core: RadialFunction  #: Partial core density
    Vloc: RadialFunction  #: Local potential (short-ranged part)
    ion_width: float  #: Gaussian width to extract long-ranged part of Vloc
    beta: RadialFunction  #: Nonlocal projectors
    psi: RadialFunction  #: Atomic orbitals
    is_relativistic: bool  #: Whether this is a relativistic pseudopotential
    j_beta: torch.Tensor | None  #: l+s of projectors (if relativistic)
    j_psi: torch.Tensor | None  #: l+s of atomic orbitals (if relativistic)
    eig_psi: np.ndarray  #: Energy eigenvalue of each atomic orbital
    D: torch.Tensor  #: Descreened nonlocal pseudopotential matrix
    q: torch.Tensor | None  #: Ultrasoft overlap coefficients (like `D`; None if NC)
    Q: RadialFunction | None  #: Ultrasoft augmentation functions, with Q.l = L
    Q_pair: torch.Tensor | None  #: Radial projectors (i <= j) of each `Q` (n_Q x 2)

    Gmax: float  #: Current reciprocal space extent of radial functions
    pqn_beta: PseudoQuantumNumbers  #: quantum numbers for projectors
    pqn_psi: PseudoQuantumNumbers  #: quantum numbers for orbitals
    pulay_data: np.ndarray | None  #: Pulay correction data for finite ke cutoff

    def __init__(self, filename: str) -> None:
        """Read pseudopotential from file.

        Parameters
        ----------
        filename : str
            File to read pseudopotential from: UPF (norm-conserving or ultrasoft)
            or Vanderbilt '.uspp' (ultrasoft, e.g. the GBRV library).
        """
        self.q = self.Q = self.Q_pair = None  # set by readers only if ultrasoft
        suffix = pathlib.Path(filename).suffix.lower()
        if suffix == ".upf":
            read_upf(self, filename)
        elif suffix == ".uspp":
            read_uspp(self, filename)
        else:
            raise ValueError(f"Unknown pseudopotential format '{suffix}' in {filename}")
        self.Gmax = 0.0  # reciprocal space versions not yet initialized
        self.pqn_beta = PseudoQuantumNumbers(self.beta.l, self.j_beta)
        self.pqn_psi = PseudoQuantumNumbers(self.psi.l, self.j_psi)
        self._read_pulay(filename)

    def update(self, Gmax: float, ion_width: float, group: dist.ProcessGroup) -> None:
        """Update to support calculation of G upto `Gmax`.
        Along with radial function transformations, also update the range
        separation of Vloc to be consistent with specified `ion_width`.
        Parallelize transformations of radial functions over `group`.
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
                RadialFunction.transform([self.Vloc], Gmax, group, self.element)
            self.ion_width = ion_width

        # Update radial transforms if necessary:
        if Gmax > self.Gmax:
            transform_list = [self.rho_atom, self.Vloc, self.beta, self.psi]
            if hasattr(self, "n_core"):
                transform_list.append(self.n_core)
            if self.Q is not None:
                transform_list.append(self.Q)  # same radial grid: same batch
            RadialFunction.transform(transform_list, Gmax, group, self.element)
            self.Gmax = Gmax

    def set_augmentation(
        self,
        r2_Q: np.ndarray,
        q_file: np.ndarray,
        r_inner: np.ndarray | None = None,
        Q_taylor: np.ndarray | None = None,
    ) -> None:
        """Set ultrasoft augmentation from data parsed in any file format.
        The readers only parse, and this does the processing common to formats:
        inner pseudization, dropping zero functions, and the overlap coefficients.

        Parameters
        ----------
        r2_Q
            :math:`r^2 Q_{ij}(r)` on the radial grid for each pair of radial
            projectors (symmetric in i, j): shape (n_beta, n_beta, n_r) if one
            function serves all L ('.uspp', and UPF with q_with_l false), or
            (n_beta, n_beta, n_L, n_r) with one function per L (UPF q_with_l true).
        q_file
            Overlap coefficients :math:`q_{ij}` stored in the file (n_beta^2).
            Only reported: `q` is instead integrated from the pseudized
            :math:`Q^0_{ij}`, so that the augmentation charge exactly equals
            :math:`\\sum_{ij}\\rho_{ij} q_{ij}` (as in JDFTx).
        r_inner, Q_taylor
            Optional inner pseudization: for r < r_inner[L], replace
            :math:`Q^L_{ij}(r)` by the series :math:`\\sum_k c_k r^{L+2k}`, with
            :math:`c_k` = Q_taylor[i, j, L, k] (shapes (n_L,), (n_beta, n_beta, n_L,
            n_qf)). This is the Vanderbilt 'qfcoef' / 'rinner' scheme.
        """
        r = self.r.to(rc.cpu).numpy()
        l_beta = self.beta.l.tolist()
        n_beta = len(l_beta)
        # Integration weights for q = int r^2 Q dr: the same Simpson rule as the
        # G = 0 limit of RadialFunction.transform, so that q matches the
        # augmentation charge on the grid to machine precision:
        w = (2.0 / 3) * (r**2) * self.dr.to(rc.cpu).numpy()
        w[1::2] *= 2.0
        w[[0, -1]] *= 0.5

        # Collect the distinct nonzero Q^L_ij(r) for i <= j (few, only at read):
        Q_f: list[np.ndarray] = []  # Q^L_ij(r) / r^L (RadialFunction convention)
        Q_l: list[int] = []  # L of each function
        Q_pair: list[tuple[int, int]] = []  # (i, j) of each function
        q = np.zeros((n_beta, n_beta))
        for i in range(n_beta):
            for j in range(i, n_beta):
                l_i, l_j = l_beta[i], l_beta[j]
                for L in range(abs(l_i - l_j), l_i + l_j + 1, 2):  # Gaunt selection
                    Q = (r2_Q[i, j] if (r2_Q.ndim == 3) else r2_Q[i, j, L]) / r**2
                    if Q_taylor is not None:
                        assert r_inner is not None
                        inner = r < r_inner[L]
                        powers = L + 2 * np.arange(Q_taylor.shape[-1])
                        Q[inner] = (r[inner, None] ** powers) @ Q_taylor[i, j, L]
                    if not np.any(Q):
                        continue  # nothing to augment (e.g. NC in ultrasoft format)
                    Q_f.append(Q / r**L)
                    Q_l.append(L)
                    Q_pair.append((i, j))
                    if L == 0:  # only possible for l_i = l_j
                        q[i, j] = q[j, i] = Q @ w

        if not Q_f:
            log.info("  No augmentation functions: treated as norm-conserving.")
            self.is_ultrasoft = False
            self.q = self.Q = self.Q_pair = None
            return
        self.is_ultrasoft = True
        self.q = torch.tensor(q, device=rc.device)
        self.Q = RadialFunction(self.r, self.dr, np.array(Q_f), np.array(Q_l))
        self.Q_pair = torch.tensor(Q_pair, device=rc.device)
        log.info(
            f"  Ultrasoft: {len(Q_f)} augmentation functions Q^L_ij(r) up to"
            f" L = {max(Q_l)}; max |q_ij - q_file| = {np.abs(q - q_file).max():.1e}"
        )

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
            log.info(f"  Loaded Pulay corrections from {pulay_filename}")
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
                    f"ke_cutoff lower than {ke_cutoff_min=} Eh for {self.element}"
                    " Pulay correction"
                )
            ke_cutoff_max = ke_cutoffs[-1]
            if ke_cutoff > ke_cutoff_max:
                log.warning(
                    f"ke_cutoff higher than {ke_cutoff_max=} Eh in {self.element}"
                    " Pulay correction."
                )
                return 0.0
            result = float(np.interp(ke_cutoff, ke_cutoffs, corrections))
            log.info(f"Pulay dE/drho_basis = {result:.6f} Eh a0^3 for {self.element}")
            return result
