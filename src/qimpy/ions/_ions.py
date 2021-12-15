from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
import pathlib
import re
from ._ions_projectors import _get_projectors
from ._ions_atomic import get_atomic_orbitals, get_atomic_density
from ._ions_update import update, _collect_ps_matrix
from typing import Optional, Union, List


class Ions(qp.TreeNode):
    """Ionic system: ionic geometry and pseudopotentials."""

    __slots__ = (
        "n_ions",
        "n_types",
        "symbols",
        "n_ions_type",
        "slices",
        "pseudopotentials",
        "positions",
        "types",
        "M_initial",
        "Z",
        "Z_tot",
        "rho_tilde",
        "Vloc_tilde",
        "n_core_tilde",
        "beta",
        "beta_full",
        "beta_version",
        "D_all",
    )
    n_ions: int  #: number of ions
    n_types: int  #: number of distinct ion types
    n_ions_type: List[int]  #: number of ions of each type
    symbols: List[str]  #: symbol for each ion type
    slices: List[slice]  #: slice to get each ion type
    pseudopotentials: List[qp.ions.Pseudopotential]  #: pseudopotential for each type
    positions: torch.Tensor  #: fractional positions of each ion (n_ions x 3)
    types: torch.Tensor  #: type of each ion (n_ions, int)
    M_initial: Optional[torch.Tensor]  #: initial magnetic moment for each ion
    Z: torch.Tensor  #: charge of each ion type (n_types, float)
    Z_tot: float  #: total ionic charge
    rho_tilde: qp.grid.FieldH  #: ionic charge density (uses coulomb.ion_width)
    Vloc_tilde: qp.grid.FieldH  #: local potential due to ions (including from rho)
    n_core_tilde: qp.grid.FieldH  #: partial core electronic density (for XC)
    beta: qp.electrons.Wavefunction  #: pseudopotential projectors (split-basis only)
    beta_full: Optional[qp.electrons.Wavefunction]  #: full-basis version of `beta`
    beta_version: int  #: version of `beta` to invalidate cached projections
    D_all: torch.Tensor  #: nonlocal pseudopotential matrix (all atoms)

    _get_projectors = _get_projectors
    get_atomic_orbitals = get_atomic_orbitals
    get_atomic_density = get_atomic_density
    update = update
    _collect_ps_matrix = _collect_ps_matrix

    def __init__(
        self,
        *,
        process_grid: qp.utils.ProcessGrid,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        coordinates: Optional[List] = None,
        pseudopotentials: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Initialize geometry and pseudopotentials.

        Parameters
        ----------
        coordinates
            :yaml:`List of [symbol, x, y, z, args] for each ion in unit cell.`
            Here, symbol is the chemical symbol of the element,
            x, y and z are in the selected coordinate system.
            Optional args is a dictionary of additional per-ion
            parameters that may include:

                * `M`: initial magnetic moment for the ion, which would be a
                  single Mz or [Mx, My, Mz] depending on if the calculation
                  is spinorial. Only specify in spin-polarized calculations.
                * Relaxation constraints: TODO

            Ions of the same type (symbol) must be specified consecutively.
        pseudopotentials
            :yaml:`Pseudopotential filenames or filename templates.`
            Templates for families of pseudopotentials are specified by
            including a $ID in the name which is replaced by the chemical
            symbol of the element. The list of specified file names and
            templates is processed in order, and the first match for
            each element takes precedence.
        """
        super().__init__()
        qp.log.info("\n--- Initializing Ions ---")

        # Read ionic coordinates:
        if coordinates is None:
            coordinates = []
        assert isinstance(coordinates, list)
        self.n_ions = 0  # number of ions
        self.n_types = 0  # number of distinct ion types
        self.symbols = []  # symbol for each ion type
        self.n_ions_type = []  # numebr of ions of each type
        self.slices = []  # slice to get each ion type
        positions = []  # position of each ion
        types = []  # type of each ion (index into symbols)
        M_initial = []  # initial magnetic moments
        type_start = 0
        for coord in coordinates:
            # Check for optional attributes:
            if len(coord) == 4:
                attrib = {}
            elif len(coord) == 5:
                attrib = coord[4]
                if not isinstance(attrib, dict):
                    raise ValueError("ion attributes must be a dict")
            else:
                raise ValueError("each ion must be 4 entries + optional dict")
            # Add new symbol or append to existing:
            symbol = str(coord[0])
            if (not self.symbols) or (symbol != self.symbols[-1]):
                self.symbols.append(symbol)
                self.n_types += 1
                if type_start != self.n_ions:
                    self.slices.append(slice(type_start, self.n_ions))
                    self.n_ions_type.append(self.n_ions - type_start)
                    type_start = self.n_ions
            # Add type and position of current ion:
            types.append(self.n_types - 1)
            positions.append([float(x) for x in coord[1:4]])
            M_initial.append(attrib.get("M", None))
            self.n_ions += 1
        if type_start != self.n_ions:
            self.slices.append(slice(type_start, self.n_ions))  # for last type
            self.n_ions_type.append(self.n_ions - type_start)

        # Check order:
        if len(set(self.symbols)) < self.n_types:
            raise ValueError("coordinates must group ions of same type together")

        # Convert to tensors before storing in class object:
        self.positions = torch.tensor(positions, device=qp.rc.device)
        self.types = torch.tensor(types, device=qp.rc.device, dtype=torch.long)
        # --- Fill in missing magnetizations (if any specified):
        M_lengths = set(
            [(len(M) if isinstance(M, list) else 1) for M in M_initial if M]
        )
        if len(M_lengths) > 1:
            raise ValueError("All M must be same type: 3-vector or scalar")
        elif len(M_lengths) == 1:
            M_length = next(iter(M_lengths))
            assert (M_length == 1) or (M_length == 3)
            M_default = [0.0, 0.0, 0.0] if (M_length == 3) else 0.0
            self.M_initial = torch.tensor(
                [(M if M else M_default) for M in M_initial],
                device=qp.rc.device,
                dtype=torch.double,
            )
        else:
            self.M_initial = None
        self.report()

        # Initialize pseudopotentials:
        self.pseudopotentials = []
        if pseudopotentials is None:
            pseudopotentials = []
        if isinstance(pseudopotentials, str):
            pseudopotentials = [pseudopotentials]
        for i_type, symbol in enumerate(self.symbols):
            fname = None  # full filename for this ion type
            symbol_variants = [symbol.lower(), symbol.upper(), symbol.capitalize()]
            # Check each filename provided in order:
            for ps_name in pseudopotentials:
                if ps_name.count("$ID"):
                    # wildcard syntax
                    for symbol_variant in symbol_variants:
                        fname_test = ps_name.replace("$ID", symbol_variant)
                        if pathlib.Path(fname_test).exists():
                            fname = fname_test  # found
                            break
                else:
                    # specific filename
                    basename = pathlib.PurePath(ps_name).stem
                    ps_symbol = re.split(r"[_\-\.]+", basename)[0]
                    if ps_symbol in symbol_variants:
                        fname = ps_name
                        if not pathlib.Path(fname).exists():
                            raise FileNotFoundError(fname)
                        break
                if fname:
                    break
            # Read pseudopotential file:
            if fname:
                self.pseudopotentials.append(qp.ions.Pseudopotential(fname))
            else:
                raise ValueError(f"no pseudopotential found for {symbol}")
        self.beta_version = 0

        # Calculate total ionic charge (needed for number of electrons):
        self.Z = torch.tensor(
            [ps.Z for ps in self.pseudopotentials], device=qp.rc.device
        )
        self.Z_tot = self.Z[self.types].sum().item()
        qp.log.info(f"\nTotal ion charge, Z_tot: {self.Z_tot:g}")

        # Initialize / check replica process grid dimension:
        n_replicas = 1  # this will eventually change for NEB / phonon DFPT
        process_grid.provide_n_tasks("r", n_replicas)

    def report(self) -> None:
        """Report ionic positions and attributes"""
        qp.log.info(f"{self.n_ions} total ions of {self.n_types} types;" " positions:")
        # Fetch to CPU for reporting:
        positions = self.positions.to(qp.rc.cpu).numpy()
        types = self.types.to(qp.rc.cpu).numpy()
        M_initial = (
            None if (self.M_initial is None) else self.M_initial.to(qp.rc.cpu).numpy()
        )
        for i_ion, position in enumerate(positions):
            # Generate attribute string:
            attrib_str = ""
            attribs = {}
            if M_initial is not None:
                M_i = M_initial[i_ion]
                if np.linalg.norm(M_i):
                    attribs["M"] = M_i
            if attribs:
                attrib_str = ", " + str(attribs).replace("'", "").replace(
                    "array(", ""
                ).replace(")", "")
            # Report:
            qp.log.info(
                f"- [{self.symbols[types[i_ion]]}, {position[0]:11.8f},"
                f" {position[1]:11.8f}, {position[2]:11.8f}{attrib_str}]"
            )

    def translation_phase(
        self, iG: torch.Tensor, atom_slice: slice = slice(None)
    ) -> torch.Tensor:
        """Get translation phases at `iG` for a slice of atoms.
        The result has atoms as the final dimension; summing over that
        dimension yields the structure factor corresponding to these atoms.
        """
        return qp.utils.cis((-2 * np.pi) * (iG @ self.positions[atom_slice].T))

    @property
    def n_projectors(self) -> int:
        """Total number of pseudopotential projectors."""
        return sum(
            (ps.n_projectors * self.n_ions_type[i_ps])
            for i_ps, ps in enumerate(self.pseudopotentials)
        )

    @property
    def n_orbital_projectors(self) -> int:
        """Total number of projectors used to generate atomic orbitals."""
        return sum(
            (ps.n_orbital_projectors * self.n_ions_type[i_ps])
            for i_ps, ps in enumerate(self.pseudopotentials)
        )

    def n_atomic_orbitals(self, n_spinor: int) -> int:
        """Total number of atomic orbitals. This depends on the number
        of spinorial components `n_spinor`."""
        return sum(
            (ps.n_atomic_orbitals(n_spinor) * self.n_ions_type[i_ps])
            for i_ps, ps in enumerate(self.pseudopotentials)
        )
