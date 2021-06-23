import qimpy as qp
import numpy as np
import torch
import pathlib
import re
from typing import Optional, Union, List, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import RunConfig
    from ._pseudopotential import Pseudopotential
    from .._system import System


class Ions:
    """Ionic system: ionic geometry and pseudopotentials. """
    __slots__ = ('rc', 'n_ions', 'n_types', 'symbols', 'slices',
                 'pseudopotentials', 'positions', 'types', 'M_initial',
                 'Z', 'Z_tot')
    rc: 'RunConfig'
    n_ions: int  #: number of ions
    n_types: int  #: number of distinct ion types
    symbols: List[str]  #: symbol for each ion type
    slices: List[slice]  #: slice to get each ion type
    pseudopotentials: List['Pseudopotential']  #: pseudopotential for each type
    positions: torch.Tensor  #: fractional positions of each ion (n_ions x 3)
    types: torch.Tensor  #: type of each ion (n_ions, int)
    M_initial: Optional[torch.Tensor]  #: initial magnetic moment for each ion
    Z: torch.Tensor  #: charge of each ion (n_ions, float)
    Z_tot: float  #: total ionic charge

    def __init__(self, *, rc: 'RunConfig',
                 coordinates: Optional[List] = None,
                 pseudopotentials: Optional[Union[str, List[str]]] = None
                 ) -> None:
        """Initialize geometry and pseudopotentials.

        Parameters
        ----------
        coordinates:
            List of [symbol, x, y, z, params] for each ion in unit cell,
            where symbol is the chemical symbol of the element,
            x, y and z are in the selected coordinate system,
            and params is an optional dictionary of additional per-ion
            parameters including initial magnetic moments, relaxation
            constraints etc. TODO
            Ions of the same type must be specified consecutively.
        pseudopotentials:
            Names of individual pseudopotential files or templates for
            families of pseudopotentials. Templates are specified by
            including a $ID in the name which is replaced by the chemical
            symbol of the element. The list of specified file names and
            templates is processed in order, and the first match for
            each element takes precedence.
        """
        self.rc = rc
        qp.log.info('\n--- Initializing Ions ---')

        # Read ionic coordinates:
        if coordinates is None:
            coordinates = []
        assert isinstance(coordinates, list)
        self.n_ions = 0  # number of ions
        self.n_types = 0  # number of distinct ion types
        self.symbols = []  # symbol for each ion type
        self.slices = []  # slice to get each ion type
        positions = []  # position of each ion
        types = []      # type of each ion (index into symbols)
        M_initial = []  # initial magnetic moments
        type_start = 0
        for coord in coordinates:
            # Check for optional attributes:
            if len(coord) == 4:
                attrib = {}
            elif len(coord) == 5:
                attrib = coord[4]
                if not isinstance(attrib, dict):
                    raise ValueError('ion attributes must be a dict')
            else:
                raise ValueError('each ion must be 4 entries + optional dict')
            # Add new symbol or append to existing:
            symbol = str(coord[0])
            if (not self.symbols) or (symbol != self.symbols[-1]):
                self.symbols.append(symbol)
                self.n_types += 1
                if type_start != self.n_ions:
                    self.slices.append(slice(type_start, self.n_ions))
                    type_start = self.n_ions
            # Add type and position of current ion:
            types.append(self.n_types-1)
            positions.append([float(x) for x in coord[1:4]])
            M_initial.append(attrib.get('M', None))
            self.n_ions += 1
        if type_start != self.n_ions:
            self.slices.append(slice(type_start, self.n_ions))  # for last type

        # Check order:
        if len(set(self.symbols)) < self.n_types:
            raise ValueError(
                'coordinates must group ions of same type together')

        # Convert to tensors before storing in class object:
        self.positions = torch.tensor(positions, device=rc.device)
        self.types = torch.tensor(types, device=rc.device)
        # --- Fill in missing magnetizations (if any specified):
        M_lengths = set([(len(M) if isinstance(M, list) else 1)
                         for M in M_initial if M])
        if len(M_lengths) > 1:
            raise ValueError('All M must be same type: 3-vector or scalar')
        elif len(M_lengths) == 1:
            M_length = next(iter(M_lengths))
            assert((M_length == 1) or (M_length == 3))
            M_default = ([0., 0., 0.] if (M_length == 3) else 0.)
            self.M_initial = torch.tensor(
                [(M if M else M_default) for M in M_initial],
                device=rc.device, dtype=torch.double)
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
            symbol_variants = [
                symbol.lower(),
                symbol.upper(),
                symbol.capitalize()]
            # Check each filename provided in order:
            for ps_name in pseudopotentials:
                if ps_name.count('$ID'):
                    # wildcard syntax
                    for symbol_variant in symbol_variants:
                        fname_test = ps_name.replace('$ID', symbol_variant)
                        if pathlib.Path(fname_test).exists():
                            fname = fname_test  # found
                            break
                else:
                    # specific filename
                    basename = pathlib.PurePath(ps_name).stem
                    ps_symbol = re.split(r'[_\-\.]+', basename)[0]
                    if ps_symbol in symbol_variants:
                        fname = ps_name
                        if not pathlib.Path(fname).exists():
                            raise FileNotFoundError(fname)
                        break
                if fname:
                    break
            # Read pseudopotential file:
            if fname:
                self.pseudopotentials.append(
                    qp.ions.Pseudopotential(fname, rc))
            else:
                raise ValueError(f'no pseudopotential found for {symbol}')

        # Calculate total ionic charge (needed for number of electrons):
        self.Z = torch.zeros(self.n_ions, device=rc.device)
        for i_type, slice_i in enumerate(self.slices):
            self.Z[slice_i] = self.pseudopotentials[i_type].Z
        self.Z_tot = self.Z.sum().item()
        qp.log.info(f'\nTotal ion charge, Z_tot: {self.Z_tot:g}')

        # Initialize / check replica process grid dimension:
        n_replicas = 1  # this will eventually change for NEB / phonon DFPT
        rc.provide_n_tasks(0, n_replicas)

    def report(self) -> None:
        """Report ionic positions and attributes"""
        qp.log.info(f'{self.n_ions} total ions of {self.n_types} types;'
                    ' positions:')
        # Fetch to CPU for reporting:
        positions = self.positions.to(self.rc.cpu).numpy()
        types = self.types.to(self.rc.cpu).numpy()
        M_initial = (None
                     if (self.M_initial is None)
                     else self.M_initial.to(self.rc.cpu).numpy())
        for i_ion, position in enumerate(positions):
            # Generate attribute string:
            attrib_str = ''
            attribs = {}
            if M_initial is not None:
                M_i = M_initial[i_ion]
                if np.linalg.norm(M_i):
                    attribs['M'] = M_i
            if attribs:
                attrib_str = ', ' + str(attribs).replace("'", '').replace(
                    'array(', '').replace(')', '')
            # Report:
            qp.log.info(
                f'- [{self.symbols[types[i_ion]]}, {position[0]:11.8f},'
                f' {position[1]:11.8f}, {position[2]:11.8f}{attrib_str}]')

    def update(self, system: 'System') -> None:
        """Update ionic potentials and energy components.
        The grids used for the potentials are derived from system,
        and the energy components are stored within system.E.
        """
        system.energy['Eewald'], _, _ = system.coulomb.ewald(self.positions,
                                                             self.Z)
        for ps in self.pseudopotentials:
            ps.update(system.grid.get_Gmax(), system.coulomb.ion_width)
