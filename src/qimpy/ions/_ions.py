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
    from ..grid import Grid, FieldR, FieldH
    from ..electrons import Wavefunction, Basis


class Ions(qp.Constructable):
    """Ionic system: ionic geometry and pseudopotentials. """
    __slots__ = ('n_ions', 'n_types', 'symbols', 'n_ions_type', 'slices',
                 'pseudopotentials', 'positions', 'types', 'M_initial',
                 'Z', 'Z_tot', 'rho', 'Vloc', 'n_core',
                 'beta', 'beta_version', 'D_all')
    n_ions: int  #: number of ions
    n_types: int  #: number of distinct ion types
    n_ions_type: List[int]  #: number of ions of each type
    symbols: List[str]  #: symbol for each ion type
    slices: List[slice]  #: slice to get each ion type
    pseudopotentials: List['Pseudopotential']  #: pseudopotential for each type
    positions: torch.Tensor  #: fractional positions of each ion (n_ions x 3)
    types: torch.Tensor  #: type of each ion (n_ions, int)
    M_initial: Optional[torch.Tensor]  #: initial magnetic moment for each ion
    Z: torch.Tensor  #: charge of each ion type (n_types, float)
    Z_tot: float  #: total ionic charge
    rho: 'FieldH'  #: ionic charge density profile (uses coulomb.ion_width)
    Vloc: 'FieldH'  #: local potential due to ions (including from rho)
    n_core: 'FieldH'  #: partial core electronic density (for inclusion in XC)
    beta: 'Wavefunction'  #: pseudopotential projectors
    beta_version: int  #: version of `beta` to invalidate cached projections
    D_all: torch.Tensor  #: nonlocal pseudopotential matrix (all atoms)

    def __init__(self, *, co: qp.ConstructOptions,
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
            :yaml:`inputfile`
        pseudopotentials:
            Names of individual pseudopotential files or templates for
            families of pseudopotentials. Templates are specified by
            including a $ID in the name which is replaced by the chemical
            symbol of the element. The list of specified file names and
            templates is processed in order, and the first match for
            each element takes precedence. :yaml:`inputfile`
        """
        super().__init__(co=co)
        rc = self.rc
        qp.log.info('\n--- Initializing Ions ---')

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
                    self.n_ions_type.append(self.n_ions - type_start)
                    type_start = self.n_ions
            # Add type and position of current ion:
            types.append(self.n_types-1)
            positions.append([float(x) for x in coord[1:4]])
            M_initial.append(attrib.get('M', None))
            self.n_ions += 1
        if type_start != self.n_ions:
            self.slices.append(slice(type_start, self.n_ions))  # for last type
            self.n_ions_type.append(self.n_ions - type_start)

        # Check order:
        if len(set(self.symbols)) < self.n_types:
            raise ValueError(
                'coordinates must group ions of same type together')

        # Convert to tensors before storing in class object:
        self.positions = torch.tensor(positions, device=rc.device)
        self.types = torch.tensor(types, device=rc.device, dtype=torch.long)
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
        self.beta_version = 0

        # Calculate total ionic charge (needed for number of electrons):
        self.Z = torch.tensor([ps.Z for ps in self.pseudopotentials],
                              device=rc.device)
        self.Z_tot = self.Z[self.types].sum().item()
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
        """Update ionic potentials, projectors and energy components.
        The grids used for the potentials are derived from system,
        and the energy components are stored within system.E.
        """
        grid = system.grid
        n_densities = system.electrons.n_densities
        self.rho = qp.grid.FieldH(grid)  # initialize zero ionic charge
        self.Vloc = qp.grid.FieldH(grid)  # initizliae zero local potential
        self.n_core = qp.grid.FieldH(grid,  # initialize zero core density
                                     shape_batch=(n_densities,))
        if not self.n_ions:
            return  # no contributions below if no ions!
        system.energy['Eewald'] = system.coulomb.ewald(self.positions,
                                                       self.Z[self.types])[0]
        # Update ionic densities and potentials:
        from .quintic_spline import Interpolator
        iG = grid.get_mesh('H').to(torch.double)  # half-space
        Gsq = ((iG @ grid.lattice.Gbasis.T)**2).sum(dim=-1)
        G = Gsq.sqrt()
        Ginterp = Interpolator(G, qp.ions.RadialFunction.DG)
        SF = torch.empty((self.n_types,) + G.shape, dtype=torch.cdouble,
                         device=G.device)  # structure factor by species
        inv_volume = 1. / grid.lattice.volume
        # --- collect radial coefficients
        Vloc_coeff = []
        n_core_coeff = []
        Gmax = system.grid.get_Gmax()
        ion_width = system.coulomb.ion_width
        for i_type, ps in enumerate(self.pseudopotentials):
            ps.update(Gmax, ion_width)
            SF[i_type] = self.translation_phase(iG, self.slices[i_type]
                                                ).sum(dim=-1) * inv_volume
            Vloc_coeff.append(ps.Vloc.f_t_coeff)
            n_core_coeff.append(ps.n_core.f_t_coeff)
        # --- interpolate to G and collect with structure factors
        self.Vloc.data = (SF * Ginterp(torch.hstack(Vloc_coeff))).sum(dim=0)
        self.n_core.data[0] = (SF * Ginterp(torch.hstack(n_core_coeff))
                               ).sum(dim=0)
        self.rho.data = ((-self.Z.view(-1, 1, 1, 1) * SF).sum(dim=0)
                         * torch.exp((-0.5*(ion_width**2)) * Gsq))
        # --- include long-range electrostatic part of Vloc:
        self.Vloc += system.coulomb(self.rho, correct_G0_width=True)

        # Update pseudopotential matrix and projectors:
        self._collect_ps_matrix(system.electrons.n_spinor)
        self.beta = self._get_projectors(system.electrons.basis)
        self.beta_version += 1  # will auto-invalidate cached projections

    def translation_phase(self, iG: torch.Tensor,
                          atom_slice: slice = slice(None)) -> torch.Tensor:
        """Get translation phases at `iG` for a slice of atoms.
        The result has atoms as the final dimension; summing over that
        dimension yields the structure factor corresponding to these atoms.
        """
        return qp.utils.cis((-2*np.pi) * (iG @ self.positions[atom_slice].T))

    def _get_projectors(self, basis: 'Basis',
                        get_psi: bool = False) -> 'Wavefunction':
        """Get projectors corresponding to specified `basis`.
        If get_psi is True, get atomic orbitals instead. This mode is only for
        internal use by :meth:`get_atomic_orbitals`, which does additional
        transformations on the spin and spinorial dimensions."""
        from .quintic_spline import Interpolator
        iGk = basis.iG[:, basis.mine] + basis.k[:, None]  # fractional G + k
        Gk = iGk @ basis.lattice.Gbasis.T  # Cartesian G + k (of this process)
        # Prepare interpolator for radial functions:
        Gk_mag = (Gk ** 2).sum(dim=-1).sqrt()
        Ginterp = Interpolator(Gk_mag, qp.ions.RadialFunction.DG)
        # Prepare output:
        nk_mine, n_basis_each = Gk_mag.shape
        n_proj_tot = (self.n_orbital_projectors if get_psi
                      else self.n_projectors)
        proj = torch.empty((1, nk_mine, n_proj_tot, 1, n_basis_each),
                           dtype=torch.complex128, device=self.rc.device)
        if not n_proj_tot:  # no ions or all local pseudopotentials
            return qp.electrons.Wavefunction(basis, coeff=proj)
        # Get harmonics (per l,m):
        l_max = max(ps.l_max for ps in self.pseudopotentials)
        Ylm_t = qp.ions.spherical_harmonics.get_harmonics_t(l_max, Gk)
        # Get per-atom translations:
        translations = (self.translation_phase(iGk).transpose(1, 2)  # k,atom,G
                        / np.sqrt(basis.lattice.volume))  # due to factor in C
        # Compute projectors by species:
        i_proj_start = 0
        for i_ps, ps in enumerate(self.pseudopotentials):
            # Select projectors (beta) or orbitals (psi) as requested:
            pqn = ps.pqn_psi if get_psi else ps.pqn_beta
            f_t_coeff = (ps.psi if get_psi else ps.beta).f_t_coeff
            # Current range:
            n_proj_cur = pqn.n_tot * self.n_ions_type[i_ps]
            i_proj_stop = i_proj_start + n_proj_cur
            # Compute atomic template:
            proj_atom = (Ginterp(f_t_coeff)[pqn.i_rf] * Ylm_t[pqn.i_lm]
                         ).transpose(0, 1)[:, None]  # k,1,i_proj,G
            # Repeat by translation to each atom:
            trans_cur = translations[:, self.slices[i_ps], None]  # k,atom,1,G
            proj[0, :, i_proj_start:i_proj_stop, 0] = (proj_atom * trans_cur
                                                       ).flatten(1, 2)
            # Prepare for next species:
            i_proj_start = i_proj_stop
        proj[basis.pad_index_mine] = 0.  # project out padded entries
        return qp.electrons.Wavefunction(basis, coeff=proj)

    def get_atomic_orbitals(self, basis: 'Basis') -> 'Wavefunction':
        """Get atomic orbitals (across all species) for specified `basis`."""
        psi = self._get_projectors(basis, get_psi=True)
        n_spinor = basis.n_spinor
        if n_spinor == 2:
            # Convert projectors to orbitals with spinor components:
            proj = psi.coeff
            n_spins, nk_mine, _, _, n_basis_each = proj.shape
            n_psi_tot = self.n_atomic_orbitals(n_spinor)
            psi_s = torch.empty((n_spins, nk_mine, n_psi_tot,
                                 n_spinor, n_basis_each),
                                dtype=torch.complex128, device=self.rc.device)
            i_proj_start = 0
            i_psi_start = 0
            for i_ps, ps in enumerate(self.pseudopotentials):
                # Slices of input projectors and output orbitals:
                n_ions_i = self.n_ions_type[i_ps]
                n_proj_each = ps.n_orbital_projectors
                n_psi_each = ps.n_atomic_orbitals(n_spinor)
                i_proj_stop = i_proj_start + n_ions_i * n_proj_each
                i_psi_stop = i_psi_start + n_ions_i * n_psi_each
                proj_cur = proj[:, :, i_proj_start:i_proj_stop, 0]  # no spinor
                psi_cur = psi_s[:, :, i_psi_start:i_psi_stop]  # spinorial
                # Convert projectors to orbitals for this species:
                if ps.is_relativistic:
                    proj_cur = proj_cur.view((nk_mine, n_ions_i,
                                              n_proj_each, n_basis_each))
                    Ylm_to_spin_angle = ps.pqn_psi.get_spin_angle_transform()
                    psi_cur[0] = torch.einsum('kipg, psb -> kibsg', proj_cur,
                                              Ylm_to_spin_angle).flatten(1, 2)
                else:
                    # Repeat twice as pure up and down spinorial orbitals:
                    psi_cur.zero_()
                    for i_spinor in range(n_spinor):
                        psi_cur[:, :, i_spinor::n_spinor, i_spinor] = proj_cur
                # Move to next species:
                i_proj_start = i_proj_stop
                i_psi_start = i_psi_stop
            return qp.electrons.Wavefunction(basis, coeff=psi_s)
        else:
            if basis.n_spins == 1:
                return psi  # no modifications needed compared to projectors
            else:  # basis.n_spins == 2:
                coeff_spin = psi.coeff.tile(2, 1, 1, 1, 1)  # repeat for spin
                return qp.electrons.Wavefunction(basis, coeff=coeff_spin)

    @property
    def n_projectors(self) -> int:
        """Total number of pseudopotential projectors."""
        return sum((ps.n_projectors * self.n_ions_type[i_ps])
                   for i_ps, ps in enumerate(self.pseudopotentials))

    @property
    def n_orbital_projectors(self) -> int:
        """Total number of projectors used to generate atomic orbitals."""
        return sum((ps.n_orbital_projectors * self.n_ions_type[i_ps])
                   for i_ps, ps in enumerate(self.pseudopotentials))

    def n_atomic_orbitals(self, n_spinor: int) -> int:
        """Total number of atomic orbitals. This depends on the number
        of spinorial components `n_spinor`."""
        return sum((ps.n_atomic_orbitals(n_spinor) * self.n_ions_type[i_ps])
                   for i_ps, ps in enumerate(self.pseudopotentials))

    def _collect_ps_matrix(self, n_spinor: int) -> None:
        """Collect pseudopotential matrices across species and atoms.
        Initializes `D_all`."""
        n_proj = self.n_projectors * n_spinor
        self.D_all = torch.zeros((n_proj, n_proj), device=self.rc.device,
                                 dtype=torch.complex128)
        i_proj_start = 0
        for i_ps, ps in enumerate(self.pseudopotentials):
            D_nlms = ps.pqn_beta.expand_matrix(ps.D, n_spinor)
            n_proj_atom = D_nlms.shape[0]
            # Set diagonal block for each atom:
            for i_atom in range(self.n_ions_type[i_ps]):
                i_proj_stop = i_proj_start + n_proj_atom
                slice_cur = slice(i_proj_start, i_proj_stop)
                self.D_all[slice_cur, slice_cur] = D_nlms
                i_proj_start = i_proj_stop

    def get_atomic_density(self, grid: 'Grid',
                           M_tot: torch.Tensor) -> 'FieldH':
        """Get atomic reference density (for LCAO) on `grid`.
        The magnetization mode and overall magnitude is set by `M_tot`."""
        from .quintic_spline import Interpolator
        iG = grid.get_mesh('H').to(torch.double)  # half-space
        G = ((iG @ grid.lattice.Gbasis.T)**2).sum(dim=-1).sqrt()
        Ginterp = Interpolator(G, qp.ions.RadialFunction.DG)
        # Compute magnetization on each atom if needed:
        n_mag = M_tot.shape[0]
        if n_mag:
            if self.M_initial is not None:
                if n_mag == 1:
                    if len(self.M_initial.shape) != 1:
                        raise ValueError('Per-ion magnetization must be a'
                                         ' scalar in non-spinorial mode')
                else:  # n_mag == 3:
                    if len(self.M_initial.shape) != 3:
                        raise ValueError('Per-ion magnetization must be a'
                                         ' 3-vector in spinorial mode')
                M_initial = self.M_initial.view((self.n_ions, n_mag))
            else:
                M_initial = torch.zeros((self.n_ions, n_mag),
                                        device=M_tot.device)
            # Get fractional magnetization of each atom:
            M_frac = M_initial / self.Z[self.types, None]
            if M_tot.norm().item():
                # Correct to match overall magnetization, if specified:
                M_frac += ((M_tot - M_initial.sum(dim=0))/self.Z_tot)[None, :]
            # Make sure fractional magnetization in range:
            M_frac_max = 0.9  # need some minority spin for numerical stability
            M_frac *= (M_frac_max
                       / M_frac.norm(dim=1).clamp(min=M_frac_max))[:, None]

        # Collect density from each atom:
        n_densities = 1 + n_mag
        n = qp.grid.FieldH(grid, shape_batch=(n_densities,))
        for i_type, ps in enumerate(self.pseudopotentials):
            rho_i = Ginterp(ps.rho_atom.f_t_coeff / grid.lattice.volume)
            SF = self.translation_phase(iG, self.slices[i_type])
            n.data[0] += rho_i[0] * SF.sum(dim=-1)
            if n_mag:
                for i_ion, M_ion in enumerate(M_frac[self.slices[i_type]]):
                    n.data[1:] += (rho_i * SF[None, ..., i_ion]
                                   * M_ion.view((n_mag, 1, 1, 1)))
        return n
