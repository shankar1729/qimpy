import qimpy as qp
import numpy as np
import torch
from ._hamiltonian import _hamiltonian
from typing import Union, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import RunConfig
    from ..lattice import Lattice
    from ..ions import Ions
    from ..symmetries import Symmetries
    from ..grid import FieldR
    from .. import System
    from ._kpoints import Kpoints, Kmesh, Kpath
    from ._fillings import Fillings
    from ._basis import Basis
    from ._davidson import Davidson
    from ._chefsi import CheFSI
    from ._wavefunction import Wavefunction


class Electrons:
    """Electronic subsystem"""
    __slots__ = ('rc', 'kpoints', 'spin_polarized', 'spinorial', 'n_spins',
                 'n_spinor', 'w_spin', 'fillings', 'n_bands', 'n_bands_extra',
                 'basis', 'diagonalize', 'C', 'eig', 'deig_max', 'V_ks')
    rc: 'RunConfig'  #: Current run configuration
    kpoints: 'Kpoints'  #: Set of kpoints (mesh or path)
    spin_polarized: bool  #: Whether calculation is spin-polarized
    spinorial: bool  #: Whether calculation is relativistic / spinorial
    n_spins: int  #: Number of spin channels
    n_spinor: int  #: Number of spinor components
    w_spin: float  #: Spin weight (degeneracy factor)
    fillings: 'Fillings'  #: Occupation factor / smearing scheme
    n_bands: int  #: Number of bands to calculate
    n_bands_extra: int  #: Number of extra bands during diagonalization
    basis: 'Basis'  #: Plane-wave basis for wavefunctions
    diagonalize: 'Davidson'  #: Hamiltonian diagonalization method
    C: 'Wavefunction'  #: Electronic wavefunctions
    eig: torch.Tensor  #: Electronic orbital eigenvalues
    deig_max: float  #: Estimate of accuracy of current `eig`
    V_ks: 'FieldR'  #: Kohn-Sham potential (local part)

    hamiltonian = _hamiltonian

    def __init__(self, *, rc: 'RunConfig', lattice: 'Lattice', ions: 'Ions',
                 symmetries: 'Symmetries',
                 k_mesh: Optional[Union[dict, 'Kmesh']] = None,
                 k_path: Optional[Union[dict, 'Kpath']] = None,
                 spin_polarized: bool = False, spinorial: bool = False,
                 fillings: Optional[Union[dict, 'Fillings']] = None,
                 basis: Optional[Union[dict, 'Basis']] = None,
                 n_bands: Optional[Union[int, str]] = None,
                 n_bands_extra: Optional[Union[int, str]] = None,
                 davidson: Optional[Union[dict, 'Davidson']] = None,
                 chefsi:  Optional[Union[dict, 'CheFSI']] = None) -> None:
        """Initialize from components and/or dictionary of options.

        Parameters
        ----------
        rc : qimpy.utils.RunConfig
            Current run configuration
        lattice : qimpy.lattice.Lattice
            Lattice (unit cell) to associate with electronic wave functions
        ions : qimpy.ions.Ions
            Ionic system interacting with the electrons
        symmetries : qimpy.symmetries.Symmetries
            Symmetries for k-point reduction and density symmetrization
        k_mesh : qimpy.electrons.Kmesh or dict, optional
            Uniform k-point mesh for Brillouin-zone integration.
            Specify only one of k_mesh or k_path.
            Default: use default qimpy.electrons.Kmesh()
        k_path : qimpy.electrons.Kpath or dict, optional
            Path of k-points through Brillouin zone, typically for band
            structure calculations. Specify only one of k_mesh or k_path.
            Default: None
        spin_polarized : bool, optional
            True, if electronic system has spin polarization / magnetization
            (i.e. breaks time reversal symmetry), else False.
            Spin polarization is treated explicitly with two sets of orbitals
            for up and down spins if spinorial = False, and implicitly by each
            orbital being spinorial if spinorial = True.
            Default: False
        spinorial : bool, optional
            True, if relativistic / spin-orbit calculations which require
            2-component spinorial wavefunctions, else False.
            Default: False
        fillings : qimpy.electrons.Fillings or None, optional
            Electron occupations and charge / chemical potential control.
            Default: use default qimpy.electrons.Fillings()
        basis : qimpy.electrons.Basis or None, optional
            Wavefunction basis set (plane waves).
            Default: use default qimpy.electrons.Basis()
        n_bands : {'x<scale>', 'atomic', int}, default: 'x1.'
            Number of bands, specified as a scale relative to the minimum
            number of bands to accommodate electrons i.e. 'x1.5' implies
            use 1.5 times the minimum number. Alternately, 'atomic' sets
            the number of bands to the number of atomic orbitals. Finally,
            an integer explicitly sets the number of bands.
        n_bands_extra : {'x<scale>', int}, default: 'x0.1'
            Number of extra bands retained by diagonalizers, necessary to
            converge any degenerate subspaces straddling n_bands. This could
            be specified as a multiple of n_bands e.g. 'x0.1' = 0.1 x n_bands,
            or could be specified as an explicit number of extra bands
        davidson : qimpy.electrons.Davidson or dict, optional
            Diagonalize Kohm-Sham Hamiltonian using the Davidson method.
            Specify only one of davidson or chefsi.
            Default: use default qimpy.electrons.Davidson()
        chefsi : qimpy.electrons.CheFSI or dict, optional
            Diagonalize Kohm-Sham Hamiltonian using the Chebyshev Filter
            Subspace Iteration (CheFSI) method.
            Specify only one of davidson or chefsi.
            Default: None
        """
        self.rc = rc
        qp.log.info('\n--- Initializing Electrons ---')

        # Initialize k-points:
        n_options = np.count_nonzero([(k is not None)
                                      for k in (k_mesh, k_path)])
        if n_options == 0:
            k_mesh = {}  # Gamma-only
        if n_options > 1:
            raise ValueError('Cannot use both k-mesh and k-path')
        if k_mesh is not None:
            self.kpoints = qp.construct(
                qp.electrons.Kmesh, k_mesh, 'k_mesh',
                rc=rc, symmetries=symmetries, lattice=lattice)
        if k_path is not None:
            self.kpoints = qp.construct(
                qp.electrons.Kpath, k_path, 'k_path',
                rc=rc, lattice=lattice)

        # Initialize spin:
        self.spin_polarized = spin_polarized
        self.spinorial = spinorial
        # --- set # spinor components, # spin channels and weight
        self.n_spinor = (2 if spinorial else 1)
        self.n_spins = (2 if (spin_polarized and not spinorial) else 1)
        self.w_spin = 2 // (self.n_spins * self.n_spinor)  # spin weight
        qp.log.info(f'n_spins: {self.n_spins}  n_spinor: {self.n_spinor}'
                    f'  w_spin: {self.w_spin}')

        # Initialize fillings:
        self.fillings = qp.construct(
            qp.electrons.Fillings, fillings, 'fillings',
            rc=rc, ions=ions, electrons=self)

        # Determine number of bands:
        if n_bands is None:
            n_bands = 'x1'
        if isinstance(n_bands, int):
            self.n_bands = n_bands
            assert(self.n_bands >= 1)
            n_bands_method = 'explicit'
        else:
            assert isinstance(n_bands, str)
            if n_bands == 'atomic':
                raise NotImplementedError('n_bands from atomic orbitals')
                n_bands_method = 'atomic'
            else:
                assert n_bands.startswith('x')
                n_bands_scale = float(n_bands[1:])
                if n_bands_scale < 1.:
                    raise ValueError('<scale> must be >=1 in n_bands')
                self.n_bands = max(1, int(np.ceil(self.fillings.n_bands_min
                                                  * n_bands_scale)))
                n_bands_method = n_bands[1:] + '*n_bands_min'
        # --- similarly for extra bands:
        if n_bands_extra is None:
            n_bands_extra = 'x0.1'
        if isinstance(n_bands_extra, int):
            self.n_bands_extra = n_bands_extra
            assert(self.n_bands_extra >= 1)
            n_bands_extra_method = 'explicit'
        else:
            assert(isinstance(n_bands_extra, str)
                   and n_bands_extra.startswith('x'))
            n_bands_extra_scale = float(n_bands_extra[1:])
            if n_bands_extra_scale <= 0.:
                raise ValueError('<scale> must be >0 in n_bands_extra')
            self.n_bands_extra = max(1, int(np.ceil(self.n_bands
                                                    * n_bands_extra_scale)))
            n_bands_extra_method = n_bands_extra[1:] + '*n_bands'
        qp.log.info(
            f'n_bands: {self.n_bands} ({n_bands_method})'
            f'  n_bands_extra: {self.n_bands_extra} ({n_bands_extra_method})')

        # Initialize wave-function basis:
        self.basis = qp.construct(
            qp.electrons.Basis, basis, 'basis',
            rc=rc, lattice=lattice, ions=ions, symmetries=symmetries,
            kpoints=self.kpoints, n_spins=self.n_spins, n_spinor=self.n_spinor)

        # Initial wavefunctions:
        qp.log.info('Initializing wavefunctions:'
                    ' bandwidth-limited random numbers')
        self.C = qp.electrons.Wavefunction(self.basis, n_bands=self.n_bands)
        self.C.randomize()
        self.C = self.C.orthonormalize()
        self.eig = torch.zeros(self.C.coeff.shape[:3], dtype=torch.double,
                               device=rc.device)
        self.deig_max = np.inf  # note that eigenvalues are completely wrong!

        # Initialize diagonalizer:
        n_options = np.count_nonzero([(d is not None)
                                      for d in (davidson, chefsi)])
        if n_options == 0:
            davidson = {}
        if n_options > 1:
            raise ValueError('Cannot use both davidson and chefsi')
        if davidson is not None:
            self.diagonalize = qp.construct(
                qp.electrons.Davidson, davidson, 'davidson',
                electrons=self)
        if chefsi is not None:
            self.diagonalize = qp.construct(
                qp.electrons.CheFSI, chefsi, 'chefsi',
                electrons=self)
        qp.log.info('diagonalization: ' + repr(self.diagonalize))

    def update(self, system: 'System') -> None:
        """Update electron density and potential"""
        self.V_ks = ~system.ions.Vloc

    def output(self) -> None:
        """Save any configured outputs (TODO: systematize this)"""
        if isinstance(self.kpoints, qp.electrons.Kpath):
            self.kpoints.plot(self.eig[..., :self.n_bands], 'bandstruct.pdf')
