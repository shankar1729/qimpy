import qimpy as qp
import numpy as np
import torch
from ._hamiltonian import _hamiltonian
from typing import Union, Optional, List, cast, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import Checkpoint, RunConfig
    from ..lattice import Lattice
    from ..ions import Ions
    from ..symmetries import Symmetries
    from ..grid import FieldR, FieldH
    from .. import System
    from ._kpoints import Kpoints, Kmesh, Kpath
    from ._fillings import Fillings
    from ._basis import Basis
    from ._davidson import Davidson
    from ._chefsi import CheFSI
    from ._scf import SCF
    from ._wavefunction import Wavefunction
    from ._lcao import LCAO
    from .xc import XC


class Electrons(qp.Constructable):
    """Electronic subsystem"""
    __slots__ = ('kpoints', 'spin_polarized', 'spinorial', 'n_spins',
                 'n_spinor', 'w_spin', 'fillings',
                 'basis', 'xc', 'diagonalize', 'scf', 'C', '_n_bands_done',
                 'lcao', 'eig', 'deig_max', 'n', 'tau', 'V_ks', 'V_tau')
    kpoints: 'Kpoints'  #: Set of kpoints (mesh or path)
    spin_polarized: bool  #: Whether calculation is spin-polarized
    spinorial: bool  #: Whether calculation is relativistic / spinorial
    n_spins: int  #: Number of spin channels
    n_spinor: int  #: Number of spinor components
    w_spin: float  #: Spin weight (degeneracy factor)
    fillings: 'Fillings'  #: Occupation factor / smearing scheme
    basis: 'Basis'  #: Plane-wave basis for wavefunctions
    xc: 'XC'  #: Exchange-correlation functional
    diagonalize: 'Davidson'  #: Hamiltonian diagonalization method
    scf: 'SCF'  #: Self-consistent field method
    C: 'Wavefunction'  #: Electronic wavefunctions
    _n_bands_done: int  #: Number of bands in C that have been initialized
    lcao: Optional['LCAO']  #: If present, use LCAO to initialize C, eig
    eig: torch.Tensor  #: Electronic orbital eigenvalues
    deig_max: float  #: Estimate of accuracy of current `eig`
    n: 'FieldH'  #: Electron density (and magnetization, if `spin_polarized`)
    tau: 'FieldH'  #: KE density (only for meta-GGAs)
    V_ks: 'FieldH'  #: Kohn-Sham potential (local part)
    V_tau: 'FieldH'  #: KE potential

    hamiltonian = _hamiltonian

    def __init__(self, *, co: qp.ConstructOptions,
                 lattice: 'Lattice', ions: 'Ions', symmetries: 'Symmetries',
                 k_mesh: Optional[Union[dict, 'Kmesh']] = None,
                 k_path: Optional[Union[dict, 'Kpath']] = None,
                 spin_polarized: bool = False, spinorial: bool = False,
                 fillings: Optional[Union[dict, 'Fillings']] = None,
                 basis: Optional[Union[dict, 'Basis']] = None,
                 xc: Optional[Union[dict, 'XC']] = None,
                 lcao: Optional[Union[dict, bool, 'LCAO']] = None,
                 davidson: Optional[Union[dict, 'Davidson']] = None,
                 chefsi:  Optional[Union[dict, 'CheFSI']] = None,
                 scf:  Optional[Union[dict, 'SCF']] = None) -> None:
        """Initialize from components and/or dictionary of options.

        Parameters
        ----------
        lattice : qimpy.lattice.Lattice
            Lattice (unit cell) to associate with electronic wave functions
        ions : qimpy.ions.Ions
            Ionic system interacting with the electrons
        symmetries : qimpy.symmetries.Symmetries
            Symmetries for k-point reduction and density symmetrization
        k_mesh : qimpy.electrons.Kmesh or dict, optional
            Uniform k-point mesh for Brillouin-zone integration.
            Specify only one of k_mesh or k_path.
            Default: use default `qimpy.electrons.Kmesh()`
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
        fillings : qimpy.electrons.Fillings or dict, optional
            Electron occupations and charge / chemical potential control.
            Default: use default `qimpy.electrons.Fillings()`
        basis : qimpy.electrons.Basis or dict, optional
            Wavefunction basis set (plane waves).
            Default: use default `qimpy.electrons.Basis()`
        xc : qimpy.electrons.xc.XC or dict, optional
            Exchange-correlation functional.
            Default: use default `qimpy.electrons.xc.XC()`
        lcao: qimpy.electrons.LCAO or dict or False, optional
            Parameters to perform linear combination of atomic orbitals to
            initialize wavefunctions, or False to disable and to start with
            bandwidth-limited random numbers instead. (If starting from a
            checkpoint with wavefunctions, this option has no effect.)
            Default: use default `qimpy.electrons.LCAO()`
        davidson : qimpy.electrons.Davidson or dict, optional
            Diagonalize Kohm-Sham Hamiltonian using the Davidson method.
            Specify only one of davidson or chefsi.
            Default: use default `qimpy.electrons.Davidson()`
        chefsi : qimpy.electrons.CheFSI or dict, optional
            Diagonalize Kohm-Sham Hamiltonian using the Chebyshev Filter
            Subspace Iteration (CheFSI) method.
            Specify only one of davidson or chefsi.
            Default: None
        """
        super().__init__(co=co)
        rc = self.rc
        qp.log.info('\n--- Initializing Electrons ---')

        # Initialize k-points:
        n_options = np.count_nonzero([(k is not None)
                                      for k in (k_mesh, k_path)])
        if n_options == 0:
            k_mesh = {}  # Gamma-only
        if n_options > 1:
            raise ValueError('Cannot use both k-mesh and k-path')
        if k_mesh is not None:
            self.construct('kpoints', qp.electrons.Kmesh, k_mesh,
                           attr_version_name='k-mesh',
                           symmetries=symmetries, lattice=lattice)
        if k_path is not None:
            self.construct('kpoints', qp.electrons.Kpath, k_path,
                           attr_version_name='k-path', lattice=lattice)

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
        self.construct('fillings', qp.electrons.Fillings, fillings,
                       ions=ions, electrons=self)

        # Initialize wave-function basis:
        self.construct('basis', qp.electrons.Basis, basis, lattice=lattice,
                       ions=ions, symmetries=symmetries, kpoints=self.kpoints,
                       n_spins=self.n_spins, n_spinor=self.n_spinor)

        # Initialize exchange-correlation functional:
        self.construct('xc', qp.electrons.xc.XC, xc,
                       spin_polarized=spin_polarized)

        # Initial wavefunctions and eigenvalues:
        self._n_bands_done = 0
        self.C = qp.electrons.Wavefunction(self.basis,
                                           n_bands=self.fillings.n_bands)
        if self._checkpoint_has('C'):
            qp.log.info('Loading wavefunctions C')
            self._n_bands_done = self.C.read(cast('Checkpoint',
                                                  self.checkpoint_in),
                                             self.path + 'C')
        self.eig = torch.zeros(self.C.coeff.shape[:3], dtype=torch.double,
                               device=rc.device)
        self.deig_max = np.nan  # eigenvalues completely wrong
        if self._checkpoint_has('eig'):
            qp.log.info('Loading band eigenvalues eig')
            if self.fillings.read_band_scalars(cast('Checkpoint',
                                                    self.checkpoint_in),
                                               self.path + 'eig', self.eig
                                               ) == self.fillings.n_bands:
                self.deig_max = np.inf  # not fully wrong, but accuracy unknown

        # Initialize LCAO subspace initializer:
        if isinstance(lcao, bool):
            if lcao:
                raise ValueError("lcao must be False or LCAO parameters")
            self.lcao = None
        else:
            self.construct('lcao', qp.electrons.LCAO, lcao)

        # Initialize diagonalizer:
        n_options = np.count_nonzero([(d is not None)
                                      for d in (davidson, chefsi)])
        if n_options == 0:
            davidson = {}
        if n_options > 1:
            raise ValueError('Cannot use both davidson and chefsi')
        if davidson is not None:
            self.construct('diagonalize', qp.electrons.Davidson, davidson,
                           attr_version_name='davidson', electrons=self)
        if chefsi is not None:
            self.construct('diagonalize', qp.electrons.CheFSI, chefsi,
                           attr_version_name='chefsi', electrons=self)
        qp.log.info('\nDiagonalization: ' + repr(self.diagonalize))

        # Initialize SCF:
        self.construct('scf', qp.electrons.SCF, scf, comm=rc.comm_kb)

    def initialize_wavefunctions(self, system: 'System') -> None:
        """Initialize wavefunctions to LCAO / random (if not from checkpoint).
        (This needs to happen after ions have been updated in order to get
        atomic orbitals, which in turn depends on electrons.__init__ being
        completed; hence this is outside the __init__.)"""
        n_atomic = 0
        if (self.lcao is not None) and not self._n_bands_done:
            n_atomic = system.ions.n_atomic_orbitals
            qp.log.info(f'Setting {n_atomic} bands of wavefunctions C'
                        ' to atomic orbitals')
            if n_atomic < self.C.n_bands():
                self.C[:, :, :n_atomic] = \
                    system.ions.get_atomic_orbitals(self.basis)
            else:
                self.C = system.ions.get_atomic_orbitals(self.basis)
            self._n_bands_done = n_atomic
        if self._n_bands_done < self.fillings.n_bands:
            qp.log.info('Randomizing {} bands of wavefunctions C '.format(
                f'{self.fillings.n_bands - self._n_bands_done}'
                if self._n_bands_done else 'all'))
            self.C.randomize(b_start=self._n_bands_done)
            self._n_bands_done = self.C.n_bands()
        # Diagonalize LCAO subspace hamiltonian:
        if n_atomic:
            qp.log.info('Setting wavefunctions to LCAO eigenvectors')
            assert self.lcao is not None
            self.lcao.update(system)
        else:
            self.C = self.C.orthonormalize()  # For random / checkpoint case

    @property
    def n_densities(self) -> int:
        """Number of electron density / magnetization components in `n`."""
        return (4 if self.spinorial else 2) if self.spin_polarized else 1

    def update_density(self, system: 'System') -> None:
        """Update electron density from wavefunctions and fillings.
        Result is in system grid in reciprocal space."""
        f = self.fillings.f
        C = self.C[:, :, :self.fillings.n_bands]  # ignore extra bands in n
        need_Mvec = (self.spinorial and self.spin_polarized)
        self.n = (~(self.basis.collect_density(C, f, need_Mvec
                                               ))).to(system.grid)
        # TODO: ultrasoft augmentation and symmetrization
        self.tau = qp.grid.FieldH(system.grid, shape_batch=(0,))
        # TODO: actually compute KE density if required

    def update_potential(self, system: 'System') -> None:
        """Update density-dependent energy terms and electron potential."""
        # Exchange-correlation contributions:
        system.energy['Exc'], self.V_ks, self.V_tau = \
            self.xc(self.n + system.ions.n_core, self.tau)
        # Hartree and local contributions:
        rho = self.n[0]  # total charge density
        VH = system.coulomb(rho)  # Hartree potential
        self.V_ks[0] += system.ions.Vloc + VH
        system.energy['Ehartree'] = 0.5 * (rho ^ VH).item()
        system.energy['Eloc'] = (rho ^ system.ions.Vloc).item()

    def update(self, system: 'System') -> None:
        """Update electronic system to current wavefunctions and eigenvalues.
        This updates occupations, density, potential and electronic energy."""
        self.fillings.update(system.energy)
        self.update_density(system)
        self.update_potential(system)
        f = self.fillings.f
        system.energy['KE'] = qp.utils.globalreduce.sum(
            self.C.band_ke()[:, :, :f.shape[2]] * self.basis.w_sk * f,
            self.rc.comm_k)
        # Nonlocal projector:
        beta_C = system.ions.beta ^ self.C[:, :, :self.fillings.n_bands]
        system.energy['Enl'] = qp.utils.globalreduce.sum(
            ((beta_C.conj() * (system.ions.D_all @ beta_C)).sum(dim=-2)
             * self.basis.w_sk * f).real, self.rc.comm_k)

    def output(self) -> None:
        """Save any configured outputs (TODO: systematize this)"""
        if isinstance(self.kpoints, qp.electrons.Kpath):
            self.kpoints.plot(self.eig[..., :self.fillings.n_bands],
                              'bandstruct.pdf')

    def _save_checkpoint(self, checkpoint: 'Checkpoint') -> List[str]:
        n_bands = self.fillings.n_bands
        self.C[:, :, :n_bands].write(checkpoint, self.path + 'C')
        self.fillings.write_band_scalars(checkpoint, self.path + 'eig',
                                         self.eig)
        return ['C', 'eig']
