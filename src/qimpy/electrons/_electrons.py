from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from ._hamiltonian import _hamiltonian
from typing import Union, Optional, List, cast


class Electrons(qp.Constructable):
    """Electronic subsystem"""
    __slots__ = ('kpoints', 'spin_polarized', 'spinorial', 'n_spins',
                 'n_spinor', 'w_spin', 'fillings',
                 'basis', 'xc', 'diagonalize', 'scf', 'C', '_n_bands_done',
                 'fixed_H', 'lcao', 'eig', 'deig_max',
                 'n_t', 'tau_t', 'V_ks_t', 'V_tau_t')
    kpoints: qp.electrons.Kpoints  #: Set of kpoints (mesh or path)
    spin_polarized: bool  #: Whether calculation is spin-polarized
    spinorial: bool  #: Whether calculation is relativistic / spinorial
    n_spins: int  #: Number of spin channels
    n_spinor: int  #: Number of spinor components
    w_spin: float  #: Spin weight (degeneracy factor)
    fillings: qp.electrons.Fillings  #: Occupation factor / smearing scheme
    basis: qp.electrons.Basis  #: Plane-wave basis for wavefunctions
    xc: qp.electrons.xc.XC  #: Exchange-correlation functional
    diagonalize: qp.electrons.Davidson  #: Hamiltonian diagonalization method
    scf: qp.electrons.SCF  #: Self-consistent field method
    C: qp.electrons.Wavefunction  #: Electronic wavefunctions
    _n_bands_done: int  #: Number of bands in C that have been initialized
    fixed_H: str  #: If given, fix Hamiltonian to checkpoint file of this name
    lcao: Optional[qp.electrons.LCAO]  #: If present, use LCAO initialization
    eig: torch.Tensor  #: Electronic orbital eigenvalues
    deig_max: float  #: Estimate of accuracy of current `eig`
    n_t: qp.grid.FieldH  \
        #: Electron density (and magnetization, if `spin_polarized`)
    tau_t: qp.grid.FieldH  #: KE density (only for meta-GGAs)
    V_ks_t: qp.grid.FieldH  #: Kohn-Sham potential (local part)
    V_tau_t: qp.grid.FieldH  #: KE potential

    hamiltonian = _hamiltonian

    def __init__(self, *, co: qp.ConstructOptions,
                 lattice: qp.lattice.Lattice, ions: qp.ions.Ions,
                 symmetries: qp.symmetries.Symmetries,
                 k_mesh: Optional[Union[dict, qp.electrons.Kmesh]] = None,
                 k_path: Optional[Union[dict, qp.electrons.Kpath]] = None,
                 spin_polarized: bool = False, spinorial: bool = False,
                 fillings: Optional[Union[dict, qp.electrons.Fillings]] = None,
                 basis: Optional[Union[dict, qp.electrons.Basis]] = None,
                 xc: Optional[Union[dict, qp.electrons.xc.XC]] = None,
                 fixed_H: str = '',
                 lcao: Optional[Union[dict, bool, qp.electrons.LCAO]] = None,
                 davidson: Optional[Union[dict, qp.electrons.Davidson]] = None,
                 chefsi:  Optional[Union[dict, qp.electrons.CheFSI]] = None,
                 scf:  Optional[Union[dict, qp.electrons.SCF]] = None) -> None:
        """Initialize from components and/or dictionary of options.

        Parameters
        ----------
        lattice
            Lattice (unit cell) to associate with electronic wave functions
        ions
            Ionic system interacting with the electrons
        symmetries
            Symmetries for k-point reduction and density symmetrization
        k_mesh
            :yaml:`Uniform k-point mesh for Brillouin-zone integration.`
            Specify only one of k_mesh or k_path.
        k_path
            :yaml:`Path of k-points through Brillouin zone.`
            (Usually for band structure calculations.)
            Specify only one of k_mesh or k_path.
        spin_polarized
            :yaml:`Whether system has spin polarization / magnetization.`
            (True if system breaks time reversal symmetry, else False.)
            Spin polarization is treated explicitly with two sets of orbitals
            for up and down spins if spinorial = False, and implicitly by each
            orbital being spinorial if spinorial = True.
        spinorial
            :yaml:`Whether to perform relativistic / spin-orbit calculations.`
            If True, calculations will use 2-component spinorial wavefunctions.
        fillings
            :yaml:`Electron occupations and charge / magnetization control.`
        basis
            :yaml:`Wavefunction basis set (plane waves).`
        xc
            :yaml:`Exchange-correlation functional.`
        fixed_H
            :yaml:`Fix Hamiltonian from checkpoint file of this name.`
            This is useful for band structure calculations along high-symmetry
            k-point paths, or for converging large numners of empty states.
            Default: don't fix Hamiltonian i.e. self-consistent calculation.
        lcao
            :yaml:`Linear combination of atomic orbitals parameters.`
            Set to False to disable and to start with bandwidth-limited
            random numbers instead. (If starting from a checkpoint with
            wavefunctions, this option has no effect.)
        davidson
            :yaml:`Davidson diagonalization of Kohm-Sham Hamiltonian.`
            Specify only one of davidson or chefsi.
        chefsi
            :yaml:`CheFSI diagonalization of Kohm-Sham Hamiltonian.`
            Uses the Chebyshev Filter Subspace Iteration (CheFSI) method,
            which can be advantageous for large number of bands being computed
            in parallel over a large number of processes.
            Specify only one of davidson or chefsi.
        scf
            :yaml:`Self-consistent field (SCF) iteration parameters.`
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
            self._n_bands_done = self.C.read(cast(qp.utils.Checkpoint,
                                                  self.checkpoint_in),
                                             self.path + 'C')
        self.eig = torch.zeros(self.C.coeff.shape[:3], dtype=torch.double,
                               device=rc.device)
        self.deig_max = np.nan  # eigenvalues completely wrong
        if self._checkpoint_has('eig'):
            qp.log.info('Loading band eigenvalues eig')
            if self.fillings.read_band_scalars(cast(qp.utils.Checkpoint,
                                                    self.checkpoint_in),
                                               self.path + 'eig', self.eig
                                               ) == self.fillings.n_bands:
                self.deig_max = np.inf  # not fully wrong, but accuracy unknown
        self.fixed_H = str(fixed_H)

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

    def initialize_wavefunctions(self, system: qp.System) -> None:
        """Initialize wavefunctions to LCAO / random (if not from checkpoint).
        (This needs to happen after ions have been updated in order to get
        atomic orbitals, which in turn depends on electrons.__init__ being
        completed; hence this is outside the __init__.)"""
        n_atomic = 0
        if (self.lcao is not None) and not self._n_bands_done:
            n_atomic = system.ions.n_atomic_orbitals(self.n_spinor)
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

    def initialize_fixed_hamiltonian(self, system: qp.System) -> None:
        """Load density/potential from checkpoint for fixed-H calculation"""
        assert self.fixed_H
        checkpoint_H = qp.utils.Checkpoint(self.fixed_H, rc=self.rc)
        # Read n and V_ks in real space from checkpoint:
        n_densities = self.n_densities
        n = qp.grid.FieldR(system.grid, shape_batch=(n_densities,))
        V_ks = qp.grid.FieldR(system.grid, shape_batch=(n_densities,))
        n.read(checkpoint_H, 'electrons/n')
        V_ks.read(checkpoint_H, 'electrons/V_ks')
        # Store in reciprocal space:
        self.n_t = ~n
        self.V_ks_t = ~V_ks

    def update_density(self, system: qp.System) -> None:
        """Update electron density from wavefunctions and fillings.
        Result is in system grid in reciprocal space."""
        f = self.fillings.f
        C = self.C[:, :, :self.fillings.n_bands]  # ignore extra bands in n
        need_Mvec = (self.spinorial and self.spin_polarized)
        self.n_t = (~(self.basis.collect_density(C, f, need_Mvec
                                                 ))).to(system.grid)
        # TODO: ultrasoft augmentation
        self.n_t.symmetrize()
        self.tau_t = qp.grid.FieldH(system.grid, shape_batch=(0,))
        # TODO: actually compute KE density if required

    def update_potential(self, system: qp.System) -> None:
        """Update density-dependent energy terms and electron potential."""
        # Exchange-correlation contributions:
        system.energy['Exc'], self.V_ks_t, self.V_tau_t = \
            self.xc(self.n_t + system.ions.n_core_t, self.tau_t)
        # Hartree and local contributions:
        rho_t = self.n_t[0]  # total charge density
        VH_t = system.coulomb(rho_t)  # Hartree potential
        self.V_ks_t[0] += system.ions.Vloc_t + VH_t
        system.energy['Ehartree'] = 0.5 * (rho_t ^ VH_t).item()
        system.energy['Eloc'] = (rho_t ^ system.ions.Vloc_t).item()
        self.V_ks_t.symmetrize()

    def update(self, system: qp.System) -> None:
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
        beta_C = self.C.proj[..., :self.fillings.n_bands]
        system.energy['Enl'] = qp.utils.globalreduce.sum(
            ((beta_C.conj() * (system.ions.D_all @ beta_C)).sum(dim=-2)
             * self.basis.w_sk * f).real, self.rc.comm_k)

    def run(self, system: qp.System) -> None:
        """Run any actions specified in the input."""
        if self.fixed_H:
            self.initialize_fixed_hamiltonian(system)
            self.initialize_wavefunctions(system)  # LCAO / randomize
            self.diagonalize()
            self.fillings.update(system.energy)
            # Replace energy with Eband:
            system.energy.clear()
            system.energy['Eband'] = self.diagonalize.get_Eband()
        else:
            self.initialize_wavefunctions(system)  # LCAO / randomize
            self.scf.update(system)
            self.scf.optimize()
        self.output()

    def output(self) -> None:
        """Save any configured outputs (TODO: systematize this)"""
        if isinstance(self.kpoints, qp.electrons.Kpath):
            self.kpoints.plot(self.eig[..., :self.fillings.n_bands],
                              'bandstruct.pdf')

    def _save_checkpoint(self, checkpoint: qp.utils.Checkpoint) -> List[str]:
        n_bands = self.fillings.n_bands
        self.C[:, :, :n_bands].write(checkpoint, self.path + 'C')
        (~self.n_t).write(checkpoint, self.path + 'n')
        (~self.V_ks_t).write(checkpoint, self.path + 'V_ks')
        self.fillings.write_band_scalars(checkpoint, self.path + 'eig',
                                         self.eig)
        return ['C', 'n', 'V_ks', 'eig']
