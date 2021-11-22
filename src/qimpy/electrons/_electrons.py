from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from ._hamiltonian import _hamiltonian
from typing import Union, Optional, List


class Electrons(qp.TreeNode):
    """Electronic subsystem"""

    __slots__ = (
        "rc",
        "kpoints",
        "spin_polarized",
        "spinorial",
        "n_spins",
        "n_spinor",
        "w_spin",
        "fillings",
        "basis",
        "xc",
        "diagonalize",
        "scf",
        "C",
        "_n_bands_done",
        "fixed_H",
        "save_wavefunction",
        "lcao",
        "eig",
        "deig_max",
        "n_tilde",
        "tau_tilde",
    )
    rc: qp.utils.RunConfig
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
    save_wavefunction: bool  #: Whether to save wavefunction in checkpoint
    lcao: Optional[qp.electrons.LCAO]  #: If present, use LCAO initialization
    eig: torch.Tensor  #: Electronic orbital eigenvalues
    deig_max: float  #: Estimate of accuracy of current `eig`
    n_tilde: qp.grid.FieldH  #: Electron density (and magnetization, if `spin_polarized`)
    tau_tilde: qp.grid.FieldH  #: KE density (only for meta-GGAs)

    hamiltonian = _hamiltonian

    def __init__(
        self,
        *,
        rc: qp.utils.RunConfig,
        lattice: qp.lattice.Lattice,
        ions: qp.ions.Ions,
        symmetries: qp.symmetries.Symmetries,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        k_mesh: Optional[Union[dict, qp.electrons.Kmesh]] = None,
        k_path: Optional[Union[dict, qp.electrons.Kpath]] = None,
        spin_polarized: bool = False,
        spinorial: bool = False,
        fillings: Optional[Union[dict, qp.electrons.Fillings]] = None,
        basis: Optional[Union[dict, qp.electrons.Basis]] = None,
        xc: Optional[Union[dict, qp.electrons.xc.XC]] = None,
        fixed_H: str = "",
        save_wavefunction: bool = True,
        lcao: Optional[Union[dict, bool, qp.electrons.LCAO]] = None,
        davidson: Optional[Union[dict, qp.electrons.Davidson]] = None,
        chefsi: Optional[Union[dict, qp.electrons.CheFSI]] = None,
        scf: Optional[Union[dict, qp.electrons.SCF]] = None,
    ) -> None:
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
        save_wavefunction
            :yaml:`Whether to save wavefunction in checkpoint.`
            Saving the wavefunction is useful for full post-processing
            capability directly from the checkpoint, at the expense of much
            larger checkpoint file. If False, calculations can still use
            the converged density / potential to resume calculations,
            but require an initial non-self-consistent calculation.
            Default: True.
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
        super().__init__()
        self.rc = rc
        qp.log.info("\n--- Initializing Electrons ---")

        # Initialize k-points:
        n_options = np.count_nonzero([(k is not None) for k in (k_mesh, k_path)])
        if n_options == 0:
            k_mesh = {}  # Gamma-only
        if n_options > 1:
            raise ValueError("Cannot use both k-mesh and k-path")
        if k_mesh is not None:
            self.add_child(
                "kpoints",
                qp.electrons.Kmesh,
                k_mesh,
                checkpoint_in,
                attr_version_name="k-mesh",
                rc=rc,
                symmetries=symmetries,
                lattice=lattice,
            )
        if k_path is not None:
            self.add_child(
                "kpoints",
                qp.electrons.Kpath,
                k_path,
                checkpoint_in,
                attr_version_name="k-path",
                rc=rc,
                lattice=lattice,
            )

        # Initialize spin:
        self.spin_polarized = spin_polarized
        self.spinorial = spinorial
        # --- set # spinor components, # spin channels and weight
        self.n_spinor = 2 if spinorial else 1
        self.n_spins = 2 if (spin_polarized and not spinorial) else 1
        self.w_spin = 2 // (self.n_spins * self.n_spinor)  # spin weight
        qp.log.info(
            f"n_spins: {self.n_spins}  n_spinor: {self.n_spinor}"
            f"  w_spin: {self.w_spin}"
        )

        # Initialize fillings:
        self.add_child(
            "fillings",
            qp.electrons.Fillings,
            fillings,
            checkpoint_in,
            rc=rc,
            ions=ions,
            electrons=self,
        )

        # Initialize wave-function basis:
        self.add_child(
            "basis",
            qp.electrons.Basis,
            basis,
            checkpoint_in,
            rc=rc,
            lattice=lattice,
            ions=ions,
            symmetries=symmetries,
            kpoints=self.kpoints,
            n_spins=self.n_spins,
            n_spinor=self.n_spinor,
        )

        # Initialize exchange-correlation functional:
        self.add_child(
            "xc",
            qp.electrons.xc.XC,
            xc,
            checkpoint_in,
            rc=rc,
            spin_polarized=spin_polarized,
        )

        # Initial wavefunctions and eigenvalues:
        self._n_bands_done = 0
        self.C = qp.electrons.Wavefunction(self.basis, n_bands=self.fillings.n_bands)
        if cp_C := checkpoint_in.member("C"):
            qp.log.info("Loading wavefunctions C")
            self._n_bands_done = self.C.read(cp_C)
        self.eig = torch.zeros(
            self.C.coeff.shape[:3], dtype=torch.double, device=rc.device
        )
        self.deig_max = np.nan  # eigenvalues completely wrong
        if cp_eig := checkpoint_in.member("eig"):
            qp.log.info("Loading band eigenvalues eig")
            if (
                self.fillings.read_band_scalars(cp_eig, self.eig)
                == self.fillings.n_bands
            ):
                self.deig_max = np.inf  # not fully wrong, but accuracy unknown
        self.fixed_H = str(fixed_H)
        self.save_wavefunction = bool(save_wavefunction)

        # Initialize LCAO subspace initializer:
        if isinstance(lcao, bool):
            if lcao:
                raise ValueError("lcao must be False or LCAO parameters")
            self.lcao = None
        else:
            self.add_child("lcao", qp.electrons.LCAO, lcao, checkpoint_in, rc=rc)

        # Initialize diagonalizer:
        n_options = np.count_nonzero([(d is not None) for d in (davidson, chefsi)])
        if n_options == 0:
            davidson = {}
        if n_options > 1:
            raise ValueError("Cannot use both davidson and chefsi")
        if davidson is not None:
            self.add_child(
                "diagonalize",
                qp.electrons.Davidson,
                davidson,
                checkpoint_in,
                attr_version_name="davidson",
                rc=rc,
                electrons=self,
            )
        if chefsi is not None:
            self.add_child(
                "diagonalize",
                qp.electrons.CheFSI,
                chefsi,
                checkpoint_in,
                attr_version_name="chefsi",
                rc=rc,
                electrons=self,
            )
        qp.log.info("\nDiagonalization: " + repr(self.diagonalize))

        # Initialize SCF:
        self.add_child(
            "scf", qp.electrons.SCF, scf, checkpoint_in, rc=rc, comm=rc.comm_kb
        )

    def initialize_wavefunctions(self, system: qp.System) -> None:
        """Initialize wavefunctions to LCAO / random (if not from checkpoint).
        (This needs to happen after ions have been updated in order to get
        atomic orbitals, which in turn depends on electrons.__init__ being
        completed; hence this is outside the __init__.)"""
        n_atomic = 0
        if (self.lcao is not None) and not self._n_bands_done:
            n_atomic = system.ions.n_atomic_orbitals(self.n_spinor)
            qp.log.info(
                f"Setting {n_atomic} bands of wavefunctions C" " to atomic orbitals"
            )
            if n_atomic < self.C.n_bands():
                self.C[:, :, :n_atomic] = system.ions.get_atomic_orbitals(self.basis)
            else:
                self.C = system.ions.get_atomic_orbitals(self.basis)
            self._n_bands_done = n_atomic
        if self._n_bands_done < self.fillings.n_bands:
            qp.log.info(
                "Randomizing {} bands of wavefunctions C ".format(
                    f"{self.fillings.n_bands - self._n_bands_done}"
                    if self._n_bands_done
                    else "all"
                )
            )
            self.C.randomize(b_start=self._n_bands_done)
            self._n_bands_done = self.C.n_bands()
        # Diagonalize LCAO subspace hamiltonian:
        if n_atomic:
            qp.log.info("Setting wavefunctions to LCAO eigenvectors")
            assert self.lcao is not None
            self.lcao.update(system)
        else:
            self.C = self.C.orthonormalize()  # For random / checkpoint case

    @property
    def n_densities(self) -> int:
        """Number of electron density / magnetization components in `n`."""
        return (4 if self.spinorial else 2) if self.spin_polarized else 1

    @property
    def need_full_projectors(self) -> bool:
        """Whether full-basis projectors are necessary."""
        return isinstance(self.diagonalize, qp.electrons.CheFSI) and (
            self.basis.division.n_procs > 1
        )

    def initialize_fixed_hamiltonian(self, system: qp.System) -> None:
        """Load density/potential from checkpoint for fixed-H calculation"""
        assert self.fixed_H
        cp_H = qp.utils.CpPath(
            checkpoint=qp.utils.Checkpoint(self.fixed_H, rc=self.rc), path="/electrons"
        )
        # Read n and V_ks (n.grad) in real space from checkpoint:
        n_densities = self.n_densities
        n = qp.grid.FieldR(system.grid, shape_batch=(n_densities,))
        V_ks = qp.grid.FieldR(system.grid, shape_batch=(n_densities,))
        n.read(cp_H.relative("n"))
        V_ks.read(cp_H.relative("V_ks"))
        # Store in reciprocal space:
        self.n_tilde = ~n
        self.n_tilde.grad = ~V_ks
        qp.log.info("  Read n and V_ks.")
        # Use mu from checkpoint for fillings:
        self.fillings.mu = cp_H.relative("fillings").attrs["mu"]
        self.fillings.mu_constrain = True  # make sure it's not updated
        qp.log.info(f"  Set mu: {self.fillings.mu}  constrained: True")

    def update_density(self, system: qp.System) -> None:
        """Update electron density from wavefunctions and fillings.
        Result is in system grid in reciprocal space."""
        f = self.fillings.f
        C = self.C[:, :, : self.fillings.n_bands]  # ignore extra bands in n
        need_Mvec = self.spinorial and self.spin_polarized
        self.n_tilde = (~(self.basis.collect_density(C, f, need_Mvec))).to(system.grid)
        # TODO: ultrasoft augmentation
        self.n_tilde.symmetrize()
        self.tau_tilde = qp.grid.FieldH(system.grid, shape_batch=(0,))
        # TODO: actually compute KE density if required

    def update_potential(self, system: qp.System, requires_grad: bool = True) -> None:
        """Update density-dependent energy terms and electron potential.
        If `requires_grad` is False, only compute the energy (skip the potentials)."""
        self.n_tilde.requires_grad_(requires_grad)
        self.tau_tilde.requires_grad_(requires_grad)
        # Exchange-correlation contributions:
        n_xc_tilde = self.n_tilde + system.ions.n_core_tilde
        n_xc_tilde.requires_grad_(requires_grad)
        system.energy["Exc"] = self.xc(n_xc_tilde, self.tau_tilde)
        if requires_grad:
            self.n_tilde.grad += n_xc_tilde.grad
            if system.ions.n_core_tilde.requires_grad:
                system.ions.n_core_tilde += n_xc_tilde.grad
        # Hartree and local contributions:
        rho_tilde = self.n_tilde[0]  # total charge density
        VH_tilde = system.coulomb(rho_tilde)  # Hartree potential
        system.energy["Ehartree"] = 0.5 * (rho_tilde ^ VH_tilde).item()
        system.energy["Eloc"] = (rho_tilde ^ system.ions.Vloc_tilde).item()
        if requires_grad:
            self.n_tilde.grad[0] += system.ions.Vloc_tilde + VH_tilde
            self.n_tilde.grad.symmetrize()

    def update(self, system: qp.System, requires_grad: bool = True) -> None:
        """Update electronic system to current wavefunctions and eigenvalues.
        This updates occupations, density, potential and electronic energy.
        If `requires_grad` is False, only compute the energy (skip the potentials)."""
        self.fillings.update(system.energy)
        self.update_density(system)
        self.update_potential(system, requires_grad)
        f = self.fillings.f
        system.energy["KE"] = qp.utils.globalreduce.sum(
            self.C.band_ke()[:, :, : f.shape[2]] * self.basis.w_sk * f, self.rc.comm_k
        )
        # Nonlocal projector:
        beta_C = self.C.proj[..., : self.fillings.n_bands]
        system.energy["Enl"] = qp.utils.globalreduce.sum(
            (
                (beta_C.conj() * (system.ions.D_all @ beta_C)).sum(dim=-2)
                * self.basis.w_sk
                * f
            ).real,
            self.rc.comm_k,
        )

    def run(self, system: qp.System) -> None:
        """Run any actions specified in the input."""
        if self.fixed_H:
            self.initialize_fixed_hamiltonian(system)
            self.initialize_wavefunctions(system)  # LCAO / randomize
            self.diagonalize()
            self.fillings.update(system.energy)
            # Replace energy with Eband:
            system.energy.clear()
            system.energy["Eband"] = self.diagonalize.get_Eband()
        else:
            self.initialize_wavefunctions(system)  # LCAO / randomize
            self.scf.update(system)
            self.scf.optimize()
        self.output()

    def output(self) -> None:
        """Save any configured outputs (TODO: systematize this)"""
        if isinstance(self.kpoints, qp.electrons.Kpath):
            self.kpoints.plot(self, "bandstruct.pdf")

    def _save_checkpoint(self, cp_path: qp.utils.CpPath) -> List[str]:
        (~self.n_tilde).write(cp_path.relative("n"))
        (~self.n_tilde.grad).write(cp_path.relative("V_ks"))
        self.fillings.write_band_scalars(cp_path.relative("eig"), self.eig)
        saved_list = ["n", "V_ks", "eig"]
        if self.save_wavefunction:
            n_bands = self.fillings.n_bands
            self.C[:, :, :n_bands].write(cp_path.relative("C"))
            saved_list.append("C")
        return saved_list
