from __future__ import annotations
from typing import Union, Optional

import numpy as np
import torch

from qimpy import log, rc, TreeNode, dft, MPI
from qimpy.io import CheckpointPath, CheckpointContext, Checkpoint
from qimpy.mpi import ProcessGrid, globalreduce, BufferView
from qimpy.math import abs_squared
from qimpy.lattice import Lattice, Kpoints, Kmesh, Kpath
from qimpy.symmetries import Symmetries
from qimpy.grid import FieldH, FieldR
from qimpy.dft.ions import Ions
from . import Fillings, Basis, Davidson, CheFSI, SCF, LCAO, Wavefunction
from .xc import XC
from ._hamiltonian import _hamiltonian
from qimpy.dft.fluid import LinearPCMFluidModel


class Electrons(TreeNode):
    """Electronic subsystem"""

    comm: MPI.Comm  #: Overall electronic communicator (k-points and bands/basis)
    kpoints: Kpoints  #: Set of kpoints (mesh or path)
    spin_polarized: bool  #: Whether calculation is spin-polarized
    spinorial: bool  #: Whether calculation is relativistic / spinorial
    n_spins: int  #: Number of spin channels
    n_spinor: int  #: Number of spinor components
    w_spin: float  #: Spin weight (degeneracy factor)
    fillings: Fillings  #: Occupation factor / smearing scheme
    basis: Basis  #: Plane-wave basis for wavefunctions
    xc: XC  #: Exchange-correlation functional
    diagonalize: Davidson  #: Hamiltonian diagonalization method
    scf: SCF  #: Self-consistent field method
    C: Wavefunction  #: Electronic wavefunctions
    _n_bands_done: int  #: Number of bands in C that have been initialized
    fixed_H: str  #: If given, fix Hamiltonian to checkpoint file of this name
    save_wavefunction: bool  #: Whether to save wavefunction in checkpoint
    lcao: Optional[LCAO]  #: If present, use LCAO initialization
    eig: torch.Tensor  #: Electronic orbital eigenvalues
    deig_max: float  #: Estimate of accuracy of current `eig`
    n_tilde: FieldH  #: Electron density (and magnetization, if `spin_polarized`)
    tau_tilde: FieldH  #: KE density (only for meta-GGAs)

    hamiltonian = _hamiltonian

    def __init__(
        self,
        *,
        process_grid: ProcessGrid,
        lattice: Lattice,
        ions: Ions,
        symmetries: Symmetries,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        k_mesh: Optional[Union[dict, Kmesh]] = None,
        k_path: Optional[Union[dict, Kpath]] = None,
        spin_polarized: bool = False,
        spinorial: bool = False,
        fillings: Optional[Union[dict, Fillings]] = None,
        basis: Optional[Union[dict, Basis]] = None,
        xc: Optional[Union[dict, XC]] = None,
        fixed_H: str = "",
        save_wavefunction: bool = True,
        lcao: Optional[Union[dict, bool, LCAO]] = None,
        davidson: Optional[Union[dict, Davidson]] = None,
        chefsi: Optional[Union[dict, CheFSI]] = None,
        scf: Optional[Union[dict, SCF]] = None,
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
        log.info("\n--- Initializing Electrons ---")

        # Initialize k-points:
        self.add_child_one_of(
            "kpoints",
            checkpoint_in,
            TreeNode.ChildOptions(
                "k-mesh",
                Kmesh,
                k_mesh,
                process_grid=process_grid,
                symmetries=symmetries,
                lattice=lattice,
            ),
            TreeNode.ChildOptions(
                "k-path",
                Kpath,
                k_path,
                process_grid=process_grid,
                lattice=lattice,
            ),
            have_default=True,
        )
        self.comm = process_grid.get_comm("kb")

        # Initialize spin:
        self.spin_polarized = spin_polarized
        self.spinorial = spinorial
        # --- set # spinor components, # spin channels and weight
        self.n_spinor = 2 if spinorial else 1
        self.n_spins = 2 if (spin_polarized and not spinorial) else 1
        self.w_spin = 2 // (self.n_spins * self.n_spinor)  # spin weight
        log.info(
            f"n_spins: {self.n_spins}  n_spinor: {self.n_spinor}"
            f"  w_spin: {self.w_spin}"
        )

        # Initialize fillings:
        self.add_child(
            "fillings",
            Fillings,
            fillings,
            checkpoint_in,
            ions=ions,
            electrons=self,
        )

        # Initialize wave-function basis:
        self.add_child(
            "basis",
            Basis,
            basis,
            checkpoint_in,
            process_grid=process_grid,
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
            XC,
            xc,
            checkpoint_in,
            spin_polarized=spin_polarized,
        )

        # Initial wavefunctions and eigenvalues:
        self._n_bands_done = 0
        self.C = Wavefunction(self.basis, n_bands=self.fillings.n_bands)
        if cp_C := checkpoint_in.member("C"):
            log.info("Loading wavefunctions C")
            self._n_bands_done = self.C.read(cp_C)
        self.eig = torch.zeros(
            self.C.coeff.shape[:3], dtype=torch.double, device=rc.device
        )
        self.deig_max = np.nan  # eigenvalues completely wrong
        if cp_eig := checkpoint_in.member("eig"):
            log.info("Loading band eigenvalues eig")
            if (
                self.fillings.read_band_scalars(cp_eig, self.eig)
                == self.fillings.n_bands
            ):
                self.deig_max = np.inf  # not fully wrong, but accuracy unknown
        self.fixed_H = str(fixed_H)
        assert self.fixed_H or (not isinstance(self.kpoints, Kpath))
        self.save_wavefunction = bool(save_wavefunction)

        # Initialize LCAO subspace initializer:
        if isinstance(lcao, bool):
            if lcao:
                raise ValueError("lcao must be False or LCAO parameters")
            self.lcao = None
        else:
            self.add_child("lcao", LCAO, lcao, checkpoint_in, comm=self.comm)

        # Initialize diagonalizer:
        self.add_child_one_of(
            "diagonalize",
            checkpoint_in,
            TreeNode.ChildOptions(
                "davidson",
                Davidson,
                davidson,
                electrons=self,
            ),
            TreeNode.ChildOptions(
                "chefsi",
                CheFSI,
                chefsi,
                electrons=self,
            ),
            have_default=True,
        )
        log.info("\nDiagonalization: " + repr(self.diagonalize))

        # Initialize SCF:
        self.add_child("scf", SCF, scf, checkpoint_in, comm=self.comm)

    def initialize_wavefunctions(self, system: dft.System) -> None:
        """Initialize wavefunctions to LCAO / random (if not from checkpoint).
        (This needs to happen after ions have been updated in order to get
        atomic orbitals, which in turn depends on electrons.__init__ being
        completed; hence this is outside the __init__.)"""
        n_atomic = 0
        if (self.lcao is not None) and not self._n_bands_done:
            n_atomic = system.ions.n_atomic_orbitals(self.n_spinor)
            log.info(
                f"Setting {n_atomic} bands of wavefunctions C" " to atomic orbitals"
            )
            if n_atomic < self.C.n_bands():
                self.C[:, :, :n_atomic] = system.ions.get_atomic_orbitals(self.basis)
            else:
                self.C = system.ions.get_atomic_orbitals(self.basis)
            self._n_bands_done = n_atomic
        if self._n_bands_done < self.fillings.n_bands:
            log.info(
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
            log.info("Setting wavefunctions to LCAO eigenvectors")
            assert self.lcao is not None
            fluid_enabled = system.fluid.enabled
            system.fluid.enabled = False
            self.lcao.update(system)
            system.fluid.enabled = fluid_enabled
        else:
            self.C = self.C.orthonormalize()  # For random / checkpoint case

    @property
    def n_densities(self) -> int:
        """Number of electron density / magnetization components in `n`."""
        return (4 if self.spinorial else 2) if self.spin_polarized else 1

    @property
    def need_full_projectors(self) -> bool:
        """Whether full-basis projectors are necessary."""
        return isinstance(self.diagonalize, CheFSI) and (
            self.basis.division.n_procs > 1
        )

    def initialize_fixed_hamiltonian(self, system: dft.System) -> None:
        """Load density/potential from checkpoint for fixed-H calculation"""
        assert self.fixed_H
        cp_H = CheckpointPath(checkpoint=Checkpoint(self.fixed_H), path="/electrons")
        # Read n and V_ks (n.grad) from checkpoint:
        n_densities = self.n_densities
        n = FieldR(system.grid, shape_batch=(n_densities,))
        V_ks = FieldR(system.grid, shape_batch=(n_densities,))
        n.read(cp_H.relative("n"))
        V_ks.read(cp_H.relative("V_ks"))
        self.n_tilde = ~n
        self.n_tilde.grad = ~V_ks
        log.info("  Read n and V_ks.")
        # Optional kinetic density and gradient:
        if self.xc.need_tau:
            tau = FieldR(system.grid, shape_batch=(n_densities,))
            V_tau = FieldR(system.grid, shape_batch=(n_densities,))
            tau.read(cp_H.relative("tau"))
            V_tau.read(cp_H.relative("V_tau"))
            self.tau_tilde = ~tau
            self.tau_tilde.grad = ~V_tau
            log.info("  Read tau and V_tau.")
        else:
            self.tau_tilde = FieldH(system.grid, shape_batch=(0,))
        # Use mu from checkpoint for fillings:
        self.fillings.mu = cp_H.relative("fillings").attrs["mu"]
        self.fillings.mu_constrain = True  # make sure it's not updated
        log.info(f"  Set mu: {self.fillings.mu}  constrained: True")

    def update_density(self, system: dft.System) -> None:
        """Update electron density from wavefunctions and fillings.
        Result is in system grid in reciprocal space."""
        f = self.fillings.f
        C = self.C[:, :, : self.fillings.n_bands]  # ignore extra bands in n
        need_Mvec = self.spinorial and self.spin_polarized
        self.n_tilde = (~(self.basis.collect_density(C, f, need_Mvec))).to(system.grid)
        # TODO: ultrasoft augmentation
        self.n_tilde.symmetrize()
        if self.xc.need_tau:
            self.tau_tilde = self.n_tilde.zeros_like()
            for i_dir in range(3):
                C_grad = self.basis.apply_gradient(C, i_dir)
                self.tau_tilde.add_(
                    ~(self.basis.collect_density(C_grad, f, need_Mvec)).to(system.grid),
                    alpha=0.5,
                )
            self.tau_tilde.symmetrize()
        else:
            self.tau_tilde = FieldH(system.grid, shape_batch=(0,))

    def update_potential(self, system: dft.System, requires_grad: bool = True) -> None:
        """Update density-dependent energy terms and electron potential.
        If `requires_grad` is False, only compute the energy (skip the potentials)."""
        self.n_tilde.requires_grad_(requires_grad, clear=True)
        self.tau_tilde.requires_grad_(requires_grad, clear=True)
        # Exchange-correlation contributions:
        n_xc_tilde = self.n_tilde + system.ions.n_core_tilde
        n_xc_tilde.requires_grad_(requires_grad, clear=True)
        system.energy["Exc"] = self.xc(n_xc_tilde, self.tau_tilde)
        if requires_grad:
            self.n_tilde.grad += n_xc_tilde.grad
        # Hartree and local contributions:
        rho_tilde = self.n_tilde[0]  # total charge density
        VH_tilde = system.coulomb.kernel(rho_tilde)  # Hartree potential
        system.energy["Ehartree"] = 0.5 * (rho_tilde ^ VH_tilde).item()
        system.energy["Eloc"] = (rho_tilde ^ system.ions.Vloc_tilde).item()
        if requires_grad:
            self.n_tilde.grad[0] += system.ions.Vloc_tilde + VH_tilde
            self.n_tilde.grad.symmetrize()

        # Fluid contributions
        if system.fluid.enabled:
            E_fluid, V_fluid, Adiel_rhoExplicitTilde = system.fluid.compute_Adiel_and_potential(self.n_tilde)
            system.energy["Efluid"] = E_fluid
            if requires_grad:
                self.n_tilde.grad[0] += V_fluid

    def update(self, system: dft.System, requires_grad: bool = True) -> None:
        """Update electronic system to current wavefunctions and eigenvalues.
        This updates occupations, density, potential and electronic energy.
        If `requires_grad` is False, only compute the energy (skip the potentials)."""
        self.fillings.update(system.energy)
        self.update_density(system)
        self.update_potential(system, requires_grad)
        f = self.fillings.f
        system.energy["KE"] = globalreduce.sum(
            self.C.band_ke()[:, :, : f.shape[2]] * self.basis.w_sk * f,
            self.kpoints.comm,
        )
        # Nonlocal projector:
        beta_C = self.C.proj[..., : self.fillings.n_bands]
        system.energy["Enl"] = globalreduce.sum(
            (
                (beta_C.conj() * (system.ions.D_all @ beta_C)).sum(dim=-2)
                * self.basis.w_sk
                * f
            ).real,
            self.kpoints.comm,
        )

    def accumulate_geometry_grad(self, system: dft.System) -> None:
        """Accumulate geometry gradient contributions of electronic energy.
        Each contribution is accumulated to a `grad` attribute,
        only if the corresponding `requires_grad` is enabled.
        Force contributions are accumulated to `system.ions.positions.grad`.
        Stress contributions are accumulated to `system.lattice.grad`.
        Gradients with respect to ionic scalar fields are accumulated to
        `system.ions.Vloc_tilde.grad` and `system.ions.n_core_tilde.grad`.
        """
        # Exchange-correlation:
        n_xc_tilde = self.n_tilde + system.ions.n_core_tilde
        n_xc_tilde.requires_grad_(True, clear=True)
        self.tau_tilde.requires_grad_(True, clear=True)
        self.xc(n_xc_tilde, self.tau_tilde)
        if system.ions.n_core_tilde.requires_grad:
            system.ions.n_core_tilde.grad += n_xc_tilde.grad

        # Coulomb / Local pseudootential:
        rho_tilde = self.n_tilde[0]  # total charge density
        if system.lattice.requires_grad:
            system.lattice.grad += 0.5 * system.coulomb.kernel.stress(
                rho_tilde, rho_tilde
            )
        if system.ions.Vloc_tilde.requires_grad:
            system.ions.Vloc_tilde.grad += rho_tilde

        # Kinetic, orthonormality constraint and volume contributions to stress:
        C = self.C[:, :, : self.fillings.n_bands]
        wf = self.fillings.f * self.basis.w_sk
        if system.lattice.requires_grad:
            # Wavefunction squared, with fillings and weights:
            wf_coeff_sq = abs_squared(C.coeff).sum(dim=3)  # sum over spinors
            wf_coeff_sq *= wf.unsqueeze(3)
            if C.basis.real_wavefunctions:
                wf_coeff_sq *= C.basis.real.Gweight_mine.view(1, 1, 1, -1)

            # Kinetic:
            lattice_grad_mine = (
                wf_coeff_sq.sum(dim=(0, 2))[:, :, None, None]
                * C.basis.get_ke_stress(C.basis.mine)
            ).sum(dim=(0, 1))

            # Orthonormality constraint:
            eig = self.eig[..., : self.fillings.n_bands]
            eye3 = torch.eye(3, device=rc.device)
            lattice_grad_mine -= (wf_coeff_sq.sum(dim=-1) * eig).sum() * eye3

            # KE-density contribution for meta-GGAs
            if self.xc.need_tau:
                for i_dir in range(3):
                    HC = C.basis.apply_potential(
                        self.tau_tilde.grad, C.basis.apply_gradient(C, i_dir)
                    )
                    for j_dir in range(i_dir, 3):
                        C_grad = C.basis.apply_gradient(C, j_dir)
                        HC_coeff = HC.coeff
                        C_grad_coeff = C_grad.coeff
                        result = -torch.einsum(
                            "ijk,ijklm->", wf, (HC_coeff * C_grad_coeff.conj()).real
                        )
                        lattice_grad_mine[i_dir, j_dir] += result
                        if i_dir != j_dir:
                            lattice_grad_mine[j_dir, i_dir] += result

            # Collect above local contributions over MPI:
            self.comm.Allreduce(MPI.IN_PLACE, BufferView(lattice_grad_mine))
            system.lattice.grad += lattice_grad_mine

            # Volume contributions:
            system.lattice.grad += eye3 * (
                system.energy["KE"] + system.energy["Ehartree"] + system.energy["Exc"]
            )

        # Nonlocal:
        if system.ions.beta.requires_grad:
            beta_C = C.proj
            beta_C_grad = (system.ions.D_all @ beta_C) * wf[:, :, None]
            if self.n_spinor == 2:
                beta_C_grad = (
                    beta_C_grad.unflatten(2, (-1, 2)).swapaxes(-2, -1).flatten(-2, -1)
                )
            system.ions.beta.grad = C.non_spinor @ beta_C_grad.transpose(-2, -1).conj()

    def run(self, system: dft.System) -> None:
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

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["spin_polarized"] = self.spin_polarized
        attrs["spinorial"] = self.spinorial
        attrs["fixed_H"] = self.fixed_H
        attrs["save_wavefunction"] = self.save_wavefunction
        (~self.n_tilde).write(cp_path.relative("n"))
        (~self.n_tilde.grad).write(cp_path.relative("V_ks"))
        self.fillings.write_band_scalars(cp_path.relative("eig"), self.eig)
        saved_list = list(attrs.keys()) + ["n", "V_ks", "eig"]
        if self.save_wavefunction and (context.stage == "end"):
            n_bands = self.fillings.n_bands
            self.C[:, :, :n_bands].write(cp_path.relative("C"))
            saved_list.append("C")
        if self.xc.need_tau:
            (~self.tau_tilde).write(cp_path.relative("tau"))
            (~self.tau_tilde.grad).write(cp_path.relative("V_tau"))
            saved_list.extend(["tau", "V_tau"])
        if self.lcao is None:
            attrs["lcao"] = False
        return saved_list
