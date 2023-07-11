from __future__ import annotations

import numpy as np
import torch

from qimpy import rc, dft
from qimpy.utils import stopwatch
from qimpy.grid import FieldH
from qimpy.dft import ions
from . import RadialFunction
from .quintic_spline import Interpolator


def update(self: ions.Ions, system: dft.System) -> None:
    """Update ionic potentials, projectors and energy components.
    The grids used for the potentials are derived from `system`,
    and the energy components are stored within `system.E`.
    """
    grid = system.grid
    n_densities = system.electrons.n_densities
    self.rho_tilde = FieldH(grid)  # initialize zero ionic charge
    self.Vloc_tilde = FieldH(grid)  # initialize zero local potential
    self.n_core_tilde = FieldH(
        grid, shape_batch=(n_densities,)  # initialize zero core density
    )
    if not self.n_ions:
        nk_mine = system.electrons.kpoints.division.n_mine
        n_basis_mine = system.electrons.basis.division.n_mine
        self.beta = dft.electrons.Wavefunction(
            system.electrons.basis,
            coeff=torch.empty(
                (1, nk_mine, 0, 1, n_basis_mine),
                dtype=torch.complex128,
                device=rc.device,
            ),
        )
        self.D_all = torch.empty((0, 0), dtype=torch.complex128, device=rc.device)
        if system.electrons.need_full_projectors:
            n_basis_tot = system.electrons.basis.n_tot
            self.beta_full = dft.electrons.Wavefunction(
                system.electrons.basis,
                coeff=torch.empty(
                    (1, nk_mine, 0, 1, n_basis_tot),
                    dtype=torch.complex128,
                    device=rc.device,
                ),
            )
        return  # no contributions below if no ions!
    system.energy["Eewald"] = system.coulomb.ewald(self.positions, self.Z[self.types])
    system.energy["Epulay"] = _update_pulay(self, system.electrons.basis)

    # Update ionic densities and potentials:
    _LocalTerms(self, system).update()

    # Update pseudopotential matrix and projectors:
    self._collect_ps_matrix(system.electrons.n_spinor)
    if system.electrons.need_full_projectors:
        beta_full = self._get_projectors(system.electrons.basis, full_basis=True)
        self.beta_full = beta_full
        self.beta = beta_full[..., system.electrons.basis.mine]
    else:
        self.beta = self._get_projectors(system.electrons.basis)
        self.beta_full = None
    self.beta_version += 1  # will auto-invalidate cached projections


def accumulate_geometry_grad(self: ions.Ions, system: dft.System) -> None:
    """Accumulate geometry gradient contributions of total energy.
    Each contribution is accumulated to a `grad` attribute,
    only if the corresponding `requires_grad` is enabled.
    Force contributions are collected in `self.positions.grad`.
    Stress contributions are collected in `system.lattice.grad`.
    Assumes Hellman-Feynman theorem, i.e., electronic system must be converged.
    Note that this invokes `system.electrons.accumulate_geometry_grad`
    as a dependency and therefore includes electronic force / stress contributions.
    """
    # Electronic contributions (direct and through ion-dependent scalar fields, beta):
    self.beta.requires_grad_(True)  # don't zero-initialize to save memory
    self.rho_tilde.requires_grad_(True, clear=True)
    self.Vloc_tilde.requires_grad_(True, clear=True)
    self.n_core_tilde.requires_grad_(True, clear=True)
    system.electrons.accumulate_geometry_grad(system)

    # Ionic contributions:
    if self.n_ions:
        self._projectors_grad(self.beta)
        _LocalTerms(self, system).update_grad()
        system.coulomb.ewald(self.positions, self.Z[self.types])
        if system.lattice.requires_grad and self.dEtot_drho_basis:
            # Pulay stress:
            eye3 = torch.eye(3, device=rc.device)
            system.lattice.grad += (
                self.dEtot_drho_basis
                * system.electrons.basis.n_avg_weighted
                / system.lattice.volume
            ) * eye3

    # Clean up intermediate gradients:
    self.beta.requires_grad_(False, clear=True)
    self.rho_tilde.requires_grad_(False, clear=True)
    self.Vloc_tilde.requires_grad_(False, clear=True)
    self.n_core_tilde.requires_grad_(False, clear=True)

    # Symmetrize:
    assert self.positions.grad is not None
    self.positions.grad = system.symmetries.symmetrize_forces(self.positions.grad)
    if system.lattice.requires_grad:
        system.lattice.grad = system.symmetries.symmetrize_matrix(
            0.5 * (system.lattice.grad + system.lattice.grad.transpose(-2, -1))
        )


class _LocalTerms:
    """
    Handle generation and gradient propagation of ionic scalar fields (local terms).
    """

    @stopwatch(name="Ions.LocalTerms.init")
    def __init__(self, ions: ions.Ions, system: dft.System):
        self.ions = ions
        self.system = system

        # Prepare interpolator for grid:
        grid = system.grid
        self.iG = grid.get_mesh("H").to(torch.double)  # half-space
        G = self.iG @ grid.lattice.Gbasis.T
        Gsq = G.square().sum(dim=-1)
        Gmag = Gsq.sqrt()
        self.Ginterp = Interpolator(Gmag, RadialFunction.DG)

        # Collect structure factor and radial coefficients:
        Vloc_coeff = []
        n_core_coeff = []
        Gmax = grid.get_Gmax()
        ion_width = system.coulomb.ion_width
        for i_type, ps in enumerate(ions.pseudopotentials):
            ps.update(Gmax, ion_width, system.electrons.comm)
            Vloc_coeff.append(ps.Vloc.f_tilde_coeff)
            n_core_coeff.append(ps.n_core.f_tilde_coeff)
        self.Vloc_coeff = torch.hstack(Vloc_coeff)
        self.n_core_coeff = torch.hstack(n_core_coeff)
        self.rho_kernel = -ions.Z.view(-1, 1, 1, 1) * torch.exp(
            (-0.5 * (ion_width**2)) * Gsq
        )

        # Extra requirements for lattice gradient:
        if ions.lattice.requires_grad:
            self.Ginterp_prime = Interpolator(Gmag, RadialFunction.DG, deriv=1)
            self.rho_kernel_prime = self.rho_kernel * (-(ion_width**2)) * Gmag
            G = G.permute(3, 0, 1, 2)  # bring gradient direction to front
            self.stress_kernel = FieldH(
                grid,
                data=(
                    torch.where(Gmag == 0.0, 0.0, -1.0 / Gmag) * G[None] * G[:, None]
                ).to(dtype=torch.cdouble),
            )

    @stopwatch(name="Ions.LocalTerms.update")
    def update(self) -> None:
        """Update ionic densities and potentials."""
        ions = self.ions
        SF = self.get_structure_factor()
        ions.Vloc_tilde.data = (SF * self.Ginterp(self.Vloc_coeff)).sum(dim=0)
        ions.n_core_tilde.data[0] = (SF * self.Ginterp(self.n_core_coeff)).sum(dim=0)
        ions.rho_tilde.data = (SF * self.rho_kernel).sum(dim=0)
        # Add long-range part of local potential from ionic charge:
        ions.Vloc_tilde += self.system.coulomb(ions.rho_tilde, correct_G0_width=True)

    @stopwatch(name="Ions.LocalTerms.update_grad")
    def update_grad(self) -> None:
        """Accumulate local-pseudopotential force / stress contributions."""
        # Propagate long-range local-potential gradient to ionic charge gradient:
        ions = self.ions
        ions.rho_tilde.grad += self.system.coulomb(
            ions.Vloc_tilde.grad, correct_G0_width=True
        )
        if ions.lattice.requires_grad:
            ions.lattice.grad += self.system.coulomb.stress(
                ions.Vloc_tilde.grad, ions.rho_tilde
            )

        # Propagate to structure factor gradient:
        SF_grad = (
            self.Ginterp(self.Vloc_coeff) * ions.Vloc_tilde.grad.data
            + self.Ginterp(self.n_core_coeff) * ions.n_core_tilde.grad.data[0]
            + self.rho_kernel * ions.rho_tilde.grad.data
        )
        # Propagate to ionic gradient:
        self.accumulate_structure_factor_forces(SF_grad)

        if ions.lattice.requires_grad:
            # Propagate to radial function gradient:
            SF = self.get_structure_factor()
            radial_part = (
                self.Ginterp_prime(self.Vloc_coeff) * ions.Vloc_tilde.grad.data
                + self.Ginterp_prime(self.n_core_coeff) * ions.n_core_tilde.grad.data[0]
                + self.rho_kernel_prime * ions.rho_tilde.grad.data
            )
            radial_grad = FieldH(
                self.system.grid, data=(radial_part * SF.conj()).sum(dim=0)
            )
            # Propagate to lattice gradient:
            ions.lattice.grad += radial_grad ^ self.stress_kernel

    def get_structure_factor(self) -> torch.Tensor:
        """Compute structure factor."""
        inv_volume = 1.0 / self.system.lattice.volume
        return torch.stack(
            [
                self.ions.translation_phase(self.iG, slice_i).sum(dim=-1) * inv_volume
                for slice_i in self.ions.slices
            ]
        )

    def accumulate_structure_factor_forces(self, SF_grad: torch.Tensor) -> None:
        """Propagate structure factor gradient to forces."""
        grid = self.system.grid
        pos_grad = self.ions.positions.grad
        assert pos_grad is not None
        inv_volume = 1.0 / grid.lattice.volume
        d_by_dpos = self.iG.permute(3, 0, 1, 2)[None] * (-2j * np.pi * inv_volume)
        for slice_i, SF_grad_i in zip(self.ions.slices, SF_grad):
            phase = self.ions.translation_phase(self.iG, slice_i)
            phase = phase.permute(3, 0, 1, 2)[:, None]  # bring atom dim to front
            dphase_by_dpos = FieldH(grid, data=d_by_dpos * phase)
            pos_grad[slice_i] += FieldH(grid, data=SF_grad_i) ^ dphase_by_dpos


def _collect_ps_matrix(self: ions.Ions, n_spinor: int) -> None:
    """Collect pseudopotential matrices across species and atoms.
    Initializes `D_all`."""
    n_proj = self.n_projectors * n_spinor
    self.D_all = torch.zeros((n_proj, n_proj), device=rc.device, dtype=torch.complex128)
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


def _update_pulay(ions: ions.Ions, basis: dft.electrons.Basis) -> float:
    "Update `ions.dEtot_drho_basis` and return Pulay correction."
    ions.dEtot_drho_basis = sum(
        n_ions_i * ps.dE_drho_basis(basis.ke_cutoff)
        for n_ions_i, ps in zip(ions.n_ions_type, ions.pseudopotentials)
    )
    return (
        ions.dEtot_drho_basis
        * (basis.n_ideal - basis.n_avg_weighted)
        / basis.lattice.volume
    )
