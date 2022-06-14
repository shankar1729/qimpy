from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from .quintic_spline import Interpolator
from typing import Tuple


def update(self: qp.ions.Ions, system: qp.System) -> None:
    """Update ionic potentials, projectors and energy components.
    The grids used for the potentials are derived from `system`,
    and the energy components are stored within `system.E`.
    """
    grid = system.grid
    n_densities = system.electrons.n_densities
    self.rho_tilde = qp.grid.FieldH(grid)  # initialize zero ionic charge
    self.Vloc_tilde = qp.grid.FieldH(grid)  # initizliae zero local potential
    self.n_core_tilde = qp.grid.FieldH(
        grid, shape_batch=(n_densities,)  # initialize zero core density
    )
    if not self.n_ions:
        return  # no contributions below if no ions!
    system.energy["Eewald"] = system.coulomb.ewald(self.positions, self.Z[self.types])

    # Update ionic densities and potentials:
    update_local(self, system)

    # Update pseudopotential matrix and projectors:
    self._collect_ps_matrix(system.electrons.n_spinor)
    if system.electrons.need_full_projectors:
        self.beta_full = self._get_projectors(system.electrons.basis, full_basis=True)
        self.beta = self.beta_full[..., system.electrons.basis.mine]
    else:
        self.beta = self._get_projectors(system.electrons.basis)
        self.beta_full = None
    self.beta_version += 1  # will auto-invalidate cached projections


def accumulate_geometry_grad(self: qp.ions.Ions, system: qp.System) -> None:
    """Accumulate geometry gradient contributions of total energy.
    Each contribution is accumulated to a `grad` attribute,
    only if the corresponding `requires_grad` is enabled.
    Force contributions are collected in `self.positions.grad`.
    Stress contributions are collected in `system.lattice.grad`.
    Assumes Hellman-Feynman theorem, i.e., electronic system must be converged.
    Note that this invokes `system.electrons.accumulate_geometry_grad`
    as a dependency and therefore includes electronic force / stress contributions.
    """
    # Electronic contributions (direct and through ion-dependent scalar fields):
    self.rho_tilde.requires_grad_(True, clear=True)
    self.Vloc_tilde.requires_grad_(True, clear=True)
    self.n_core_tilde.requires_grad_(True, clear=True)
    system.electrons.accumulate_geometry_grad(system)

    # Ionic contributions:
    update_local_grad(self, system)  # propagate ionic scalar fields
    system.coulomb.ewald(self.positions, self.Z[self.types])


def update_local(ions: qp.ions.Ions, system: qp.System) -> None:
    """Update ionic densities and potentials."""
    Ginterp, iG, Vloc_coeff, n_core_coeff, ion_gauss = get_local_coeff(ions, system)
    SF = get_structure_factor(ions, system.grid, iG)
    ions.Vloc_tilde.data = (SF * Ginterp(Vloc_coeff)).sum(dim=0)
    ions.n_core_tilde.data[0] = (SF * Ginterp(n_core_coeff)).sum(dim=0)
    ions.rho_tilde.data = (-ions.Z.view(-1, 1, 1, 1) * SF).sum(dim=0) * ion_gauss
    # Add long-range part of local potential from ionic charge:
    ions.Vloc_tilde += system.coulomb(ions.rho_tilde, correct_G0_width=True)


def update_local_grad(ions: qp.ions.Ions, system: qp.System) -> None:
    """Accumulate local-pseudopotential force / stress contributions."""
    # Propagate long-range local-potential gradient to ionic charge gradient:
    ions.rho_tilde.grad += system.coulomb(ions.Vloc_tilde.grad, correct_G0_width=True)
    # Propagate to structure factor gradient:
    Ginterp, iG, Vloc_coeff, n_core_coeff, ion_gauss = get_local_coeff(ions, system)
    SF_grad = Ginterp(Vloc_coeff) * ions.Vloc_tilde.grad.data
    SF_grad += Ginterp(n_core_coeff) * ions.n_core_tilde.grad.data[0]
    SF_grad -= (ions.rho_tilde.grad.data * ion_gauss) * ions.Z.view(-1, 1, 1, 1)
    # Propagate to ionic gradient:
    accumulate_structure_factor_forces(ions, system.grid, iG, SF_grad)


def get_local_coeff(
    ions: qp.ions.Ions, system: qp.System
) -> Tuple[Interpolator, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get interpolator and radial coefficients,
    shared between `update_local` and `update_local_grad`.
    """
    # Prepare interpolator for grid:
    grid = system.grid
    iG = grid.get_mesh("H").to(torch.double)  # half-space
    Gsq = ((iG @ grid.lattice.Gbasis.T) ** 2).sum(dim=-1)
    Ginterp = Interpolator(Gsq.sqrt(), qp.ions.RadialFunction.DG)

    # Collect structure factor and radial coefficients:
    Vloc_coeff = []
    n_core_coeff = []
    Gmax = grid.get_Gmax()
    ion_width = system.coulomb.ion_width
    for i_type, ps in enumerate(ions.pseudopotentials):
        ps.update(Gmax, ion_width, system.electrons.comm)
        Vloc_coeff.append(ps.Vloc.f_tilde_coeff)
        n_core_coeff.append(ps.n_core.f_tilde_coeff)
    ion_gauss = torch.exp((-0.5 * (ion_width ** 2)) * Gsq)
    return Ginterp, iG, torch.hstack(Vloc_coeff), torch.hstack(n_core_coeff), ion_gauss


def get_structure_factor(
    ions: qp.ions.Ions, grid: qp.grid.Grid, iG: torch.Tensor
) -> torch.Tensor:
    """Compute structure factor."""
    return torch.stack(
        [
            ions.translation_phase(iG, slice_i).sum(dim=-1) / grid.lattice.volume
            for slice_i in ions.slices
        ]
    )


def accumulate_structure_factor_forces(
    ions: qp.ions.Ions, grid: qp.grid.Grid, iG: torch.Tensor, SF_grad: torch.Tensor
) -> None:
    """Propagate structure factor gradient to forces."""
    pos_grad = ions.positions.grad
    d_by_dpos = iG.permute(3, 0, 1, 2)[None] * (-2j * np.pi / grid.lattice.volume)
    for slice_i, SF_grad_i in zip(ions.slices, SF_grad):
        phase_bcast = ions.translation_phase(iG, slice_i).permute(3, 0, 1, 2)[:, None]
        dphase_by_dpos = qp.grid.FieldH(grid, data=d_by_dpos * phase_bcast)
        pos_grad[slice_i] += qp.grid.FieldH(grid, data=SF_grad_i) ^ dphase_by_dpos


def _collect_ps_matrix(self: qp.ions.Ions, n_spinor: int) -> None:
    """Collect pseudopotential matrices across species and atoms.
    Initializes `D_all`."""
    n_proj = self.n_projectors * n_spinor
    self.D_all = torch.zeros(
        (n_proj, n_proj), device=qp.rc.device, dtype=torch.complex128
    )
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
