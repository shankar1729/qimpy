from __future__ import annotations
import qimpy as qp
import torch
from .quintic_spline import Interpolator
from typing import Optional, Tuple


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


def update_grad(self: qp.ions.Ions, system: qp.System) -> None:
    """Update ionic gradients in `self.forces`, and optionally in `self.stress`,
    depending on `self.compute_stress`.
    Assumes Hellman-Feynman theorem, i.e., electronic system must be converged.
    The corresponding energy components are stored within `system.E`.
    """
    E_pos = torch.zeros_like(self.forces)  # fractional forces
    E_RRT: Optional[torch.Tensor] = (
        torch.zeros_like(self.stress) if self.compute_stress else None
    )  # lattice derivative of energy (stress * vol)

    # Collect contributions:
    system.coulomb.ewald(self.positions, self.Z[self.types], E_pos, E_RRT)
    update_local_grad(self, system, E_pos, E_RRT)
    update_nonlocal_grad(self, system, E_pos, E_RRT)

    # Store in Cartesian form:
    self.forces = E_pos @ torch.linalg.inv(system.lattice.Rbasis)
    if E_RRT is not None:
        self.stress = E_RRT / system.lattice.volume


def update_local(self: qp.ions.Ions, system: qp.System) -> None:
    """Update ionic densities and potentials."""
    Ginterp, SF, Vloc_coeff, n_core_coeff, ion_gauss = get_local_coeff(self, system)
    self.Vloc_tilde.data = (SF * Ginterp(Vloc_coeff)).sum(dim=0)
    self.n_core_tilde.data[0] = (SF * Ginterp(n_core_coeff)).sum(dim=0)
    self.rho_tilde.data = (-self.Z.view(-1, 1, 1, 1) * SF).sum(dim=0) * ion_gauss
    self.Vloc_tilde += system.coulomb(self.rho_tilde, correct_G0_width=True)


def update_local_grad(
    self: qp.ions.Ions,
    system: qp.System,
    E_pos: torch.Tensor,
    E_RRT: Optional[torch.Tensor],
) -> None:
    """Accumulate local-pseudopotential force / stress contributions."""
    # Compute gradient with respect to ionic potentials / charges:
    self.rho_tilde.requires_grad_(True)
    self.Vloc_tilde.requires_grad_(True)
    self.n_core_tilde.requires_grad_(True)
    system.electrons.update_potential(system, True)


def get_local_coeff(
    self: qp.ions.Ions, system: qp.System
) -> Tuple[Interpolator, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get interpolator, structure factor and radial coefficients,
    shared between `update_local` and `update_local_grad`.
    """
    # Prepare interpolator for grid:
    grid = system.grid
    iG = grid.get_mesh("H").to(torch.double)  # half-space
    Gsq = ((iG @ grid.lattice.Gbasis.T) ** 2).sum(dim=-1)
    G = Gsq.sqrt()
    Ginterp = Interpolator(G, qp.ions.RadialFunction.DG)

    # Collect structure factor and radial coefficients:
    SF = torch.empty(
        (self.n_types,) + G.shape, dtype=torch.cdouble, device=G.device
    )  # structure factor by species
    Vloc_coeff = []
    n_core_coeff = []
    Gmax = grid.get_Gmax()
    ion_width = system.coulomb.ion_width
    inv_volume = 1.0 / grid.lattice.volume
    for i_type, ps in enumerate(self.pseudopotentials):
        ps.update(Gmax, ion_width, system.electrons.comm)
        SF[i_type] = (
            self.translation_phase(iG, self.slices[i_type]).sum(dim=-1) * inv_volume
        )
        Vloc_coeff.append(ps.Vloc.f_tilde_coeff)
        n_core_coeff.append(ps.n_core.f_tilde_coeff)
    ion_gauss = torch.exp((-0.5 * (ion_width ** 2)) * Gsq)
    return Ginterp, SF, torch.hstack(Vloc_coeff), torch.hstack(n_core_coeff), ion_gauss


def update_nonlocal_grad(
    self: qp.ions.Ions,
    system: qp.System,
    E_pos: torch.Tensor,
    E_RRT: Optional[torch.Tensor],
) -> None:
    """Accumulate local-pseudopotential force / stress contributions."""


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
