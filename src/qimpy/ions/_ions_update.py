from __future__ import annotations
import qimpy as qp
import torch


def update(self: qp.ions.Ions, system: qp.System) -> None:
    """Update ionic potentials, projectors and energy components.
    The grids used for the potentials are derived from system,
    and the energy components are stored within system.E.
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
    system.energy["Eewald"] = system.coulomb.ewald(self.positions, self.Z[self.types])[
        0
    ]
    # Update ionic densities and potentials:
    from .quintic_spline import Interpolator

    iG = grid.get_mesh("H").to(torch.double)  # half-space
    Gsq = ((iG @ grid.lattice.Gbasis.T) ** 2).sum(dim=-1)
    G = Gsq.sqrt()
    Ginterp = Interpolator(G, qp.ions.RadialFunction.DG)
    SF = torch.empty(
        (self.n_types,) + G.shape, dtype=torch.cdouble, device=G.device
    )  # structure factor by species
    inv_volume = 1.0 / grid.lattice.volume
    # --- collect radial coefficients
    Vloc_coeff = []
    n_core_coeff = []
    Gmax = system.grid.get_Gmax()
    ion_width = system.coulomb.ion_width
    for i_type, ps in enumerate(self.pseudopotentials):
        ps.update(Gmax, ion_width)
        SF[i_type] = (
            self.translation_phase(iG, self.slices[i_type]).sum(dim=-1) * inv_volume
        )
        Vloc_coeff.append(ps.Vloc.f_tilde_coeff)
        n_core_coeff.append(ps.n_core.f_tilde_coeff)
    # --- interpolate to G and collect with structure factors
    self.Vloc_tilde.data = (SF * Ginterp(torch.hstack(Vloc_coeff))).sum(dim=0)
    self.n_core_tilde.data[0] = (SF * Ginterp(torch.hstack(n_core_coeff))).sum(dim=0)
    self.rho_tilde.data = (-self.Z.view(-1, 1, 1, 1) * SF).sum(dim=0) * torch.exp(
        (-0.5 * (ion_width ** 2)) * Gsq
    )
    # --- include long-range electrostatic part of Vloc:
    self.Vloc_tilde += system.coulomb(self.rho_tilde, correct_G0_width=True)

    # Update pseudopotential matrix and projectors:
    self._collect_ps_matrix(system.electrons.n_spinor)
    if system.electrons.need_full_projectors:
        self.beta_full = self._get_projectors(system.electrons.basis, full_basis=True)
        self.beta = self.beta_full[..., system.electrons.basis.mine]
    else:
        self.beta = self._get_projectors(system.electrons.basis)
        self.beta_full = None
    self.beta_version += 1  # will auto-invalidate cached projections


def _collect_ps_matrix(self: qp.ions.Ions, n_spinor: int) -> None:
    """Collect pseudopotential matrices across species and atoms.
    Initializes `D_all`."""
    n_proj = self.n_projectors * n_spinor
    self.D_all = torch.zeros(
        (n_proj, n_proj), device=self.rc.device, dtype=torch.complex128
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
