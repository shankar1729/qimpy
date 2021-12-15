from __future__ import annotations
import qimpy as qp
import torch


def get_atomic_orbitals(
    self: qp.ions.Ions, basis: qp.electrons.Basis
) -> qp.electrons.Wavefunction:
    """Get atomic orbitals (across all species) for specified `basis`."""
    psi = self._get_projectors(basis, get_psi=True)
    n_spinor = basis.n_spinor
    if n_spinor == 2:
        # Convert projectors to orbitals with spinor components:
        proj = psi.coeff
        n_spins, nk_mine, _, _, n_basis_each = proj.shape
        n_psi_tot = self.n_atomic_orbitals(n_spinor)
        psi_s = torch.empty(
            (n_spins, nk_mine, n_psi_tot, n_spinor, n_basis_each),
            dtype=torch.complex128,
            device=qp.rc.device,
        )
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
                proj_cur = proj_cur.view((nk_mine, n_ions_i, n_proj_each, n_basis_each))
                Ylm_to_spin_angle = ps.pqn_psi.get_spin_angle_transform()
                psi_cur[0] = torch.einsum(
                    "kipg, psb -> kibsg", proj_cur, Ylm_to_spin_angle
                ).flatten(1, 2)
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


def get_atomic_density(
    self: qp.ions.Ions, grid: qp.grid.Grid, M_tot: torch.Tensor
) -> qp.grid.FieldH:
    """Get atomic reference density (for LCAO) on `grid`.
    The magnetization mode and overall magnitude is set by `M_tot`."""
    from .quintic_spline import Interpolator

    iG = grid.get_mesh("H").to(torch.double)  # half-space
    G = ((iG @ grid.lattice.Gbasis.T) ** 2).sum(dim=-1).sqrt()
    Ginterp = Interpolator(G, qp.ions.RadialFunction.DG)
    # Compute magnetization on each atom if needed:
    n_mag = M_tot.shape[0]
    if n_mag:
        if self.M_initial is not None:
            if n_mag == 1:
                if len(self.M_initial.shape) != 1:
                    raise ValueError(
                        "Per-ion magnetization must be a"
                        " scalar in non-spinorial mode"
                    )
            else:  # n_mag == 3:
                if len(self.M_initial.shape) != 3:
                    raise ValueError(
                        "Per-ion magnetization must be a" " 3-vector in spinorial mode"
                    )
            M_initial = self.M_initial.view((self.n_ions, n_mag))
        else:
            M_initial = torch.zeros((self.n_ions, n_mag), device=M_tot.device)
        # Get fractional magnetization of each atom:
        M_frac = M_initial / self.Z[self.types, None]
        if M_tot.norm().item():
            # Correct to match overall magnetization, if specified:
            M_frac += ((M_tot - M_initial.sum(dim=0)) / self.Z_tot)[None, :]
        # Make sure fractional magnetization in range:
        M_frac_max = 0.9  # need some minority spin for numerical stability
        M_frac *= (M_frac_max / M_frac.norm(dim=1).clamp(min=M_frac_max))[:, None]

    # Collect density from each atom:
    n_densities = 1 + n_mag
    n = qp.grid.FieldH(grid, shape_batch=(n_densities,))
    for i_type, ps in enumerate(self.pseudopotentials):
        rho_i = Ginterp(ps.rho_atom.f_tilde_coeff / grid.lattice.volume)
        SF = self.translation_phase(iG, self.slices[i_type])
        n.data[0] += rho_i[0] * SF.sum(dim=-1)
        if n_mag:
            for i_ion, M_ion in enumerate(M_frac[self.slices[i_type]]):
                n.data[1:] += (
                    rho_i * SF[None, ..., i_ion] * M_ion.view((n_mag, 1, 1, 1))
                )
    return n
