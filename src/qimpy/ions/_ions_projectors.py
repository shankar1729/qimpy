from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from .quintic_spline import Interpolator


def _get_projectors(
    self: qp.ions.Ions,
    basis: qp.electrons.Basis,
    get_psi: bool = False,
    full_basis: bool = False,
) -> qp.electrons.Wavefunction:
    """Get projectors corresponding to specified `basis`.
    If get_psi is True, get atomic orbitals instead. This mode is only for
    internal use by :meth:`get_atomic_orbitals`, which does additional
    transformations on the spin and spinorial dimensions.
    If `full_basis` is True, return projectors or orbitals on the entire basis,
    rather than on the slice of basis local to current process for the default
    case when `full_basis` is False."""
    basis_slice = slice(None) if full_basis else basis.mine
    iGk = basis.iG[:, basis_slice] + basis.k[:, None]  # fractional G + k
    Gk = iGk @ basis.lattice.Gbasis.T  # Cartesian G + k (of this process)
    # Prepare interpolator for radial functions:
    Gk_mag = (Gk ** 2).sum(dim=-1).sqrt()
    Ginterp = Interpolator(Gk_mag, qp.ions.RadialFunction.DG)
    # Prepare output:
    nk_mine, n_basis = Gk_mag.shape
    n_proj_tot = self.n_orbital_projectors if get_psi else self.n_projectors
    proj = torch.empty(
        (1, nk_mine, n_proj_tot, 1, n_basis),
        dtype=torch.complex128,
        device=self.rc.device,
    )
    if not n_proj_tot:  # no ions or all local pseudopotentials
        return qp.electrons.Wavefunction(basis, coeff=proj)
    # Get harmonics (per l,m):
    l_max = max(ps.l_max for ps in self.pseudopotentials)
    Ylm_tilde = qp.ions.spherical_harmonics.get_harmonics_tilde(l_max, Gk)
    # Get per-atom translations:
    translations = self.translation_phase(iGk).transpose(1, 2) / np.sqrt(  # k,atom,G
        basis.lattice.volume
    )  # due to factor in C
    # Compute projectors by species:
    i_proj_start = 0
    for i_ps, ps in enumerate(self.pseudopotentials):
        # Select projectors (beta) or orbitals (psi) as requested:
        pqn = ps.pqn_psi if get_psi else ps.pqn_beta
        f_t_coeff = (ps.psi if get_psi else ps.beta).f_tilde_coeff
        # Current range:
        n_proj_cur = pqn.n_tot * self.n_ions_type[i_ps]
        i_proj_stop = i_proj_start + n_proj_cur
        # Compute atomic template:
        proj_atom = (Ginterp(f_t_coeff)[pqn.i_rf] * Ylm_tilde[pqn.i_lm]).transpose(
            0, 1
        )[
            :, None
        ]  # k,1,i_proj,G
        # Repeat by translation to each atom:
        trans_cur = translations[:, self.slices[i_ps], None]  # k,atom,1,G
        proj[0, :, i_proj_start:i_proj_stop, 0] = (proj_atom * trans_cur).flatten(1, 2)
        # Prepare for next species:
        i_proj_start = i_proj_stop
    # Project out padded entries:
    pad_index = basis.pad_index if full_basis else basis.pad_index_mine
    proj[pad_index] = 0.0
    return qp.electrons.Wavefunction(basis, coeff=proj)