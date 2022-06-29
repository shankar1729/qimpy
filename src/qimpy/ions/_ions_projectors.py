from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from .quintic_spline import Interpolator
from .spherical_harmonics import get_harmonics_tilde, get_harmonics_tilde_and_prime


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
    Gk = iGk @ self.lattice.Gbasis.T  # Cartesian G + k (of this process)
    # Prepare interpolator for radial functions:
    Gk_mag = (Gk ** 2).sum(dim=-1).sqrt()
    Ginterp = Interpolator(Gk_mag, qp.ions.RadialFunction.DG)
    # Prepare output:
    nk_mine, n_basis = Gk_mag.shape
    n_proj_tot = self.n_orbital_projectors if get_psi else self.n_projectors
    proj = torch.empty(
        (1, nk_mine, n_proj_tot, 1, n_basis),
        dtype=torch.complex128,
        device=qp.rc.device,
    )
    if not n_proj_tot:  # no ions or all local pseudopotentials
        return qp.electrons.Wavefunction(basis, coeff=proj)
    # Get harmonics (per l,m):
    l_max = max(ps.l_max for ps in self.pseudopotentials)
    Ylm_tilde = get_harmonics_tilde(l_max, Gk)
    # Get per-atom translations:
    translations = self.translation_phase(iGk).transpose(1, 2)  # k, atom, G
    translations *= 1.0 / np.sqrt(self.lattice.volume)  # due to factor in C
    # Compute projectors by species:
    i_proj_start = 0
    for ps, n_ions_i, slice_i in zip(
        self.pseudopotentials, self.n_ions_type, self.slices
    ):
        # Select projectors (beta) or orbitals (psi) as requested:
        pqn = ps.pqn_psi if get_psi else ps.pqn_beta
        f_t_coeff = (ps.psi if get_psi else ps.beta).f_tilde_coeff
        # Current range:
        n_proj_cur = pqn.n_tot * n_ions_i
        i_proj_stop = i_proj_start + n_proj_cur
        # Compute atomic template:
        proj_atom = Ginterp(f_t_coeff)[pqn.i_rf] * Ylm_tilde[pqn.i_lm]  # i_proj,k,G
        proj_atom = proj_atom.transpose(0, 1)[:, None]  # k,1,i_proj,G
        # Repeat by translation to each atom:
        trans_cur = translations[:, slice_i, None]  # k,atom,1,G
        proj[0, :, i_proj_start:i_proj_stop, 0] = (proj_atom * trans_cur).flatten(1, 2)
        # Prepare for next species:
        i_proj_start = i_proj_stop
    # Project out padded entries:
    pad_index = basis.pad_index if full_basis else basis.pad_index_mine
    proj[pad_index] = 0.0
    return qp.electrons.Wavefunction(basis, coeff=proj)


def _projectors_grad(
    self: qp.ions.Ions, proj: qp.electrons.Wavefunction, is_psi: bool = False
) -> None:
    """Propagate `proj.grad` to forces and stresses.
    Each contribution is accumulated to a `grad` attribute,
    only if the corresponding `requires_grad` is enabled.
    Force contributions are collected in `self.positions.grad`.
    Stress contributions are collected in `self.lattice.grad`.
    """
    basis = proj.basis
    iGk = basis.iG[:, basis.mine] + basis.k[:, None]  # fractional G + k

    # Reduce spin and spinor dimensions in grad (projectors are spin-indep):
    if proj.grad.coeff.shape[0] * proj.grad.coeff.shape[3] != 1:
        proj.grad.coeff = proj.grad.coeff.sum(dim=(0, 3), keepdim=True)

    # Projector forces:
    if self.positions.requires_grad:
        # Combine all terms except for the iG from d/dpos of translation phase:
        # (sum over spin and spinors here, as projectors don't depend on them)
        pp_grad = proj.coeff.conj() * proj.grad.coeff
        d_by_dpos = qp.electrons.Wavefunction(
            basis, coeff=(iGk * (-2j * np.pi)).transpose(1, 2)[None, :, :, None]
        )  # shape like a 3-band wavefunction

        # Collect forces by species:
        pos_grad = torch.zeros_like(self.positions)
        i_proj_start = 0
        for ps, n_ions_i, slice_i in zip(
            self.pseudopotentials, self.n_ions_type, self.slices
        ):
            # Get projector range for this species:
            pqn = ps.pqn_psi if is_psi else ps.pqn_beta
            n_proj_cur = pqn.n_tot * n_ions_i
            i_proj_stop = i_proj_start + n_proj_cur
            # Reduce pp_grad over projectors on each atom:
            pp_grad_cur = qp.electrons.Wavefunction(
                basis,
                coeff=pp_grad[:, :, i_proj_start:i_proj_stop]
                .view(pp_grad.shape[:2] + (n_ions_i, pqn.n_tot) + pp_grad.shape[3:])
                .sum(dim=3),
            )
            # Convert to forces:
            pos_grad[slice_i] += 2.0 * (pp_grad_cur ^ d_by_dpos).wait().real.sum(
                dim=(0, 1)
            )
            # Prepare for next species:
            i_proj_start = i_proj_stop

        basis.kpoints.comm.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(pos_grad))
        self.positions.grad += pos_grad

    # Projector stress:
    if self.lattice.requires_grad:
        # Prepare interpolators for radial functions (and derivatives):
        Gk = iGk @ self.lattice.Gbasis.T  # Cartesian G + k (of this process)
        Gk_mag = Gk.norm(dim=-1)
        Gk_hat = Gk * torch.where(Gk_mag == 0.0, 0.0, 1.0 / Gk_mag).unsqueeze(2)
        Gk_hat = Gk_hat.permute(2, 0, 1)[:, None]  # 3,1,k,G
        Ginterp = Interpolator(Gk_mag, qp.ions.RadialFunction.DG)
        Ginterp_prime = Interpolator(Gk_mag, qp.ions.RadialFunction.DG, deriv=1)
        minus_Gk = qp.electrons.Wavefunction(
            basis, coeff=(-Gk).transpose(1, 2)[None, :, :, None].to(torch.cdouble)
        )  # shape like a 3-band wavefunction

        # Get harmonics (and derivatives):
        l_max = max(ps.l_max for ps in self.pseudopotentials)
        Ylm_tilde, Ylm_tilde_prime = get_harmonics_tilde_and_prime(l_max, Gk)
        # Get per-atom translations:
        translations = self.translation_phase(iGk).transpose(1, 2)  # k,atom,G
        translations *= 1.0 / np.sqrt(self.lattice.volume)  # due to factor in C
        # Collect stresses by species:
        lattice_grad = torch.zeros_like(self.lattice.grad)
        i_proj_start = 0
        for ps, n_ions_i, slice_i in zip(
            self.pseudopotentials, self.n_ions_type, self.slices
        ):
            # Get projector range for this species:
            pqn = ps.pqn_psi if is_psi else ps.pqn_beta
            n_proj_cur = pqn.n_tot * n_ions_i
            i_proj_stop = i_proj_start + n_proj_cur
            p_grad_cur = proj.grad.coeff[:, :, i_proj_start:i_proj_stop]
            # Reduce over atoms using structure factor:
            p_grad_cur = p_grad_cur.unflatten(2, (n_ions_i, -1))  # 1,k,atom,i_proj,1,G
            trans_cur = translations[None, :, slice_i, None, None]  # 1,k,atom,1,1,G
            p_grad_atom = (trans_cur.conj() * p_grad_cur).sum(dim=2)  # 1,k,i_proj,1,G
            # Propagate to derivative w.r.t Gk:
            f_t_coeff = (ps.psi if is_psi else ps.beta).f_tilde_coeff
            proj_atom_prime = (
                Gk_hat * (Ginterp_prime(f_t_coeff)[pqn.i_rf] * Ylm_tilde[pqn.i_lm])
                + Ginterp(f_t_coeff)[pqn.i_rf] * Ylm_tilde_prime[:, pqn.i_lm]
            )  # 3,i_proj,k,G
            Gk_grad = qp.electrons.Wavefunction(
                basis,
                coeff=(
                    p_grad_atom.conj() * proj_atom_prime.transpose(1, 2).unsqueeze(3)
                ).sum(dim=2, keepdim=True),
            )
            # Propagate to lattice derivative:
            lattice_grad += 2.0 * (minus_Gk ^ Gk_grad).wait().real.sum(dim=1).squeeze()
            # Prepare for next species:
            i_proj_start = i_proj_stop

        basis.kpoints.comm.Allreduce(qp.MPI.IN_PLACE, qp.utils.BufferView(lattice_grad))
        self.lattice.grad += lattice_grad
