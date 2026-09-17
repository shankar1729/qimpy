"""k-space representations (numerical backends) for the FermiSurface model.

The physical model (`FermiSurface`) owns the collision physics; a representation
owns the discrete k-grid and the map between stored delta-f and the modal e-e
basis, plus the matching boundary operators and observables.  The model applies
the collision through ``representation.apply_collision(rho, modal_op)``, where
``modal_op`` is the model's representation-agnostic modal operator
``a -> -rates*a + ee.a_dot(a) + k_speed*G(a)``.

Two backends:
  * DeltaK    -- delta-k collocation on (radial x angular) nodes near the Fermi
                 surface, |v| = vF, lab-centred tensor-product transform.  The
                 Fermi-circle / small-drift regime (Nr=1 = pure Fermi circle).
  * Cartesian -- a fixed uniform 2D k-grid, v = k/m*, with a drift-centred
                 per-cell projection onto the SAME modal basis.  The large-drift
                 full-f regime.  (see _cartesian.py)
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import torch

from qimpy import rc, TreeNode
from qimpy.mpi import ProcessGrid
from qimpy.io import CheckpointPath, CheckpointContext

# Shared output schema (same names for every representation, so the checkpoint
# and plotter are backend-independent).  Cell scalars are local fields; fluxes
# are vector moments emitted as face-normal fluxes on edges.
CELL_SCALAR_NAMES = ["density", "energy_density", "temperature",
                     "chemical_potential", "velocity_x", "velocity_y"]
FLUX_NAMES = ["particle_current", "momentum_flux_x", "momentum_flux_y", "energy_flux"]


class KRepresentation(TreeNode):
    """Numerical k-space backend for `FermiSurface`.

    Subclasses build the grid in ``__init__`` and set ``Nk``, ``wk``, ``v``
    (transport velocity, shape ``(Nk, 2)``), then provide ``apply_collision``,
    the boundary operators, and the observables.
    """

    Nk: int
    wk: float
    v: torch.Tensor            # (Nk, 2) transport velocity

    def apply_collision(self, rho: torch.Tensor, modal_op: Callable) -> torch.Tensor:
        raise NotImplementedError

    def get_contactor(self, n: torch.Tensor, **kwargs) -> Callable:
        raise NotImplementedError

    def get_reflector(self, n: torch.Tensor) -> Callable:
        raise NotImplementedError

    # ---- observables: scalars on cells, vector moments as face fluxes ----
    def get_density_weight(self) -> torch.Tensor:   # (Nk,) per-k density measure
        raise NotImplementedError

    def get_cell_scalar_names(self) -> list[str]:
        raise NotImplementedError

    def get_cell_scalars(self, rho: torch.Tensor) -> torch.Tensor:  # (K, n_scalar)
        raise NotImplementedError

    def get_flux_names(self) -> list[str]:
        raise NotImplementedError

    def get_flux_weights(self) -> torch.Tensor:     # (n_flux, Nk) per-channel weight g
        raise NotImplementedError


class DeltaK(KRepresentation):
    """delta-k collocation on (k_r, theta_q); |v| = vF; lab-centred modal transform.

    Storage is ``Nr * N_theta`` collocation points; the modal transform is the
    tensor product of the angular Fourier and radial polynomial bases (the model's
    ``angular``/``radial``).  ``Nr=1`` is the pure Fermi-circle limit.
    """

    def __init__(self, *, fermi_surface, process_grid: ProcessGrid,
                 checkpoint_in: CheckpointPath = CheckpointPath()) -> None:
        super().__init__()
        fs = fermi_surface
        self.fs = fs
        Ntheta = fs.angular.N_theta
        self.Nk = fs.Nr * Ntheta
        self.wk = 1.0
        # per-collocation velocity: |v| = vF for every radial point (shell linearization)
        theta_q = fs.angular.theta
        v_per_theta = fs.vF * torch.stack([torch.cos(theta_q), torch.sin(theta_q)], dim=-1)
        self.v = v_per_theta.repeat(fs.Nr, 1)                # (Nr*N_theta, 2)

    # ---- modal transforms (tensor product of angular and radial) ----
    def to_modes(self, f: torch.Tensor) -> torch.Tensor:
        """Nodal (..., Nr*N_theta) -> modal (..., Nr*(2M+1))."""
        fs = self.fs; Nr = fs.Nr; Ntheta = fs.angular.N_theta
        s = f.shape
        f4 = f.reshape(*s[:-1], Nr, Ntheta)
        a_t = torch.einsum("cq,...rq->...rc", fs.angular.T_to_modes, f4)
        a = torch.einsum("nr,...rc->...nc", fs.radial.T_to_modes, a_t)
        return a.reshape(*s[:-1], Nr * fs.angular.dim)

    def from_modes(self, a: torch.Tensor) -> torch.Tensor:
        """Modal (..., Nr*(2M+1)) -> nodal (..., Nr*N_theta)."""
        fs = self.fs; Nr = fs.Nr; dim_t = fs.angular.dim
        s = a.shape
        a4 = a.reshape(*s[:-1], Nr, dim_t)
        f_r = torch.einsum("rn,...nc->...rc", fs.radial.T_from_modes, a4)
        f = torch.einsum("qc,...rc->...rq", fs.angular.T_from_modes, f_r)
        return f.reshape(*s[:-1], Nr * fs.angular.N_theta)

    def apply_collision(self, rho: torch.Tensor, modal_op: Callable) -> torch.Tensor:
        return self.from_modes(modal_op(self.to_modes(rho)))

    # ---- observables: scalar fields on cells; vector moments as face fluxes ----
    def _density_weight(self) -> torch.Tensor:
        fs = self.fs
        w_r = fs.radial.quad_w / torch.sqrt(fs.radial.quad_w.sum())    # (Nr,)
        one_q = torch.full((fs.angular.N_theta,), 1.0 / fs.angular.N_theta,
                           dtype=w_r.dtype, device=w_r.device)
        return (w_r[:, None] * one_q[None, :]).reshape(-1)             # (Nk,)

    def _eps_node(self) -> torch.Tensor:
        fs = self.fs                                                   # per-node energy
        return (fs.mu + fs.radial.xi * fs.T_temp).repeat_interleave(fs.angular.N_theta)

    def get_density_weight(self) -> torch.Tensor:
        return self._density_weight()                                 # (Nk,)

    def get_cell_scalar_names(self) -> list[str]:
        return CELL_SCALAR_NAMES

    def get_cell_scalars(self, rho: torch.Tensor) -> torch.Tensor:
        # Linearized shell model (delta-f about the uniform Fermi sea): density and
        # energy are DEVIATIONS from the uniform reference, velocity is the linear
        # drift <v delta-f> (the density weight sums to 1), and T/mu are uniform.
        fs = self.fs
        df = rho.reshape(-1, self.Nk)
        n_op = self._density_weight()
        dn = df @ n_op
        dE = df @ (self._eps_node() * n_op)
        ux = df @ (n_op * self.v[:, 0]); uy = df @ (n_op * self.v[:, 1])
        Te = torch.full_like(dn, fs.T_temp); mu = torch.full_like(dn, fs.mu)
        return torch.stack([dn, dE, Te, mu, ux, uy], dim=1)           # (K, 6)

    def get_flux_names(self) -> list[str]:
        return FLUX_NAMES

    def get_flux_weights(self) -> torch.Tensor:
        fs = self.fs
        n_op = self._density_weight()                                 # particle weight
        px = fs.m_star * self.v[:, 0] * n_op                          # x-momentum weight
        py = fs.m_star * self.v[:, 1] * n_op                          # y-momentum weight
        return torch.stack([n_op, px, py, self._eps_node() * n_op], dim=0)  # (4, Nk)

    def get_contactor(self, n: torch.Tensor, **kwargs) -> Callable:
        return _DeltaKContactor(self.fs, n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable:
        return _DeltaKReflector(self.fs, n, self.fs.specularity)


class _DeltaKContactor:
    """Reservoir ghost for the delta-k representation.

    The reservoir is a drifted Fermi-Dirac at (mu + dmu, T) with PHYSICAL inward
    drift velocity ``vD`` (along -n, into the device), i.e. a shell shift
    s(theta) = (dmu - kF vD cos(theta-phi) - m* vD^2/2) / T.

    nonlinear=False (default): the EXACT LINEARIZATION -- nodal storage
        Phi = dmu - kF vD cos(theta-phi), built in modes (m=0 and m=1 only).
    nonlinear=True: the exact Pauli-bounded deviation of the shifted FD,
        Phi(xi, theta) = 2T (1 + cosh xi) [expit(s - xi) - expit(-xi)]
        (at Nr=1 this is the saturating 2T tanh(s/2) contact), band-limited once
        through the representation's modal transform.  Its small-s limit is
        exactly the linear ghost, so the flag changes fidelity, not convention.
    """
    def __init__(self, fs, n: torch.Tensor, *, dmu: float = 0.0, vD: float = 0.0,
                 nonlinear: bool = False) -> None:
        n = n.to(rc.device); Nsel = n.shape[0]
        phi = torch.atan2(n[:, 1], n[:, 0])
        rep = fs.representation
        if not nonlinear:
            dim_t = fs.angular.dim
            cm = torch.zeros((Nsel, fs.Nr, dim_t), device=rc.device, dtype=n.dtype)
            cm[:, 0, 0] = dmu
            cm[:, 0, 1] = -(fs.kF * vD) * torch.cos(phi)
            cm[:, 0, 2] = -(fs.kF * vD) * torch.sin(phi)
            self.rho_contact = rep.from_modes(cm.reshape(Nsel, fs.Nr * dim_t))
        else:
            theta = fs.angular.theta                          # (N_theta,)
            xi = fs.radial.xi                                 # (Nr,)
            m_star = fs.kF / fs.vF
            s = (dmu - fs.kF * vD * torch.cos(theta[None, :] - phi[:, None])
                 - 0.5 * m_star * vD * vD) / fs.T_temp        # (Nsel, N_theta)
            xi_b = xi[None, :, None]                          # (1, Nr, 1)
            s_b = s[:, None, :]                               # (Nsel, 1, N_theta)
            dfe = torch.special.expit(s_b - xi_b) - torch.special.expit(-xi_b)
            Phi = 2.0 * fs.T_temp * (1.0 + torch.cosh(xi_b)) * dfe
            Phi = Phi.reshape(Nsel, fs.Nr * fs.angular.N_theta)
            self.rho_contact = rep.from_modes(rep.to_modes(Phi))   # band-limit once

    def __call__(self, t: float) -> torch.Tensor:
        return self.rho_contact


class _DeltaKReflector:
    """Specular block-rotation per harmonic (fraction s) + diffuse (1-s) refill,
    with mass and tangential-momentum conservation, all in modes."""
    def __init__(self, fs, n: torch.Tensor, specularity: float) -> None:
        n = n.to(rc.device)
        self.fs = fs; self.s = float(specularity)
        self.M_theta = fs.M_theta; self.Nr = fs.Nr
        self.dim_theta = fs.angular.dim; self.N_theta = fs.angular.N_theta
        self.phi = torch.atan2(n[:, 1], n[:, 0])
        self.angle = 2.0 * self.phi + np.pi
        theta = fs.angular.theta
        v_dot_n = fs.vF * (n[:, 0:1] * torch.cos(theta)[None, :]
                           + n[:, 1:2] * torch.sin(theta)[None, :])
        self.adn_pos = v_dot_n.clamp(min=0.0)
        self.adn_neg = v_dot_n.clamp(max=0.0)
        self.w_in = (-self.adn_neg).sum(-1).clamp(min=1e-300)
        self.T_to_rad_0 = fs.radial.T_to_modes[0, :]
        self.psi_0 = fs.radial.T_from_modes[:, 0]

    def _specular_modal(self, a_modal: torch.Tensor) -> torch.Tensor:
        # Per-harmonic block rotation, vectorized over all m at once (a Python
        # loop over m is launch-bound and dominates the per-step cost at large
        # M_theta): out_c + i*out_s = e^{i m angle} (a_c - i a_s) for m = 1..M.
        s = a_modal.shape
        a4 = a_modal.reshape(*s[:-1], self.Nr, self.dim_theta)
        out = a4.clone()
        m = torch.arange(1, self.M_theta + 1, device=a_modal.device,
                         dtype=a_modal.dtype)
        ang = self.angle[:, None] * m                    # (Nsel, M)
        c = torch.cos(ang).unsqueeze(-2)                 # (Nsel, 1, M) -> bcast Nr
        sn = torch.sin(ang).unsqueeze(-2)
        a_c = a4[..., 1::2]                              # (..., Nsel, Nr, M) cos coeffs
        a_s = -a4[..., 2::2]                             # (..., Nsel, Nr, M) -sin coeffs
        out[..., 1::2] = c * a_c - sn * a_s
        out[..., 2::2] = sn * a_c + c * a_s
        return out.reshape(s)

    def __call__(self, uM_dk: torch.Tensor) -> torch.Tensor:
        fs = self.fs; rep = fs.representation
        uM_modal = rep.to_modes(uM_dk)
        spec_dk = rep.from_modes(self._specular_modal(uM_modal))
        s = uM_dk.shape
        uM4 = uM_dk.reshape(*s[:-1], self.Nr, self.N_theta)
        sp4 = spec_dk.reshape(*s[:-1], self.Nr, self.N_theta)
        T_to_r = fs.radial.T_to_modes; T_from_r = fs.radial.T_from_modes
        uM_n_q = torch.einsum("nr,...arq->...anq", T_to_r, uM4)
        sp_n_q = torch.einsum("nr,...arq->...anq", T_to_r, sp4)
        sin_q = (torch.cos(self.phi)[:, None] * torch.sin(fs.angular.theta)[None, :]
                 - torch.sin(self.phi)[:, None] * torch.cos(fs.angular.theta)[None, :])
        beta = (self.adn_neg * sin_q).sum(-1)
        gamma = (self.adn_neg * sin_q ** 2).sum(-1)
        adn_pos = self.adn_pos[:, None, :]; adn_neg = self.adn_neg[:, None, :]
        sin_b = sin_q[:, None, :]; vF = fs.vF
        F_out_mass_n = (adn_pos * uM_n_q).sum(-1)
        F_in_spec_mass_n = (adn_neg * sp_n_q).sum(-1)
        F_out_tang_n = vF * (adn_pos * sin_b * uM_n_q).sum(-1)
        F_in_spec_tang_n = vF * (adn_neg * sin_b * sp_n_q).sum(-1)
        F_M_n = F_out_mass_n + self.s * F_in_spec_mass_n
        F_T_n = F_out_tang_n + F_in_spec_tang_n
        det = -vF * (self.w_in * gamma + beta * beta)
        det_b = det[:, None]
        b1 = -F_M_n; b2 = -self.s * F_T_n
        D = (b1 * (vF * gamma)[:, None] - b2 * beta[:, None]) / det_b
        T = ((-self.w_in)[:, None] * b2 - (vF * beta)[:, None] * b1) / det_b
        D_r = torch.einsum("rn,...an->...ar", T_from_r, D)
        T_r = torch.einsum("rn,...an->...ar", T_from_r, T)
        u_added = D_r.unsqueeze(-1) + T_r.unsqueeze(-1) * sin_q.unsqueeze(-2)
        return self.s * spec_dk + u_added.reshape(s)
