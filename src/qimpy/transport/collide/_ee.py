"""Microscopic e-e collision operator for the FermiSurface material."""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from qimpy import log, rc, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from . import _kernels

if TYPE_CHECKING:
    from qimpy.transport.material import FermiSurface


class EECollisions(TreeNode):
    """Electron-electron collisions of an isotropic 2D Fermi liquid.

    Linear part: block-diagonal in angular harmonic ``m``; within each
    ``m``, a matrix over the radial (energy) modes computed at
    initialization from the exact kinematic reduction of the screened
    Coulomb collision integral (``rates="exact"``, includes the
    finite-T thermal-shell average and the collinear-log enhancement of
    energy/heat modes), or the leading-order closed form
    ``gamma_m = m*^2 T^2 K_m/(16 pi E_F)`` (``rates="closed_form"``,
    ``Nr = 1`` only).  Number, energy and momentum are conserved exactly
    by construction (null-space projection) and the operator is
    symmetric positive-semidefinite (eigenvalue clipping).

    Nonlinear part (optional): the cubic pure-momentum-exchange vertex
    acting on the ``n = 0`` radial (surface) modes, evaluated in the
    manifestly-finite combination
    ``Phi_dot = A/(4T)^2 [ (g+h) K*(gh) - gh K*(g+h) ]``,
    ``h(phi) = g(phi+pi)``, where ``K*`` is the angular convolution with
    eigenvalues ``K_m`` (even ``m`` only) -- equivalent to the mode-sum
    vertex ``-2A (-1)^t (K_s - K_{s+t})`` of the derivation notes but
    free of its individually-divergent odd-``m`` coefficients.  Evaluated
    on a dealiased angular grid (exact through the cubic order).
    """

    fermi_surface: FermiSurface
    epsilon_bg: float  #: background dielectric constant
    kappa: float  #: 2D Thomas-Fermi screening wavevector
    m_star: float  #: effective mass
    E_F: float  #: Fermi energy
    well_width: float  #: quantum-well width for form factor (0 = ideal 2D)
    nonlinear: bool  #: include cubic momentum-exchange vertex
    rates: str  #: "exact" or "closed_form"
    K: torch.Tensor  #: angular kernels K_m, m = 0 .. 2 M_theta
    L_coeff: torch.Tensor  #: (dim_theta, Nr, Nr) linear decay blocks

    def __init__(
        self,
        *,
        fermi_surface: FermiSurface,
        epsilon_bg: float,
        kappa: float = 0.0,
        m_star: float = 0.0,
        well_width: float = 0.0,
        nonlinear: bool = True,
        cubic_scale: float = 1.0,
        rates: str = "exact",
        n_alpha: int = 4096,
        n_xi: int = 32,
        xi_cut: float = 10.0,
        n_phi: int = 1024,
        n_xi_proj: int = 16,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize microscopic e-e collisions.

        Parameters
        ----------
        epsilon_bg
            :yaml:`Background dielectric constant.`
        kappa
            :yaml:`2D Thomas-Fermi screening wavevector.`
            Defaults to 2 m*/epsilon_bg (degeneracy-2 2DEG).
        m_star
            :yaml:`Effective mass.`  Defaults to kF/vF (parabolic band).
        well_width
            :yaml:`Quantum-well width (a.u.) for the finite-thickness form factor.`
            0 (default) is the ideal zero-thickness 2DEG of the notes.
        nonlinear
            :yaml:`Include the cubic momentum-exchange vertex.`
        cubic_scale
            :yaml:`Overall scale applied to the cubic vertex.`
            The vertex is the Fermi-surface-confined (T = 0 kinematics)
            limit; against the exact nonlinear operator at T/E_F = 0.032
            the thermal-shell dressing is ~0.75 (m=2 self-interaction)
            to ~0.61 (2+2+2 -> 6 pumping).  Default 1 (undressed).
        rates
            :yaml:`Linear rates: "exact" (init-time quadrature) or "closed_form".`
            "closed_form" is leading order in T/E_F (pointwise at the
            Fermi level; the exact thermal-shell rate is ~30% larger for
            the shear mode at T/E_F ~ 0.03) and requires Nr = 1.
        n_alpha
            :yaml:`Quadrature points for the angular kernels K_m.`
        n_xi, xi_cut, n_phi
            :yaml:`Quadrature for exact-rate evaluation.`
            Energy points / cut (units of T) and angular points; the
            collinear structures require n_phi >> E_F/T.
        n_xi_proj
            :yaml:`Radial Galerkin projection points for exact rates.`
        """
        super().__init__()
        fs = fermi_surface
        self.fermi_surface = fs
        self.epsilon_bg = epsilon_bg
        self.m_star = m_star if m_star else fs.kF / fs.vF
        self.kappa = kappa if kappa else 2 * self.m_star / epsilon_bg
        self.E_F = 0.5 * fs.kF**2 / self.m_star
        self.well_width = well_width
        self.nonlinear = nonlinear
        self.cubic_scale = cubic_scale
        self.rates = rates
        self.n_alpha = n_alpha
        self.n_xi, self.xi_cut, self.n_phi = n_xi, xi_cut, n_phi
        self.n_xi_proj = n_xi_proj

        T = fs.T_temp
        t_ratio = T / self.E_F
        log.info("\n--- Initializing e-e collisions (2D Fermi liquid) ---")
        log.info(
            f"m* = {self.m_star:.4g}, E_F = {self.E_F:.4g}, "
            f"kappa = {self.kappa:.4g}, T/E_F = {t_ratio:.4g}"
        )
        if not (0.0 < t_ratio < 0.3):
            raise InvalidInputException(
                f"T/E_F = {t_ratio:.3g} outside the degenerate Fermi-liquid"
                " regime (need 0 < T/E_F < 0.3); set the material T"
            )

        M = fs.M_theta
        dim = fs.angular.dim
        Nr = fs.Nr
        dtype = fs.v.dtype
        device = rc.device

        # Angular kernels (up to 2M for the cubic vertex):
        self.K = _kernels.K_table(
            2 * M,
            kF=fs.kF,
            epsilon_bg=epsilon_bg,
            kappa=self.kappa,
            well_width=well_width,
            n_alpha=n_alpha,
        )
        gamma_cf = _kernels.gamma_linear(
            self.K, m_star=self.m_star, T=T, E_F=self.E_F
        )
        log.info(
            f"K_2 = {self.K[2]:.4g}, gamma_2 (closed form) = {gamma_cf[2]:.4g}"
        )

        # ---- Linear rates ----
        if rates == "closed_form":
            if Nr != 1:
                raise InvalidInputException(
                    "rates='closed_form' supports Nr = 1 only (the radial"
                    " tower requires rates='exact')"
                )
            L = torch.zeros(dim, 1, 1, dtype=torch.float64)
            for m in range(1, M + 1):
                L[2 * m - 1, 0, 0] = L[2 * m, 0, 0] = gamma_cf[m]
        elif rates == "exact":
            L = self._exact_L_blocks(T)
        else:
            raise InvalidInputException(f"Unrecognized {rates=}")
        self.L_coeff = L.to(dtype=dtype, device=device)
        gam2 = self.L_coeff[3, 0, 0] if M >= 2 else None
        if gam2 is not None:
            log.info(
                f"gamma_2 (linear, as applied) = {gam2:.4g}"
                + (
                    f" = {gam2 / gamma_cf[2]:.3f} x closed form"
                    if gamma_cf[2] > 0
                    else ""
                )
            )

        # ---- Cubic vertex setup (dealiased fine angular grid) ----
        if nonlinear:
            from qimpy.transport.material._fermi_surface import AngularBasis

            N2 = -(-(4 * M + 1) // 4) * 4  # multiple of 4, >= 4M+1
            fine = AngularBasis(2 * M, n_quad=N2, dtype=torch.float64)
            kvec2 = torch.zeros(fine.dim, dtype=torch.float64)
            for m in range(1, 2 * M + 1):
                kvec2[2 * m - 1] = kvec2[2 * m] = self.K[m]
            Kop = fine.T_from_modes @ (kvec2[:, None] * fine.T_to_modes)
            self._T_up = fine.T_from_modes[:, :dim].to(dtype=dtype, device=device)
            self._T_down = fine.T_to_modes[:dim, :].to(dtype=dtype, device=device)
            self._Kop = Kop.to(dtype=dtype, device=device)
            self._N2 = N2
            self._cubic_prefac = cubic_scale * _kernels.cubic_prefactor(
                m_star=self.m_star, E_F=self.E_F
            )
            log.info(
                f"Cubic vertex enabled: prefactor = {self._cubic_prefac:.4g},"
                f" dealiased angular grid N = {N2}"
            )

    def _exact_L_blocks(self, T: float) -> torch.Tensor:
        """Exact-kinematics linear blocks, Galerkin in the code's radial basis."""
        fs = self.fermi_surface
        M, Nr = fs.M_theta, fs.Nr
        t_ratio = T / self.E_F

        # Radial basis as polynomials in x = xi/T (exact: it is polynomial):
        xi_c = fs.radial.xi.to(torch.float64).cpu()  # collocation nodes
        Tfm = fs.radial.T_from_modes.to(torch.float64).cpu()  # psi_l(xi_c)
        V = torch.vander(xi_c, Nr, increasing=True)  # (Nr, Nr)
        psi_coeff = torch.linalg.solve(V, Tfm) if Nr > 1 else torch.ones(1, 1, dtype=torch.float64)

        # Fine radial quadrature for the Galerkin projection.  Keep all
        # nodes above the band bottom (xi/T > -1/t for parabolic bands):
        xg, xw = np.polynomial.legendre.leggauss(self.n_xi_proj)
        x_span = max(8.0, float(xi_c.abs().max()) + 2.0) if Nr > 1 else 8.0
        x_span = min(x_span, 0.9 / t_ratio)
        if float(xi_c.abs().max()) >= 0.95 / t_ratio:
            raise InvalidInputException(
                f"Radial truncation xi_max = {float(xi_c.abs().max()):g} T"
                f" reaches the band bottom (E_F/T = {1/t_ratio:g});"
                " reduce xi_max or T"
            )
        x_fine = torch.tensor(x_span * xg, dtype=torch.float64)
        w_fine = torch.tensor(x_span * xw, dtype=torch.float64)
        w_eq = 0.25 / torch.cosh(x_fine / 2) ** 2 / T  # (1/4T) sech^2(x/2)

        # psi values on the fine grid:
        def psi_eval(x):
            res = torch.zeros(x.shape + (Nr,), dtype=torch.float64)
            for p in range(psi_coeff.shape[0] - 1, -1, -1):
                res = res * x[..., None] + psi_coeff[p]
            return res

        Psi = psi_eval(x_fine)  # (n_fine, Nr)
        P = Psi.T * (w_fine * w_eq)  # (Nr, n_fine): <psi_l| . >_w
        G = P @ Psi  # Gram in the fine measure (~ identity)

        log.info(
            f"Computing exact e-e rates: m = 0..{M}, quadrature"
            f" {self.n_xi}^2 x {self.n_phi} x {self.n_xi_proj}"
        )
        R = _kernels.L_blocks(
            x_nodes=x_fine,
            psi_coeff=psi_coeff,
            m_list=range(M + 1),
            kF=fs.kF,
            m_star=self.m_star,
            T=T,
            epsilon_bg=self.epsilon_bg,
            kappa=self.kappa,
            well_width=self.well_width,
            n_xi=self.n_xi,
            xi_cut=self.xi_cut,
            n_phi=self.n_phi,
        )  # (M+1, n_fine, Nr): minus Phi_dot at fine nodes

        Ginv = torch.linalg.inv(G)
        L_m = torch.einsum("ij,jf,mfl->mil", Ginv, P, R)  # (M+1, Nr, Nr)

        # Exact conservation: null-space projection per harmonic, in the
        # code's discrete radial measure; then symmetrize and clip to PSD.
        quad_w = fs.radial.quad_w.to(torch.float64).cpu()
        Ttm_r = fs.radial.T_to_modes.to(torch.float64).cpu()
        nulls: dict[int, list[torch.Tensor]] = {0: [], 1: []}
        ones_c = Ttm_r @ torch.ones(Nr, dtype=torch.float64)
        nulls[0].append(ones_c)  # particle number
        if Nr > 1:
            nulls[0].append(Ttm_r @ xi_c)  # energy
        k_c = Ttm_r @ torch.sqrt(1.0 + t_ratio * xi_c)  # momentum ~ k(xi)
        nulls[1].append(k_c)
        for m in range(M + 1):
            Lm = 0.5 * (L_m[m] + L_m[m].T)
            if m in nulls:
                Vn = torch.stack(nulls[m], dim=1)  # (Nr, n_null)
                Q, _ = torch.linalg.qr(Vn)
                Proj = torch.eye(Nr, dtype=torch.float64) - Q @ Q.T
                Lm = Proj @ Lm @ Proj
            evals, evecs = torch.linalg.eigh(Lm)
            evals = evals.clamp(min=0.0)
            L_m[m] = (evecs * evals) @ evecs.T
        # Scatter per-harmonic blocks onto coefficient ordering
        # (a_0; a_1, b_1; ...; a_M, b_M):
        dim = fs.angular.dim
        L = torch.zeros(dim, Nr, Nr, dtype=torch.float64)
        L[0] = L_m[0]
        for m in range(1, M + 1):
            L[2 * m - 1] = L_m[m]
            L[2 * m] = L_m[m]
        return L

    def a_dot(self, a: torch.Tensor) -> torch.Tensor:
        """Collision contribution to modal coefficients' time derivative.

        ``a``: modal coefficients, shape ``(..., Nr * dim_theta)`` in the
        FermiSurface flattened (radial, angular) ordering.
        """
        fs = self.fermi_surface
        Nr, dim = fs.Nr, fs.angular.dim
        shape_in = a.shape
        a4 = a.reshape(*shape_in[:-1], Nr, dim)
        # Linear: block-diagonal in harmonic, matrix over radial modes:
        out = -torch.einsum("cij,...jc->...ic", self.L_coeff, a4)
        if self.nonlinear:
            a0 = a4[..., 0, :]  # surface (n = 0) modes
            g = a0 @ self._T_up.T  # (..., N2) nodal on fine grid
            h = g.roll(self._N2 // 2, dims=-1)  # phi -> phi + pi (exact)
            s = g + h
            p = g * h
            Ks = s @ self._Kop.T
            Kp = p @ self._Kop.T
            cubic = (s * Kp - p * Ks) @ self._T_down.T  # (..., dim)
            out = out.clone()
            out[..., 0, :] += self._cubic_prefac * cubic
        return out.reshape(shape_in)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["epsilon_bg"] = self.epsilon_bg
        attrs["kappa"] = self.kappa
        attrs["m_star"] = self.m_star
        attrs["well_width"] = self.well_width
        attrs["nonlinear"] = self.nonlinear
        attrs["cubic_scale"] = self.cubic_scale
        attrs["rates"] = self.rates
        attrs["n_alpha"] = self.n_alpha
        attrs["n_xi"] = self.n_xi
        attrs["xi_cut"] = self.xi_cut
        attrs["n_phi"] = self.n_phi
        attrs["n_xi_proj"] = self.n_xi_proj
        return list(attrs.keys())
