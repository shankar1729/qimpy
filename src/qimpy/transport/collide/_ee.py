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

    Nonlinear part (optional): the exact finite-temperature reduction of
    ``(B - F)`` beyond linear order, precomputed at initialization from the
    same thermal-shell kinematics as the linear blocks (no free parameter --
    every coefficient comes from the kinematic integral):

    * cubic (``f0``-independent) vertex
      ``C3 = d1 d2 (d3+d4) - d3 d4 (d1+d2)`` on the ``l = 0`` surface
      angular modes (the physical surface-deformation self-interaction),
      projected onto all radial output modes; odd output harmonics are
      gated to zero (parity selection -- odd harmonics do not relax);
    * quadratic particle-hole-odd (thermoelectric) vertex ``Q2``, which
      vanishes on the Fermi surface (``O(T/E_F)``) and couples opposite
      energy parities -- full radial on both inputs and outputs.

    Both are projected onto the conservation null space (number, momentum,
    energy) so that the nonlinear terms conserve exactly, like the linear
    matrix.
    """

    fermi_surface: FermiSurface
    epsilon_bg: float  #: background dielectric constant
    kappa: float  #: 2D Thomas-Fermi screening wavevector
    m_star: float  #: effective mass
    E_F: float  #: Fermi energy
    well_width: float  #: quantum-well width for form factor (0 = ideal 2D)
    nonlinear: bool  #: include the exact cubic and quadratic e-e vertices
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
            :yaml:`Include the exact cubic and quadratic e-e vertices.`
            Both are computed from the thermal-shell kinematics at
            initialization (the finite-T dressing is computed, not scaled);
            there is no free parameter.
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

        # ---- Nonlinear vertices (exact cubic + quadratic) ----
        if nonlinear:
            V_cubic, V_quad = self._nonlinear_vertices(T)
            self._V_cubic = V_cubic.to(dtype=dtype, device=device)
            self._V_quad = V_quad.to(dtype=dtype, device=device)
            log.info(
                "Nonlinear e-e vertices enabled (exact cubic + quadratic):"
                f" cubic {tuple(self._V_cubic.shape)},"
                f" quadratic {tuple(self._V_quad.shape)}"
            )

    def _radial_galerkin(self, T: float):
        """Shared radial Galerkin machinery for the exact-kinematics path.

        Returns ``(psi_coeff, x_fine, P, Ginv, psi0_norm)`` where
        ``psi_coeff[p, l]`` are the power-basis coefficients of the radial
        basis ``psi_l(x) = sum_p psi_coeff[p, l] x^p`` (``x = xi/T``),
        ``x_fine`` the fine Galerkin nodes, ``P[l, f]`` the projection
        covector ``<psi_l | . >_w`` in the fine measure, ``Ginv`` the inverse
        Gram (``~ identity``), and ``psi0_norm`` the constant value of the
        ``l = 0`` radial basis function (the surface mode).
        """
        fs = self.fermi_surface
        Nr = fs.Nr
        t_ratio = T / self.E_F
        xi_c = fs.radial.xi.to(torch.float64).cpu()  # collocation nodes
        Tfm = fs.radial.T_from_modes.to(torch.float64).cpu()  # psi_l(xi_c)
        V = torch.vander(xi_c, Nr, increasing=True)  # (Nr, Nr)
        psi_coeff = (
            torch.linalg.solve(V, Tfm)
            if Nr > 1
            else torch.ones(1, 1, dtype=torch.float64)
        )
        psi0_norm = float(Tfm[0, 0])  # constant l=0 basis value

        # Fine radial quadrature for the Galerkin projection.  Keep all nodes
        # above the band bottom (xi/T > -1/t for parabolic bands):
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

        def psi_eval(x):
            res = torch.zeros(x.shape + (Nr,), dtype=torch.float64)
            for p in range(psi_coeff.shape[0] - 1, -1, -1):
                res = res * x[..., None] + psi_coeff[p]
            return res

        Psi = psi_eval(x_fine)  # (n_fine, Nr)
        P = Psi.T * (w_fine * w_eq)  # (Nr, n_fine): <psi_l| . >_w
        G = P @ Psi  # Gram in the fine measure (~ identity)
        return psi_coeff, x_fine, P, torch.linalg.inv(G), psi0_norm

    def _null_covectors(self, T: float) -> "dict[int, list[torch.Tensor]]":
        """Conservation null covectors per angular harmonic, in the code's
        discrete radial measure (number/energy at m=0, momentum at m=1)."""
        fs = self.fermi_surface
        Nr = fs.Nr
        t_ratio = T / self.E_F
        xi_c = fs.radial.xi.to(torch.float64).cpu()
        Ttm_r = fs.radial.T_to_modes.to(torch.float64).cpu()
        nulls: dict[int, list[torch.Tensor]] = {0: [], 1: []}
        nulls[0].append(Ttm_r @ torch.ones(Nr, dtype=torch.float64))  # number
        if Nr > 1:
            nulls[0].append(Ttm_r @ xi_c)  # energy
        nulls[1].append(Ttm_r @ torch.sqrt(1.0 + t_ratio * xi_c))  # momentum
        return nulls

    def _exact_L_blocks(self, T: float) -> torch.Tensor:
        """Exact-kinematics linear blocks, Galerkin in the code's radial basis."""
        fs = self.fermi_surface
        M, Nr = fs.M_theta, fs.Nr
        t_ratio = T / self.E_F
        xi_c = fs.radial.xi.to(torch.float64).cpu()
        psi_coeff, x_fine, P, Ginv, _ = self._radial_galerkin(T)

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

        L_m = torch.einsum("ij,jf,mfl->mil", Ginv, P, R)  # (M+1, Nr, Nr)

        # Exact conservation: null-space projection per harmonic, in the
        # code's discrete radial measure; then symmetrize and clip to PSD.
        nulls = self._null_covectors(T)
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

    def _radial_null_projectors(self, T: float) -> torch.Tensor:
        """Per-angular-mode radial null projectors ``Proj[co]`` (shape
        ``(dim_theta, Nr, Nr)``) that annihilate the conservation nulls on the
        output radial axis: number/energy at ``m = 0`` (``co = 0``), momentum
        at ``m = 1`` (``co = 1, 2``); identity for all other output modes."""
        fs = self.fermi_surface
        M, Nr = fs.M_theta, fs.Nr
        dim = fs.angular.dim
        nulls = self._null_covectors(T)
        eye = torch.eye(Nr, dtype=torch.float64)
        proj_by_harm: dict[int, torch.Tensor] = {}
        for m, vecs in nulls.items():
            if vecs:
                Vn = torch.stack(vecs, dim=1)  # (Nr, n_null)
                Q, _ = torch.linalg.qr(Vn)
                proj_by_harm[m] = eye - Q @ Q.T
        Proj = torch.zeros(dim, Nr, Nr, dtype=torch.float64)
        Proj[0] = proj_by_harm.get(0, eye)  # m = 0 output mode
        for m in range(1, M + 1):
            Pm = proj_by_harm.get(m, eye)
            Proj[2 * m - 1] = Pm
            Proj[2 * m] = Pm
        return Proj

    def _nonlinear_vertices(self, T: float):
        """Exact cubic and quadratic vertices, Galerkin-projected onto the
        radial modes and null-projected for exact conservation.

        Returns ``(V_cubic, V_quad)``:

        * ``V_cubic[lo, co, a, b, c]`` (shape ``(Nr, dim, dim, dim, dim)``):
          contract with the ``l = 0`` surface coefficients three times to get
          ``a_dot[lo, co]`` (the cubic surface self-interaction);
        * ``V_quad[lo, co, la, a, lb, b]`` (shape
          ``(Nr, dim, Nr, dim, Nr, dim)``): contract with the full modal
          coefficients twice to get the thermoelectric quadratic
          ``a_dot[lo, co]``.
        """
        fs = self.fermi_surface
        M, Nr = fs.M_theta, fs.Nr
        dim = fs.angular.dim
        psi_coeff, x_fine, P, Ginv, psi0_norm = self._radial_galerkin(T)
        kin = dict(
            kF=fs.kF,
            m_star=self.m_star,
            T=T,
            epsilon_bg=self.epsilon_bg,
            kappa=self.kappa,
            well_width=self.well_width,
            n_xi=self.n_xi,
            xi_cut=self.xi_cut,
            n_phi=self.n_phi,
        )
        log.info(
            f"Computing exact nonlinear e-e vertices (M = {M}, Nr = {Nr}),"
            f" quadrature {self.n_xi}^2 x {self.n_phi} x {self.n_xi_proj}"
        )
        # Cubic: V_node[f, co, a, b, c] -> Galerkin project node f onto lo.
        Vc_node = _kernels.cubic_vertex(
            x_nodes=x_fine, psi0_norm=psi0_norm, M=M, **kin
        )
        GP = Ginv @ P  # (Nr, n_fine): radial Galerkin projector
        V_cubic = torch.einsum("lf,fcabd->lcabd", GP, Vc_node)
        # Quadratic: V_node[f, co, la, a, lb, b] -> project node f onto lo.
        Vq_node = _kernels.quadratic_vertex(
            x_nodes=x_fine, psi_coeff=psi_coeff, M=M, **kin
        )
        V_quad = torch.einsum("lf,fcxayb->lcxayb", GP, Vq_node)

        # Exact conservation: project the OUTPUT (lo, co) onto the null
        # complement.  Number/energy gate the m=0 output; momentum gates m=1.
        Proj = self._radial_null_projectors(T)  # (dim, Nr, Nr)
        V_cubic = torch.einsum("cLl,lcabd->Lcabd", Proj, V_cubic)
        V_quad = torch.einsum("cLl,lcxayb->Lcxayb", Proj, V_quad)
        return V_cubic, V_quad

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
            # Cubic surface self-interaction (l = 0 inputs, all radial outputs):
            a0 = a4[..., 0, :]  # surface (n = 0) angular modes
            cubic = torch.einsum(
                "lcabd,...a,...b,...d->...lc", self._V_cubic, a0, a0, a0
            )
            # Quadratic thermoelectric (full modal inputs and outputs):
            quad = torch.einsum(
                "lcxayb,...xa,...yb->...lc", self._V_quad, a4, a4
            )
            out = out + cubic + quad
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
        attrs["rates"] = self.rates
        attrs["n_alpha"] = self.n_alpha
        attrs["n_xi"] = self.n_xi
        attrs["xi_cut"] = self.xi_cut
        attrs["n_phi"] = self.n_phi
        attrs["n_xi_proj"] = self.n_xi_proj
        return list(attrs.keys())
