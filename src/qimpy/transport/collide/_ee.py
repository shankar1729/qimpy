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
      ``C3 = d1 d2 (d3+d4) - d3 d4 (d1+d2)`` over the FULL modal basis --
      every radial/energy mode and angular harmonic on all three input legs
      (complete to cubic order), projected onto all radial output modes.
      The exact integral carries the angular parity selection on its own
      (a purely even input field produces purely even output), so no parity
      gate is imposed; odd output harmonics are kept where the kinematics
      genuinely populate them (a field with odd content relaxes through the
      cubic);
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
            cdtype = (
                torch.complex64 if dtype == torch.float32 else torch.complex128
            )
            verts = self._nonlinear_vertices(T)
            self._Tc_full = verts["Tc_full"].to(dtype=cdtype, device=device)
            self._Tc_leg1 = verts["Tc_leg1"].to(dtype=cdtype, device=device)
            self._Qc_full = verts["Qc_full"].to(dtype=cdtype, device=device)
            self._Qc_leg1 = verts["Qc_leg1"].to(dtype=cdtype, device=device)
            self._build_apply_tables()
            log.info(
                "Nonlinear e-e vertices enabled (exact cubic + quadratic,"
                " complex-harmonic convolution form):"
                f" cubic-full {tuple(self._Tc_full.shape)},"
                f" cubic-leg1 {tuple(self._Tc_leg1.shape)},"
                f" quad-full {tuple(self._Qc_full.shape)},"
                f" quad-leg1 {tuple(self._Qc_leg1.shape)}"
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

    def _null_projectors_by_harm(self, T: float) -> "dict[int, torch.Tensor]":
        """Radial null projectors keyed by complex output harmonic ``mo``.

        Maps ``mo = 0`` to the number/energy projector, ``mo = +-1`` to the
        momentum projector (both annihilating the conservation nulls on the
        output radial axis); every other ``mo`` uses the identity (omitted).
        Equivalent, on the binned complex output ``(lo, mo)``, to the real-basis
        per-output-mode ``Proj[co]`` of ``_radial_null_projectors`` -- the
        radial projector commutes with the (mo -> co) angular mixing of ``R``,
        and ``Proj`` is harmonic-diagonal (co = 0 <-> m = 0; co = 1, 2 <-> m = 1).
        """
        fs = self.fermi_surface
        Nr = fs.Nr
        nulls = self._null_covectors(T)
        proj: dict[int, torch.Tensor] = {}
        for m, vecs in nulls.items():
            if vecs:
                Vn = torch.stack(vecs, dim=1)  # (Nr, n_null)
                Q, _ = torch.linalg.qr(Vn)
                Pm = (torch.eye(Nr, dtype=torch.float64) - Q @ Q.T)
                proj[m] = Pm.to(torch.complex128)
                if m >= 1:  # momentum projector also gates mo = -m
                    proj[-m] = proj[m]
        return proj

    def _nonlinear_vertices(self, T: float):
        """Exact cubic and quadratic vertices as compact complex harmonic
        tensors, Galerkin-projected onto the radial output modes.

        Exploits the rotational selection rule ``mo = sum of input harmonics``
        to drop the redundant output-harmonic axis: instead of the dense rank-8
        real ``V_cubic[lo, co, la, a, lb, b, lc, c]`` we store the complex
        rank-3-angular vertex (``(2M+1)x`` less storage), applied by harmonic
        convolution.  The output leg (leg 1, ``phi1 = 0``) carries a constant
        phase, so the leg-triples / pair-terms containing it are stored without
        a leg-1 harmonic axis (leg-1 reduction).  Returns the dict of stored
        tensors; the per-output-harmonic null projection is deferred to
        ``a_dot`` (small matrices on ``mo in {0, +-1}`` only).
        """
        fs = self.fermi_surface
        M, Nr = fs.M_theta, fs.Nr
        psi_coeff, x_fine, P, Ginv, _ = self._radial_galerkin(T)
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
        # Compact complex vertices (BEFORE conv/sign and projection):
        Tc_full, Tc_leg1, conv = _kernels._cubic_complex(
            x_nodes=x_fine, psi_coeff=psi_coeff, M=M, **kin
        )
        Qc_full, Qc_leg1, _ = _kernels._quadratic_complex(
            x_nodes=x_fine, psi_coeff=psi_coeff, M=M, **kin
        )
        # Galerkin node->lo projector with the -conv solver-field scaling folded
        # into the node axis (matches cubic/quadratic_vertex returning -conv*V):
        GP = Ginv @ P  # (Nr, n_fine)
        GPc = (GP * (-conv)[None, :]).to(torch.complex128)  # (Nr, n_fine)
        # Galerkin-project the output-energy node axis f onto the radial mode o.
        # Tc_full[f, la,A, lb,B, lc,C]; Tc_leg1[f, la, lp,P, lq,Q];
        # Qc_full[f, la,A, lb,B];      Qc_leg1[f, la, lq,Q].
        Tc_full_lo = torch.einsum("of,faAbBcC->oaAbBcC", GPc, Tc_full)
        Tc_leg1_lo = torch.einsum("of,fapPqQ->oapPqQ", GPc, Tc_leg1)
        Qc_full_lo = torch.einsum("of,faAbB->oaAbB", GPc, Qc_full)
        Qc_leg1_lo = torch.einsum("of,faqQ->oaqQ", GPc, Qc_leg1)
        return {
            "Tc_full": Tc_full_lo,
            "Tc_leg1": Tc_leg1_lo,
            "Qc_full": Qc_full_lo,
            "Qc_leg1": Qc_leg1_lo,
        }

    def _build_apply_tables(self) -> None:
        """Precompute the harmonic transforms, mo-binning index tables and the
        per-harmonic null projectors used by the convolution apply."""
        fs = self.fermi_surface
        M = fs.M_theta
        nh = 2 * M + 1
        dtype, device = fs.v.dtype, rc.device
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        ps = torch.arange(-M, M + 1)
        U = _kernels._real_to_complex(M)  # (nh, nh) ahat[m] = sum_c U[m,c] a[c]
        R = _kernels._complex_to_real(M)  # (nh, nh) out[co] = R[co,M+mo] Fhat[mo]
        self._U = U.to(dtype=cdtype, device=device)
        self._R = R.to(dtype=cdtype, device=device)
        # Pair-sum bin: s = pA + pB over (A, B), kept in [-2M, 2M] (always in
        # range), index sP = s + 2M  (length 4M+1).
        npair = 4 * M + 1
        sp_idx = (ps[:, None] + ps[None, :]) + 2 * M  # (nh, nh) in [0, 4M]
        self._pair_bin = sp_idx.reshape(-1).to(device)  # (nh*nh,)
        self._npair = npair
        # Dense binning operator Bin3[mo, A, B, C] = 1 where pA+pB+pC = mo and
        # |mo| <= M, else 0.  Folded straight into the cubic-full contraction so
        # the output is binned to mo (size nh) WITHOUT materializing the dense
        # (g, Nr, nh^3) intermediate an explicit scatter would need.
        t_idx = (ps[:, None, None] + ps[None, :, None]
                 + ps[None, None, :])  # (nh,nh,nh) = pA + pB + pC
        Bin3 = torch.zeros(nh, nh, nh, nh, dtype=torch.float64)  # (mo,A,B,C)
        ok3 = t_idx.abs() <= M
        idx3 = torch.nonzero(ok3, as_tuple=False)  # (#, 3) over (A,B,C)
        Bin3[M + t_idx[ok3], idx3[:, 0], idx3[:, 1], idx3[:, 2]] = 1.0
        self._Bin3 = Bin3.to(dtype=cdtype, device=device)
        # Leg-1 convolution: from pair-sum sP (harmonic s in [-2M,2M]) and the
        # leg-1 harmonic pA1 = mo - s, output mo in [-M, M].  Precompute, for
        # each (mo, s), the leg-1 harmonic index (M + pA1) if |pA1| <= M else
        # dump; built as a gather table conv_src[mo, s].
        s_vals = torch.arange(-2 * M, 2 * M + 1)  # length npair
        pA1 = ps[:, None] - s_vals[None, :]  # (nh, npair), mo - s
        conv_ok = pA1.abs() <= M
        conv_src = torch.where(conv_ok, M + pA1, torch.full_like(pA1, nh))
        self._conv_src = conv_src.to(device)  # (nh, npair) index into ahat (nh+1)
        # Per-harmonic null radial projectors (mo in {0, +-1}); identity else.
        proj = self._null_projectors_by_harm(self.fermi_surface.T_temp)
        self._null_proj = {
            mo: Pm.to(dtype=cdtype, device=device) for mo, Pm in proj.items()
        }
        self._nh = nh
        self._ps = ps.to(device)

    def _bin_pairs(self, Gpair: torch.Tensor) -> torch.Tensor:
        """Bin a pair-harmonic tensor ``Gpair[..., A, B]`` (last two axes the
        two complex harmonics) by ``s = pA + pB`` into ``[..., sP]`` of length
        ``4M+1``.  Vectorized scatter (index_add) over the flattened (A, B)."""
        lead = Gpair.shape[:-2]
        flat = Gpair.reshape(*lead, -1)  # (..., nh*nh)
        out = torch.zeros(*lead, self._npair, dtype=Gpair.dtype,
                          device=Gpair.device)
        out.index_add_(-1, self._pair_bin, flat)
        return out

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
            ahat = self._to_complex(a4)
            # Binned complex output Fhat[..., lo, mo] (mo = output harmonic):
            Fhat = self._apply_cubic(ahat) + self._apply_quadratic(ahat)
            out = out + self._finalize_nonlinear(Fhat).to(out.dtype)
        return out.reshape(shape_in)

    # ---- nonlinear convolution apply (compact complex vertices) ----

    def _to_complex(self, a4: torch.Tensor) -> torch.Tensor:
        """Real modal field ``a4[..., l, c]`` -> complex harmonics
        ``ahat[..., l, m] = sum_c U[m, c] a[l, c]``."""
        return torch.einsum("mc,...lc->...lm", self._U, a4.to(self._U.dtype))

    def _leg1_conv_field(self, ahat: torch.Tensor) -> torch.Tensor:
        """Gathered leg-1 field for the convolution apply:
        ``a_conv[..., l, mo, s] = ahat[l, mo - s]`` (zero if ``|mo - s| > M``),
        with ``s`` the pair-sum harmonic index and ``mo`` the output harmonic.
        Leg 1 (the output leg at ``phi1 = 0``) has its harmonic fixed by the
        selection rule ``mo = (leg-1 harmonic) + s``."""
        lead = ahat.shape[:-2]
        Nr = ahat.shape[-2]
        ahat_ext = torch.cat(
            [ahat, torch.zeros(*lead, Nr, 1, dtype=ahat.dtype,
                               device=ahat.device)], dim=-1)  # (..., Nr, nh+1)
        return ahat_ext[..., self._conv_src]  # (..., Nr, nh, npair)

    def _apply_cubic(self, ahat: torch.Tensor) -> torch.Tensor:
        """Cubic contribution to the binned complex output ``Fhat[..., o, mo]``.

        The single full-rank-3 triple ``(2,3,4)`` is contracted with the field
        and binned to ``mo`` in one shot (the ``|mo| <= M`` selection folded into
        ``Bin3`` so no dense ``(..., Nr, (2M+1)^3)`` intermediate forms); the
        three leg-1 triples are literal harmonic convolutions."""
        # (2,3,4) full triple:
        Fhat = torch.einsum(
            "oaAbBcC,nABC,...aA,...bB,...cC->...on",
            self._Tc_full, self._Bin3, ahat, ahat, ahat,
        )  # (..., Nr, nh) over mo (n)
        # leg-1 triples: contract the partner pair, bin its sum-harmonic
        # s = pP + pQ, then convolve with leg 1 (harmonic mo - s):
        Hpair = torch.einsum(
            "oapPqQ,...pP,...qQ->...oaPQ", self._Tc_leg1, ahat, ahat,
        )  # (..., o, a, P, Q)
        Hs = self._bin_pairs(Hpair)  # (..., o, a, sP)  s in [-2M, 2M]
        a_conv = self._leg1_conv_field(ahat)  # (..., Nr, nh, npair)
        Fhat = Fhat + torch.einsum("...oas,...ams->...om", Hs, a_conv)
        return Fhat

    def _apply_quadratic(self, ahat: torch.Tensor) -> torch.Tensor:
        """Quadratic (thermoelectric) contribution to ``Fhat[..., o, mo]``.

        Non-leg-1 pairs are contracted and binned by ``mo = pA + pB``; the three
        leg-1 pairs are convolutions (leg 1 at harmonic ``mo - pQ``)."""
        M, nh = self.fermi_surface.M_theta, self._nh
        # non-leg-1 pairs (rank-2 angular):
        Gq = torch.einsum(
            "oaAbB,...aA,...bB->...oAB", self._Qc_full, ahat, ahat,
        )  # (..., o, A, B)
        Fq = self._bin_pairs(Gq)  # (..., o, sP) over s in [-2M, 2M]
        Fhat = Fq[..., M:(M + nh)]  # restrict pair-sum to |mo| <= M
        # leg-1 pairs (convolution): partner harmonic pQ acts as the pair-sum,
        # so select the conv columns at sP = pQ + 2M:
        Hq = torch.einsum(
            "oaqQ,...qQ->...oaQ", self._Qc_leg1, ahat,
        )  # (..., o, a, Q)
        a_conv = self._leg1_conv_field(ahat)
        a_conv_q = a_conv[..., self._ps + 2 * M]  # (..., Nr, nh, nh) over Q
        Fhat = Fhat + torch.einsum("...oaQ,...amQ->...om", Hq, a_conv_q)
        return Fhat

    def _finalize_nonlinear(self, Fhat: torch.Tensor) -> torch.Tensor:
        """Apply the per-output-harmonic null projection on the binned complex
        output and convert to real coefficients ``[..., o, co]``."""
        M = self.fermi_surface.M_theta
        for mo, Pm in self._null_proj.items():
            idx = M + mo
            Fhat = Fhat.clone()
            Fhat[..., idx] = torch.einsum("Ll,...l->...L", Pm, Fhat[..., idx])
        return torch.einsum("cm,...om->...oc", self._R, Fhat).real

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
