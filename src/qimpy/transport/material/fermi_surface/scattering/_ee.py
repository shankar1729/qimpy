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


class EEScattering(TreeNode):
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
    on_shell: bool  #: linear rate: closed-form surface (True) or exact (False)
    tol: float  #: target relative quadrature accuracy (drives auto resolution)
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
        on_shell: bool = False,
        tol: float = 1e-3,
        n_alpha: int = 0,
        n_xi: int = 0,
        xi_cut: float = 0.0,
        n_phi: int = 0,
        n_xi_proj: int = 0,
        check_convergence: bool = True,
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
        on_shell
            :yaml:`Linear rate: closed-form surface (True) or exact (False).`
            on_shell=True uses the leading-order T/E_F surface rate
            gamma_m = m*^2 T^2 K_m / (16 pi E_F) (quasiparticles at the Fermi
            level, requires Nr = 1); the default False uses the exact
            thermal-shell quadrature (finite T/E_F, full radial tower, ~30%
            larger for the shear mode at T/E_F ~ 0.03).  Affects only the linear
            block -- the nonlinear vertices are always exact.
        tol
            :yaml:`Target relative quadrature accuracy (default 1e-3).`
            Drives the automatic resolution of all quadrature parameters
            below from the truncation (M, Nr) and the degeneracy T/E_F; a
            construction-time convergence check warns if it is not met.
            Lower `tol` -> finer (and slower) quadrature.
        n_alpha, n_xi, xi_cut, n_phi, n_xi_proj
            :yaml:`Optional explicit quadrature overrides (0 = auto from tol).`
            Energy points / cut (units of T), angular points, and radial
            Galerkin projection nodes.  Each is derived automatically when
            left at 0; set a nonzero value only to override a specific axis.
            n_phi is the binding (physics-tied) one -- it must dealias the
            cubic's 3M harmonics and resolve the collinear edge of angular
            width ~T/E_F.
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
        self.on_shell = on_shell
        self.tol = tol

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

        # Quadrature: derive from the truncation (M, Nr), the degeneracy T/E_F
        # and the target relative accuracy `tol`.  Any explicitly-set (nonzero)
        # input overrides its auto value; the construction-time convergence
        # check below verifies the result.  Rationale: xi_cut/n_alpha are fixed
        # constants; n_xi tracks the radial degree (~Nr); n_xi_proj >= Nr; n_phi
        # must dealias the cubic's 3M harmonics AND resolve the collinear edge
        # of angular width ~T/E_F, so it is the only physics-tied knob.
        s = max(1.0, 1.0 + float(np.log10(1e-3 / max(tol, 1e-12))))
        self.n_xi = n_xi if n_xi else int(np.ceil(max(16, 4 * Nr) * s))
        self.n_xi_proj = n_xi_proj if n_xi_proj else max(Nr + 6, 8)
        nphi_auto = max(6 * M + 2, int(np.ceil(8.0 * s / max(t_ratio, 1e-3))))
        self.n_phi = n_phi if n_phi else nphi_auto + (nphi_auto % 2)
        self.xi_cut = xi_cut if xi_cut else 10.0
        self.n_alpha = n_alpha if n_alpha else 4096
        log.info(
            f"quadrature (tol = {tol:.0e}): n_xi = {self.n_xi}, n_phi ="
            f" {self.n_phi}, n_xi_proj = {self.n_xi_proj},"
            f" xi_cut = {self.xi_cut:g}"
        )

        # Angular kernels (up to 2M for the cubic vertex):
        self.K = _kernels.K_table(
            2 * M,
            kF=fs.kF,
            epsilon_bg=epsilon_bg,
            kappa=self.kappa,
            well_width=well_width,
            n_alpha=self.n_alpha,
        )
        gamma_cf = _kernels.gamma_linear(
            self.K, m_star=self.m_star, T=T, E_F=self.E_F
        )
        log.info(
            f"K_2 = {self.K[2]:.4g}, gamma_2 (closed form) = {gamma_cf[2]:.4g}"
        )

        # ---- Linear rates: closed-form surface (on_shell) or exact shell ----
        if on_shell:
            if Nr != 1:
                raise InvalidInputException(
                    "on_shell=True (closed-form surface rate) supports Nr = 1"
                    " only; use on_shell=False (exact) for the radial tower"
                )
            L = torch.zeros(dim, 1, 1, dtype=torch.float64)
            for m in range(1, M + 1):
                L[2 * m - 1, 0, 0] = L[2 * m, 0, 0] = gamma_cf[m]
        else:
            L = self._exact_L_blocks(T)
        self.L_coeff = L.to(dtype=dtype, device=device)
        if check_convergence and not on_shell:
            self._check_quadrature_convergence(T)
        gam2 = float(self.L_coeff[3, 0, 0]) if M >= 2 else 0.0
        if gam2 > 0.0:
            # Characteristic transport scales from the m=2 shear rate gamma_2.
            # qimpy works in atomic units; convert to SI transport units:
            PS = 2.4188843e-5       # a.u. time           -> ps
            UM = 5.29177211e-5      # a.u. length (Bohr)  -> um
            UMPS = 2.18769126       # a.u. velocity       -> um/ps
            MEV = 27211.386         # Hartree -> meV       (= mV for e = 1)
            UA_UM = 1.25170e8       # a.u. current/length -> uA/um
            tau_ee = 1.0 / gam2
            l_ee = fs.vF * tau_ee
            v_nl = T / (self.m_star * fs.vF)            # drift where Phi ~ 1
            j_nl = (fs.kF**2 / (2.0 * np.pi)) * v_nl    # 2D current/width, spin 2
            rule = "  " + "─" * 52

            def _row(name: str, sym: str, val: str) -> str:
                return f"   {name:<17}{sym:<8}= {val}"

            lines = [
                rule,
                "   e-e scattering: characteristic transport scales",
                rule,
                _row("Fermi energy", "E_F", f"{self.E_F * MEV:.4g} meV"),
                _row("Fermi velocity", "vF", f"{fs.vF * UMPS:.4g} um/ps"),
                _row("degeneracy", "T/E_F", f"{t_ratio:.4g}"),
                _row("scattering time", "tau_ee", f"{tau_ee * PS:.4g} ps"),
                _row("mean free path", "l_ee", f"{l_ee * UM:.4g} um"),
            ]
            if nonlinear:
                lines += [
                    rule,
                    "   nonlinear onset (drive beyond which transport nonlinear):",
                    _row("drift velocity", "v_d*", f"{v_nl * UMPS:.4g} um/ps"),
                    _row("current/width", "J*", f"{j_nl * UA_UM:.4g} uA/um"),
                    _row("bias", "V*", f"{T * MEV:.4g} mV"),
                ]
            lines.append(rule)
            log.info("\n" + "\n".join(lines))

        # ---- Nonlinear operator (exact cubic + quadratic, matrix-free) ----
        if nonlinear:
            self._build_matrix_free_generator(T)
            ngen = self._mf_Wk.numel()
            log.info(
                "Nonlinear e-e operator enabled (matrix-free kinematic"
                " generator, L-independent storage):"
                f" {self._mf_nf} output nodes x {self._mf_nq} quadrature"
                f" points, {4 * ngen * 8 / 1e6:.1f} MB generator"
                " (independent of Nr beyond the psi table)"
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

    def _check_quadrature_convergence(self, T: float) -> None:
        """Verify the (auto or explicit) quadrature is converged: recompute the
        representative linear rates at a refined ``n_phi`` and warn if they move
        by more than ``tol``.  The angular resolution ``n_phi`` (collinear edge +
        dealiasing) is the binding axis; ``n_xi``/``n_xi_proj`` converge
        spectrally and faster, so refining ``n_phi`` is the decisive test."""
        fs = self.fermi_surface
        M = fs.M_theta
        if M < 2:
            return  # only m=0,1 (conserved nulls); nothing rate-bearing to check
        m_chk = [m for m in (2, 4) if m <= M]
        psi_coeff, x_fine, P, Ginv, _ = self._radial_galerkin(T)

        def block_norms(n_phi: int) -> torch.Tensor:
            R = _kernels.L_blocks(
                x_nodes=x_fine, psi_coeff=psi_coeff, m_list=m_chk,
                kF=fs.kF, m_star=self.m_star, T=T, epsilon_bg=self.epsilon_bg,
                kappa=self.kappa, well_width=self.well_width,
                n_xi=self.n_xi, xi_cut=self.xi_cut, n_phi=n_phi,
            )  # (len(m_chk), n_fine, Nr)
            L = torch.einsum("ij,jf,mfl->mil", Ginv, P, R)  # (len, Nr, Nr)
            return torch.stack([torch.linalg.norm(L[i]) for i in range(len(m_chk))])

        n_phi_ref = int(np.ceil(1.5 * self.n_phi))
        g0, g1 = block_norms(self.n_phi), block_norms(n_phi_ref)
        rel = ((g1 - g0).abs() / g0.clamp_min(1e-300)).max().item()
        if rel > self.tol:
            log.info(
                f"WARNING: e-e quadrature may be under-resolved -- gamma_m moved"
                f" {rel:.1e} when n_phi {self.n_phi} -> {n_phi_ref}"
                f" (tol = {self.tol:.0e}). Lower `tol` or set `n_phi` explicitly."
            )
        else:
            log.info(
                f"quadrature convergence OK: max d(gamma_m) = {rel:.1e}"
                f" <= tol = {self.tol:.0e} (n_phi {self.n_phi} -> {n_phi_ref})"
            )

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

    def _build_harmonic_tables(self) -> None:
        """Real<->complex harmonic transforms and per-harmonic null projectors
        (all O(nh^2)).  These are the ONLY apply tables the matrix-free path
        needs; it must NOT build the dense convolution tables, whose ``Bin3`` is
        O(nh^4) (~70 GB at M=128) and is never touched by ``_apply_matrix_free``.
        """
        fs = self.fermi_surface
        M = fs.M_theta
        nh = 2 * M + 1
        dtype, device = fs.v.dtype, rc.device
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        U = _kernels._real_to_complex(M)  # (nh, nh) ahat[m] = sum_c U[m,c] a[c]
        R = _kernels._complex_to_real(M)  # (nh, nh) out[co] = R[co,M+mo] Fhat[mo]
        self._U = U.to(dtype=cdtype, device=device)
        self._R = R.to(dtype=cdtype, device=device)
        # Per-harmonic null radial projectors (mo in {0, +-1}); identity else.
        proj = self._null_projectors_by_harm(self.fermi_surface.T_temp)
        self._null_proj = {
            mo: Pm.to(dtype=cdtype, device=device) for mo, Pm in proj.items()
        }
        self._nh = nh
        self._ps = torch.arange(-M, M + 1).to(device)

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
            # Matrix-free: evaluate the reduced (cubic + quadratic) operator by
            # quadrature over the stored field-independent generator -- no mode
            # tensor formed.  The linear part stays in L_coeff above.
            out = out + self._apply_matrix_free(a4).to(out.dtype)
        return out.reshape(shape_in)

    def _finalize_nonlinear(self, Fhat: torch.Tensor) -> torch.Tensor:
        """Apply the per-output-harmonic null projection on the binned complex
        output and convert to real coefficients ``[..., o, co]``."""
        M = self.fermi_surface.M_theta
        for mo, Pm in self._null_proj.items():
            idx = M + mo
            Fhat = Fhat.clone()
            Fhat[..., idx] = torch.einsum("Ll,...l->...L", Pm, Fhat[..., idx])
        return torch.einsum("cm,...om->...oc", self._R, Fhat).real

    # ---- matrix-free / kinematic-generator path (L-independent storage) ----

    def _build_matrix_free_generator(self, T: float) -> None:
        """Precompute the field-independent kinematic generator.

        For each output radial collocation node ``xi_1`` (the fine Galerkin
        nodes ``x_fine``) and each quadrature point ``(xi_2, xi_3, phi_3, root)``
        store the leg energies (``xi_1, xi_2, xi_3, xi_4 = xi_1 + xi_2 - xi_3``),
        the leg RELATIVE angles (``dphi_1 = 0``, ``dphi_3 = phi_3``, and
        ``dphi_2, dphi_4`` from the two energy-shell roots), and the full
        kinematic weight ``Wk = |M_q|^2 / (k_2 k_4 |sin(phi_4 - phi_2)|)`` times
        the quadrature weight ``w_2 w_3 w_phi T^2`` and prefactor
        ``(m*)^3/(2 pi)^3`` (the band/edge masks are already baked into ``Wk``,
        which is zero off the kinematically allowed shell).  NONE of this depends
        on ``L = Nr`` -- only the tiny radial-basis table ``psi_coeff`` does.

        Also stores the radial Galerkin node->mode projector ``GPc`` (with the
        ``-conv`` solver-field scaling folded in, exactly as the dense path) and
        builds the harmonic transforms + null projectors (shared with the dense
        apply) so ``_finalize_nonlinear`` can be reused.
        """
        fs = self.fermi_surface
        device = rc.device
        dtype = fs.v.dtype
        psi_coeff, x_fine, P, Ginv, _ = self._radial_galerkin(T)
        t = T / self.E_F
        # Galerkin node->mode projector with -conv (decay sign) folded in -- the
        # output node quantity is the raw kinematic integral (occupation rate
        # f_dot, since the legs carry the physical weight w_eq); GPc maps it to
        # modal coefficients exactly as in _nonlinear_vertices.  The solver field
        # obeys delta_f = w_eq Phi_code, so Phi_code_dot = f_dot / w_eq =
        # 4 T cosh^2(x1/2) * f_dot:
        conv = 4 * T * torch.cosh(x_fine / 2) ** 2  # 1 / w_eq(x1)
        GPc = (Ginv @ P) * (-conv)[None, :]  # (Nr, n_fine)

        log.info(
            f"Building matrix-free e-e generator (M = {fs.M_theta},"
            f" Nr = {fs.Nr}), quadrature {self.n_xi}^2 x 2 x {self.n_phi}"
            f" at {len(x_fine)} output nodes"
        )
        x2, w2, beta, wbeta, cosB, sinB = _kernels._shell_quadrature(
            T, self.E_F, self.n_xi, self.xi_cut, self.n_phi
        )
        kin = dict(
            T=T, t=t, kF=fs.kF, m_star=self.m_star, epsilon_bg=self.epsilon_bg,
            kappa=self.kappa, well_width=self.well_width,
            n_xi=self.n_xi, n_phi=self.n_phi,
        )
        Nf = len(x_fine)
        ng = self.n_xi * self.n_phi
        # Per-(f, i3, root) we get an (n_xi, n_phi) grid; accumulate the four
        # point-dependent fields (x4, dphi2, dphi4, Wk) as [Nf, n_xi(i3), 2, ng].
        x4_l, dphi2_l, dphi4_l, Wk_l = [], [], [], []
        for f in range(Nf):
            x1v = x_fine[f]
            k1 = _kernels.k_fermi(x1v, fs.kF, t)
            X2 = x2[:, None]  # (n_xi, 1)
            k2 = _kernels.k_fermi(X2, fs.kF, t)
            for i3 in range(self.n_xi):
                x3v = x2[i3]
                w3v = w2[i3]
                weight_phase, X4 = _kernels._shell_geometry(
                    x1v=x1v, X2=X2, k1=k1, k2=k2, x3v=x3v, w2=w2, w3v=w3v,
                    beta=beta, wbeta=wbeta, cosB=cosB, sinB=sinB, **kin,
                )
                for sgn in (+1.0, -1.0):
                    Wk, phi2, phi4 = weight_phase(sgn)  # (n_xi, n_phi)
                    x4_l.append(X4.expand(self.n_xi, self.n_phi).reshape(ng))
                    dphi2_l.append(phi2.reshape(ng))
                    dphi4_l.append(phi4.reshape(ng))
                    Wk_l.append(Wk.reshape(ng))
        # Flatten the quadrature axis: nq = n_xi(i3) * 2(root) * n_xi(i2) * n_phi.
        nq = len(x4_l) // Nf * ng
        x4 = torch.stack(x4_l, 0).reshape(Nf, -1)  # (Nf, nq)
        dphi2 = torch.stack(dphi2_l, 0).reshape(Nf, -1)
        dphi4 = torch.stack(dphi4_l, 0).reshape(Nf, -1)
        Wk = torch.stack(Wk_l, 0).reshape(Nf, -1)
        # x1 is constant per output node; x2, x3 and dphi3 = beta tile across the
        # quadrature axis.  Build the per-quadrature-point x2/x3/dphi3 once.
        x1_col = x_fine.to(torch.float64)  # (Nf,)
        # ordering of the quadrature axis matches the append order:
        #   for i3 in n_xi: for sgn in 2: (i2 in n_xi, i_phi in n_phi flattened)
        x3_blk = torch.repeat_interleave(x2, 2 * ng)  # (nq,)
        x2_tile = x2.repeat_interleave(self.n_phi)  # (ng,) i2-major
        dphi3_tile = beta.repeat(self.n_xi)  # (ng,) i_phi-minor
        nblk = self.n_xi * 2
        x2_q = x2_tile.repeat(nblk)  # (nq,)
        dphi3_q = dphi3_tile.repeat(nblk)  # (nq,)

        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        def _to(x):
            return x.to(dtype=torch.float64, device=device)

        # x1 is constant per output node (stored as (Nf, 1) -> broadcasts over
        # the quadrature axis); x2, x3, dphi3 = beta tile across it (stored as
        # 1D (nq,) -> broadcast over the node axis); only x4, dphi2, dphi4, Wk
        # are genuinely (Nf, nq).  This is the minimal, L-independent footprint.
        self._mf_x1 = _to(x1_col)[:, None]  # (Nf, 1)
        self._mf_x2 = _to(x2_q)[None, :]  # (1, nq)
        self._mf_x3 = _to(x3_blk)[None, :]  # (1, nq)
        self._mf_x4 = _to(x4)  # (Nf, nq)
        self._mf_dphi2 = _to(dphi2)  # (Nf, nq)
        self._mf_dphi3 = _to(dphi3_q)[None, :]  # (1, nq)
        self._mf_dphi4 = _to(dphi4)  # (Nf, nq)
        self._mf_Wk = _to(Wk)  # (Nf, nq)
        self._mf_psi_coeff = _to(psi_coeff)
        self._mf_GPc = GPc.to(dtype=cdtype, device=device)
        self._mf_nf = Nf
        self._mf_nq = nq
        # Precompute the (field-independent) radial-basis tables psi_l and the
        # equilibrium weight w_eq and Fermi factor f0 at every leg/quad point --
        # these are L-small (psi) / scalar and reused across all spatial cells:
        self._mf_psi = {
            leg: self._mf_psi_eval(getattr(self, f"_mf_{leg}"))
            for leg in ("x1", "x2", "x3", "x4")
        }
        self._mf_weq = {
            leg: 0.25 / torch.cosh(getattr(self, f"_mf_{leg}") / 2) ** 2 / T
            for leg in ("x1", "x2", "x3", "x4")
        }
        self._mf_f0 = {
            leg: torch.sigmoid(-getattr(self, f"_mf_{leg}"))
            for leg in ("x1", "x2", "x3", "x4")
        }
        # Output-angle grid for resolving the output harmonics mo = sum of input
        # harmonics.  The cubic reaches harmonic 3M; a uniform grid of >= 6M+1
        # points integrates e^{-i mo phi1} exactly (we use 6M+2, even):
        M = fs.M_theta
        Nout = 6 * M + 2
        phi1 = 2 * np.pi * torch.arange(Nout, dtype=torch.float64) / Nout
        ps = torch.arange(-M, M + 1, dtype=torch.float64)
        # complex-harmonic analysis covector: Fhat[mo] = (1/Nout) sum_n
        # out(phi1_n) e^{-i mo phi1_n}, matching the dense binned Fhat[lo, mo]:
        self._mf_phi1 = phi1.to(device)
        self._mf_expmn = (
            torch.exp(-1j * phi1[None, :] * ps[:, None]) / Nout
        ).to(dtype=cdtype, device=device)  # (nh, Nout)
        self._mf_Nout = Nout
        # Only the light harmonic transforms + null projectors -- NOT the dense
        # convolution tables (the O(nh^4) Bin3 would be ~70 GB at M=128 and is
        # never used by _apply_matrix_free):
        self._build_harmonic_tables()

    def _mf_psi_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Radial basis ``psi_l(x)`` (power-basis Horner), shape ``x.shape+(Nr,)``."""
        pc = self._mf_psi_coeff
        res = torch.zeros(x.shape + (pc.shape[1],), dtype=torch.float64,
                          device=x.device)
        for p in range(pc.shape[0] - 1, -1, -1):
            res = res * x[..., None] + pc[p]
        return res

    def _apply_matrix_free(self, a4: torch.Tensor) -> torch.Tensor:
        """Batch-chunked matrix-free apply.  The per-cell apply allocates
        ``N_out x N_f x n_q`` leg-reconstruction intermediates; for a large
        spatial batch these are processed in chunks sized to a fixed memory
        budget so peak memory stays bounded regardless of grid size.  The result
        is independent of the chunk size (no reduction couples the batch), so
        this changes only memory, not the output or the total work."""
        Nr, dim = a4.shape[-2], a4.shape[-1]
        batch_shape = a4.shape[:-2]
        B = int(np.prod(batch_shape)) if batch_shape else 1
        # cells per chunk from a ~4 GB budget; the per-cell peak is dominated by
        # the (N_out, N_f, n_q) reconstructions (~100 B/element at peak):
        per_cell = self._mf_Nout * self._mf_nf * self._mf_nq * 128
        chunk = max(1, int((16 * 1024**3) // max(per_cell, 1)))
        if chunk >= B:  # fits in one pass -- no loop / concat overhead
            return self._apply_matrix_free_chunk(a4)
        a_flat = a4.reshape(B, Nr, dim)
        outs = [self._apply_matrix_free_chunk(a_flat[i:i + chunk])
                for i in range(0, B, chunk)]
        return torch.cat(outs, dim=0).reshape(*batch_shape, Nr, dim)

    def _apply_matrix_free_chunk(self, a4: torch.Tensor) -> torch.Tensor:
        """Evaluate the reduced (cubic + quadratic) e-e operator by quadrature
        over the stored kinematic generator -- no mode tensor formed.

        ``a4``: modal field ``[..., Nr, dim_theta]``.  Reconstructs ``delta_f``
        at the four legs from the complex harmonics, evaluates the cubic
        ``C3 = d1 d2 (d3+d4) - d3 d4 (d1+d2)`` and the particle-hole-odd
        quadratic ``Q2`` of ``(B-F)`` at an output-angle grid, integrates the
        quadrature, projects to output radial modes (``GPc``) and to output
        angular harmonics (DFT in ``phi_1``), then applies the SAME null
        projection + real fold as the dense path.  Returns ``[..., Nr, dim]``.
        """
        fs = self.fermi_surface
        M, Nr = fs.M_theta, fs.Nr
        cdt = self._U.dtype
        ps = torch.arange(-M, M + 1, device=a4.device)
        # complex harmonics of the modal field: ahat[..., l, m]:
        ahat = torch.einsum("mc,...lc->...lm", self._U, a4.to(cdt))  # (..,Nr,nh)

        Nf, nq, Nout = self._mf_nf, self._mf_nq, self._mf_Nout
        batch = ahat.shape[:-2]
        nbatch = int(np.prod(batch)) if batch else 1
        legs = ("x1", "x2", "x3", "x4")
        # Full (Nf, nq) views of the field-independent leg tables (broadcast axes
        # expanded as views, no copy; x1 const per node, x2/x3/dphi3 const):
        psi_f = {leg: self._mf_psi[leg].to(cdt).expand(Nf, nq, -1) for leg in legs}
        weq_f = {leg: self._mf_weq[leg].expand(Nf, nq) for leg in legs}
        f0_f = {leg: self._mf_f0[leg].expand(Nf, nq) for leg in legs}
        zero = self._mf_dphi3.new_zeros(1, nq)  # dphi1 = 0 (broadcast)
        dphi_f = {"x1": zero.expand(Nf, nq), "x2": self._mf_dphi2,
                  "x3": self._mf_dphi3.expand(Nf, nq), "x4": self._mf_dphi4}

        # nq-axis chunking: the quadrature is a sum over q, so accumulate it in
        # q-blocks sized to a memory budget; the (Nout, Nf, nq) leg
        # reconstructions then never exceed it regardless of M / n_phi.  The
        # block is the full nq when it fits (single pass -> bit-identical and no
        # loop overhead); only larger problems split (then summation reorders at
        # the ~1e-15 level).  Reusing one zero-padded FFT buffer across the four
        # legs avoids re-zeroing it each leg.
        per_q = Nout * Nf * 128 * nbatch  # ~peak bytes per quadrature point
        qchunk = max(1, min(nq, int((4 * 1024**3) // max(per_q, 1))))
        fdot = 0
        for q0 in range(0, nq, qchunk):
            q1 = min(q0 + qchunk, nq)
            pad = ahat.new_zeros(*batch, Nf, q1 - q0, Nout)  # reused over legs
            d = []
            for leg in legs:
                B = torch.einsum("...lm,fql->...fqm", ahat, psi_f[leg][:, q0:q1])
                B = B * torch.exp(1j * dphi_f[leg][:, q0:q1, None] * ps)
                pad[..., : M + 1] = B[..., M:]      # m = 0..M  -> bins 0..M
                pad[..., Nout - M:] = B[..., :M]     # m = -M..-1; middle stays 0
                Phi = torch.fft.ifft(pad, dim=-1) * Nout  # (..,Nf,nb,Nout)
                d.append((weq_f[leg][:, q0:q1] * torch.movedim(Phi, -1, -3)).real)
            d1, d2, d3, d4 = d
            f1, f2, f3, f4 = (f0_f[leg][:, q0:q1] for leg in legs)
            # cubic (f0-independent) + particle-hole-odd quadratic of (B-F):
            C3 = d1 * d2 * (d3 + d4) - d3 * d4 * (d1 + d2)
            Q2 = (d1 * d2 * (f3 + f4 - 1.0) + d1 * d3 * (f2 - f4)
                  + d1 * d4 * (f2 - f3) + d2 * d3 * (f1 - f4)
                  + d2 * d4 * (f1 - f3) - d3 * d4 * (f1 + f2 - 1.0))
            fdot = fdot + torch.einsum(
                "...nfq,fq->...nf", C3 + Q2, self._mf_Wk[:, q0:q1])
        # Galerkin-project the output-energy node axis onto radial modes (GPc
        # folds the -conv solver-field scaling); then DFT in phi1 to harmonics:
        out_rad = torch.einsum(
            "of,...nf->...no", self._mf_GPc, fdot.to(self._mf_GPc.dtype))
        Fhat = torch.einsum("mn,...no->...om", self._mf_expmn, out_rad)
        # SAME per-mo null projection + real fold as the dense path:
        return self._finalize_nonlinear(Fhat)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["epsilon_bg"] = self.epsilon_bg
        attrs["kappa"] = self.kappa
        attrs["m_star"] = self.m_star
        attrs["well_width"] = self.well_width
        attrs["nonlinear"] = self.nonlinear
        attrs["on_shell"] = self.on_shell
        attrs["tol"] = self.tol
        attrs["n_alpha"] = self.n_alpha
        attrs["n_xi"] = self.n_xi
        attrs["xi_cut"] = self.xi_cut
        attrs["n_phi"] = self.n_phi
        attrs["n_xi_proj"] = self.n_xi_proj
        return list(attrs.keys())
