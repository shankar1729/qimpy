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
        g_s: float = 2.0,
        m_star: float = 0.0,
        well_width: float = 0.0,
        nonlinear: bool = True,
        on_shell: bool = False,
        T_build: float = 0.0,
        tol: float = 1e-3,
        n_alpha: int = 0,
        n_xi: int = 0,
        xi_cut: float = 0.0,
        n_phi: int = 0,
        n_xi_proj: int = 0,
        check_convergence: bool = True,
        backend: str = "auto",
        recon: str = "auto",
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
        backend
            :yaml:`Nonlinear apply backend: 'auto' (default), 'dense' or
            'matrix_free'.`  Both backends are the SAME operator (agree to
            roundoff); they only trade speed vs memory.  'dense' precontracts the
            vertices into a SPARSE-SYMMETRIC packed kernel and applies it as a
            gather + scatter; storage and per-cell apply both scale as
            ~ Nr (Nr dim)^3 / 16 -- the four exact symmetries (additive harmonic
            selection + output truncation, input-leg permutation, output reality,
            phi -> -phi reflection) cut ~16 dim off the naive (Nr dim)^4 dense
            contraction.  The apply is orders of magnitude cheaper per cell than
            matrix-free, but storage grows as (Nr dim)^3.  The kernel is BUILT
            directly in packed form (slab-blocked over the first leg index), so
            build memory is ~ the packed kernel, not the (Nr dim)^3 vertex.
            'matrix_free' evaluates the reduced operator by quadrature over the
            stored kinematic generator -- per-cell apply ~ n_xi_proj * nq * M *
            Nout, but storage flat in Nr; the only feasible option at large
            (M, Nr).  'auto' uses 'dense' when its packed kernel fits a fixed cap
            (~0.5 GiB) AND the one-time build is quick, else 'matrix_free'.
            For large-M matrix-free runs prefer ``precision='float32'`` (the
            transport-level knob) on a GPU: the apply is ~2x faster and uses half
            the memory, with ~1e-6 relative accuracy (well within the ~1e-3
            quadrature tolerance) and conservation still exact (the null
            projection is applied regardless of dtype).
        recon
            :yaml:`Angular leg-reconstruction backend: 'auto' (default), 'gemm'
            or 'fft'.`  Only used by the matrix-free backend.  The nonlinear
            apply rebuilds delta_f at the output-angle grid from its harmonics
            every call.  The field is real, so the reconstruction is done from
            the m >= 0 harmonics alone (Hermitian symmetry).  'gemm' uses two
            real (M+1, Nout) synthesis matmuls (cos/sin) -- faster for
            small/moderate M and far better on GPU (where the many size-Nout
            transforms are launch-bound).  'fft' uses a half-spectrum inverse
            real FFT (irfft), with asymptotically fewer FLOPs at very large M
            (Nout log Nout vs Nout*M).  'auto' micro-benchmarks both on
            this machine/device at construction and keeps the faster -- no user
            tuning, and it adapts to M, dtype and CPU-vs-GPU.  All three are
            mathematically identical (agree to float64 roundoff); the choice
            only trades speed.
        """
        super().__init__()
        fs = fermi_surface
        self.fermi_surface = fs
        self.epsilon_bg = epsilon_bg
        # Spin degeneracy of the PARTNER electron in the golden-rule sum.
        # It multiplies the whole collision integral:
        #   C = (2 pi/hbar) g_s Int d2k2 d2k3 d2k4/(2 pi)^6 |M|^2 dd (B - F)
        # and after the two deltas are resolved it survives as the overall
        # factor on (m*)^3/(2 pi)^3.  It was previously absent (g_s = 1)
        # while the screening constant kappa = g_s m*/eps_bg was built with
        # the spin-DEGENERATE 2D density of states -- the two were
        # inconsistent.  g_s = 2 for an unpolarized 2DEG (GaAs/AlGaAs) with
        # the direct (Hartree) matrix element; use g_s = 1 only if the
        # same-spin exchange channel is being excluded deliberately.
        self.g_s = float(g_s)
        self.m_star = m_star if m_star else fs.kF / fs.vF
        # kappa = 2 pi e^2 nu_2D/eps_bg with nu_2D = g_s m*/(2 pi):
        self.kappa = kappa if kappa else self.g_s * self.m_star / epsilon_bg
        self.E_F = 0.5 * fs.kF**2 / self.m_star
        self.well_width = well_width
        self.nonlinear = nonlinear
        self.on_shell = on_shell
        self.tol = tol
        if backend not in ("auto", "dense", "matrix_free"):
            raise InvalidInputException(
                f"backend must be 'auto', 'dense' or 'matrix_free',"
                f" got {backend!r}"
            )
        self.backend = backend
        if recon not in ("auto", "gemm", "fft"):
            raise InvalidInputException(
                f"recon must be 'auto', 'gemm' or 'fft', got {recon!r}"
            )
        self.recon = recon

        # T_build: optional build-temperature override (local-T_e ensembles
        # tabulate the operator at Chebyshev nodes T_i around the material T;
        # 0 = use the material temperature).  All internal quadratures,
        # vertices and null structures below are built at this temperature.
        T = float(T_build) if T_build else fs.T_temp
        self.T_build = T
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
        # >= 96 so ALL hybrid Gram entries are machine-converged (memo):
        self.n_xi_proj = max(n_xi_proj, 96, Nr + 6)
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
            self.K, m_star=self.m_star, T=T, E_F=self.E_F,
            g_s=self.g_s,
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

        # ---- Nonlinear operator (exact cubic + quadratic) ----
        # Two apply backends, same operator: 'dense' precontracts the vertices
        # into selection-compact complex kernels (fast convolution apply, storage
        # ~ Nr^4 dim^3) and 'matrix_free' evaluates by quadrature (storage flat
        # in Nr).  'auto' uses dense when the kernel fits a fixed cap, else
        # matrix-free.
        if nonlinear:
            # Sparse-symmetric dense kernel built DIRECTLY in packed form
            # (cubic_packed_node slab-blocks the (2,3,4) triple and gathers only
            # the kept unordered triples -- no (Nr dim)^3 vertex, no g*(Nr dim)^2
            # intermediate).  Peak build memory is therefore ~ the packed kernel
            # itself: N_tri ~ (Nr dim)^3 / 16 cubic triples (additive selection +
            # permutation + reality), each carried over Nr output modes.  Gate
            # 'auto' on BOTH (a) that packed kernel fitting a fixed cap, and (b)
            # the one-time build (slab matmuls ~ n_xi_proj * (Nr dim)^3 * nq)
            # being quick -- so 'auto' precomputes dense only when it is both
            # memory-feasible and fast (the dense apply is then orders of
            # magnitude cheaper per cell).  Larger sizes remain available on
            # demand via backend='dense' (feasible now, just a slower build).
            Pf = Nr * dim
            N_tri = max(1, Pf**3 // 16)
            citem = 8 if dtype == torch.float32 else 16  # complex build itemsize
            kernel_bytes = Nr * N_tri * citem            # build accumulator peak
            nq = 2 * self.n_xi**2 * self.n_phi
            build_flops = self.n_xi_proj * Pf**3 * nq    # slab-matmul work
            mem_cap = 512 * 1024**2   # ~0.5 GiB packed-kernel build cap
            flop_cap = 1.0e13         # ~ build that finishes in <~ a minute
            if self.backend == "auto":
                self.backend = (
                    "dense"
                    if (kernel_bytes <= mem_cap and build_flops <= flop_cap)
                    else "matrix_free"
                )
            if self.backend == "dense":
                self._build_dense_vertices(T)
                vb = (self._sp_S.numel() * self._sp_S.element_size()
                      + self._sq_S.numel() * self._sq_S.element_size())
                log.info(
                    "Nonlinear e-e operator enabled (dense sparse-symmetric"
                    f" kernel): backend=dense, {self._sp_S.shape[1]} cubic +"
                    f" {self._sq_S.shape[1]} quadratic packed triples,"
                    f" {vb / 1e6:.2f} MB (selection + permutation + reality)"
                )
            else:
                self._build_matrix_free_generator(T)
                if self.recon == "auto":  # self-calibrate the recon backend
                    self.recon = self._benchmark_recon()
                ngen = self._mf_Wk.numel()
                log.info(
                    "Nonlinear e-e operator enabled (matrix-free kinematic"
                    " generator, L-independent storage):"
                    f" {self._mf_nf} output nodes x {self._mf_nq} quadrature"
                    f" points, {ngen * self._mf_Wk.element_size() / 1e6:.1f} MB"
                    f" generator (independent of Nr beyond the psi table),"
                    f" recon={self.recon}"
                )

    def _radial_galerkin(self, T: float):
        """Shared radial Galerkin machinery for the exact-kinematics path.

        Returns ``(psi_coeff, x_fine, P, Ginv, psi0_norm)`` where
        ``psi_coeff[p, l]`` are hybrid-FEATURE coefficients of the radial
        basis ``psi_l(x) = sum_p psi_coeff[p, l] phi_p(x)`` with features
        ``phi = [1, x, v^2, v, v^4, v^3, ...]``, ``v = tanh(x/2)``
        (``x = xi/T``; the {1, x} block keeps mass/energy shapes exact),
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

        def _feats(x):
            # hybrid features [1, x, v^2, v, v^4, v^3, ...], v = tanh(x/2)
            v = torch.tanh(0.5 * x)
            cols = [torch.ones_like(x), x]
            pe, po = 2, 1
            while len(cols) < Nr:
                cols.append(v ** pe); pe += 2
                if len(cols) < Nr:
                    cols.append(v ** po); po += 2
            return torch.stack(cols[:Nr], dim=-1)

        psi_coeff = (
            torch.linalg.solve(_feats(xi_c), Tfm)
            if Nr > 1
            else torch.ones(1, 1, dtype=torch.float64)
        )
        psi0_norm = float(Tfm[0, 0])  # constant l=0 basis value

        # Fine radial quadrature for the Galerkin projection.  Keep all nodes
        # above the band bottom (xi/T > -1/t for parabolic bands):
        xg, xw = np.polynomial.legendre.leggauss(self.n_xi_proj)
        # DOMAIN.  The Galerkin overlap integrand is w_eq psi_l Phi_dot.  Phi_dot
        # is O(1) at large |x| and w_eq ~ e^{-|x|}, while psi_1 ~ xi GROWS and the
        # tanh-power modes only saturate around |x| ~ 6 -- so the integrand decays
        # like |x| e^{-|x|} and truncating at X leaves a relative tail
        # ~ (1 + X) e^{-X}.  The old fixed X = 8 leaves 3e-3 of it, and because
        # the HIGHER modes carry more tail weight the induced error GROWS WITH Nr:
        # measured against a reference projected over a converged domain, the
        # linear block was off by 5.4 % at Nr = 3, 13 % at Nr = 4 and 22 % at
        # Nr = 6, all of which vanished when the domain was matched.  X = 16 puts
        # the tail at 2e-6; the n_xi_proj >= 96 Gauss rule still converges there
        # (nearest pole of the Fermi factor is at +-i pi, giving a Bernstein
        # parameter 1.22 and an error ~1.22^-192).  Still capped short of the band
        # bottom, where the 1/(k2 k4) factors are quadrature-hostile.
        x_span = max(16.0, float(xi_c.abs().max()) + 2.0) if Nr > 1 else 16.0
        x_span = min(x_span, 0.9 / t_ratio)
        if float(xi_c.abs().max()) >= 0.95 / t_ratio:
            raise InvalidInputException(
                f"Radial truncation xi_max = {float(xi_c.abs().max()):g} T"
                f" reaches the band bottom (E_F/T = {1/t_ratio:g});"
                " reduce xi_max or T"
            )
        # Bare-x Gauss rule with n >= 96: x-block entries of the hybrid Gram
        # are exact (polynomials in x), tanh-power entries converge at the
        # sech^2 pole rate rho ~ 1.3-1.7 => <= 1e-13 at n = 96.  (A u-mapped
        # rule is DISQUALIFIED for the hybrid: atanh's log endpoint stalls the
        # x-entries at rho ~ 1.01-1.10.)
        x_fine = torch.tensor(x_span * xg, dtype=torch.float64)
        w_meas = torch.tensor(x_span * xw, dtype=torch.float64) * (
            0.25 / torch.cosh(x_fine / 2) ** 2 / T)

        def psi_eval(x):
            return _feats(x) @ psi_coeff

        Psi = psi_eval(x_fine)  # (n_fine, Nr)
        P = Psi.T * w_meas  # (Nr, n_fine): <psi_l| . >_w
        G = P @ Psi  # Gram in the fine measure (~ identity)
        if Nr > 1 and float(torch.linalg.cond(G)) > 1e3:
            raise RuntimeError(
                f"radial Galerkin Gram ill-conditioned (cond={float(torch.linalg.cond(G)):.1e});"
                " increase n_xi_proj"
            )
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
        representative rates at a refined ``n_phi`` and warn if they move by more
        than ``tol``.  The angular resolution ``n_phi`` (collinear edge +
        dealiasing) is the binding axis; ``n_xi``/``n_xi_proj`` converge
        spectrally and faster, so refining ``n_phi`` is the decisive test.

        BOTH the linear blocks and, when ``nonlinear``, the quadratic and cubic
        vertices are checked.  Checking only the linear part is NOT sufficient
        and was a real hole: measured on the GaAs 2DEG of the notes at the auto
        ``n_phi = 254``, the linear rates are converged to 0.4 % while the cubic
        ``(n=0, m=2)`` modal coefficient still moves **3.5 %** on ``n_phi ->
        508`` (and sits ~9 % below the reduction-free definition).  The vertices
        carry up to ``3 M`` harmonics and the same collinear edge, so they bind
        harder than ``L_blocks`` does.  The nonlinear leg is priced at ONE
        output node (the build itself sweeps ``n_xi_proj ~ 96``), so the check
        costs ~2 % of the build rather than 2.25x it."""
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
                n_xi=self.n_xi, xi_cut=self.xi_cut, n_phi=n_phi, g_s=self.g_s,
            )  # (len(m_chk), n_fine, Nr)
            L = torch.einsum("ij,jf,mfl->mil", Ginv, P, R)  # (len, Nr, Nr)
            return torch.stack([torch.linalg.norm(L[i]) for i in range(len(m_chk))])

        def vertex_norms(n_phi: int) -> torch.Tensor:
            """Quadratic + cubic vertex magnitudes at ONE representative output
            node (xi_1 = 0, the thermal shell) -- enough to resolve the angular
            convergence, which is node-independent, at 1/n_xi_proj of the cost."""
            kin = dict(
                x_nodes=x_fine[len(x_fine) // 2].reshape(1), psi_coeff=psi_coeff,
                M=M, kF=fs.kF, m_star=self.m_star, T=T,
                epsilon_bg=self.epsilon_bg, kappa=self.kappa,
                well_width=self.well_width, n_xi=self.n_xi,
                xi_cut=self.xi_cut, n_phi=n_phi, g_s=self.g_s,
            )
            Qc, _ = _kernels.quadratic_kernel_complex(**kin)
            Tc, Tc1, _ = _kernels._cubic_complex(**kin)
            return torch.stack([torch.linalg.norm(v.reshape(-1).abs().double())
                                for v in (Qc, Tc, Tc1)])

        n_phi_ref = int(np.ceil(1.5 * self.n_phi))
        parts = {"gamma_m": (block_norms(self.n_phi), block_norms(n_phi_ref))}
        if self.nonlinear:
            parts["vertex"] = (vertex_norms(self.n_phi), vertex_norms(n_phi_ref))
        moves = {k: ((b - a).abs() / a.clamp_min(1e-300)).max().item()
                 for k, (a, b) in parts.items()}
        rel = max(moves.values())
        detail = ", ".join(f"d({k}) = {v:.1e}" for k, v in moves.items())
        if rel > self.tol:
            log.info(
                f"WARNING: e-e quadrature may be under-resolved -- {detail}"
                f" when n_phi {self.n_phi} -> {n_phi_ref}"
                f" (tol = {self.tol:.0e}). Lower `tol` or set `n_phi` explicitly."
            )
        else:
            log.info(
                f"quadrature convergence OK: {detail}"
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
            n_phi=self.n_phi, g_s=self.g_s,
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
        proj = self._null_projectors_by_harm(self.T_build)
        self._null_proj = {
            mo: Pm.to(dtype=cdtype, device=device) for mo, Pm in proj.items()
        }
        self._nh = nh
        self._ps = torch.arange(-M, M + 1).to(device)

    def _build_dense_vertices(self, T: float) -> None:
        """Precontract the cubic + quadratic vertices into a fully-reduced SPARSE
        SYMMETRIC modal kernel for the dense backend, exploiting every exact
        symmetry simultaneously:

        * additive angular selection ``mo = sum of input m`` AND output
          truncation (only triples with ``0 <= mo <= M`` are kept -- harmonics
          outside the retained band are projected out anyway);
        * permutation symmetry of the (identical-field) input legs -- store one
          entry per UNORDERED ``(l, m)`` triple, with the leg-ordering
          multiplicity folded into the value;
        * output reality (``f_dot`` is real -> Hermitian harmonics), so only the
          ``mo >= 0`` half is stored and the ``mo < 0`` half is the conjugate;
        * reflection (``phi -> -phi``) symmetry of the isotropic operator, which
          with reality makes the complex-harmonic kernel REAL -> stored real
          (another 2x, and the apply is real-kernel x complex-field).

        Storage and apply both scale as ``~ Nr (Nr dim)^3 / 16`` (measured
        compression ~16 dim, -> 18 dim asymptotically; vs the naive dense
        ``(Nr dim)^4``).  The energy-parity ('even sigma') radial selection is
        NOT exploited: it is only leading order in ``T/E_F`` (broken by band
        curvature), so using it would be an approximation, not an exact symmetry.
        Built in fp64, stored in the working precision; the per-output-harmonic
        null projection + real fold are shared with the matrix-free path via
        ``_finalize_nonlinear``."""
        fs = self.fermi_surface
        device = rc.device
        self._build_harmonic_tables()  # _U, _R, _null_proj
        M = fs.M_theta
        nh = 2 * M + 1
        Nr = fs.Nr
        Pflat = Nr * nh  # flat leg index p = l*nh + (m + M)
        psi_coeff, x_fine, P, Ginv, _ = self._radial_galerkin(T)
        # Run the build GEMM/accumulator in the working precision and on
        # rc.device: complex64 -> ~2x faster, half-memory fp32 build (q-sum error
        # << tol); an accelerator device -> the GEMM (the build bottleneck) runs
        # on the GPU (kinematics stay fp64 on host for accuracy).
        wdt = torch.complex64 if fs.v.dtype == torch.float32 else torch.complex128
        kin = dict(
            M=M, kF=fs.kF, m_star=self.m_star, T=T, epsilon_bg=self.epsilon_bg,
            kappa=self.kappa, well_width=self.well_width, n_xi=self.n_xi,
            xi_cut=self.xi_cut, n_phi=self.n_phi,
            work_dtype=wdt, work_device=device,
        )
        pm = (torch.arange(Pflat, device=device) % nh) - M  # m of each leg index

        # Enumerate the kept UNORDERED triples / pairs once (output-node
        # independent): selection + output truncation + reality (0 <= sum m <= M)
        # AND permutation packing (p1 <= p2 <= p3).
        tri = torch.combinations(
            torch.arange(Pflat, device=device), r=3, with_replacement=True)
        sm = pm[tri[:, 0]] + pm[tri[:, 1]] + pm[tri[:, 2]]
        keep = (sm >= 0) & (sm <= M)
        tri, sm = tri[keep], sm[keep]
        p1, p2, p3 = tri[:, 0], tri[:, 1], tri[:, 2]
        e1, e2 = (p1 == p2), (p2 == p3)
        mult = torch.where(e1 & e2, 1, torch.where(e1 | e2, 3, 6))
        pr = torch.combinations(
            torch.arange(Pflat, device=device), r=2, with_replacement=True)
        smq = pm[pr[:, 0]] + pm[pr[:, 1]]
        keepq = (smq >= 0) & (smq <= M)
        pr, smq = pr[keepq], smq[keepq]
        q1, q2 = pr[:, 0], pr[:, 1]
        multq = torch.where(q1 == q2, 1, 2)

        # radial node -> output-mode projection with the +conv solver-
        # field scaling folded in (conv = 1 / w_eq = 4 T cosh^2(x1/2)):
        conv = 4.0 * T * torch.cosh(x_fine.to(torch.float64) / 2) ** 2
        # SIGN: the raw packed pieces accumulate +f_dot (occupation rate);
        # folding +conv (= 1/w_eq) makes nl = +Phi_dot_NL so that
        # a_dot = Phi_dot_lin + Phi_dot_NL is convention-consistent (the linear
        # path un-negates the decay-form L_coeff explicitly).  Verified signed
        # against the governing equation (test_a_dot_nonlinear_signed).  NOTE:
        # the standalone cubic_vertex/quadratic_vertex wrappers keep their own
        # documented -Phi_dot convention -- do not conflate the two.
        GPc = ((Ginv @ P) * conv[None, :]).to(device)  # (Nr, n_fine), real
        Nx1 = x_fine.shape[0]
        Sc = torch.zeros(Nr, p1.shape[0], dtype=wdt, device=device)
        Sq = torch.zeros(Nr, q1.shape[0], dtype=wdt, device=device)
        # CONSTRUCTION: assemble the packed kernel DIRECTLY -- the cubic part is
        # built one output node at a time by `cubic_packed_node`, which never
        # materializes the (Nr dim)^3 per-node vertex nor the g*(Nr dim)^2 GEMM
        # intermediate: it slab-blocks the (2,3,4) triple over the first leg index
        # and gathers only the kept unordered triples (plus the small leg-1
        # tensor).  Peak build memory is therefore ~ the packed kernel itself, not
        # (Nr dim)^3.  The quadratic kernel is only (Nr dim)^2 (tiny), so it is
        # formed in full and gathered.  Per-node contributions are independent, so
        # summing GPc[:, f] * packed(node f) reproduces the full-tensor build
        # exactly (verified to roundoff against the cubic_kernel_complex gather).
        for f in range(Nx1):
            s0 = _kernels.cubic_packed_node(
                x_node=x_fine[f], psi_coeff=psi_coeff,
                ti=p1, tj=p2, tk=p3, **kin, g_s=self.g_s)
            Sc += GPc[:, f:f + 1].to(s0.dtype) * s0[None, :]
            del s0
            Qc1, _c = _kernels.quadratic_kernel_complex(
                x_nodes=x_fine[f:f + 1], psi_coeff=psi_coeff, **kin, g_s=self.g_s)
            Qc1 = Qc1.reshape(Pflat, Pflat)
            q0 = 0.5 * (Qc1[q1, q2] + Qc1[q2, q1])
            Sq += GPc[:, f:f + 1].to(q0.dtype) * q0[None, :]
            del Qc1, q0
        # reflection (phi -> -phi) symmetry of the isotropic operator + reality
        # make the complex-harmonic kernel REAL (verified to ~1e-14) -> store real
        # (2x less memory, real x complex applies):
        self._sp_S = (Sc * mult[None, :]).real.to(dtype=fs.v.dtype, device=device)
        self._sp_p1, self._sp_p2, self._sp_p3 = p1, p2, p3
        self._sp_mo = sm  # output harmonic in 0..M (the mo >= 0 half)
        self._sq_S = (Sq * multq[None, :]).real.to(dtype=fs.v.dtype, device=device)
        self._sq_q1, self._sq_q2 = q1, q2
        self._sq_mo = smq
        self._dc_M, self._dc_nh = M, nh

    def _apply_dense(self, a4: torch.Tensor) -> torch.Tensor:
        """Sparse-symmetric dense apply.  Gather the field at each packed
        ``(l, m)`` triple/pair, multiply by the (multiplicity-folded) symmetric
        kernel, scatter into the ``mo >= 0`` half by the additive rule, mirror
        the ``mo < 0`` half by conjugation (output reality), and finish with the
        SAME null projection + real fold as the matrix-free path.  ``a4``:
        (..., Nr, dim) -> (..., Nr, dim)."""
        cdt = self._U.dtype
        M, nh = self._dc_M, self._dc_nh
        Nr = a4.shape[-2]
        ahat = torch.einsum("mc,...lc->...lm", self._U, a4.to(cdt))
        v = ahat.reshape(*ahat.shape[:-2], -1)  # (.., P) flat (l, m)
        Fhalf = ahat.new_zeros(*v.shape[:-1], Nr, M + 1)  # mo = 0..M
        # cubic:
        vvv = v[..., self._sp_p1] * v[..., self._sp_p2] * v[..., self._sp_p3]
        Fhalf.index_add_(-1, self._sp_mo, self._sp_S * vvv[..., None, :])
        # quadratic:
        vv = v[..., self._sq_q1] * v[..., self._sq_q2]
        Fhalf.index_add_(-1, self._sq_mo, self._sq_S * vv[..., None, :])
        # assemble full Hermitian Fhat[.., Nr, nh]: mo>=0 direct, mo<0 conjugate:
        Fhat = ahat.new_zeros(*Fhalf.shape[:-1], nh)
        Fhat[..., M:] = Fhalf                       # mo = 0..M -> idx M..2M
        Fhat[..., :M] = Fhalf[..., 1:].flip(-1).conj()  # mo = -M..-1
        return self._finalize_nonlinear(Fhat)

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
            # Add the exact cubic + quadratic operator via the selected backend
            # (dense precontracted vertex, or matrix-free quadrature).  The
            # linear part stays in L_coeff above.
            nl = (self._apply_dense(a4) if self.backend == "dense"
                  else self._apply_matrix_free(a4))
            out = out + nl.to(out.dtype)
        return out.reshape(shape_in)

    def a_dot_breakdown(self, a: torch.Tensor):
        """Split the collision contribution into (linear, quadratic, cubic).

        Returns three tensors with the same shape as ``a`` (``(..., Nr*dim)``):

        * ``lin``  -- the linearized collision operator ``-L_coeff @ a4``;
        * ``quad`` -- the part of the exact nonlinear operator that is even in
          ``a`` (quadratic-in-deltaf vertex), isolated as ``½(NL[a]+NL[-a])``;
        * ``cub``  -- the part odd in ``a`` (cubic-in-deltaf vertex), isolated
          as ``½(NL[a]-NL[-a])``.

        Using the +/- symmetrization rather than re-deriving the per-degree
        vertices keeps this consistent with the single ``NL`` path used in
        :meth:`a_dot` (and matches the warm in-loop apply, avoiding the
        standalone-apply slow path).  ``lin + quad + cub`` reproduces
        ``a_dot(a)`` to round-off.
        """
        fs = self.fermi_surface
        Nr, dim = fs.Nr, fs.angular.dim
        shape_in = a.shape
        a4 = a.reshape(*shape_in[:-1], Nr, dim)
        lin = -torch.einsum("cij,...jc->...ic", self.L_coeff, a4)
        if self.nonlinear:
            if self.backend == "dense":
                nlp = self._apply_dense(a4)
                nlm = self._apply_dense(-a4)
            else:
                nlp = self._apply_matrix_free(a4)
                nlm = self._apply_matrix_free(-a4)
            nlp = nlp.to(lin.dtype)
            nlm = nlm.to(lin.dtype)
            cub = 0.5 * (nlp - nlm)
            quad = 0.5 * (nlp + nlm)
        else:
            cub = torch.zeros_like(lin)
            quad = torch.zeros_like(lin)
        return (lin.reshape(shape_in),
                quad.reshape(shape_in),
                cub.reshape(shape_in))

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
        ``+conv`` solver-field scaling folded in, exactly as the dense path) and
        builds the harmonic transforms + null projectors (shared with the dense
        apply) so ``_finalize_nonlinear`` can be reused.
        """
        fs = self.fermi_surface
        device = rc.device
        dtype = fs.v.dtype
        psi_coeff, x_fine, P, Ginv, _ = self._radial_galerkin(T)
        t = T / self.E_F
        # Galerkin node->mode projector with +conv folded in (nl = +Phi_dot) -- the
        # output node quantity is the raw kinematic integral (occupation rate
        # f_dot, since the legs carry the physical weight w_eq); GPc maps it to
        # modal coefficients exactly as in the dense vertex build.  The solver field
        # obeys delta_f = w_eq Phi_code, so Phi_code_dot = f_dot / w_eq =
        # 4 T cosh^2(x1/2) * f_dot:
        conv = 4 * T * torch.cosh(x_fine / 2) ** 2  # 1 / w_eq(x1)
        GPc = (Ginv @ P) * conv[None, :]  # (Nr, n_fine); +conv: nl = +Phi_dot_NL

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
            n_xi=self.n_xi, n_phi=self.n_phi, g_s=self.g_s,
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
            # Store the generator in the WORKING precision (fp64 default, fp32
            # when requested): the kinematics are built in fp64 above, this only
            # sets storage/apply precision.  fp32 ~halves memory and roughly
            # doubles apply throughput; the q-sum error (~sqrt(nq) eps32 ~ 1e-4)
            # stays well under the ~1e-3 quadrature tolerance, and conservation
            # is exact by the null projection regardless of dtype.
            return x.to(dtype=dtype, device=device)

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
        # Real (Hermitian) SYNTHESIS of the leg field at the output-angle grid.
        # The modal field is real, so ahat[-m] = conj(ahat[m]); the leg phase
        # e^{i m dphi} preserves this (B[-m] = conj(B[m])) and the reconstructed
        # Phi(phi1) = sum_m B_m e^{i m phi1} is REAL.  Synthesize it from the
        # m >= 0 half alone:
        #   Phi = Re B_0 + 2 sum_{m>0} [Re B_m cos(m phi1) - Im B_m sin(m phi1)]
        #       = ReB . cos_syn - ImB . sin_syn ,
        # with the doubling (1, 2, 2, ...) folded into the (M+1, Nout) real
        # matrices.  This halves the leg contraction (only the m >= 0 half of
        # ahat enters) and replaces the complex (2M+1, Nout) synthesis with two
        # real (M+1, Nout) GEMMs -- ~2-4x less work, real arithmetic, and Phi is
        # real with no discarded imaginary part.  The fft backend uses the
        # equivalent half-spectrum irfft (see _apply_matrix_free_chunk):
        ps_pos = torch.arange(0, M + 1, dtype=torch.float64)  # m = 0..M
        cfac = torch.full((M + 1,), 2.0, dtype=torch.float64)
        cfac[0] = 1.0
        ang = ps_pos[:, None] * phi1[None, :]  # (M+1, Nout)
        self._mf_cos = (cfac[:, None] * torch.cos(ang)).to(
            dtype=dtype, device=device)  # (M+1, Nout) real
        self._mf_sin = (cfac[:, None] * torch.sin(ang)).to(
            dtype=dtype, device=device)  # (M+1, Nout) real
        self._mf_ps_pos = ps_pos.to(dtype=dtype, device=device)  # (M+1,) phase
        self._mf_Nout = Nout
        # Only the light harmonic transforms + null projectors -- NOT the dense
        # convolution tables (the O(nh^4) Bin3 would be ~70 GB at M=128 and is
        # never used by _apply_matrix_free):
        self._build_harmonic_tables()

    def _mf_psi_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Radial basis ``psi_l(x)`` via hybrid features
        ``[1, x, v^2, v, v^4, v^3, ...]``, ``v = tanh(x/2)``;
        shape ``x.shape+(Nr,)``."""
        pc = self._mf_psi_coeff
        n = pc.shape[0]
        v = torch.tanh(0.5 * x)
        cols = [torch.ones_like(x), x]
        pe, po = 2, 1
        while len(cols) < n:
            cols.append(v ** pe); pe += 2
            if len(cols) < n:
                cols.append(v ** po); po += 2
        return torch.stack(cols[:n], dim=-1).to(pc.dtype) @ pc

    def _benchmark_recon(self) -> str:
        """Time both leg-reconstruction backends on a small representative slice
        and return the faster ('gemm' or 'fft').  Self-calibrating: adapts to M,
        dtype and CPU/GPU with no hardcoded crossover.  The backend ranking is
        independent of the spatial batch and q-block size (the reconstruction is
        linear in both), so a small slice predicts the full apply.  Falls back to
        'gemm' (the safe, GPU-friendly choice) on any error."""
        import time

        try:
            M = self.fermi_surface.M_theta
            Nout, Nf, nq = self._mf_Nout, self._mf_nf, self._mf_nq
            cos_syn, sin_syn = self._mf_cos, self._mf_sin
            rdt, dev = cos_syn.dtype, cos_syn.device
            cdt = torch.complex64 if rdt == torch.float32 else torch.complex128
            Nfreq = Nout // 2 + 1
            qb = min(nq, 1024)  # small q-slice; ranking is q-linear
            ReB = torch.randn(1, Nf, qb, M + 1, dtype=rdt, device=dev)
            ImB = torch.randn(1, Nf, qb, M + 1, dtype=rdt, device=dev)

            def gemm():  # two real (M+1, Nout) synthesis GEMMs
                return (torch.einsum("...fqm,mn->...nfq", ReB, cos_syn)
                        - torch.einsum("...fqm,mn->...nfq", ImB, sin_syn))

            def fft():  # half-spectrum irfft on the m >= 0 harmonics
                spec = torch.zeros(1, Nf, qb, Nfreq, dtype=cdt, device=dev)
                spec[..., : M + 1] = torch.complex(ReB, ImB)
                return torch.movedim(
                    torch.fft.irfft(spec, n=Nout, dim=-1) * Nout, -1, -3)

            def clock(fn, reps=5):
                fn()  # warm up (allocations, FFT plan, caches)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    fn()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                return (time.perf_counter() - t0) / reps

            return "gemm" if clock(gemm) <= clock(fft) else "fft"
        except Exception:
            return "gemm"

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
        at the four legs from the modal harmonics, evaluates the cubic
        ``C3 = d1 d2 (d3+d4) - d3 d4 (d1+d2)`` and the particle-hole-odd
        quadratic ``Q2`` of ``(B-F)`` at an output-angle grid, integrates the
        quadrature, projects to output radial modes (``GPc``) and to output
        angular harmonics (DFT in ``phi_1``), then applies the SAME null
        projection + real fold as the dense path.  Returns ``[..., Nr, dim]``.

        Two exact optimizations vs. a naive evaluation:

        * REAL (Hermitian) reconstruction -- the field is real so each leg's
          harmonics ``B_m`` are Hermitian and the reconstructed ``Phi`` is real;
          it is synthesized from the ``m >= 0`` half with the real
          ``cos``/``sin`` matrices (see ``_build_matrix_free_generator``),
          halving the leg contraction and the synthesis;
        * LEG-1 HOISTING -- the output-energy leg sits at ``dphi_1 = 0`` with a
          ``q``-independent radial factor, so ``d1`` is independent of the
          quadrature point; it is reconstructed ONCE and broadcast, instead of
          being rebuilt for every quadrature point.

        NOT exploited: the reflection ``phi -> -phi`` (which maps the quadrature
        point ``(beta, root)`` to ``(2 pi - beta, other root)``).  It is an exact
        symmetry of the OPERATOR, but it relates the operator on ``delta_f`` to
        the operator on the reflected field ``delta_f(-phi)``; for a general
        (non-reflection-symmetric) input the two halves of the ``beta`` grid give
        genuinely different contributions (the leg harmonics transform as
        ``C_m e^{-i m dphi}``, not the conjugate/reverse of the originals), so it
        does NOT halve the quadrature without approximating.  Like the
        energy-parity selection in the dense path, it is therefore left out.
        """
        fs = self.fermi_surface
        M, Nr = fs.M_theta, fs.Nr
        cdt = self._U.dtype
        # complex harmonics of the modal field; only the m >= 0 half enters the
        # Hermitian (real) reconstruction:
        ahat = torch.einsum("mc,...lc->...lm", self._U, a4.to(cdt))  # (..,Nr,nh)
        aRe = ahat[..., M:].real  # (.., Nr, M+1)  m = 0..M
        aIm = ahat[..., M:].imag

        Nf, nq, Nout = self._mf_nf, self._mf_nq, self._mf_Nout
        batch = ahat.shape[:-2]
        nbatch = int(np.prod(batch)) if batch else 1
        legs = ("x2", "x3", "x4")  # leg x1 is hoisted out (q-independent)
        psi_r = {leg: self._mf_psi[leg].expand(Nf, nq, -1) for leg in legs}
        weq_q = {leg: self._mf_weq[leg].expand(Nf, nq) for leg in legs}
        f0_q = {leg: self._mf_f0[leg].expand(Nf, nq)
                for leg in ("x1", "x2", "x3", "x4")}
        dphi_q = {"x2": self._mf_dphi2, "x3": self._mf_dphi3.expand(Nf, nq),
                  "x4": self._mf_dphi4}
        cos_syn, sin_syn = self._mf_cos, self._mf_sin  # (M+1, Nout) real
        ps_pos = self._mf_ps_pos  # (M+1,) = 0..M
        use_fft = self.recon == "fft"
        Nfreq = Nout // 2 + 1  # rfft half-spectrum length

        def synth(ReB, ImB, spec, axes, mv):
            """Real reconstruction of Phi from the m>=0 harmonics (ReB, ImB).
            ``axes`` is the einsum spec for the gemm path; ``spec`` is a
            preallocated complex half-spectrum buffer and ``mv`` the movedim
            destination (placing the angle axis) for the fft path."""
            if use_fft:
                spec[..., : M + 1] = torch.complex(ReB, ImB)
                Phi = torch.fft.irfft(spec, n=Nout, dim=-1) * Nout
                return torch.movedim(Phi, -1, mv)
            return (torch.einsum(axes, ReB, cos_syn)
                    - torch.einsum(axes, ImB, sin_syn))

        # ---- leg 1 (output-energy leg): q-independent, dphi1 = 0, build once --
        psi1 = self._mf_psi["x1"][:, 0, :]  # (Nf, Nr) real
        cR1 = torch.einsum("...lm,fl->...fm", aRe, psi1)  # (.., Nf, M+1)
        cI1 = torch.einsum("...lm,fl->...fm", aIm, psi1)
        spec1 = (ahat.new_zeros(*batch, Nf, Nfreq) if use_fft else None)
        Phi1 = synth(cR1, cI1, spec1, "...fm,mn->...nf", -2)  # (.., Nout, Nf)
        d1 = (self._mf_weq["x1"][:, 0] * Phi1).unsqueeze(-1)  # (.., Nout, Nf, 1)
        f1_all = f0_q["x1"]  # (Nf, nq)

        # nq-axis chunking: the quadrature is a sum over q, accumulated in
        # q-blocks sized to a memory budget; the (Nout, Nf, nq) leg
        # reconstructions then never exceed it regardless of M / n_phi.  The
        # block is the full nq when it fits (single pass -> bit-identical and no
        # loop overhead); only larger problems split (then summation reorders at
        # the ~1e-15 level).
        per_q = Nout * Nf * 128 * nbatch  # ~peak bytes per quadrature point
        qchunk = max(1, min(nq, int((4 * 1024**3) // max(per_q, 1))))
        fdot = 0
        for q0 in range(0, nq, qchunk):
            q1 = min(q0 + qchunk, nq)
            qb = q1 - q0
            spec = (ahat.new_zeros(*batch, Nf, qb, Nfreq) if use_fft else None)
            d = []
            for leg in legs:
                psi_leg = psi_r[leg][:, q0:q1]  # (Nf, qb, Nr)
                cR = torch.einsum("...lm,fql->...fqm", aRe, psi_leg)
                cI = torch.einsum("...lm,fql->...fqm", aIm, psi_leg)
                mdphi = dphi_q[leg][:, q0:q1, None] * ps_pos  # (Nf, qb, M+1)
                cmd, smd = torch.cos(mdphi), torch.sin(mdphi)
                ReB = cR * cmd - cI * smd  # leg phase e^{i m dphi}, real part
                ImB = cR * smd + cI * cmd  # ... imaginary part
                Phi = synth(ReB, ImB, spec, "...fqm,mn->...nfq", -3)  # (..,Nout,Nf,q)
                d.append(weq_q[leg][:, q0:q1] * Phi)
            d2, d3, d4 = d
            f1 = f1_all[:, q0:q1]
            f2, f3, f4 = (f0_q[leg][:, q0:q1] for leg in legs)
            # cubic (f0-independent) + particle-hole-odd quadratic of (B-F):
            C3 = d1 * d2 * (d3 + d4) - d3 * d4 * (d1 + d2)
            Q2 = (d1 * d2 * (f3 + f4 - 1.0) + d1 * d3 * (f2 - f4)
                  + d1 * d4 * (f2 - f3) + d2 * d3 * (f1 - f4)
                  + d2 * d4 * (f1 - f3) - d3 * d4 * (f1 + f2 - 1.0))
            fdot = fdot + torch.einsum(
                "...nfq,fq->...nf", C3 + Q2, self._mf_Wk[:, q0:q1])
        # Galerkin-project the output-energy node axis onto radial modes (GPc
        # folds the +conv solver-field scaling); then DFT in phi1 to harmonics:
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
