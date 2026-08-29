"""Cartesian k-grid representation for the FermiSurface model.

A fixed uniform 2D k-grid (v = k/m*) that streams stably at arbitrary drift; the
e-e collision is applied by projecting each spatial cell's deviation onto the
model's modal basis in the LOCAL drifted-heated frame, running the model's modal
operator, reconstructing, and projecting out (n, J, E) for exact conservation.

This is the large-drift full-f backend (formerly the FermiCartesian material).
It supports the microscopic e-e collision + conservation; the lab-frame
phenomenological tau_p and cyclotron terms are NOT frame-covariant here and are
rejected at construction (use the delta-k representation for those).
"""
from __future__ import annotations
from typing import Callable, Optional
import os
import numpy as np
import torch

from qimpy import log, rc
from qimpy.mpi import ProcessGrid
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from ._representation import KRepresentation, CELL_SCALAR_NAMES, FLUX_NAMES


class Cartesian(KRepresentation):
    kF: float; vF: float; m_star: float; mu: float; T_temp: float; xi_max: float

    @staticmethod
    def recommended_grid(kF, vF, T, *, xi_max=6.0, dmu_max=0.0, kD_max=0.0,
                         te_fac_max=1.0, annulus_xi=0.0,
                         dk=None, safety_cells=3, m_star=None):
        """Minimal (k_max, n_k) whose box holds the occupied shell for the given BC.

        The OUTER margin is sized to xi_out = max(xi_max*te_fac_max, annulus_xi):
        heated cells (T_e up to te_fac_max*T) occupy out to xi_loc ~ xi_max,
        i.e. xi ~ xi_max*te_fac_max in material units, and when an annulus
        freezes the interior at depth annulus_xi the outer truncation should
        meet the same e^{-annulus_xi} standard (symmetric margins).  A drift
        shifts the shell centre by |k_D|; dmu shifts mu.  Streaming + specular
        walls preserve |k|.  dk defaults to the thermal width T/vF."""
        m = float(m_star) if m_star is not None else kF / vF
        EF = 0.5 * kF * kF / m
        dk = float(dk) if dk else T / vF
        xi_out = max(xi_max * max(te_fac_max, 1.0), annulus_xi)
        k_shell = (2 * m * (EF + abs(dmu_max) + xi_out * T)) ** 0.5
        k_max = abs(kD_max) + k_shell + safety_cells * dk
        return float(k_max), int(np.ceil(2 * k_max / dk))

    def __init__(
        self, *, fermi_surface,
        k_max: Optional[float] = None, n_k: Optional[int] = None,
        dk: Optional[float] = None, dmu_max: float = 0.0, kD_max: float = 0.0,
        grid_safety_cells: int = 3, spin: float = 2.0, circular: bool = True,
        annulus_xi: float = 0.0, te_fac_max: float = 1.0,
        local_te_rates: bool = True, projection_iters: int = 8,
        projection_tol: float = 1e-14,
        newton_iters: int = 8, frame_polish: int = 3,
        cell_chunk: int = 4096, mem_budget_gb: float = 3.0,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        super().__init__()
        fs = fermi_surface
        self.fs = fs
        if fs.tau_inv_p != 0.0 or fs.k_speed != 0.0:
            raise InvalidInputException(
                "The Cartesian representation is drift-centred; the lab-frame "
                "tau_p and cyclotron terms are unsupported.  Use tau_p=inf, r_c=inf "
                "(microscopic ee only) or the delta-k representation."
            )
        self.kF, self.vF, self.m_star = fs.kF, fs.vF, fs.m_star
        self.mu, self.T_temp, self.xi_max = fs.mu, fs.T_temp, fs.xi_max
        self.local_te_rates = bool(local_te_rates)
        self.projection_iters = int(projection_iters)
        self.projection_tol = float(projection_tol)
        self.newton_iters, self.frame_polish = newton_iters, frame_polish
        self.cell_chunk, self.mem_budget_gb = cell_chunk, mem_budget_gb
        m_star = fs.m_star; dtype = torch.get_default_dtype()

        if k_max is None or n_k is None:                     # auto-size from BC
            k_max, n_k = self.recommended_grid(
                fs.kF, fs.vF, fs.T_temp, xi_max=fs.xi_max, dmu_max=dmu_max,
                kD_max=kD_max, te_fac_max=te_fac_max, annulus_xi=annulus_xi,
                dk=dk, safety_cells=grid_safety_cells, m_star=m_star)
        dk = 2.0 * k_max / n_k
        kg = torch.arange(n_k, device=rc.device) * dk - k_max + 0.5 * dk
        KX, KY = torch.meshgrid(kg, kg, indexing="ij")
        k = torch.stack([KX.reshape(-1), KY.reshape(-1)], dim=-1).to(dtype)
        Nk = n_k * n_k
        # Square-grid bookkeeping (the wall reflector's bilinear stencil indexes
        # the underlying uniform grid even when the active set is masked):
        self.n_k_grid = n_k
        self._k_min = float(kg[0])
        self._dk_grid = dk
        # Circular active set: |k| conservation (streaming + specular walls) and
        # the bounded contact shell mean k-points beyond |k| ~ k_max are NEVER
        # occupied -- but their box-corner speed sqrt(2)*k_max/m* sets the CFL dt.
        # Dropping them is exact physics: dt grows x sqrt(2) and Nk falls ~21%.
        # Keep a 1.5*dk guard ring so reflection stencils of active points find
        # their neighbors (occupancy there < e^-7: dropped stencil corners beyond
        # it are negligible).
        self.annulus_xi = float(annulus_xi)
        if annulus_xi > 0.0 and not circular:
            raise InvalidInputException("annulus_xi requires circular=True")
        self._n_frozen = self._E_frozen = 0.0
        self._p_frozen = torch.zeros(2, device=rc.device, dtype=dtype)
        self._eps_dos_full = None
        self._k_polish_full = None
        self._annulus_on = False
        if circular:
            r_act = k_max + 1.5 * dk
            act = k.square().sum(-1) <= r_act * r_act
            if annulus_xi > 0.0:
                # Annulus active set: the deep Fermi sea is FROZEN at exactly
                # f0_lab (deviations there are bounded by e^-annulus_xi of the
                # shell scale: streaming preserves a k-uniform state and both
                # contacts and collisions only touch the shell).  Its moments
                # enter frame recovery and observables as precomputed
                # constants; its (isotropic) flux sums vanish to roundoff.
                # NOT bit-identical to the full grid -- roundoff-equivalent
                # (interior dynamics are ~1e-15); opt-in knob, default off.
                r_in = float(np.sqrt(max(
                    2 * m_star * (self.mu - abs(dmu_max)
                                  - annulus_xi * self.T_temp), 0.0)))
                r_in = max(r_in - abs(kD_max) - 1.5 * dk, 0.0)
                # Heated cells reach the frozen zone as e^{-annulus_xi*T/Te}:
                # for Te_max/T ~ 2 the deviation clipped there is e^{-axi/2}
                # (1.2e-3 at axi=12).  Size axi >= xi_max*(Te_max/T) + ~4.
                if annulus_xi < 2.0 * self.xi_max + 4.0:
                    log.warning(
                        f"annulus_xi = {annulus_xi:g} < 2*xi_max+4 ="
                        f" {2*self.xi_max+4:g}: heated cells (Te ~ 2T) leak"
                        f" ~e^(-annulus_xi/2) into the frozen sea")
                frozen = act & (k.square().sum(-1) < r_in * r_in)
                act = act & ~frozen
                wk0 = float(spin) * dk * dk / (2.0 * np.pi) ** 2
                eps_full = k.square().sum(-1) / (2 * m_star)
                f0_full = torch.special.expit(-(eps_full - self.mu) / self.T_temp)
                self._n_frozen = float(wk0 * f0_full[frozen].sum())
                self._p_frozen = wk0 * (f0_full[frozen, None]
                                        * k[frozen]).sum(0).to(dtype)
                self._E_frozen = float(wk0 * (f0_full[frozen]
                                              * eps_full[frozen]).sum())
                # frame-recovery Newton needs the FULL-band density of states
                # (the FD model's moments must be comparable to the totals):
                self._eps_dos_full = eps_full[act | frozen].to(dtype)
                # k over the SAME full-band point set -- MUST be sliced here,
                # before `k = k[act]` below rebinds k to the active subset.
                # (Indexing the reduced k with the full-grid mask raised
                # IndexError: annulus + frame_polish had no test coverage.)
                self._k_polish_full = k[act | frozen]
                self._annulus_on = bool(int(frozen.sum()))
            self._full2act = torch.full((Nk,), -1, dtype=torch.long, device=rc.device)
            self._full2act[act] = torch.arange(int(act.sum().item()), device=rc.device)
            k = k[act]
            Nk = k.shape[0]
        else:
            self._full2act = None
        self.dk_area = dk * dk
        self.Nk = Nk
        # Envelope-guard rings: innermost/outermost active bands (2.5 dk wide).
        # A healthy run never carries significant deviation there; reaching the
        # edge means the declared envelope (annulus_xi / kD_max / box) is
        # exceeded and the state is being silently clipped.
        kr2 = k.square().sum(-1)
        r_hi = float(kr2.max().sqrt())
        self._ring_out = torch.where(kr2 > (r_hi - 2.5 * dk) ** 2)[0]
        if circular and annulus_xi > 0.0 and self._annulus_on:
            r_lo = float(kr2.min().sqrt())
            self._ring_in = torch.where(kr2 < (r_lo + 2.5 * dk) ** 2)[0]
        else:
            self._ring_in = None
        self._env_warn_count = 0
        # Physical BZ phase-space weight: n = spin * int d^2k/(2pi)^2 f  ->  per-k
        # weight spin*dk^2/(2pi)^2.  (Gives n = kF^2/2pi at equilibrium, and makes
        # contact currents physical so I_set is in a.u. current, 20uA=3.02e-3.)
        self.wk = float(spin) * self.dk_area / (2.0 * np.pi) ** 2
        self.k = k
        self.eps_k = (k.square().sum(-1) / (2 * m_star)).to(dtype)
        self.v = (k / m_star).to(dtype)                      # (Nk, 2) transport velocity

        # density of states for O(nb) frame recovery (Newton moments are 1D in
        # eps).  Under the annulus the FD-model moments must span the FULL band
        # (targets include the frozen constants), so bin the full-grid energies
        # and use a finer histogram for the polish stage as well.
        eps_dos = self.eps_k if self._eps_dos_full is None else self._eps_dos_full
        nb = min(4096 if self._eps_dos_full is None else 16384,
                 int(eps_dos.shape[0]))
        edges = torch.linspace(float(eps_dos.min()), float(eps_dos.max()),
                               nb + 1, device=rc.device, dtype=dtype)
        self._dos_eps = 0.5 * (edges[1:] + edges[:-1])
        idx = torch.bucketize(eps_dos, edges[1:-1])
        self._dos_g = torch.zeros(nb, device=rc.device, dtype=dtype).scatter_add_(
            0, idx, torch.ones_like(eps_dos))
        # Under the annulus keep the full-band energies for the EXACT polish
        # stage (the 16k histogram alone leaves a ~2e-4 systematic in Te --
        # amplified (mu/T)^2-fold from total-E binning noise; the full-eps
        # polish restores machine recovery, validated to 9e-14 T):
        self._eps_polish = eps_dos if self._annulus_on else None
        # k over the SAME point set the polish sums over, so the
        # polish can use drift-centred energies (see _recover_frame).
        self._k_polish = self._k_polish_full if self._annulus_on else None
        self._eps_dos_full = None

        # torch.compile the per-step table builders: they are long elementwise
        # chains over (cells, Nk) and Inductor fuses them into single kernels.
        # Measured on the mixer grid at M=24: _fourier 123 -> 31 ms/1792cells,
        # output bit-identical.  Opt out with QIMPY_COLL_COMPILE=0.
        self._fourier_fn = self._fourier
        self._psi_fn = self._psi
        if os.environ.get("QIMPY_COLL_COMPILE", "1") != "0":
            try:
                # NOT plain max-autotune: cudagraphs reuse output buffers,
                # and psi would be clobbered by the fourier call that follows.
                mode = os.environ.get("QIMPY_COLL_COMPILE_MODE",
                                      "max-autotune-no-cudagraphs")
                self._fourier_fn = torch.compile(self._fourier, mode=mode)
                self._psi_fn = torch.compile(self._psi, mode=mode)
            except Exception:
                self._fourier_fn, self._psi_fn = self._fourier, self._psi
        self._psi_coeff = self._radial_poly_coeffs(dtype)
        self._ang_norm = torch.tensor(
            [1.0 / (2 * np.pi)] + [1.0 / np.pi] * (2 * fs.M_theta),
            dtype=dtype, device=rc.device)
        self._f0_lab = torch.special.expit(-(self.eps_k - self.mu) / self.T_temp)
        # Radial Gram.  The raw projection returns (G @ c) / T rather than the
        # operator contract's coefficients c of Phi = delta_f/(f0(1-f0)/T), so
        # apply_collision left-applies T * G^-1 and the reconstruction divides
        # its weight by T.
        #
        # WHICH G matters.  The forward projection is a discrete sum over THIS
        # k-grid; for the round trip to be the identity, G must be the Gram of
        # that same sum.  G_band below is instead a 4001-point 1-D rule in xi
        # -- the CONTINUUM Gram.  The two disagree because the Cartesian grid
        # is coarse in xi (dxi ~ 1 at the shell for the default dk = T/vF, so
        # ~12 points across |xi| < xi_max) and because the mask cuts along a
        # staircase of lattice points that the smooth 1-D rule knows nothing
        # about.  Result: Pi = R P = G_band^-1 G_disc = I +- eps, a few percent
        # -- which is why test_cartesian_projection_contract only asserts 2e-2,
        # and it does so at 3x finer dk than production.
        #
        # `discrete_gram` uses the per-cell G_disc actually realized by the
        # k-sum instead, which fixes the RADIAL half by construction.
        #
        # MEASURED (production dk = T/vF, kD = 0.37 cell, Nr = 4, m >= 2 probes;
        # /tmp/gram_probe.py, gram_vsM.py in session):
        #
        #     M     ||Pi - I||inf    lam(sym Pi)        fine-rule -> discrete
        #      6      1.9e-3      [0.9896, 1.0076]
        #     12      3.7e-3      [0.9751, 1.0194]      3.69e-3 -> 3.69e-3
        #     24      4.2e-3      [0.9487, 1.0395]      0.0395  -> 0.0393
        #     48      4.4e-3      [0.9099, 1.0804]      0.0804  -> 0.0803
        #
        # Two conclusions.  (1) discrete_gram is a MEASURED NO-OP at production
        # dk: the residual is entirely ANGULAR quadrature, which a radial Gram
        # cannot touch.  Default OFF so prior results stay bit-identical; kept
        # because it is the correct normalization and does help at finer dk
        # (1.4e-3 -> 9.7e-4 at dk/3, kD = 0).  (2) The eigenvalue spread GROWS
        # WITH M -- so raising M does not monotonically improve C[f] on this
        # grid, and the residual closure's -g(1 - Pi) can inject at up to
        # (lam_max - 1) * gamma_res: 3% of gamma_res at M = 6 but 8% at M = 48.
        xi_f = torch.linspace(-fs.xi_max, fs.xi_max, 4001,
                              device=rc.device, dtype=dtype)
        w_f = 0.25 / torch.cosh(0.5 * xi_f) ** 2
        psi_f = self._psi(xi_f)                              # (nfine, Nr)
        G_band = torch.einsum("xn,xm,x->nm", psi_f, psi_f, w_f) \
            * float(xi_f[1] - xi_f[0])
        self._Ginv_band = torch.linalg.inv(G_band)

        # Even-m selector over the flattened (radial, angular) modal ordering.
        # Angular layout: index 0 is m = 0, then (cos, sin) pairs at 2m-1, 2m,
        # so the harmonic of angular index c is m = (c + 1) // 2.  Used by the
        # residual closure, which may only touch even harmonics.
        dim_t = self.fs.angular.dim
        m_of_c = (np.arange(dim_t) + 1) // 2
        self._even_m_flat = torch.as_tensor(
            np.tile((m_of_c % 2 == 0).astype(float), self.fs.Nr),
            dtype=dtype, device=rc.device)

    # ---- radial polynomial evaluation at arbitrary xi (reuse the model's psi_n) ----
    def _radial_poly_coeffs(self, dtype) -> torch.Tensor:
        Nr = self.fs.Nr
        if Nr == 1:
            return torch.ones((1, 1), dtype=dtype, device=rc.device)
        xi_c = self.fs.radial.xi.cpu().numpy()
        v = np.tanh(0.5 * xi_c)
        cols = [np.ones_like(xi_c), xi_c]
        pe, po = 2, 1
        while len(cols) < Nr:
            cols.append(v ** pe); pe += 2
            if len(cols) < Nr:
                cols.append(v ** po); po += 2
        F = np.stack(cols[:Nr], axis=-1)     # hybrid features [1, x, v^2, v, ...]
        Tfm = self.fs.radial.T_from_modes.cpu().numpy()
        return torch.as_tensor(np.linalg.solve(F, Tfm), dtype=dtype, device=rc.device)

    def _psi(self, xi: torch.Tensor) -> torch.Tensor:
        Nr = self.fs.Nr
        if Nr == 1:
            return torch.ones((*xi.shape, 1), dtype=xi.dtype, device=xi.device)
        v = torch.tanh(0.5 * xi)
        cols = [torch.ones_like(xi), xi]
        pe, po = 2, 1
        while len(cols) < Nr:
            cols.append(v ** pe); pe += 2
            if len(cols) < Nr:
                cols.append(v ** po); po += 2
        return torch.stack(cols[:Nr], dim=-1) @ self._psi_coeff

    def _fourier(self, th: torch.Tensor) -> torch.Tensor:
        """Real harmonics [1, cos t, sin t, cos 2t, sin 2t, ...] at each angle."""
        M = self.fs.M_theta
        m = torch.arange(1, M + 1, device=th.device, dtype=th.dtype)
        mth = th.unsqueeze(-1) * m
        cols = torch.empty((*th.shape, 2 * M + 1), dtype=th.dtype, device=th.device)
        cols[..., 0] = 1.0
        cols[..., 1::2] = torch.cos(mth)
        cols[..., 2::2] = torch.sin(mth)
        return cols

    # ---- local drifted-heated frame recovery from moments ----
    def _recover_frame(self, f: torch.Tensor):
        w = self.wk
        n = (f.sum(-1) * w + self._n_frozen).clamp_min(1e-30)
        p = torch.einsum("ck,kd->cd", f, self.k) * w + self._p_frozen
        E = torch.einsum("ck,k->c", f, self.eps_k) * w + self._E_frozen
        kD = p / n[:, None]
        E_rest = E - (kD.square().sum(-1) / (2 * self.m_star)) * n
        Te = torch.full_like(n, self.T_temp); mu = torch.full_like(n, self.mu)
        Te, mu = self._newton_TeMu(Te, mu, n, E_rest,
                                   self._dos_eps[None, :], self._dos_g[None, :] * w,
                                   self.newton_iters)
        # POLISH: full 4x4 Newton on (kD_x, kD_y, Te, mu) against the EXACT
        # grid moments (n, p_x, p_y, E).
        #
        # Two things were wrong with the 2x2 version.  (1) It fitted the model
        # using the LAB energies |k|^2/2m*, but the model it represents is
        # centred at kD, with energies |k - kD|^2/2m*; the two agree only at
        # kD = 0.  (2) kD itself was never solved for -- kD = p/n is exact only
        # in the continuum, where the model's momentum moment is n*kD by
        # symmetry; on a lattice it is n*kD plus a quadrature error.
        #
        # Newton had CONVERGED in both cases (more polish iterations changed
        # nothing to every digit) -- it was solving the wrong equations.
        # Measured on an exact drifted-heated FD at kD = 3 cells, Te = 1.6 T,
        # dmu = 2 T, the leftover |df_loc|/|df| was 1.2e-2.
        k_pol = self.k if self._k_polish is None else self._k_polish
        return self._newton_frame(kD, Te, mu, n, p, E, k_pol, w,
                                  self.frame_polish)

    def _newton_frame(self, kD, Te, mu, n, p, E, k_pol, w, iters):
        """Match (n, p_x, p_y, E) exactly by Newton on (kD, Te, mu).

        Derivatives of f_le = sigma(-(|k-kD|^2/2m* - mu)/Te):
            df/dmu    = w_eq/Te
            df/dTe    = w_eq x/Te
            df/dkD_j  = w_eq (k_j - kD_j)/(m* Te)
        so the Jacobian is J[a,b] = sum_k w phi_a df/dparam_b with the moment
        set phi = (1, k_x, k_y, eps_lab) -- 16 reductions per pass, cheap next
        to everything else in the apply.
        """
        m2 = 2.0 * self.m_star
        eps_lab = k_pol.square().sum(-1) / m2
        tgt = torch.stack([n, p[:, 0], p[:, 1], E], dim=-1)          # (c,4)
        eye = torch.eye(4, device=kD.device, dtype=kD.dtype)
        for _ in range(max(int(iters), 1)):
            kp = k_pol[None] - kD[:, None]                           # (c,Nk,2)
            x = (kp.square().sum(-1) / m2 - mu[:, None]) / Te[:, None]
            f = torch.special.expit(-x)
            c_mu = (f * (1 - f)) / Te[:, None]                       # w_eq/Te
            drv = (c_mu.unsqueeze(-1)
                   * torch.stack([kp[..., 0] / self.m_star,
                                  kp[..., 1] / self.m_star,
                                  x, torch.ones_like(x)], dim=-1))   # (c,Nk,4)
            phis = (None, k_pol[:, 0], k_pol[:, 1], eps_lab)         # None = 1
            J = torch.empty(kD.shape[0], 4, 4, device=kD.device, dtype=kD.dtype)
            cur = torch.empty(kD.shape[0], 4, device=kD.device, dtype=kD.dtype)
            for a, ph in enumerate(phis):
                if ph is None:                                       # phi = 1
                    cur[:, a] = f.sum(-1) * w
                    J[:, a, :] = drv.sum(1) * w
                else:
                    cur[:, a] = (f * ph).sum(-1) * w
                    J[:, a, :] = torch.einsum("k,ckb->cb", ph, drv) * w
            J = J + 1e-13 * float(J.diagonal(dim1=-2, dim2=-1).abs().mean()) * eye
            d = torch.linalg.solve(J, (tgt - cur).unsqueeze(-1))[..., 0]
            kD = kD + d[:, :2]
            Te = (Te + d[:, 2]).clamp_min(0.05 * self.T_temp)
            mu = mu + d[:, 3]
        return kD, Te, mu

    def _newton_TeMu(self, Te, mu, n, E_rest, eps, gw, iters):
        ge = gw * eps
        for _ in range(iters):
            xi = (eps - mu[:, None]) / Te[:, None]
            f0 = torch.special.expit(-xi); weq = f0 * (1 - f0)
            nn = (gw * f0).sum(-1); EE = (ge * f0).sum(-1)
            g_mu = weq / Te[:, None]; g_Te = weq * xi / Te[:, None]
            J11 = (gw * g_mu).sum(-1); J12 = (gw * g_Te).sum(-1)
            J21 = (ge * g_mu).sum(-1); J22 = (ge * g_Te).sum(-1)
            det = J11 * J22 - J12 * J21
            det = torch.where(det.abs() < 1e-30, torch.ones_like(det), det)
            r1 = n - nn; r2 = E_rest - EE
            mu = mu + (J22 * r1 - J12 * r2) / det
            Te = (Te + (-J21 * r1 + J11 * r2) / det).clamp_min(0.05 * self.T_temp)
        return Te, mu

    # ---- collision: drift-centred projection -> model's modal_op -> reconstruct ----
    @torch.no_grad()
    def apply_collision(self, rho: torch.Tensor, modal_op: Callable) -> torch.Tensor:
        shape_in = rho.shape
        df_lab = rho.reshape(-1, shape_in[-1])               # (C, Nk) delta-f about f0_lab
        C = df_lab.shape[0]
        out = torch.empty_like(df_lab)
        dim = self.fs.angular.dim; Nr = self.fs.Nr; Nk = df_lab.shape[-1]
        k = self.k; f0_lab = self._f0_lab
        # +10 rows when the residual closure is on: the parity reflection builds
        # (c,Nk,2) grid coordinates plus per-corner index/weight temporaries.
        closure_rows = 14 if self.fs.gamma_residual() is not None else 0
        bytes_per_row = Nk * (2 * dim + 3 * Nr + 12 + closure_rows) * 8
        chunk = max(1, min(self.cell_chunk,
                           int(self.mem_budget_gb * (2 ** 30) / bytes_per_row)))
        for lo in range(0, C, chunk):
            hi = min(lo + chunk, C)
            df = df_lab[lo:hi]
            f = f0_lab[None, :] + df
            kD, Te, mu = self._recover_frame(f)
            kp = k[None] - kD[:, None]
            eps_p = kp.square().sum(-1) / (2 * self.m_star)
            xi = (eps_p - mu[:, None]) / Te[:, None]
            th = torch.atan2(kp[..., 1], kp[..., 0])
            f0_loc = torch.special.expit(-xi); weq = f0_loc * (1 - f0_loc)
            df_loc = f - f0_loc
            psi = self._psi_fn(xi)                           # (c, Nk, Nr)
            fou = self._fourier_fn(th)                       # (c, Nk, dim)
            mask = (xi.abs() < self.xi_max)
            jac = self.dk_area / (self.m_star * Te)
            gp = (df_loc * mask).unsqueeze(-1) * psi
            a = torch.bmm(gp.transpose(1, 2), fou * self._ang_norm)
            a = a * jac[:, None, None]                       # raw (G @ c)/T
            # Operator contract: coefficients c of Phi = delta_f/(f0(1-f0)/T).
            # Normalize by the Gram the k-sum ACTUALLY realizes, per cell, on
            # the same masked point set:
            #   G_disc[n,m] = (jac/2pi) sum_k mask w_eq psi_n psi_m,
            # whose continuum limit is exactly G_band (the angular factor
            # F_d^2 N_d has mean 1/2pi for EVERY d, m = 0 included, so one
            # radial Gram serves all angular channels and only the angular
            # quadrature error survives).  This makes R.P the identity by
            # construction instead of to a few percent.
            if self.projection_iters:
                # EXACT projection by iterative refinement.
                #
                # R.P is the identity only if the normalization is the Gram
                # that the k-sum ACTUALLY realizes, G = B^T W B over the JOINT
                # (radial, angular) index -- not a 1-D rule in xi, and not the
                # radial block alone: the dominant error is ANGULAR (a square
                # lattice samples a ring at uneven angles, worse the more
                # wiggles the harmonic has, which is why ||Pi - I|| GREW with
                # M: lam(sym Pi) spread [0.990,1.008] at M=6 -> [0.910,1.080]
                # at M=48).  Building G costs Nk*D^2 and is far too slow.
                #
                # But APPLYING G is free: G a = fwd(rec(a)), i.e. reconstruct
                # then re-project, both already here and both reusing the psi
                # and fourier tables built once above.  So solve G a = raw by
                # Richardson iteration preconditioned with the old fine-rule
                # normalization -- which the measurement shows is already
                # within 4-8%, so the iteration contracts by ~0.04-0.08 and
                # reaches round-off in a handful of passes.
                #
                # This makes Pi^2 = Pi to solver tolerance, hence (1 - Pi) a
                # true projector and the residual closure's -g(1 - Pi)
                # EXACTLY dissipative (eigenvalues 0 and -g) rather than
                # merely small.  It also removes the upper bound on M.
                fou_n = fou * self._ang_norm
                jac3 = jac[:, None, None]

                def fwd(u):                                  # (c,Nk) -> (c,Nr,dim)
                    return torch.bmm(((u * mask).unsqueeze(-1) * psi
                                      ).transpose(1, 2), fou_n) * jac3

                def rec(am):                                 # (c,Nr,dim) -> (c,Nk)
                    return (weq / self.T_temp) * (
                        psi * torch.bmm(fou, am.transpose(1, 2))).sum(-1)

                def prec(r):
                    return self.T_temp * torch.einsum(
                        "nm,cmd->cnd", self._Ginv_band, r)

                # MEASURED with the frame PINNED (so _recover_frame cannot
                # contaminate the number): ||Pi - I|| tracks the solver
                # tolerance exactly -- 1.0e-10 at tol 1e-10, 4.1e-12 at 1e-14
                # (M=24) and 4.5e-12 (M=48).  4e-12 is the fp64 accumulation
                # floor for a reduction over Nk ~ 3e4 points, so the projection
                # is EXACT to round-off and nothing further is recoverable
                # here.  With the frame merely recovered rather than pinned the
                # apparent error sits at 1.1e-5 -- that is frame recovery, a
                # separate and now-dominant floor, NOT the projection.
                #
                # Preconditioned CG on the normal equations is not needed:
                # G is symmetric positive definite in the (jac*mask*w_eq/T)
                # metric, so plain PCG applies.  Richardson stalls near a
                # contraction of ~0.6 on the worst mode; CG's Krylov
                # acceleration takes the same preconditioner to round-off.
                raw = a
                scale = raw.abs().amax().clamp(min=1e-300)
                a = prec(raw)
                r = raw - fwd(rec(a))
                z = prec(r); pdir = z.clone()
                rz = (r * z).sum()
                self._proj_it, self._proj_resid = 0, float(
                    r.abs().amax() / scale)
                for it in range(1, self.projection_iters + 1):
                    if self._proj_resid <= self.projection_tol:
                        break
                    Gp = fwd(rec(pdir))
                    pGp = (pdir * Gp).sum()
                    if float(pGp.abs()) < 1e-300:
                        break
                    alpha = rz / pGp
                    a = a + alpha * pdir
                    r = r - alpha * Gp
                    z = prec(r)
                    rz_new = (r * z).sum()
                    pdir = z + (rz_new / rz) * pdir
                    rz = rz_new
                    self._proj_it = it
                    self._proj_resid = float(r.abs().amax() / scale)
            else:
                a = self.T_temp * torch.einsum("nm,cmd->cnd", self._Ginv_band, a)
            a = a.reshape(hi - lo, Nr * dim)
            # Local-T_e rates: hand the per-cell recovered T_e to the modal
            # operator -- evaluated EXACTLY there when the model has a local_te
            # ensemble (analytic T_e^2 T^(1-d) prefactors x tabulated shape),
            # else by the leading-order uniform (T_e/T)^2 rescale.
            te = Te if self.local_te_rates else None
            a_dot = modal_op(a, te=te)                       # <-- model's modal collision
            # Residual closure: the modes above M_theta / N_r are NOT filtered
            # -- the increment is built from the retained basis, so it has zero
            # component there and simply leaves them undamped.  Relax the EVEN
            # part of that residual at the top retained even rate:
            #     ddf -= g * (even.df - Pi_even df).
            # Since the reconstruction R is linear,
            #     R(a_dot) - g*(even.df - R(a_even))
            #       = R(a_dot + g*a_even) - g*even.df,
            # so the mode-space half costs one axpy on the (tiny) coefficients
            # and needs NO second reconstruction.  Neither side is masked: R is
            # evaluated at every k, so masking only the df side would leave a
            # bare +g*Pi.df outside the band -- a one-signed secular source
            # rather than a damping.
            # _project_conserved runs on the total, so conservation is exact by
            # construction regardless of this term's size or sign.
            # NOTE: dissipativity requires ||Pi|| <= 1, i.e. Pi idempotent in the
            # w_eq inner product.  _Ginv_band is a 1-D fine rule, NOT the actual
            # discrete masked Gram of this k-grid, so Pi is only approximately a
            # projector (the repo's own P.R contract tests hold it to 2e-2 at 3x
            # finer dk than production).  MEASURE lambda_max(Pi) on the
            # production grid before treating this term as strictly dissipative.
            g_res = self.fs.gamma_residual(Te if self.local_te_rates else None)
            if g_res is not None:
                g_col = (g_res.unsqueeze(-1) if torch.is_tensor(g_res)
                         else g_res)
                # PARITY: only the EVEN-m residual may be damped.  Odd angular
                # harmonics are gated to zero at leading order (K_m = 0 for odd
                # m) and are only weakly relaxed by the exact operator -- these
                # are the long-lived tomographic modes, and a single rate would
                # destroy exactly the physics worth resolving.  So the closure
                # acts on  even(df_loc) - Pi_even df_loc.
                # The mode-space half is free: zero the odd-m coefficients.
                # The k-space half needs one reflection about k_D, since
                # even(df_loc)(k') = 1/2 [df_loc(k') + df_loc(-k')].
                # f0_loc depends only on |k'|, so it is even and drops out of
                # the reflection.
                # What is gathered is df_loc, NOT the lab-frame df: df_loc is
                # identically zero at the local drifted-heated equilibrium, so
                # ANY linear interpolation of it is exactly zero there and the
                # closure vanishes on the null state C[f_le] = 0 by
                # construction.  Gathering the lab df instead would put the
                # interpolation error on the O(1) shifted-shell structure --
                # an ABSOLUTE source ~1e-2 independent of the residual, and
                # invisible at kD = 0 where the reflection is an exact grid
                # permutation.
                # Stencil weight that falls outside the active set is simply
                # dropped (df_loc treated as 0 there), which is right on both
                # edges: in the frozen sea BOTH f and f0_loc are saturated, so
                # df_loc ~ e^-annulus_xi (1e-7 at the recommended axi >= 16),
                # and beyond the outer guard ring occupancy is < e^-7.  An
                # earlier version completed the dropped weight with
                # f0_lab(k_r) - f0_loc; that is the deviation about the LAB
                # sea, not the local one, and it injected a source 15x the
                # true residual at the drifted-heated equilibrium.
                r = torch.zeros_like(df_loc)
                for idx, wt in self._reflect_about(kD):
                    r = r + wt * torch.gather(df_loc, -1, idx)
                dfl_even = 0.5 * (df_loc + r)
                a_dot = a_dot + g_col * (a * self._even_m_flat)
            ad = a_dot.reshape(hi - lo, Nr, dim)
            B = torch.bmm(fou, ad.transpose(1, 2))
            # w_eq_RB = f0(1-f0)/T -- the /T completes the contract (audit bug 2):
            ddf = (weq / self.T_temp) * (psi * B).sum(-1)
            if g_res is not None:
                ddf = ddf - g_col * dfl_even
            out[lo:hi] = self._project_conserved(ddf, kp, eps_p, weq)
        return out.reshape(shape_in)

    def _reflect_about(self, kD: torch.Tensor):
        """Bilinear stencil for the drift-centred parity map k' -> -k'.

        In lab coordinates the reflected point is ``k_r = 2 k_D - k``.  Same
        construction as the specular wall reflector, but k_D is per-cell so the
        stencil is built per chunk rather than once.  Yields (index, weight)
        pairs over the ACTIVE set; stencil corners outside it (guard ring, or
        the frozen sea when the annulus is on) get zero weight, which is exact
        for the quantity gathered here -- see :meth:`apply_collision`, where
        what is gathered is delta-f about f0_lab and delta-f is identically
        zero in the frozen sea.
        """
        n_k = self.n_k_grid
        g = ((2.0 * kD[:, None, :] - self.k[None] - self._k_min)
             / self._dk_grid)                                # (c, Nk, 2)
        # Mask BEFORE clamping.  The specular reflector may clamp because
        # mirror reflection preserves |k|, so its images stay in the box; the
        # DRIFT reflection preserves |k - kD| and moves |k| by up to 2|kD|, so
        # images do leave the box and clamping would silently gather an edge
        # value at full weight instead of dropping it.
        ok = ((g >= 0.0) & (g <= n_k - 1.0)).all(-1)         # (c, Nk)
        g = g.clamp(0, n_k - 1.0001)
        i0 = g.floor()
        fx, fy = (g[..., 0] - i0[..., 0]), (g[..., 1] - i0[..., 1])
        i0 = i0.long(); ix, iy = i0[..., 0].clone(), i0[..., 1].clone()
        del g, i0                                            # free (c,Nk,2) temps
        for idx, wt in ((ix * n_k + iy, (1 - fx) * (1 - fy)),
                        ((ix + 1) * n_k + iy, fx * (1 - fy)),
                        (ix * n_k + iy + 1, (1 - fx) * fy),
                        ((ix + 1) * n_k + iy + 1, fx * fy)):
            wt = wt * ok
            if self._full2act is not None:
                a = self._full2act[idx]
                wt = wt * (a >= 0)
                idx = a.clamp(min=0)
            yield idx, wt

    def _project_conserved(self, ddf, kp, eps_p, weq):
        w = self.wk
        px, py = kp[..., 0], kp[..., 1]
        psi = torch.stack([torch.ones_like(px), px, py, eps_p], dim=1)   # (c,4,Nk)
        g = psi * weq.unsqueeze(1)
        M = torch.einsum("cak,cbk->cab", g, psi) * w
        U = torch.einsum("cak,ck->ca", psi, ddf) * w
        eye = 1e-30 * torch.eye(4, device=ddf.device, dtype=ddf.dtype)
        lam = torch.linalg.solve(M + eye, U.unsqueeze(-1))[..., 0]
        return ddf - torch.einsum("ca,cak->ck", lam, g)

    @torch.no_grad()
    def check_envelope(self, rho: torch.Tensor) -> None:
        """Warn (throttled) when the state carries significant deviation at the
        k-grid edges -- the declared envelope is being exceeded and clipped."""
        df = rho.reshape(-1, self.Nk)
        m = float(df.abs().max())
        if m <= 0.0:
            return
        for ring, nm, fix in ((self._ring_in, "inner (annulus)",
                               "increase annulus_xi and/or kD_max"),
                              (self._ring_out, "outer (box)",
                               "increase te_fac_max/dmu_max/kD_max box margins")):
            if ring is None:
                continue
            r = float(df[:, ring].abs().max()) / m
            if r > 1e-2:
                if self._env_warn_count % 100 == 0:
                    log.warning(
                        f"k-grid {nm} edge carries {r:.1%} of the peak"
                        f" deviation -- state is being clipped; {fix}")
                self._env_warn_count += 1

    def get_density_weight(self) -> torch.Tensor:
        return torch.full_like(self.eps_k, self.wk)       # (Nk,) BZ weight -> physical current

    def get_cell_scalar_names(self) -> list[str]:
        return CELL_SCALAR_NAMES

    @torch.no_grad()
    def get_cell_scalars(self, rho: torch.Tensor) -> torch.Tensor:
        """Full local fields from the drifted-heated frame recovery of f = f0 + df:
        [density, energy_density, temperature, chemical_potential, velocity_x,
        velocity_y], (K, 6), physical (phase-space weight wk included)."""
        df_lab = rho.reshape(-1, self.Nk)
        C = df_lab.shape[0]
        w = self.wk
        out = torch.empty((C, 6), device=df_lab.device, dtype=df_lab.dtype)
        per_row = self.Nk * 8 * 8                          # ~8 (C,Nk) temporaries
        chunk = max(1, min(self.cell_chunk,
                           int(self.mem_budget_gb * (2 ** 30) / per_row)))
        for lo in range(0, C, chunk):
            hi = min(lo + chunk, C)
            f = self._f0_lab[None, :] + df_lab[lo:hi]
            kD, Te, mu = self._recover_frame(f)
            n = f.sum(-1) * w + self._n_frozen
            E = torch.einsum("ck,k->c", f, self.eps_k) * w + self._E_frozen
            u = kD / self.m_star                           # drift velocity <v>
            out[lo:hi] = torch.stack([n, E, Te, mu, u[:, 0], u[:, 1]], dim=1)
        return out

    def get_flux_names(self) -> list[str]:
        return FLUX_NAMES

    def get_flux_weights(self) -> torch.Tensor:
        # Physical face fluxes (phase-space weight wk): particle j=int f v,
        # momentum flux Pi_i = int f (m* v_i) v = int f k_i v, heat q = int f eps v.
        w = self.wk
        one = torch.ones_like(self.eps_k)
        kx, ky = self.k[:, 0], self.k[:, 1]                # m* v = k (momentum)
        return w * torch.stack([one, kx, ky, self.eps_k], dim=0)  # (4, Nk)

    def get_contactor(self, n: torch.Tensor, **kwargs) -> Callable:
        return _CartesianContactor(self, n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable:
        return _CartesianReflector(self, n, self.fs.specularity)


class _CartesianContactor:
    """Reservoir ghost = drifted FD at (mu + dmu, T) with PHYSICAL inward drift
    velocity ``vD`` (along -n, into the device; kD = -m* vD n), as delta-f about
    f0_lab.  nonlinear=True (default): the exact Pauli-bounded deviation.
    nonlinear=False: its exact linearization,
        delta-f = (dmu - vD (k.n)) f0(1-f0)/T,
    matching the delta-k linear contactor's convention on the shell."""
    def __init__(self, rep: "Cartesian", n: torch.Tensor, *, dmu: float = 0.0,
                 vD: float = 0.0, nonlinear: bool = True):
        n = n.to(rc.device)
        f0 = rep._f0_lab
        if nonlinear:
            kD = -(rep.m_star * vD) * n                      # inward drift
            kp = rep.k[None] - kD[:, None]
            eps_p = kp.square().sum(-1) / (2 * rep.m_star)
            f_res = torch.special.expit(-(eps_p - (rep.mu + dmu)) / rep.T_temp)
            self.df_contact = f_res - f0[None, :]
        else:
            k_dot_n = rep.k @ n.t()                          # (Nk, Nsel)
            self.df_contact = ((dmu - vD * k_dot_n.t())
                               * (f0 * (1.0 - f0))[None, :] / rep.T_temp)

    def __call__(self, t: float) -> torch.Tensor:
        return self.df_contact


class _CartesianReflector:
    """Specular wall: delta-f(k) <- delta-f(k - 2(k.n)n) via bilinear interp on the
    underlying uniform grid (exact for axis-aligned walls).  With a circular
    active set, stencil indices are translated full-grid -> active; a stencil
    corner beyond the guard ring is dropped (weight zeroed) -- reflection
    preserves |k|, so every reflected ACTIVE point itself lies inside the active
    radius, but the four GRID CORNERS around it need not be.

    EXACT PARTICLE-FLUX CONSERVATION.  Bilinear interpolation makes the rows sum
    to 1, which conserves OCCUPANCY per output node.  What a wall must conserve
    is the particle FLUX -- the net normal current through the face is zero:

        sum_{v.n<0} |v.n| out  ==  sum_{v.n>0} |v.n| u                       (*)

    (`wk` is a scalar, so (*) involves only |v.n|.)  That is a condition on the
    |v.n|-weighted COLUMNS, which bilinear weights do not satisfy: the four
    corners carry different |v.n| from the point being interpolated, and dropped
    corners remove weight outright.  Left uncorrected the wall is ~0.4% ABSORBING
    per bounce (1.9% worst face), and a closed cavity leaks 2.0e-6 of its
    deviation mass per step, linearly: 5.9e-3 over 3000 steps.  The modal
    (_DeltaKReflector) path does not have this problem -- the mirror image of a
    quadrature node IS a node -- which is why the four shipped conservation tests
    never caught it.

    A wall has a SECOND invariant, and the same interpolation breaks it too --
    the tangential momentum flux, i.e. the shear stress it exerts:

        sum_{v.n<0} |v.n| (v.t) out  ==  s * sum_{v.n>0} |v.n| (v.t) u .      (**)

    Uncorrected residual at s = 1: 4.0e-3 on a drifted Fermi-Dirac trace.

    ARBITRARY SPECULARITY.  (*) is s-INDEPENDENT -- a wall passes zero net
    current whatever it does to momentum -- and (**) carries the entire s
    dependence: at s = 1 tangential momentum is conserved (a specular wall
    exerts no drag), at s = 0 the target is zero (a diffuse wall exerts full
    drag).  So the ghost is

        out = s * (interpolated specular)  +  (diffuse refill)

    with the refill fixed by (*) and (**).

    Corrections, in order:
      1. renormalise the surviving bilinear weights back to sum 1, undoing the
         dropped corners;
      2. a TWO-STAGE closure, separating the numerics from the physics.  Both
         stages use correction vectors biorthogonal to the constraint
         functionals L_a(x) = sum_I |v.n| mu_a x, with mu = (1, v.t, eps, |v.n|)
         and basis b_a = mu_a 1_I: [c_1..c_r] = [b_1..b_r] G^-1, G_ab = L_a(b_b).

         STAGE 1 (rank 4, NUMERICS).  The exact specular map preserves |k| and
         |v.n| and flips only sign(v.n), so it conserves the particle flux, the
         tangential momentum flux, the ENERGY flux and the NORMAL PRESSURE --
         all four targets are simply the outflow-side sums, and none depends on
         s.  Correcting the interpolated operator to hit all four removes the
         interpolation error itself.

         STAGE 2 (rank 2, PHYSICS).  The diffuse refill supplies the flux the
         specular part did not return, constraining MASS and TANGENTIAL
         MOMENTUM only -- exactly what the modal _DeltaKReflector does.

         ⛔ The energy row must be CENTRED on mu.  eps ~ mu across the active
         shell, so a raw energy row is nearly parallel to the constant row and
         the 4x4 Gram goes singular; (eps - mu)/(xi_max T) keeps cond(G4) at
         ~217, and the stage-2 Gram at 5.07.  Falls back to the particle-only
         correction if a solve fails.

         ⛔ Do NOT extend stage 2.  Applying the rank-4 closure to the WHOLE
         ghost pins all four moments at s = 1 but over-constrains the diffuse
         limit -- at s < 1 a diffuse wall's pressure is DETERMINED by isotropy
         plus the mass flux, not free.  Measured s = 0 refill: 63% angular
         structure with the pressure row, 2.2% with an energy row, against
         1.4e-16 deviation from constant (the Maxwell law) with two rows.

    ⛔ The basis must be the inflow INDICATOR, not the flux weight w_in = |v.n|.
    At s = 1 the correction is a ~0.4% residual and the basis is immaterial, but
    at s = 0 the refill IS the entire ghost, and a refill proportional to |v.n|
    is not the Maxwell law -- a diffuse wall re-emits ISOTROPICALLY, f = const,
    whose FLUX then goes like |v.n|.  With the indicator basis the s = 0 ghost
    is constant over each face's inflow set to 7.5e-16.  This is the same span
    the modal _DeltaKReflector uses ({1, sin(theta-phi)}); the (v.t) admixture
    absorbs the asymmetry of the DISCRETE inflow set so the refill carries
    exactly zero tangential momentum rather than approximately zero.

    Measured.  At s = 1 (fully specular, the production setting) all four
    moments are exact: particle 1.6e-16, tangential 1.8e-16, energy 4.8e-16,
    pressure 2.4e-16.  The energy and pressure residuals no longer converge with
    dk -- they are flat at ~1.5e-16 across a 4x refinement, where the earlier
    two-moment closure left them at order 2 (energy 1.40e-2 -> 7.77e-4).
    Across s = 0, 0.25, 0.5, 0.75, 1 the particle, tangential and pressure
    residuals stay <= 4.1e-14 and the per-face drag ratio
    (returned j_t)/(incident j_t) equals s to every printed digit, matching the
    modal reflector exactly at each s.  At s = 0 the refill is constant over
    each face's inflow set to 1.4e-16 -- the Maxwell law.

    ⚠ The ENERGY flux is exact only at s = 1.  At s < 1 the refill carries
    whatever energy the two-row diffuse model gives it, which is the same status
    as the modal reflector -- neither imposes an energy condition on the diffuse
    part.  Adding one restores exactness at the cost of the isotropy above.

    Step 2 is deliberately LINEAR.  The obvious alternative -- rescaling by
    beta = src/ret -- also makes (*) exact but is a ratio of two linear
    functionals, hence nonlinear in u (measured additivity defect 4.8e-3), and
    `FiniteVolume._setup_boundary` builds its dense `_refl_mat` cache by pushing
    the Nk basis vectors through this operator, i.e. it ASSUMES linearity.  A
    nonlinear reflector would silently disagree with its own cache at small Nk.

    Measured at the production configuration (n_k = 224, real mixer mesh):
    closed-cavity |dM|/M over 400 steps 3.06e-4 (linear in step count) -> 0
    EXACTLY; net wall current 4.5e-3 -> 4.1e-16; shear 4.0e-3 -> 5.3e-16;
    additivity defect 4.4e-16 (still linear); global overshoot unchanged at
    2.2e-16; undershoot 3.12e-3 -> 4.96e-14.  On the driven device the contact
    currents move by ~4e-7 relative and the f range and interior DMP are
    unchanged to all printed digits.
    """
    def __init__(self, rep: "Cartesian", n: torch.Tensor,
                 specularity: float = 1.0):
        n = n.to(rc.device); self.rep = rep; self.Ns = n.shape[0]
        self.s = float(specularity)
        k = rep.k                                       # active points
        kdotn = (k[None] * n[:, None]).sum(-1)
        k_ref = k[None] - 2.0 * kdotn.unsqueeze(-1) * n[:, None]
        n_k = rep.n_k_grid
        g = ((k_ref - rep._k_min) / rep._dk_grid).clamp(0, n_k - 1.0001)
        i0 = g.floor().long(); fr = g - i0
        ix, iy = i0[..., 0], i0[..., 1]; fx, fy = fr[..., 0], fr[..., 1]
        idx_full = (ix * n_k + iy, (ix + 1) * n_k + iy,
                    ix * n_k + iy + 1, (ix + 1) * n_k + iy + 1)
        wts = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
        self._idx = []; self._wt = []
        for idx, wt in zip(idx_full, wts):
            if rep._full2act is not None:
                a = rep._full2act[idx]
                wt = wt * (a >= 0)                      # outside active set: drop
                idx = a.clamp(min=0)
            self._idx.append(idx)
            self._wt.append(wt)
        # (1) undo the dropped corners: the surviving weights must still sum to 1
        tot = sum(self._wt)
        scale = torch.where(tot > 1e-12, 1.0 / tot.clamp(min=1e-12),
                            torch.zeros_like(tot))
        self._wt = [w * scale for w in self._wt]
        # (2) rank-2 flux + shear closure (see the class docstring)
        v = k / rep.m_star
        vn = kdotn / rep.m_star                         # v.n on the active set
        t_hat = torch.stack([-n[:, 1], n[:, 0]], -1)    # wall tangent
        vmax = max(float(v.norm(dim=1).max()), 1e-300)   # scale v.t to O(1)
        self._vt = (v[None] * t_hat[:, None]).sum(-1) / vmax
        self._w_in = vn.abs() * (vn < 0)                # (Ns, Nk)
        self._w_out = vn.abs() * (vn > 0)
        eps = k.square().sum(-1) / (2 * rep.m_star)
        self._vna = vn.abs() / vmax                     # |v.n|, scaled
        self._eps = ((eps - rep.mu) / (rep.xi_max * rep.T_temp))[None].expand_as(
            self._vt)                                   # CENTRED: eps ~ mu on
        #   the shell, so the raw energy row is nearly parallel to the constant
        #   row and the Gram becomes singular.  Centring on mu fixes that.
        self._mu_rows = (torch.ones_like(self._vt), self._vt, self._eps,
                         self._vna)
        infl = (vn < 0).to(v.dtype)

        def biorth(rows):
            """[c_1..c_r] with L_a(c_b) = delta_ab; falls back to the
            particle-only correction if the Gram is singular."""
            Bm = torch.stack([m * infl for m in rows], dim=1)    # (Ns, r, Nk)
            Mm = torch.stack(list(rows), dim=1)
            Gm = torch.einsum("sak,sbk->sab", Mm * self._w_in[:, None, :], Bm)
            try:
                return torch.linalg.solve(Gm, Bm)
            except Exception:
                C = torch.zeros_like(Bm)
                C[:, 0, :] = infl / (self._w_in * infl).sum(
                    -1, keepdim=True).clamp(min=1e-300)
                return C

        self._C4 = biorth(self._mu_rows)                # stage 1: numerics
        self._C2 = biorth(self._mu_rows[:2])            # stage 2: diffuse model

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        spec = torch.zeros_like(u)
        for idx, wt in zip(self._idx, self._wt):
            spec += wt[None] * torch.gather(u, -1, idx[None].expand_as(u))
        # Targets: the EXACT specular reflection returns, on the inflow side,
        # exactly the outflow-side sum of each of these four moments.  None of
        # them depends on s.
        T = [(self._w_out * m * u).sum(-1, keepdim=True) for m in self._mu_rows]
        # ---- stage 1: NUMERICS.  Remove the interpolation error, so the
        # specular operator reproduces the exact specular map in all four.
        for a in range(4):
            got = (self._w_in * self._mu_rows[a] * spec).sum(-1, keepdim=True)
            spec = spec + self._C4[:, a, :] * (T[a] - got)
        # ---- stage 2: PHYSICS.  The diffuse refill supplies the flux the
        # specular part did not return.  It constrains MASS and TANGENTIAL
        # MOMENTUM only -- exactly what the modal _DeltaKReflector does.
        # ⛔ Do NOT add the energy or pressure rows here: at s < 1 a diffuse
        # wall's pressure is DETERMINED by isotropy plus the mass flux, not
        # free, and imposing it forces the right number with the wrong shape --
        # measured 63% (with the pressure row) or 2.2% (with an energy row) of
        # angular structure in the s = 0 refill, which is not the Maxwell law
        # and not any physical wall.  With two rows the s = 0 refill is constant
        # over the inflow set to 1.4e-16.
        out = self.s * spec
        want = (T[0], self.s * T[1])
        for b in range(2):
            got = (self._w_in * self._mu_rows[b] * out).sum(-1, keepdim=True)
            out = out + self._C2[:, b, :] * (want[b] - got)
        return out
