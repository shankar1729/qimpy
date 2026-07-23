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
import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from ._representation import KRepresentation, CELL_SCALAR_NAMES, FLUX_NAMES


class Cartesian(KRepresentation):
    kF: float; vF: float; m_star: float; mu: float; T_temp: float; xi_max: float

    @staticmethod
    def recommended_grid(kF, vF, T, *, xi_max=6.0, dmu_max=0.0, kD_max=0.0,
                         dk=None, safety_cells=3, m_star=None):
        """Minimal (k_max, n_k) whose box holds the occupied shell for the given BC.

        f is negligible (< e^{-xi_max}) beyond |k| = sqrt(2 m*(E_F+dmu_max+xi_max*T));
        a drift shifts the shell centre by |k_D|.  Streaming + specular walls
        preserve |k|, so for ballistic this is exact; the drift enters only through
        kD_max.  dk defaults to the thermal width T/vF."""
        m = float(m_star) if m_star is not None else kF / vF
        EF = 0.5 * kF * kF / m
        dk = float(dk) if dk else T / vF
        k_shell = (2 * m * (EF + abs(dmu_max) + xi_max * T)) ** 0.5
        k_max = abs(kD_max) + k_shell + safety_cells * dk
        return float(k_max), int(np.ceil(2 * k_max / dk))

    def __init__(
        self, *, fermi_surface,
        k_max: Optional[float] = None, n_k: Optional[int] = None,
        dk: Optional[float] = None, dmu_max: float = 0.0, kD_max: float = 0.0,
        grid_safety_cells: int = 3, spin: float = 2.0, circular: bool = True,
        local_te_rates: bool = True,
        newton_iters: int = 8, frame_polish: int = 2,
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
        self.newton_iters, self.frame_polish = newton_iters, frame_polish
        self.cell_chunk, self.mem_budget_gb = cell_chunk, mem_budget_gb
        m_star = fs.m_star; dtype = torch.get_default_dtype()

        if k_max is None or n_k is None:                     # auto-size from BC
            k_max, n_k = self.recommended_grid(
                fs.kF, fs.vF, fs.T_temp, xi_max=fs.xi_max, dmu_max=dmu_max,
                kD_max=kD_max, dk=dk, safety_cells=grid_safety_cells, m_star=m_star)
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
        if circular:
            r_act = k_max + 1.5 * dk
            act = k.square().sum(-1) <= r_act * r_act
            self._full2act = torch.full((Nk,), -1, dtype=torch.long, device=rc.device)
            self._full2act[act] = torch.arange(int(act.sum().item()), device=rc.device)
            k = k[act]
            Nk = k.shape[0]
        else:
            self._full2act = None
        self.dk_area = dk * dk
        self.Nk = Nk
        # Physical BZ phase-space weight: n = spin * int d^2k/(2pi)^2 f  ->  per-k
        # weight spin*dk^2/(2pi)^2.  (Gives n = kF^2/2pi at equilibrium, and makes
        # contact currents physical so I_set is in a.u. current, 20uA=3.02e-3.)
        self.wk = float(spin) * self.dk_area / (2.0 * np.pi) ** 2
        self.k = k
        self.eps_k = (k.square().sum(-1) / (2 * m_star)).to(dtype)
        self.v = (k / m_star).to(dtype)                      # (Nk, 2) transport velocity

        # density of states for O(nb) frame recovery (Newton moments are 1D in eps)
        nb = min(4096, Nk)
        edges = torch.linspace(float(self.eps_k.min()), float(self.eps_k.max()),
                               nb + 1, device=rc.device, dtype=dtype)
        self._dos_eps = 0.5 * (edges[1:] + edges[:-1])
        idx = torch.bucketize(self.eps_k, edges[1:-1])
        self._dos_g = torch.zeros(nb, device=rc.device, dtype=dtype).scatter_add_(
            0, idx, torch.ones_like(self.eps_k))

        self._psi_coeff = self._radial_poly_coeffs(dtype)
        self._ang_norm = torch.tensor(
            [1.0 / (2 * np.pi)] + [1.0 / np.pi] * (2 * fs.M_theta),
            dtype=dtype, device=rc.device)
        self._f0_lab = torch.special.expit(-(self.eps_k - self.mu) / self.T_temp)
        # Radial Gram of the CONTINUUM measure the k-sum realizes.  The psi
        # basis is orthonormal only under the DISCRETE Nr-point quadrature, so
        # the raw projection returns (G_band @ c) / T rather than the operator
        # contract's coefficients c of Phi = delta_f / (f0(1-f0)/T).  Invert
        # G_band (fine quadrature of the dimensionless 0.25 sech^2 measure over
        # the projection mask |xi| < xi_max); apply_collision left-applies
        # T * Ginv_band and the reconstruction divides its weight by T, making
        # projection -> reconstruction the exact identity (audit bugs 2+3).
        xi_f = torch.linspace(-fs.xi_max, fs.xi_max, 4001,
                              device=rc.device, dtype=dtype)
        w_f = 0.25 / torch.cosh(0.5 * xi_f) ** 2
        psi_f = self._psi(xi_f)                              # (nfine, Nr)
        G_band = torch.einsum("xn,xm,x->nm", psi_f, psi_f, w_f) \
            * float(xi_f[1] - xi_f[0])
        self._Ginv_band = torch.linalg.inv(G_band)

    # ---- radial polynomial evaluation at arbitrary xi (reuse the model's psi_n) ----
    def _radial_poly_coeffs(self, dtype) -> torch.Tensor:
        Nr = self.fs.Nr
        if Nr == 1:
            return torch.ones((1, 1), dtype=dtype, device=rc.device)
        v = np.tanh(0.5 * self.fs.radial.xi.cpu().numpy())
        V = np.vander(v, Nr, increasing=True)
        Tfm = self.fs.radial.T_from_modes.cpu().numpy()
        return torch.as_tensor(np.linalg.solve(V, Tfm), dtype=dtype, device=rc.device)

    def _psi(self, xi: torch.Tensor) -> torch.Tensor:
        Nr = self.fs.Nr
        if Nr == 1:
            return torch.ones((*xi.shape, 1), dtype=xi.dtype, device=xi.device)
        v = torch.tanh(0.5 * xi)
        powers = [torch.ones_like(v), v]
        for _ in range(2, Nr):
            powers.append(powers[-1] * v)
        return torch.stack(powers, dim=-1) @ self._psi_coeff

    def _fourier(self, th: torch.Tensor) -> torch.Tensor:
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
        n = (f.sum(-1) * w).clamp_min(1e-30)
        p = torch.einsum("ck,kd->cd", f, self.k) * w
        E = torch.einsum("ck,k->c", f, self.eps_k) * w
        kD = p / n[:, None]
        E_rest = E - (kD.square().sum(-1) / (2 * self.m_star)) * n
        Te = torch.full_like(n, self.T_temp); mu = torch.full_like(n, self.mu)
        Te, mu = self._newton_TeMu(Te, mu, n, E_rest,
                                   self._dos_eps[None, :], self._dos_g[None, :] * w,
                                   self.newton_iters)
        Te, mu = self._newton_TeMu(Te, mu, n, E_rest,
                                   self.eps_k[None, :], w, self.frame_polish)
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
        bytes_per_row = Nk * (2 * dim + 3 * Nr + 12) * 8
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
            psi = self._psi(xi)                              # (c, Nk, Nr)
            fou = self._fourier(th)                          # (c, Nk, dim)
            mask = (xi.abs() < self.xi_max)
            jac = self.dk_area / (self.m_star * Te)
            gp = (df_loc * mask).unsqueeze(-1) * psi
            a = torch.bmm(gp.transpose(1, 2), fou * self._ang_norm)
            a = a * jac[:, None, None]                       # raw (G_band @ c)/T
            # Operator contract: coefficients c of Phi = delta_f/(f0(1-f0)/T):
            a = self.T_temp * torch.einsum("nm,cmd->cnd", self._Ginv_band, a)
            a = a.reshape(hi - lo, Nr * dim)
            # Local-T_e rates: hand the per-cell recovered T_e to the modal
            # operator -- evaluated EXACTLY there when the model has a local_te
            # ensemble (analytic T_e^2 T^(1-d) prefactors x tabulated shape),
            # else by the leading-order uniform (T_e/T)^2 rescale.
            te = Te if self.local_te_rates else None
            a_dot = modal_op(a, te=te)                       # <-- model's modal collision
            ad = a_dot.reshape(hi - lo, Nr, dim)
            B = torch.bmm(fou, ad.transpose(1, 2))
            # w_eq_RB = f0(1-f0)/T -- the /T completes the contract (audit bug 2):
            ddf = (weq / self.T_temp) * (psi * B).sum(-1)
            out[lo:hi] = self._project_conserved(ddf, kp, eps_p, weq)
        return out.reshape(shape_in)

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
            n = f.sum(-1) * w
            E = torch.einsum("ck,k->c", f, self.eps_k) * w
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
        return _CartesianReflector(self, n)


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
    active set, stencil indices are translated full-grid -> active; the rare
    stencil corner beyond the guard ring (occupancy < e^-7) is dropped (weight
    zeroed) -- reflection preserves |k|, so every reflected ACTIVE point itself
    lies inside the active radius."""
    def __init__(self, rep: "Cartesian", n: torch.Tensor):
        n = n.to(rc.device); self.rep = rep; self.Ns = n.shape[0]
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

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(u)
        for idx, wt in zip(self._idx, self._wt):
            out += wt[None] * torch.gather(u, -1, idx[None].expand_as(u))
        return out
