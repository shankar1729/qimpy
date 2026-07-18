"""FermiCartesian: drifted Fermi gas on a FIXED uniform Cartesian k-grid.

Streaming is exact per-k real-space advection (v = k / m*, hbar=1) on the fixed
grid -- stable at arbitrary drift, no moving frame.  The e-e collision is applied
by, per spatial cell:

  1. recover the local drifted-heated Fermi-Dirac frame (k_D, T_e, mu) from the
     cell's (n, J, E) moments  [k_D = <k> exact; T_e, mu by a 2x2 Newton],
  2. project the deviation delta-f about that frame onto the code's OWN modal
     basis (RadialBasis psi_n(xi) x AngularBasis Fourier), drift-CENTRED,
  3. apply the EXISTING EEScattering.a_dot unchanged,
  4. reconstruct the increment on the Cartesian grid and project out (n, J, E)
     so collision conserves particle number, momentum and energy exactly.

State stored per channel = delta-f = f - f0_lab (deviation about the uniform
lab equilibrium); streaming advects it, rho_dot returns the collision term.
"""
from __future__ import annotations
from typing import Callable, Optional, Union
import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from ..fermi_surface.scattering import EEScattering
from ..fermi_surface import AngularBasis, RadialBasis
from .._material import Material


class FermiCartesian(Material):
    kF: float; vF: float; m_star: float; mu: float
    M_theta: int; Nr: int; T_temp: float; xi_max: float
    angular: AngularBasis
    radial: RadialBasis

    def __init__(
        self, *, kF: float, vF: float, k_max: float, n_k: int, M_theta: int,
        Nr: int = 4, T: float = 1.0, xi_max: float = 6.0,
        m_star: Optional[float] = None, spin: float = 2.0,
        newton_iters: int = 8, cell_chunk: int = 256,
        ee_scattering: Optional[Union[EEScattering, dict]] = None,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        super().__init__()
        m_star = float(m_star) if m_star is not None else kF / vF
        self.kF, self.vF, self.m_star = kF, vF, m_star
        self.M_theta, self.Nr = M_theta, Nr
        self.T_temp, self.xi_max = T, xi_max
        self.mu = 0.5 * kF * kF / m_star                      # E_F (degenerate)
        self.newton_iters, self.cell_chunk = newton_iters, cell_chunk

        dk = 2.0 * k_max / n_k                                # cell-centred uniform grid
        kg = torch.arange(n_k, device=rc.device) * dk - k_max + 0.5 * dk
        KX, KY = torch.meshgrid(kg, kg, indexing="ij")
        k = torch.stack([KX.reshape(-1), KY.reshape(-1)], dim=-1)  # (Nk, 2)
        Nk = n_k * n_k
        self.dk_area = dk * dk                               # Cartesian cell area (no spin)
        self.initialize(wk=float(spin) * self.dk_area, nk=Nk, n_bands=1, n_dim=2,
                        process_grid=process_grid)
        if self.comm.size > 1:
            raise InvalidInputException(
                "FermiCartesian couples k-channels in the collision; the k "
                "process-grid dimension must be 1 (parallelize over space)."
            )
        dtype = self.v.dtype
        self.k = k.to(dtype)                                 # (Nk, 2)
        self.eps_k = (k.square().sum(-1) / (2 * m_star)).to(dtype)   # (Nk,)
        self.E = self.eps_k.reshape(Nk, 1)                   # Material.E: (Nk, n_bands)
        self.v = (k / m_star).to(dtype)                      # (Nk, 2) transport velocity

        # code's own modal bases (identical construction to FermiSurface)
        self.angular = AngularBasis(M_theta, dtype=dtype)
        self.radial = RadialBasis(Nr, T_temp=T, xi_max=xi_max, dtype=dtype)
        self._psi_coeff = self._radial_poly_coeffs(dtype)    # (Nr, Nr): monomials in u=xi/xi_max
        self._ang_norm = torch.tensor(                       # 1/norm_c for a_c
            [1.0 / (2 * np.pi)] + [1.0 / np.pi] * (2 * M_theta),
            dtype=dtype, device=rc.device)

        # uniform lab equilibrium f0_lab (k_D=0, T, mu) and delta-f=0 start
        self._f0_lab = torch.special.expit(-(self.eps_k - self.mu) / T)   # (Nk,)
        self.rho0 = torch.zeros((Nk, 1, 1), device=rc.device, dtype=dtype)

        if (ee_scattering is not None) or checkpoint_in.member("ee_scattering"):
            self.add_child("ee_scattering", EEScattering, ee_scattering,
                           checkpoint_in, fermi_surface=self)

    # ---- radial polynomial evaluation at arbitrary xi (reuse code's psi_n) ----
    def _radial_poly_coeffs(self, dtype) -> torch.Tensor:
        if self.Nr == 1:
            return torch.ones((1, 1), dtype=dtype, device=rc.device)
        u = (self.radial.xi / self.xi_max).cpu().numpy()
        V = np.vander(u, self.Nr, increasing=True)           # (Nr, Nr)
        Tfm = self.radial.T_from_modes.cpu().numpy()         # psi_n(xi_r): (Nr, Nr)
        coeff = np.linalg.solve(V, Tfm)                      # (Nr_pow, Nr_mode)
        return torch.as_tensor(coeff, dtype=dtype, device=rc.device)

    def _psi(self, xi: torch.Tensor) -> torch.Tensor:
        """psi_n(xi) for all n, shape (..., Nr).  xi any shape."""
        if self.Nr == 1:
            return torch.ones((*xi.shape, 1), dtype=xi.dtype, device=xi.device)
        u = (xi / self.xi_max).unsqueeze(-1)                 # (..., 1)
        powers = u ** torch.arange(self.Nr, device=xi.device)  # (..., Nr_pow)
        return powers @ self._psi_coeff                       # (..., Nr_mode)

    def _fourier(self, th: torch.Tensor) -> torch.Tensor:
        """[1, cos th, sin th, ..., cos M th, sin M th] : (..., 2M+1)."""
        cols = [torch.ones_like(th)]
        for m in range(1, self.M_theta + 1):
            cols += [torch.cos(m * th), torch.sin(m * th)]
        return torch.stack(cols, dim=-1)

    @property
    def transport_velocity(self) -> torch.Tensor:
        return self.v

    # ---- local drifted-heated frame recovery from moments ----
    def _recover_frame(self, f: torch.Tensor):
        """f: (C, Nk) full distribution -> (kD:(C,2), Te:(C,), mu:(C,))."""
        w = self.wk
        n = (f.sum(-1) * w).clamp_min(1e-30)                 # (C,)
        p = torch.einsum("ck,kd->cd", f, self.k) * w         # (C, 2)
        E = torch.einsum("ck,k->c", f, self.eps_k) * w       # (C,)
        kD = p / n[:, None]                                  # <k> exact drift
        E_rest = E - (kD.square().sum(-1) / (2 * self.m_star)) * n  # internal energy
        # 2x2 Newton on (Te, mu) matching (n, E_rest) via rest-frame FD grid moments
        Te = torch.full_like(n, self.T_temp)
        mu = torch.full_like(n, self.mu)
        eps = self.eps_k[None, :]                            # (1, Nk)
        for _ in range(self.newton_iters):
            xi = (eps - mu[:, None]) / Te[:, None]
            f0 = torch.special.expit(-xi)
            weq = f0 * (1 - f0)
            nn = f0.sum(-1) * w
            EE = (f0 * eps).sum(-1) * w
            g_mu = weq / Te[:, None]                         # df0/dmu
            g_Te = weq * xi / Te[:, None]                    # df0/dTe
            J11 = g_mu.sum(-1) * w;              J12 = g_Te.sum(-1) * w
            J21 = (g_mu * eps).sum(-1) * w;      J22 = (g_Te * eps).sum(-1) * w
            det = (J11 * J22 - J12 * J21)
            det = torch.where(det.abs() < 1e-30, torch.ones_like(det), det)
            r1 = n - nn; r2 = E_rest - EE
            dmu = (J22 * r1 - J12 * r2) / det
            dTe = (-J21 * r1 + J11 * r2) / det
            mu = mu + dmu
            Te = (Te + dTe).clamp_min(0.05 * self.T_temp)
        return kD, Te, mu

    # ---- the projection-collision (reuses EEScattering unchanged) ----
    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        if not hasattr(self, "ee_scattering"):
            return torch.zeros_like(rho)
        shape_in = rho.shape
        df_lab = rho.reshape(-1, shape_in[-1])               # (C, Nk) delta-f about f0_lab
        C = df_lab.shape[0]
        out = torch.empty_like(df_lab)
        dim = self.angular.dim
        eps = self.eps_k; k = self.k; f0_lab = self._f0_lab
        Ttm_r = self.radial.T_to_modes                       # (Nr, Nr) modal<-nodal radial
        for lo in range(0, C, self.cell_chunk):
            hi = min(lo + self.cell_chunk, C)
            df = df_lab[lo:hi]                               # (c, Nk)
            f = f0_lab[None, :] + df
            kD, Te, mu = self._recover_frame(f)             # (c,2),(c,),(c,)
            kp = k[None] - kD[:, None]                       # (c, Nk, 2)
            eps_p = kp.square().sum(-1) / (2 * self.m_star)  # (c, Nk)
            xi = (eps_p - mu[:, None]) / Te[:, None]         # local xi
            th = torch.atan2(kp[..., 1], kp[..., 0])         # local theta
            f0_loc = torch.special.expit(-xi)
            weq = f0_loc * (1 - f0_loc)
            df_loc = f - f0_loc                              # deviation about local frame
            # forward: Cartesian -> modal a[c, Nr*dim]  (drift-centred projection)
            mask = (xi.abs() < self.xi_max)
            psi = self._psi(xi) * mask.unsqueeze(-1)         # (c, Nk, Nr)
            fou = self._fourier(th) * self._ang_norm         # (c, Nk, dim)
            jac = self.dk_area / (self.m_star * Te)          # (c,)
            a = torch.einsum("ck,ckn,ckd->cnd", df_loc, psi, fou) * jac[:, None, None]
            a = a.reshape(hi - lo, self.Nr * dim)
            a_dot = self.ee_scattering.a_dot(a)             # <-- REUSED UNCHANGED
            # backward: modal -> Cartesian increment  dδf = weq * sum a_dot psi fourier
            ad = a_dot.reshape(hi - lo, self.Nr, dim)
            psi_all = self._psi(xi)                          # (c, Nk, Nr) unmasked for recon
            fou_pl = self._fourier(th)                       # (c, Nk, dim)
            Phi_dot = torch.einsum("cnd,ckn,ckd->ck", ad, psi_all, fou_pl)
            ddf = weq * Phi_dot                              # (c, Nk)
            # exact conservation: project (n, J, E) out of the increment
            ddf = self._project_conserved(ddf, kp, eps_p, weq)
            out[lo:hi] = ddf
        return out.reshape(shape_in)

    def _project_conserved(self, ddf, kp, eps_p, weq):
        """Remove (1, k'x, k'y, eps') moments from ddf using a weq-localized basis."""
        w = self.wk
        # basis g_a = weq * psi_a,  psi = (1, k'x, k'y, eps')
        px, py = kp[..., 0], kp[..., 1]
        psi = torch.stack([torch.ones_like(px), px, py, eps_p], dim=1)  # (c,4,Nk)
        g = psi * weq.unsqueeze(1)                           # (c,4,Nk)
        M = torch.einsum("cak,cbk->cab", g, psi) * w         # (c,4,4)
        U = torch.einsum("cak,ck->ca", psi, ddf) * w         # (c,4)
        eye = 1e-30 * torch.eye(4, device=ddf.device, dtype=ddf.dtype)
        lam = torch.linalg.solve(M + eye, U.unsqueeze(-1))[..., 0]  # (c,4)
        return ddf - torch.einsum("ca,cak->ck", lam, g)

    # ---- observables: n, jx, jy on the Cartesian grid ----
    def get_observable_names(self) -> list[str]:
        return ["n", "jx", "jy"]

    @stopwatch
    def get_observables(self, t: float) -> torch.Tensor:
        one = torch.ones_like(self.eps_k)
        return torch.stack([one, self.v[:, 0], self.v[:, 1]], dim=0)  # (3, Nk)

    # ---- boundaries ----
    def get_contactor(self, n: torch.Tensor, **kwargs) -> Callable:
        return _CartesianContactor(self, n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable:
        return _CartesianReflector(self, n)

    def initialize_fields(self, rho, params, patch_id) -> None:
        pass

    def _save_checkpoint(self, cp_path: CheckpointPath, context: CheckpointContext) -> list[str]:
        a = cp_path.attrs
        a["kF"], a["vF"], a["m_star"] = self.kF, self.vF, self.m_star
        a["M_theta"], a["Nr"], a["T"], a["xi_max"] = self.M_theta, self.Nr, self.T_temp, self.xi_max
        return list(a.keys())


class _CartesianContactor:
    """Contact ghost = a drifted-heated FD reservoir, as delta-f about f0_lab."""
    def __init__(self, mat: "FermiCartesian", n: torch.Tensor, *, dmu: float = 0.0, vD: float = 0.0):
        n = n.to(rc.device)
        phi = torch.atan2(n[:, 1], n[:, 0])                  # outward normal
        kD = (mat.m_star * vD) * torch.stack([torch.cos(phi), torch.sin(phi)], -1)  # (Ns,2)
        kp = mat.k[None] - kD[:, None]                       # (Ns, Nk, 2)
        eps_p = kp.square().sum(-1) / (2 * mat.m_star)
        f_res = torch.special.expit(-(eps_p - (mat.mu + dmu)) / mat.T_temp)  # (Ns, Nk)
        self.df_contact = f_res - mat._f0_lab[None, :]       # delta-f about lab

    def __call__(self, t: float) -> torch.Tensor:
        return self.df_contact


class _CartesianReflector:
    """Specular wall: delta-f(k) <- delta-f(k - 2(k.n)n), via bilinear interp on
    the uniform Cartesian grid (exact for any wall angle)."""
    def __init__(self, mat: "FermiCartesian", n: torch.Tensor):
        n = n.to(rc.device); self.mat = mat; self.Ns = n.shape[0]
        k = mat.k                                            # (Nk,2)
        kdotn = (k[None] * n[:, None]).sum(-1)               # (Ns, Nk)
        k_ref = k[None] - 2.0 * kdotn.unsqueeze(-1) * n[:, None]  # (Ns, Nk, 2)
        n_k = int(round(np.sqrt(k.shape[0])))
        kmin = k[:, 0].min()
        self._nk = n_k
        self._dk = (k[:, 0].max() - kmin) / (n_k - 1)
        # bilinear weights for k_ref on the (n_k x n_k) grid
        g = (k_ref - kmin) / self._dk                        # (Ns, Nk, 2) fractional index
        g = g.clamp(0, n_k - 1.0001)
        i0 = g.floor().long(); fr = g - i0
        ix, iy = i0[..., 0], i0[..., 1]; fx, fy = fr[..., 0], fr[..., 1]
        self._idx = (ix * n_k + iy, (ix + 1) * n_k + iy, ix * n_k + iy + 1, (ix + 1) * n_k + iy + 1)
        self._wt = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        # u: (Nghost, Ns, Nk) delta-f trace; reflect along the last axis per wall node
        out = torch.zeros_like(u)
        for idx, wt in zip(self._idx, self._wt):
            out += wt[None] * torch.gather(u, -1, idx[None].expand_as(u))
        return out
