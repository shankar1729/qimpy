"""Specular wall reflection on the Cartesian k-grid as an FFT unitary involution.

WHY.  A row-stochastic stencil W >= 0 with W^2 = I has a non-negative inverse,
hence is monomial, hence a permutation -- so a bounded, exactly-involutive
reflector exists only where the mirror image of every node is a node:
0/45/90/135 deg on a square lattice (no 2-D lattice has more than 6 mirror
lines).  Every non-D4 stencil is therefore either smearing (bias, ||W|| = 1,
W^2 != I) or amplifying (||W|| > 1); the shipped cubic+RADEX stencil measured
4x per bounce.  Positivity cannot be structural; it is accuracy-limited.

WHAT.  Write the mirror as M = R(2 alpha) F_x (alpha = wall-line angle),
reduce R(2 alpha) = R(theta') R(90 k) with theta' in (-45, 45], do the D4 part
as an index permutation and the residual rotation as three FFT shears
R(theta') = S_x(-tan theta'/2) S_y(sin theta') S_x(-tan theta'/2).  Each shear
is a phase multiplication along one axis, i.e. orthogonal, so on an ODD grid
(no Nyquist bin) W is exactly orthogonal, ||W||_2 = 1, and every product of
wall reflections is orthogonal: nothing compounds, ever.  W^2 = I exactly for
k even, and to the band-limit for k odd (the two D4 conjugates of the
shear product agree on band-limited content).

ACCURACY = the state's spectral content beyond ~0.8 k_Nyq, single-shot, never
accumulating.  A Fermi step w grid cells wide aliases like exp(-0.8 pi^2 w):
at the default dk = T/v_F (w = 1) a cold drifted ring reads 1.6e-4; at
dk = T/(3 v_F) it reads 1e-12 (measured cells/T 1/2/3/4 -> 1.6e-4 / 1.4e-8 /
1.2e-12 / 7e-14).  Hot cells are limited by the box truncation e^-xi_max
instead (state ~1e-5 at the rim for xi_max = 12), handled by a smooth radial
taper over the last 6 cells of the box so nothing wraps around the FFT.

CONSERVATION, without breaking the involution: conjugate, do not add.  With
a_m = |v.n| mu_m on the inflow half-plane, b_m the same on the outflow, and
c_m biorthogonal (c_m.a_l = 0, c_m.b_l = delta_ml),

    H = I + sum_m (W^T a_m - b_m) c_m^T,      W' = H^T W H^-T

gives W'^T a_m = b_m exactly (all four discrete flux moments to 1e-15) and
W'^2 = I whenever W^2 = I.  W' - W on a physical state is ~1e-8 -- the
half-plane quadrature defect of the EXACT ghost, which is unavoidable.

Measured (numpy prototype, 36 states x 3 walls, dk = T/(3 v_F), xi_max = 12):
Te = T / 2.5 T: 2e-12 ... 1e-10; Te = 5.69 T: 4e-6 ... 2e-5 (= truncation);
all moments <= 4e-15; min f >= -2e-6 (hot rim) / -1e-10 (cold); 64 alternating
17/63 deg bounces: error flat, norm 1 - 1e-11.

COST.  Matrix-free: per step, per wall edge, six 1-D FFT passes over the
n_k x n_k box (batched over edges sharing a normal) plus a rank-4 dot; there
is no dense cache.  Memory: 8 vectors of length Nk per distinct normal.

BOUNDS (QIMPY_REFL_FFT_BOUND, default 1).  A linear reflector at a non-D4
angle cannot preserve [0, 1] (bounded + involution => permutation), so the
bound is restored NONLINEARLY but LOCALLY, per wall edge and per step, with a
correction no larger than the violation it removes: clip the ghost occupancy
f0 + g into [0, 1], then put the four flux moments back exactly with a rank-4
correction carried on the clipped state's own HEADROOM min(f, 1-f) over the
inflow set (so it cannot leave [0, 1] as long as the moment defect is small,
which it is: ~1e-6), and repeat until both hold.  This is a per-edge FCT-style
repair -- nothing reduces over cells, so unlike `limit_positivity` it cannot
move the device's trajectory beyond the O(1e-5) it corrects.  The operator is
then nonlinear; it is only ever applied per step (no dense cache).

Not implemented: specularity < 1 (the diffuse refill of _CartesianReflector).
"""
from __future__ import annotations

import os
import numpy as np
import torch

from qimpy import rc
from qimpy.io import InvalidInputException


class _CartesianFFTReflector:
    def __init__(self, rep, n: torch.Tensor, specularity: float = 1.0):
        if float(specularity) != 1.0:
            raise InvalidInputException(
                "QIMPY_REFL_FFT=1 supports specularity = 1 only")
        n_k = int(rep.n_k_grid)
        if n_k % 2 == 0:
            raise InvalidInputException(
                f"QIMPY_REFL_FFT=1 needs an ODD n_k (got {n_k}): an even grid "
                "carries a Nyquist bin whose phase shift is not orthogonal, so "
                "the reflector would not be an exact involution")
        dev = rc.device
        dtype = rep.k.dtype
        self.rep = rep
        self.n_k = n_k
        n = n.to(device=dev, dtype=dtype)
        self.Ns = n.shape[0]
        dk = float(rep._dk_grid)
        self._dk = dk
        kg = torch.arange(n_k, device=dev, dtype=dtype) * dk + float(rep._k_min)
        self._kg = kg
        k_box = float(-rep._k_min) + 0.5 * dk           # box half-width
        KX, KY = torch.meshgrid(kg, kg, indexing="ij")
        kk = torch.sqrt(KX ** 2 + KY ** 2)
        wt = 6.0 * dk
        s = ((kk - (k_box - wt)) / wt).clamp(0.0, 1.0)
        self._taper = (1.0 - s * s * (3.0 - 2.0 * s))    # (n_k, n_k)
        self._act = (torch.where(rep._full2act >= 0)[0]
                     if rep._full2act is not None else None)
        self.Nk = rep.k.shape[0]
        self._q = 2.0 * np.pi * torch.fft.fftfreq(n_k, d=dk).to(device=dev, dtype=dtype)
        self._bound = os.environ.get("QIMPY_REFL_FFT_BOUND", "1") == "1"
        self._f0 = rep._f0_lab

        # ---- per-edge operator data, BATCHED over all wall edges ---------------
        # (a per-normal Python loop was launch-bound on the mixer: ~300 distinct
        # normals x ~30 kernels x 3 RK stages = 1.2 s/step; batched it is ~40
        # kernels per apply on (Ns, n_k, n_k) tensors)
        k = rep.k
        v = k / rep.m_star
        t_hat = torch.stack([-n[:, 1], n[:, 0]], -1)
        vmax = max(float(v.norm(dim=1).max()), 1e-300)
        eps = k.square().sum(-1) / (2 * rep.m_star)
        eps_c = (eps - rep.mu) / (rep.xi_max * rep.T_temp)
        phi = torch.atan2(n[:, 1], n[:, 0])
        theta = 2.0 * (phi + 0.5 * np.pi)                 # 2 x wall-line angle
        kq = torch.round(theta / (0.5 * np.pi))
        thp = theta - kq * 0.5 * np.pi                    # residual in (-45, 45]
        self._kq = (kq.to(torch.int64) % 4)
        # node permutation P = R(90k) F_x per edge: out[i, j] = in[perm[i, j]]
        n2 = n_k * n_k
        base = torch.arange(n2, device=dev).reshape(n_k, n_k)
        def rot(M, kk):
            for _ in range(kk % 4):
                M = M.flip(-1).transpose(-2, -1)
            return M
        fwd = torch.stack([rot(base.flip(-1), int(kk)).reshape(-1) for kk in self._kq])
        inv = torch.empty_like(fwd)
        inv.scatter_(1, fwd, torch.arange(n2, device=dev)[None].expand_as(fwd))
        self._perm_fwd, self._perm_inv = fwd, inv                 # (Ns, n^2)
        a = -torch.tan(0.5 * thp); b = torch.sin(thp)     # (Ns,)
        # x-shear shifts row y_j by a*y_j (fft along dim -2); y-shear shifts
        # column x_i by b*x_i (fft along dim -1); exact permutations get a = b = 0
        self._phx = torch.exp(-1j * self._q[None, :, None] * (a[:, None] * kg[None, :])[:, None, :])
        self._phy = torch.exp(-1j * self._q[None, None, :] * (b[:, None] * kg[None, :])[:, :, None])
        # conservation conjugation on the active-set flux moments, per edge
        vn = v @ n.t()                                    # (Nk, Ns)
        vt = (v @ t_hat.t()) / vmax
        vn, vt = vn.t().contiguous(), vt.t().contiguous()  # (Ns, Nk)
        mus = torch.stack([torch.ones_like(vn), vt, eps_c[None].expand_as(vn),
                           vn.abs() / vmax], 1)           # (Ns, 4, Nk)
        w_in = (vn.abs() * (vn < 0))[:, None, :]
        w_out = (vn.abs() * (vn > 0))[:, None, :]
        A = w_in * mus; B = w_out * mus                   # (Ns, 4, Nk)
        WtA = torch.stack([self._gather(self._apply_T(self._scatter(A[:, m_])))
                           for m_ in range(4)], 1)        # (Ns, 4, Nk): taper W^T a
        U = WtA - B
        basis = torch.cat([A, B], 1)                      # (Ns, 8, Nk)
        G = torch.einsum("eak,ebk->eab", basis, basis)    # (Ns, 8, 8)
        rhs = torch.zeros(self.Ns, 8, 4, device=dev, dtype=dtype)
        rhs[:, 4:] = torch.eye(4, device=dev, dtype=dtype)
        C = torch.einsum("eab,eak->ebk", torch.linalg.solve(G, rhs), basis)  # (Ns, 4, Nk)
        K = torch.eye(4, device=dev, dtype=dtype) + torch.einsum("eak,ebk->eab", U, C)
        self._U, self._C, self._Kinv = U, C, torch.linalg.inv(K)
        self._A, self._B = A, B
        self._mu_in = (vn < 0).to(dtype)[:, None, :] * mus

    def _repair(self, g: torch.Tensor, x: torch.Tensor, n_iter: int = 2) -> torch.Tensor:
        """Clip f0 + g into [0, 1]; restore the four flux moments on the clipped
        state's headroom; repeat.  Local per edge, exact moments on exit.
        Fixed iteration count and no host syncs, so it stays inside torch.compile
        (a data-dependent loop broke the compiled RHS and cost 6x per step)."""
        f0 = self._f0
        lo, hi = -f0, 1.0 - f0                                  # bounds on delta-f
        A, B, mu_in = self._A, self._B, self._mu_in
        target = torch.einsum("eak,...ek->...ea", B, x)         # outflow moments
        for it in range(n_iter):
            g = torch.minimum(torch.maximum(g, lo), hi)
            h = torch.minimum(g - lo, hi - g).clamp(min=0.0)    # headroom
            basis = mu_in * h[..., None, :]                      # (..., Ns, 4, Nk)
            r = target - torch.einsum("eak,...ek->...ea", A, g)  # defect
            G = torch.einsum("eak,...ebk->...eab", A, basis)     # (..., Ns, 4, 4)
            lam = torch.linalg.solve(G, r.unsqueeze(-1)).squeeze(-1)
            g = g + torch.einsum("...ea,...eak->...ek", lam, basis)
        return g

    # ---- full-grid <-> active-set --------------------------------------------
    def _scatter(self, x: torch.Tensor) -> torch.Tensor:
        """(..., Nk) active -> (..., n_k, n_k) full grid (zero off the active set)."""
        n2 = self.n_k * self.n_k
        if self._act is None:
            return x.reshape(*x.shape[:-1], self.n_k, self.n_k)
        F = torch.zeros(*x.shape[:-1], n2, device=x.device, dtype=x.dtype)
        F[..., self._act] = x
        return F.reshape(*x.shape[:-1], self.n_k, self.n_k)

    def _gather(self, F: torch.Tensor) -> torch.Tensor:
        F = F.reshape(*F.shape[:-2], self.n_k * self.n_k)
        return F if self._act is None else F[..., self._act]

    # ---- the mirror, batched over edges (dim -3) --------------------------------
    @staticmethod
    def _shear_x(F, ph):
        return torch.fft.ifft(torch.fft.fft(F, dim=-2) * ph, dim=-2).real

    @staticmethod
    def _shear_y(F, ph):
        return torch.fft.ifft(torch.fft.fft(F, dim=-1) * ph, dim=-1).real

    def _perm(self, F, idx):
        """Apply a per-edge node permutation (flip + 90k rotation) in one gather."""
        n2 = self.n_k * self.n_k
        G = F.reshape(*F.shape[:-2], n2)
        return torch.gather(G, -1, idx.expand(*G.shape[:-2], *idx.shape)).reshape(F.shape)

    def _apply(self, F: torch.Tensor) -> torch.Tensor:
        """W (taper f): flip y, rotate by 90k (permutation), three shears."""
        F = self._perm(F * self._taper, self._perm_fwd)
        F = self._shear_x(F, self._phx)
        F = self._shear_y(F, self._phy)
        return self._shear_x(F, self._phx)

    def _apply_T(self, F: torch.Tensor) -> torch.Tensor:
        """(W taper)^T = taper W^T: inverse shears, inverse permutation, flip."""
        F = self._shear_x(F, self._phx.conj())
        F = self._shear_y(F, self._phy.conj())
        F = self._shear_x(F, self._phx.conj())
        return self._perm(F, self._perm_inv) * self._taper

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """u: (..., Ns, Nk) wall traces (delta-f) -> reflected ghost, same shape."""
        U, C, Kinv = self._U, self._C, self._Kinv
        s = torch.einsum("...ea,eba->...eb",                     # K^-1 (U u)
                         torch.einsum("eak,...ek->...ea", U, u), Kinv)
        x1 = u - torch.einsum("...ea,eak->...ek", s, C)
        g = self._gather(self._apply(self._scatter(x1)))
        g = g + torch.einsum("...ea,eak->...ek",
                             torch.einsum("eak,...ek->...ea", U, g), C)
        # the dense-cache build pushes the Nk identity vectors through as
        # (Nk, Ns, Nk); the repair is nonlinear and must not see those
        if self._bound and not (u.dim() == 3 and u.shape[0] == self.Nk):
            g = self._repair(g, u)
        return g
