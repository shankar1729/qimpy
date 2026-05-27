from __future__ import annotations

import numpy as np
import torch

from qimpy import rc


class AngularModes:
    """Angular-harmonic (modal) representation of a Fermi-circle distribution.

    Instead of `N_theta` discrete directions (delta functions in angle), the
    distribution rho(theta) is expanded in real Fourier harmonics up to order M:

        rho(theta) = a_0 + sum_{m=1..M} [a_m cos(m theta) + b_m sin(m theta)]

    stored as the 2M+1 real coefficients f = (a_0, a_1, b_1, a_2, b_2, ...).
    In this basis the kinetic operators take their natural form:

    * Collisions are diagonal per harmonic: m=0 (density) is conserved, m=1
      (momentum/current) relaxes only through tau_p, and m>=2 relax through
      tau_p^-1 + tau_ee^-1. This is exactly the moment projection the
      discrete-ordinates collision term performs, but without any quadrature.
    * The cyclotron (momentum-space) advection omega_c d/dtheta is an *exact*
      block rotation of each harmonic pair at frequency m*omega_c -- no upwind
      scheme, no CFL limit, machine precision.
    * Real-space advection vF (cos theta d/dx + sin theta d/dy) couples
      m <-> m+-1 via the ladder matrices Ax, Ay (multiplication by vF cos/sin
      theta projected back onto the kept harmonics). It is therefore a coupled
      linear hyperbolic system A_x d/dx + A_y d/dy, advected on the spatial mesh
      with a characteristic/Lax-Friedrichs flux (max wave speed vF).
    * Observables read directly off the low modes: n ~ a_0, jx ~ vF a_1 / 2,
      jy ~ vF b_1 / 2.
    """

    def __init__(self, M: int, vF: float, n_quad: int | None = None,
                 dtype=torch.double):
        self.M = M
        self.vF = vF
        self.dim = 2 * M + 1
        # de-aliased quadrature: resolve up to harmonic M+1 (the ladder leakage)
        Nq = n_quad if n_quad is not None else max(4 * M + 4, 8)
        self.Nq = Nq
        theta = 2 * np.pi * np.arange(Nq) / Nq

        # mode -> node (T) and node -> mode (Tinv) transforms
        T = np.zeros((Nq, self.dim)); T[:, 0] = 1.0
        Tinv = np.zeros((self.dim, Nq)); Tinv[0, :] = 1.0 / Nq
        for m in range(1, M + 1):
            c, s = np.cos(m * theta), np.sin(m * theta)
            T[:, 2 * m - 1] = c; T[:, 2 * m] = s
            Tinv[2 * m - 1, :] = (2.0 / Nq) * c
            Tinv[2 * m, :] = (2.0 / Nq) * s

        # ladder matrices: multiply by vF cos/sin(theta), project onto kept modes
        Ax = Tinv @ np.diag(vF * np.cos(theta)) @ T
        Ay = Tinv @ np.diag(vF * np.sin(theta)) @ T
        # cyclotron generator G: df/dt = omega_c G f  (block rotation per harmonic)
        G = np.zeros((self.dim, self.dim))
        for m in range(1, M + 1):
            G[2 * m - 1, 2 * m] = -m
            G[2 * m, 2 * m - 1] = +m

        # |Ax| for the characteristic (upwind) flux. Ax is self-adjoint under the
        # harmonic metric W = diag(2*pi, pi, pi, ...), so symmetrize, take |.| via
        # eigh, map back. Because A_n(phi) = Rot(phi) Ax Rot(-phi) and W is
        # rotation-invariant, |A_n(phi)| = Rot(phi) |Ax| Rot(-phi): one eigen-
        # decomposition serves every face normal (see `abs_flux`).
        w = np.full(self.dim, np.pi); w[0] = 2 * np.pi
        wh, wih = np.sqrt(w), 1.0 / np.sqrt(w)
        Sx = (wh[:, None] * Ax) * wih[None, :]
        ev, Uev = np.linalg.eigh(Sx)
        absAx = wih[:, None] * ((Uev * np.abs(ev)) @ Uev.T) * wh[None, :]

        dev = rc.device
        self.theta = theta
        self.T = torch.tensor(T, device=dev, dtype=dtype)
        self.Tinv = torch.tensor(Tinv, device=dev, dtype=dtype)
        self.Ax = torch.tensor(Ax, device=dev, dtype=dtype)
        self.Ay = torch.tensor(Ay, device=dev, dtype=dtype)
        self.absAx = torch.tensor(absAx, device=dev, dtype=dtype)
        self.G = torch.tensor(G, device=dev, dtype=dtype)
        self.max_speed = vF  # spectral radius of n_x Ax + n_y Ay (for LF flux/CFL)

    def rotate(self, s: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Apply the angular-rotation operator Rot(phi) = exp(phi G) to modal
        coefficients s (..., dim); harmonic m rotates by m*phi. `phi` broadcasts
        over the leading dimensions of s."""
        out = s.clone()
        for m in range(1, self.M + 1):
            c, sn = torch.cos(m * phi), torch.sin(m * phi)
            a, b = s[..., 2 * m - 1], s[..., 2 * m]
            out[..., 2 * m - 1] = c * a - sn * b
            out[..., 2 * m] = sn * a + c * b
        return out

    def abs_flux(self, s: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """|A_n| s for face normals at angle phi, via |A_n| = Rot(phi)|Ax|Rot(-phi).
        s is (..., dim); phi (...) is the per-node outward-normal angle."""
        return self.rotate(self.rotate(s, -phi) @ self.absAx.T, phi)

    # --- transforms ---
    def to_nodes(self, f: torch.Tensor) -> torch.Tensor:
        """Modal coefficients (..., dim) -> nodal values (..., Nq)."""
        return torch.einsum("qc,...c->...q", self.T, f)

    def to_modes(self, u: torch.Tensor) -> torch.Tensor:
        """Nodal values (..., Nq) -> modal coefficients (..., dim)."""
        return torch.einsum("cq,...q->...c", self.Tinv, u)

    # --- diagonal momentum-space operators ---
    def collision_rates(self, tau_p: float, tau_ee: float) -> torch.Tensor:
        """Per-coefficient relaxation rates (diagonal collision operator)."""
        rp = 0.0 if not np.isfinite(tau_p) else 1.0 / tau_p
        ree = 0.0 if not np.isfinite(tau_ee) else 1.0 / tau_ee
        r = np.zeros(self.dim)
        for m in range(1, self.M + 1):
            r[2 * m - 1] = r[2 * m] = rp if m == 1 else (rp + ree)
        return torch.tensor(r, device=rc.device, dtype=self.G.dtype)

    def cyclotron(self, f: torch.Tensor, omega_c: float) -> torch.Tensor:
        """Exact momentum-space advection contribution omega_c * G f."""
        return omega_c * torch.einsum("cd,...d->...c", self.G, f)

    def advect_x(self, f: torch.Tensor) -> torch.Tensor:
        return torch.einsum("cd,...d->...c", self.Ax, f)

    def advect_y(self, f: torch.Tensor) -> torch.Tensor:
        return torch.einsum("cd,...d->...c", self.Ay, f)

    # --- observables (continuum normalization) ---
    def observables(self, f: torch.Tensor) -> torch.Tensor:
        """(n, jx, jy) from modal coefficients (..., dim)."""
        n = f[..., 0]
        jx = 0.5 * self.vF * f[..., 1]
        jy = 0.5 * self.vF * f[..., 2]
        return torch.stack([n, jx, jy], dim=-1)
