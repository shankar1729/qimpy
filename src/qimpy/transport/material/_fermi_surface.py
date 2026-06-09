"""FermiSurface: unified Fermi-surface / Fermi-circle material.

Storage: delta-k collocation in (k_r, theta_q), shape (Np, K, Nr*N_theta).
Modal transforms: tensor product of an angular Fourier basis (over theta) and a
radial polynomial basis (over xi = (E-mu)/T) orthonormal under the equilibrium
fluctuation weight w(xi) = (1/4T) sech^2(xi/2). The radial basis collapses to
the identity at Nr=1, recovering the Fermi-circle limit of a single Fermi-
surface state with angular Fourier modes.

Per-operator basis is chosen for each operator's natural form:
    - real-space advection      : delta-k, scalar per-collocation upwind
    - cyclotron / collision     : modal, diagonal in (l, n)
    - specular / diffuse walls  : modal (handled inside the reflector object)
    - contacts                  : constructed in modal, transformed to delta-k once
    - observables {n, jx, jy}   : delta-k weighted sums
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from . import Material


# ----------------------------------------------------------------------------
# Angular basis: real Fourier on [0, 2*pi), nodal storage at theta_q = 2*pi*q/Nq
# ----------------------------------------------------------------------------
class AngularBasis:
    """Real Fourier transforms with nodal (delta-k) storage.

    Mode ordering: ``(a_0, a_1, b_1, a_2, b_2, ..., a_M, b_M)`` for ``2M+1`` modes.
    Nodes are the *midpoint* grid ``theta_q = 2*pi*(q+1/2)/Nq``; the transforms
    are exact inverses for any ``n_quad = Nq >= 2M+1``.  The midpoint grid is
    invariant under both ``theta->-theta`` and (for even ``Nq``) ``theta->pi-theta``
    -- the ``v_y->-v_y`` and ``v_x->-v_x`` wall reflections -- so callers that need
    left-right contact symmetry pass an even ``Nq`` (see ``FermiSurface``).
    """

    def __init__(self, M: int, n_quad: int | None = None,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device | None = None) -> None:
        Nq = (2 * M + 1) if n_quad is None else n_quad
        if Nq < 2 * M + 1:
            raise ValueError(f"AngularBasis n_quad={Nq} < 2M+1={2*M+1}")
        self.M = M
        self.dim = 2 * M + 1
        self.N_theta = Nq
        # Midpoint (half-offset) nodes.  Unlike the endpoint grid 2*pi*q/Nq, this
        # set is symmetric under theta->-theta and (even Nq) theta->pi-theta, i.e.
        # the v_y->-v_y and v_x->-v_x reflections; the endpoint grid breaks
        # v_x->-v_x for odd Nq, destroying left-right contact symmetry in the
        # collisionless limit.  Discrete Fourier orthogonality holds for Nq>=2M+1.
        theta = 2.0 * np.pi * (np.arange(Nq) + 0.5) / Nq
        # modes -> nodes:  T_from_modes[q, c] s.t. f(theta_q) = sum_c T_fm[q,c] a_c
        Tfm = np.zeros((Nq, self.dim))
        Tfm[:, 0] = 1.0
        for m in range(1, M + 1):
            Tfm[:, 2 * m - 1] = np.cos(m * theta)
            Tfm[:, 2 * m]     = np.sin(m * theta)
        # nodes -> modes (discrete Fourier coefficients)
        Ttm = np.zeros((self.dim, Nq))
        Ttm[0, :] = 1.0 / Nq
        for m in range(1, M + 1):
            Ttm[2 * m - 1, :] = (2.0 / Nq) * np.cos(m * theta)
            Ttm[2 * m, :]     = (2.0 / Nq) * np.sin(m * theta)
        # cyclotron generator G: block-skew per harmonic; (G a)_m = m*(b_m,-a_m)
        G = np.zeros((self.dim, self.dim))
        for m in range(1, M + 1):
            G[2 * m - 1, 2 * m] = -m
            G[2 * m,     2 * m - 1] = +m
        dev = device or rc.device
        self.theta        = torch.as_tensor(theta, dtype=dtype, device=dev)
        self.T_from_modes = torch.as_tensor(Tfm,   dtype=dtype, device=dev)
        self.T_to_modes   = torch.as_tensor(Ttm,   dtype=dtype, device=dev)
        self.G            = torch.as_tensor(G,     dtype=dtype, device=dev)


# ----------------------------------------------------------------------------
# Radial basis: polynomials in xi orthonormal under  w(xi) = (1/4T) sech^2(xi/2)
# ----------------------------------------------------------------------------
class RadialBasis:
    """Polynomial transforms orthonormal under the equilibrium-fluctuation
    weight ``w(xi) = (1/4T) sech^2(xi/2)``.

    For ``Nr == 1`` the basis collapses to the identity at a single point
    ``xi = 0`` (Fermi surface); this is the regime where the material is the
    pure Fermi-circle limit -- one Fermi-surface state, angular Fourier modes
    only.  For ``Nr > 1`` we use ``Nr`` Gauss-Legendre points on
    ``[-xi_max, xi_max]`` as collocation, build the polynomial powers and
    orthonormalize them by Cholesky of the discrete mass matrix under the
    combined ``(Gauss-Legendre weight) * (sech^2 weight)``; this guarantees the
    discrete transforms ``T_to_modes @ T_from_modes = I`` exactly.
    """

    def __init__(self, Nr: int, T_temp: float = 1.0, xi_max: float = 6.0,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device | None = None) -> None:
        self.Nr     = Nr
        self.T_temp = T_temp
        self.xi_max = xi_max
        dev = device or rc.device
        if Nr == 1:
            # Trivial: single point at the Fermi surface, identity transforms.
            self.xi           = torch.zeros(1, dtype=dtype, device=dev)
            self.quad_w       = torch.ones(1,  dtype=dtype, device=dev)
            self.T_from_modes = torch.ones((1, 1), dtype=dtype, device=dev)
            self.T_to_modes   = torch.ones((1, 1), dtype=dtype, device=dev)
            return
        # Gauss-Legendre nodes on [-1, 1] scaled to [-xi_max, xi_max].
        x_std, w_std = np.polynomial.legendre.leggauss(Nr)
        xi  = xi_max * x_std
        w_x = xi_max * w_std                                 # Jacobian
        w_eq = (1.0 / (4.0 * T_temp)) / np.cosh(0.5 * xi) ** 2
        w_q  = w_x * w_eq                                    # discrete measure
        # Vandermonde V[q, p] = xi_q^p
        V = np.vander(xi, Nr, increasing=True)
        # Mass matrix G[p, p'] = sum_q w_q xi_q^(p+p')
        G = (V.T * w_q) @ V
        # Cholesky G = L L^T, so V @ L^{-T} is orthonormal under the discrete <,>.
        L = np.linalg.cholesky(G)
        Linv_T = np.linalg.solve(L.T, np.eye(Nr))
        Tfm = V @ Linv_T                                     # nodes <- modes
        Ttm = Tfm.T * w_q                                    # modes <- nodes
        # Sanity: Ttm @ Tfm should be identity (orthonormality under <,>_w).
        eye_check = Ttm @ Tfm
        err = float(np.max(np.abs(eye_check - np.eye(Nr))))
        if err > 1e-10:
            raise RuntimeError(
                f"RadialBasis: orthonormality check failed (max |Ttm@Tfm - I| = {err:.2e})"
            )
        self.xi           = torch.as_tensor(xi,  dtype=dtype, device=dev)
        self.quad_w       = torch.as_tensor(w_q, dtype=dtype, device=dev)
        self.T_from_modes = torch.as_tensor(Tfm, dtype=dtype, device=dev)
        self.T_to_modes   = torch.as_tensor(Ttm, dtype=dtype, device=dev)


# ----------------------------------------------------------------------------
# FermiSurface material
# ----------------------------------------------------------------------------
class FermiSurface(Material):
    """Unified Fermi-surface material; ``Nr=1`` recovers the Fermi-circle limit.

    Parameters
    ----------
    kF, vF
        Fermi wave-vector and Fermi velocity (a.u.).
    M_theta
        Highest angular harmonic retained (storage carries ``2*M_theta+1`` per
        radial point).
    Nr
        Radial mode count (defaults to 1 = Fermi-circle limit).
    T
        Temperature in energy units (a.u.).  Used only when ``Nr > 1``.
    xi_max
        Radial truncation in units of ``T`` (defaults to 6: tails of sech^2 are
        ``< 1e-5``).
    tau_p, tau_ee, r_c, specularity
        Phenomenological momentum-relaxation time, electron-electron-collision
        time, cyclotron radius (``inf`` for no field), and wall specularity
        ``s in [0, 1]`` (``s=1`` pure specular, ``s=0`` fully diffuse).
    """

    kF: float; vF: float; M_theta: int; Nr: int
    T_temp: float; xi_max: float
    tau_inv_p: float; tau_inv_ee: float
    r_c: float; specularity: float
    k_speed: float
    angular: AngularBasis
    radial:  RadialBasis

    def __init__(
        self, *, kF: float, vF: float, M_theta: int,
        Nr: int = 1, T: float = 1.0, xi_max: float = 6.0,
        tau_p: float = np.inf, tau_ee: float = np.inf,
        r_c: float = np.inf, specularity: float = 1.0,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        super().__init__()
        self.kF, self.vF = kF, vF
        self.M_theta, self.Nr = M_theta, Nr
        self.T_temp, self.xi_max = T, xi_max
        self.r_c = r_c
        self.tau_inv_p  = 1.0 / tau_p
        self.tau_inv_ee = 1.0 / tau_ee
        self.specularity = specularity
        # Even angular-node count (rounded up to a multiple of 4, >= 2M+1): with
        # the midpoint quadrature this is symmetric under both v_x->-v_x and
        # v_y->-v_y and places no node tangent to an axis-aligned wall (avoids a
        # 0/0 in the specular reflector).  Keeps source/drain (left-right)
        # symmetry exact even in the ballistic limit; an odd 2M+1 grid breaks it.
        # The retained mode count (2M+1) is unchanged.
        N_theta = -(-(2 * M_theta + 1) // 4) * 4
        N_k = Nr * N_theta
        self.initialize(wk=1.0, nk=N_k, n_bands=1, n_dim=2,
                        process_grid=process_grid)
        if self.comm.size > 1:
            raise InvalidInputException(
                "FermiSurface couples k-channels at the boundary; the k "
                "process-grid dimension must be 1 (parallelize over space)."
            )
        dtype = self.v.dtype
        self.angular = AngularBasis(M_theta, n_quad=N_theta, dtype=dtype)
        self.radial  = RadialBasis(Nr, T_temp=T, xi_max=xi_max, dtype=dtype)
        # Per-collocation transport velocity: same for every radial point
        # (Fermi-surface linearization holds |v| = vF independent of energy).
        theta_q = self.angular.theta
        v_per_theta = vF * torch.stack(
            [torch.cos(theta_q), torch.sin(theta_q)], dim=-1
        )                                                       # (N_theta, 2)
        self.v = v_per_theta.repeat(Nr, 1)                      # (Nr*N_theta, 2)
        # Scalar advection -> no coupling object exposed to the DG layer
        self.coupling = None
        self.k_speed = (vF / r_c) if np.isfinite(r_c) else 0.0
        # Per-mode collision rates in flattened (radial n, angular m) ordering.
        # Angular: m=0 conserved (rate 0); m=1 decays through tau_p only;
        # m>=2 decays through tau_p + tau_ee (both impurity and viscous channels).
        # Radial: n=0 is the equilibrium-shape mode (no extra decay); n>=1 are
        # higher energy moments that we damp with 1/tau_ee as a placeholder
        # until the microscopic-L hook is added.
        ang = np.zeros(self.angular.dim)
        for m in range(1, M_theta + 1):
            ang[2 * m - 1] = ang[2 * m] = (
                self.tau_inv_p if m == 1 else (self.tau_inv_p + self.tau_inv_ee)
            )
        rad = np.zeros(Nr)
        rad[1:] = self.tau_inv_ee
        rates = ang[None, :] + rad[:, None]                  # (Nr, dim_theta)
        # Particle conservation at (n=0, m=0): override to 0.
        rates[0, 0] = 0.0
        self.rates_modal = torch.as_tensor(
            rates.reshape(-1), dtype=dtype, device=rc.device
        )

    # ---- transforms (tensor product of radial and angular pieces) ----

    def to_modes(self, f: torch.Tensor) -> torch.Tensor:
        """Nodal ``(..., Nr*N_theta)`` -> modal ``(..., Nr*(2M_theta+1))``."""
        Ntheta = self.angular.N_theta
        Nr = self.Nr
        shape_in = f.shape
        f4 = f.reshape(*shape_in[:-1], Nr, Ntheta)
        # angular: (..., r, q) -> (..., r, c_theta)
        a_t = torch.einsum("cq,...rq->...rc", self.angular.T_to_modes, f4)
        # radial: (..., r, c_theta) -> (..., n_r, c_theta)
        a   = torch.einsum("nr,...rc->...nc", self.radial.T_to_modes, a_t)
        return a.reshape(*shape_in[:-1], Nr * self.angular.dim)

    def from_modes(self, a: torch.Tensor) -> torch.Tensor:
        """Modal ``(..., Nr*(2M_theta+1))`` -> nodal ``(..., Nr*N_theta)``."""
        dim_theta = self.angular.dim
        Nr = self.Nr
        shape_in = a.shape
        a4 = a.reshape(*shape_in[:-1], Nr, dim_theta)
        # radial: (..., n_r, c_theta) -> (..., r, c_theta)
        f_r = torch.einsum("rn,...nc->...rc", self.radial.T_from_modes, a4)
        # angular: (..., r, c_theta) -> (..., r, q)
        f   = torch.einsum("qc,...rc->...rq", self.angular.T_from_modes, f_r)
        return f.reshape(*shape_in[:-1], Nr * self.angular.N_theta)

    # ---- the rest, stubbed for steps 2-7 ----

    @property
    def transport_velocity(self) -> torch.Tensor:
        return self.v

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        """Cyclotron + collision in modal space; identity in delta-k storage.

        Transforms rho (delta-k) -> a (modal), applies diagonal collision rates
        and (if r_c is finite) the exact cyclotron generator G acting on the
        angular block within each radial mode, then transforms back.
        """
        if self.rates_modal.abs().sum() == 0 and self.k_speed == 0.0:
            return torch.zeros_like(rho)                       # ballistic, no field
        a = self.to_modes(rho)                                 # (..., Nr*dim_theta)
        a_dot = -self.rates_modal * a
        if self.k_speed:
            Nr, dim_t = self.Nr, self.angular.dim
            a4 = a.reshape(*a.shape[:-1], Nr, dim_t)
            Ga4 = torch.einsum("dc,...nc->...nd", self.angular.G, a4)
            a_dot = a_dot + self.k_speed * Ga4.reshape(*a.shape)
        return self.from_modes(a_dot)

    def get_observable_names(self) -> list[str]:
        return ["n", "jx", "jy"]

    @stopwatch
    def get_observables(self, t: float) -> torch.Tensor:
        """Per-channel coefficients for [n, jx, jy] in delta-k storage.

        n[r,q]  = w_r / N_theta
        jx[r,q] = w_r * vF * cos(theta_q) / N_theta
        jy[r,q] = w_r * vF * sin(theta_q) / N_theta

        For Nr=1 (w_r = 1) these are the standard Fermi-circle integrals; for
        Nr>1 the radial weights w_r are the equilibrium-fluctuation-weighted
        Gauss-Legendre weights from RadialBasis.
        """
        N_theta = self.angular.N_theta
        theta = self.angular.theta
        w_r = self.radial.quad_w                              # (Nr,)
        cos_q = torch.cos(theta) / N_theta                    # (N_theta,)
        sin_q = torch.sin(theta) / N_theta
        one_q = torch.full_like(cos_q, 1.0 / N_theta)
        # (Nr, N_theta) -> flatten to (Nr * N_theta,)
        n_rq  = (w_r[:, None] * one_q[None, :]).reshape(-1)
        jx_rq = (w_r[:, None] * (self.vF * cos_q)[None, :]).reshape(-1)
        jy_rq = (w_r[:, None] * (self.vF * sin_q)[None, :]).reshape(-1)
        return torch.stack([n_rq, jx_rq, jy_rq], dim=0)

    def get_contactor(self, n: torch.Tensor, **kwargs) -> Callable:
        return _FermiSurfaceContactor(self, n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable:
        return _FermiSurfaceReflector(self, n, self.specularity)

    def initialize_fields(self, rho, params, patch_id) -> None:
        pass

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        a = cp_path.attrs
        a["kF"], a["vF"] = self.kF, self.vF
        a["M_theta"], a["Nr"] = self.M_theta, self.Nr
        a["T"], a["xi_max"]   = self.T_temp, self.xi_max
        a["tau_p"]  = (1.0 / self.tau_inv_p)  if self.tau_inv_p  else np.inf
        a["tau_ee"] = (1.0 / self.tau_inv_ee) if self.tau_inv_ee else np.inf
        a["r_c"], a["specularity"] = self.r_c, self.specularity
        return list(a.keys())


# ----------------------------------------------------------------------------
# Contactor: voltage dmu + drift current vD (Dirichlet ghost in delta-k)
# ----------------------------------------------------------------------------
class _FermiSurfaceContactor:
    """Contact distribution constructed in modes, then transformed to delta-k.

    Voltage shift dmu sets the equilibrium-shape (n=0, m=0) mode.  Drift current
    vD into the device sets the (n=0, m=1) cos/sin pair, rotated to the wall
    outward normal.  Higher modes are left at zero.  For Nr=1 the n=0 mode is
    the only radial mode and this matches ``ModalContactor`` identically; for
    Nr>1 the equilibrium-shape part lives entirely in the n=0 radial mode (the
    particle-number null mode of L^(ee)) and the equilibrium derivative w.r.t.
    chemical potential / drift is *exactly* representable there.
    """

    def __init__(
        self, fs: "FermiSurface", n: torch.Tensor, *,
        dmu: float = 0.0, vD: float = 0.0,
    ) -> None:
        # Build the modal contact distribution in (Nr_modes, dim_theta).
        n = n.to(rc.device)  # accept normals supplied on any device
        Nsel = n.shape[0]
        dim_theta = fs.angular.dim
        contact_modal = torch.zeros(
            (Nsel, fs.Nr, dim_theta), device=rc.device, dtype=n.dtype,
        )
        phi = torch.atan2(n[:, 1], n[:, 0])                  # outward normal
        # Voltage: set (n=0, m=0) to dmu (isotropic, equilibrium-shaped).
        contact_modal[:, 0, 0] = dmu
        # Drift: set (n=0, m=1) cos/sin to -(vD/vF) (cos phi, sin phi).
        contact_modal[:, 0, 1] = -(vD / fs.vF) * torch.cos(phi)
        contact_modal[:, 0, 2] = -(vD / fs.vF) * torch.sin(phi)
        # Transform once to delta-k.  Flatten radial axis for from_modes.
        contact_modal_flat = contact_modal.reshape(Nsel, fs.Nr * dim_theta)
        self.rho_contact = fs.from_modes(contact_modal_flat)  # (Nsel, Nr*N_theta)

    def __call__(self, t: float) -> torch.Tensor:
        return self.rho_contact


# ----------------------------------------------------------------------------
# Reflector: specular block-rotation per harmonic + diffuse (n=0,m=0) refill,
# both done in modes inside this class so the geometry layer just substitutes
# the returned u^P as the Dirichlet ghost.
# ----------------------------------------------------------------------------
class _FermiSurfaceReflector:
    """Boundary reflection for FermiSurface: specular fraction s, diffuse (1-s).

    Specular in modes: identical to ``ModalReflector._specular`` -- flip the
    sin(m theta) coefficients, then rotate harmonic m by ``m * (2 phi + pi)`` --
    applied independently on each radial slice (radial index commutes with
    angular rotation).  Exact at any wall angle.

    Diffuse in modes: outgoing is set to ``D`` in the (n=0, m=0) mode only,
    everything else zero, where ``D`` is fixed by zero net normal mass flux:
        D = sum_q (v_q . n)_+  u^M_{r=0}(theta_q)  /  sum_q (v_q . n)_- (with sign).
    Here ``u^M_{r=0}(theta_q) = sum_r T_to_radial[0, r] u^M(k_r, theta_q)`` is
    the n=0 radial projection of u^M -- only this projection contributes to the
    mass-flux balance (higher radial modes carry no mass).  For ``Nr = 1`` this
    is just ``u^M(theta_q)`` and the formula reduces to the scalar one used by
    ``_dg_torch`` for discrete-ordinate single_band materials.
    """

    def __init__(
        self, fs: "FermiSurface", n: torch.Tensor, specularity: float,
    ) -> None:
        n = n.to(rc.device)  # accept normals supplied on any device
        self.fs = fs
        self.s = float(specularity)
        self.M_theta = fs.M_theta
        self.Nr = fs.Nr
        self.dim_theta = fs.angular.dim
        self.N_theta = fs.angular.N_theta
        # outward-normal angle and the rotation angle 2 phi + pi
        self.phi   = torch.atan2(n[:, 1], n[:, 0])
        self.angle = 2.0 * self.phi + np.pi
        # per-(wall, ordinate) v dot n (same for every radial point)
        theta = fs.angular.theta
        v_dot_n = fs.vF * (
            n[:, 0:1] * torch.cos(theta)[None, :] +
            n[:, 1:2] * torch.sin(theta)[None, :]
        )                                                    # (Nsel, N_theta)
        self.adn_pos = v_dot_n.clamp(min=0.0)
        self.adn_neg = v_dot_n.clamp(max=0.0)
        self.w_in    = (-self.adn_neg).sum(-1).clamp(min=1e-300)
        # n=0 radial projector  T_to_radial[0, :] (length Nr); for Nr=1 this is [1.0]
        self.T_to_rad_0 = fs.radial.T_to_modes[0, :]         # (Nr,)
        # n=0 radial basis function psi_0(xi_r) = T_from_radial[:, 0] (length Nr)
        self.psi_0 = fs.radial.T_from_modes[:, 0]            # (Nr,)

    # ---- specular: block-rotation R(phi) per harmonic, independent in radial ----
    def _specular_modal(self, a_modal: torch.Tensor) -> torch.Tensor:
        shape_in = a_modal.shape
        a4 = a_modal.reshape(*shape_in[:-1], self.Nr, self.dim_theta)
        out = a4.clone()                                     # (n=0, m=0) untouched
        angle = self.angle                                   # (Nsel,)
        for m in range(1, self.M_theta + 1):
            c   = torch.cos(m * angle)
            sn  = torch.sin(m * angle)
            a_c = a4[..., 2 * m - 1]
            a_s = -a4[..., 2 * m]                            # flip sin coeff first
            # rotate (a_c, a_s) by m*angle:  (c*a - sn*b, sn*a + c*b)
            out[..., 2 * m - 1] = c[..., None] * a_c - sn[..., None] * a_s
            out[..., 2 * m]     = sn[..., None] * a_c + c[..., None] * a_s
        return out.reshape(shape_in)

    # ---- main call ----
    def __call__(self, uM_dk: torch.Tensor) -> torch.Tensor:
        # 1) Specular component in modes, transformed back to delta-k.
        uM_modal  = self.fs.to_modes(uM_dk)
        spec_modal = self._specular_modal(uM_modal)
        spec_dk   = self.fs.from_modes(spec_modal)
        # 2) (D_n, T_n) per radial mode -- enforce mass AND tangential-momentum
        #    conservation at the wall to machine precision.
        #
        # The outgoing addition has the form
        #     u_added(r, q) = sum_n  psi_n(xi_r) * [D_n + T_n * sin(theta_q - phi)]
        # so D_n controls the (n, m=0) mode (mass-like) and T_n controls the
        # (n, m=1 tangential) mode (tang-momentum-like) of the outgoing.
        #
        # Per radial n we solve the 2x2 system
        #     [-w_in     beta    ] [D_n]   [-F_M_n     ]
        #     [vF beta   vF gamma] [T_n] = [-s * F_T_n ]
        # where
        #   beta  = sum_{q in inflow} (v_q.n) * sin(theta_q - phi)     (<=0)
        #   gamma = sum_{q in inflow} (v_q.n) * sin^2(theta_q - phi)   (<=0)
        #   F_M_n = F_out_mass_n + s * F_in_spec_mass_n  (discrete mass flux)
        #   F_T_n = F_out_tang_n + F_in_spec_tang_n     (discrete tang flux)
        # The RHS for tang enforces the discrete identity
        #   F_total_tang = (1 - s) * F_out_tang_n
        # which is the continuum tang flux balance at the wall (specular
        # preserves tang momentum, diffuse fraction dumps it into the wall).
        # At s=1 this drives the specular's discrete-quadrature tang leak to
        # zero; at s=0 it cancels the discrete artifact  D*beta  in the tang
        # flux integral, leaving the gas-loses-F_out_tang continuum behavior.
        #
        # The 2x2 blocks decouple per n because the velocity v_q is r-indep:
        # same (beta, gamma, w_in) for every radial level.
        shape_in = uM_dk.shape
        uM4 = uM_dk.reshape(*shape_in[:-1], self.Nr, self.N_theta)
        sp4 = spec_dk.reshape(*shape_in[:-1], self.Nr, self.N_theta)
        # radial-n projection: u^M_n(q) = sum_r T_to_radial[n, r] u^M(r, q)
        T_to_r   = self.fs.radial.T_to_modes                # (Nr, Nr)
        T_from_r = self.fs.radial.T_from_modes              # (Nr, Nr)
        uM_n_q = torch.einsum("nr,...arq->...anq", T_to_r, uM4)
        sp_n_q = torch.einsum("nr,...arq->...anq", T_to_r, sp4)
        # angular basis sin(theta_q - phi) per wall node
        sin_q = (torch.cos(self.phi)[:, None] * torch.sin(self.fs.angular.theta)[None, :]
                 - torch.sin(self.phi)[:, None] * torch.cos(self.fs.angular.theta)[None, :])
        # 2x2 wall-geometry coefficients (Nsel,) -- same for every n
        beta  = (self.adn_neg * sin_q).sum(-1)
        gamma = (self.adn_neg * sin_q ** 2).sum(-1)
        # discrete fluxes per (Nsel, Nr)
        adn_pos = self.adn_pos[:, None, :]
        adn_neg = self.adn_neg[:, None, :]
        sin_b   = sin_q[:, None, :]
        vF = self.fs.vF
        F_out_mass_n     = (adn_pos * uM_n_q).sum(-1)
        F_in_spec_mass_n = (adn_neg * sp_n_q).sum(-1)
        F_out_tang_n     = vF * (adn_pos * sin_b * uM_n_q).sum(-1)
        F_in_spec_tang_n = vF * (adn_neg * sin_b * sp_n_q).sum(-1)
        F_M_n = F_out_mass_n + self.s * F_in_spec_mass_n
        F_T_n = F_out_tang_n + F_in_spec_tang_n
        # 2x2 Cramer per (Nsel), broadcast over Nr (and any leading batch).
        det = -vF * (self.w_in * gamma + beta * beta)        # (Nsel,)  < 0
        det_b = det[:, None]
        b1 = -F_M_n
        b2 = -self.s * F_T_n
        D = (b1 * (vF * gamma)[:, None] - b2 * beta[:, None]) / det_b
        T = ((-self.w_in)[:, None] * b2 - (vF * beta)[:, None] * b1) / det_b
        # 3) u_added(r, q) = sum_n psi_n(r) [D_n + T_n sin_q],  via T_from_modes.
        D_r = torch.einsum("rn,...an->...ar", T_from_r, D)
        T_r = torch.einsum("rn,...an->...ar", T_from_r, T)
        u_added = D_r.unsqueeze(-1) + T_r.unsqueeze(-1) * sin_q.unsqueeze(-2)
        return self.s * spec_dk + u_added.reshape(shape_in)
