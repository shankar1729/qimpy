"""Microscopic e-e collision kernels for a 2D Fermi liquid (atomic units).

Implements the electron-electron collision operator of an isotropic 2D
Fermi liquid with statically-screened Coulomb interaction
``M_q = 2 pi F(q) / (epsilon_b (q + kappa F(q)))``, where ``F(q)`` is an
optional quantum-well form factor (``F = 1`` for an ideal 2D sheet) and
``kappa`` the 2D Thomas-Fermi wavevector.

Conventions follow the derivation notes "The e-e Collision Operator in a
2D Fermi Liquid" (exact nonlinear reduction): two electrons 1,2 scatter
to 3,4; the kinematic reduction resolves the momentum delta and the
energy delta exactly, leaving integrals over (xi_2, xi_3, phi_3) with two
azimuthal roots for phi_2.  Rates carry the Fermi-liquid (kB T)^2 phase
space; the closed-form even-harmonic rate is

    gamma_m = m*^2 (kB T)^2 K_m / (16 pi E_F),
    K_m = int_0^{2 pi} |M_q|^2 (1 - cos m alpha) / |sin alpha| d alpha,
    q = 2 kF sin(alpha/2).

Note: K_m is finite only for even m (for odd m the integrand is
non-integrable at alpha = pi); odd harmonics do not relax at leading
order and their entries are gated to zero here.  All inputs/outputs in
Hartree atomic units (hbar = kB = 1).
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence

import numpy as np
import torch

#: Relative cut at the azimuthal root-coalescence edge |cos_arg| = 1
#: (kinematic van Hove edge).  The edge is inverse-square-root
#: integrable: excluding the sliver biases integrals by O(sqrt(eps))
#: ~ 1e-3 relative, while preventing O(1/sqrt(eps)) node spikes.
EDGE_EPS = 1e-6


def well_form_factor(qW: torch.Tensor) -> torch.Tensor:
    """Intra-subband Coulomb form factor of an infinite square well.

    For the ground subband ``psi(z) = sqrt(2/W) sin(pi z / W)``:
    ``F(q) = int dz dz' |psi(z)|^2 |psi(z')|^2 exp(-q |z - z'|)``
    evaluated in closed form as a function of ``x = q W``.
    ``F(0) = 1`` and ``F -> 3/x`` for large ``x``.
    """
    x = qW
    four_pi_sq = 4 * np.pi**2
    den = x**2 + four_pi_sq
    # Closed form (Ando-Fowler-Stern); regularize x -> 0 where the
    # individually-divergent 2/x pieces cancel against each other:
    x_safe = torch.where(x > 1e-6, x, torch.ones_like(x))
    F = (
        3 * x_safe / den
        + 8 * np.pi**2 / (x_safe * den)
        - 32 * np.pi**4 * (1 - torch.exp(-x_safe)) / (x_safe**2 * den**2)
    )
    return torch.where(x > 1e-6, F, torch.ones_like(x))


def matrix_element_sq(
    q: torch.Tensor, *, epsilon_bg: float, kappa: float, well_width: float = 0.0
) -> torch.Tensor:
    """``|M_q|^2`` for the statically screened 2D Coulomb interaction.

    ``M_q = 2 pi F(q) / (epsilon_bg (q + kappa F(q)))``: bare 2D Coulomb
    ``2 pi F(q) / (epsilon_bg q)`` with quasi-2D Thomas-Fermi screening
    ``epsilon(q) = 1 + kappa F(q) / q``.  ``well_width = 0`` gives the
    ideal-sheet limit ``F = 1`` used in the derivation notes.
    """
    if well_width > 0.0:
        F = well_form_factor(q * well_width)
    else:
        F = torch.ones_like(q)
    return (2 * np.pi * F / (epsilon_bg * (q + kappa * F))) ** 2


def K_table(
    M_max: int,
    *,
    kF: float,
    epsilon_bg: float,
    kappa: float,
    well_width: float = 0.0,
    n_alpha: int = 4096,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Angular collision kernels ``K_m`` for ``m = 0 .. M_max``.

    ``K_m = int_0^{2pi} |M_q|^2 (1 - cos m alpha)/|sin alpha| d alpha``
    with ``q = 2 kF sin(alpha/2)``, by Gauss-Legendre quadrature on
    ``(0, pi)`` (doubled by symmetry).  Odd-``m`` entries are set to zero:
    the corresponding integrals are log-divergent at ``alpha = pi`` and
    odd harmonics are conserved at leading order (parity gate
    ``[1 + (-1)^m]/2`` of the derivation notes).
    """
    x, w = np.polynomial.legendre.leggauss(n_alpha)
    alpha = 0.5 * np.pi * (x + 1.0)  # (0, pi)
    w_alpha = 0.5 * np.pi * w
    alpha_t = torch.tensor(alpha, dtype=torch.float64)
    q = 2 * kF * torch.sin(alpha_t / 2)
    Msq = matrix_element_sq(
        q, epsilon_bg=epsilon_bg, kappa=kappa, well_width=well_width
    )
    base = (torch.tensor(w_alpha) * Msq / torch.sin(alpha_t)).numpy()
    m = np.arange(M_max + 1)
    one_minus_cos = 1.0 - np.cos(np.outer(alpha, m))  # (n_alpha, M_max+1)
    K = 2.0 * (base @ one_minus_cos)
    K[1::2] = 0.0  # odd m: gated (see docstring)
    return torch.tensor(K, dtype=dtype, device=device)


def gamma_linear(
    K: torch.Tensor, *, m_star: float, T: float, E_F: float
) -> torch.Tensor:
    """Closed-form leading-order rates ``gamma_m`` from the ``K_m`` table.

    ``gamma_m = m*^2 T^2 K_m / (16 pi E_F)`` for even ``m`` (zero for
    ``m = 0`` since ``K_0 = 0``, and for odd ``m`` by the parity gate).
    """
    return (m_star**2 * T**2 / (16 * np.pi * E_F)) * K


def cubic_prefactor(*, m_star: float, E_F: float) -> float:
    """Prefactor of the cubic vertex acting on code-units fields.

    The vertex in on-circle occupation units (g = delta-f at xi = 0) is
    ``f_dot|_cubic = A [ (g+h) K*(gh) - gh K*(g+h) ]`` with
    ``A = m*^2 T^2/(8 pi E_F)``, ``h(phi) = g(phi + pi)`` and ``K*`` the
    angular convolution diagonal in harmonics with eigenvalues ``K_m``
    (even ``m`` only; products ``gh``, ``g+h`` are pi-periodic so only
    even harmonics enter -- this form is manifestly free of the odd-m
    divergences of the term-by-term mode sum).  The solver field Phi
    (energy units) relates by ``g = Phi / (4 T)``; the cubic is
    homogeneous of degree 3, so ``Phi_dot = A/(4T)^2 [ ... in Phi ... ]``
    and the ``T^2`` cancels: the code-units prefactor is T-independent.
    """
    return m_star**2 / (128 * np.pi * E_F)


# ----------------------------------------------------------------------------
# Exact nonlinear reduced operator (reference implementation for verification)
# ----------------------------------------------------------------------------
def exact_collision_reference(
    delta_f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x1: torch.Tensor,
    phi1: torch.Tensor,
    *,
    kF: float,
    m_star: float,
    T: float,
    epsilon_bg: float,
    kappa: float,
    well_width: float = 0.0,
    n_xi: int = 48,
    xi_cut: float = 12.0,
    n_phi: int = 1024,
    linearize: bool = False,
    chunk: int = 16,
) -> torch.Tensor:
    """Brute-force evaluation of the exact reduced collision operator.

    Evaluates ``f_dot(x1, phi1)`` (occupation per unit time) for the
    deviation ``delta_f(x, phi)`` (occupation units; ``x = xi / T``) via

    ``f_dot_1 = m*^3/(2 pi)^3 int dxi2 dxi3 dphi3 sum_roots
                |M_q|^2 / (k2 k4 |sin(phi4 - phi2)|) (B - F)``

    with ``B - F = f3 f4 (1-f1)(1-f2) - f1 f2 (1-f3)(1-f4)`` and
    ``k4 = k1 + k2 - k3`` resolved on the energy shell
    (two azimuthal roots for ``phi2``).  ``linearize=True`` replaces
    ``B - F`` by its exact first variation around equilibrium.

    This is O(n_xi^2 n_phi) per evaluation point - verification only.
    """
    E_F = 0.5 * kF**2 / m_star
    t = T / E_F
    dd = dict(dtype=torch.float64)

    # Quadrature grids (Gauss-Legendre).  The lower energy limit stays
    # above the band bottom (x > -1/t), where the 1/(k2 k4) kinematic
    # factors have integrable but quadrature-hostile edges; the excluded
    # strip is occupation-suppressed by ~exp(-E_F/T):
    lo = -min(xi_cut, 0.8 / t)
    xg, xw = np.polynomial.legendre.leggauss(n_xi)
    x2 = torch.tensor(0.5 * (xi_cut + lo) + 0.5 * (xi_cut - lo) * xg, **dd)
    w2 = torch.tensor(0.5 * (xi_cut - lo) * xw, **dd)
    x3, w3 = x2.clone(), w2.clone()
    pg, pw = np.polynomial.legendre.leggauss(n_phi)
    phi3 = torch.tensor(np.pi * (pg + 1.0), **dd)
    wphi = torch.tensor(np.pi * pw, **dd)

    def f0(x):
        return torch.sigmoid(-x)  # 1/(e^x + 1)

    def k_of(x):
        return kF * torch.sqrt(torch.clamp(1.0 + t * x, min=0.0))

    pref = m_star**3 / (2 * np.pi) ** 3
    out = torch.empty(len(x1), **dd)

    for i0 in range(0, len(x1), chunk):
        sl = slice(i0, min(i0 + chunk, len(x1)))
        X1 = x1[sl, None, None, None].to(torch.float64)  # (B,1,1,1)
        P1 = phi1[sl, None, None, None].to(torch.float64)
        X2 = x2[None, :, None, None]
        X3 = x3[None, None, :, None]
        P3 = phi3[None, None, None, :]
        X4 = X1 + X2 - X3
        k1, k2, k3, k4 = k_of(X1), k_of(X2), k_of(X3), k_of(X4)
        band_ok = (1.0 + t * X4) > 0.05  # exclude deep band-bottom edge

        # Momentum transfer q = |k1 - k3| and its azimuth:
        dphi31 = P3 - P1
        q_sq = k1**2 + k3**2 - 2 * k1 * k3 * torch.cos(dphi31)
        q = torch.sqrt(torch.clamp(q_sq, min=1e-300))
        Px = k1 * torch.cos(P1) - k3 * torch.cos(P3)
        Py = k1 * torch.sin(P1) - k3 * torch.sin(P3)
        phiP = torch.atan2(Py, Px)

        # Azimuthal roots for phi2 from |P + k2| = k4.  Exclude a tiny
        # sliver at the root-coalescence edge |cos_arg| -> 1, where the
        # 1/|sin(phi4 - phi2)| Jacobian has its (integrable) inverse-
        # square-root singularity: the sliver contributes O(sqrt(eps))
        # while unregularized nodes there produce O(1/sqrt(eps)) spikes.
        cos_arg = (k4**2 - q_sq - k2**2) / torch.clamp(2 * q * k2, min=1e-300)
        root_ok = band_ok & (cos_arg.abs() <= 1.0 - EDGE_EPS)
        dlt = torch.arccos(torch.clamp(cos_arg, -1.0, 1.0))

        Msq = matrix_element_sq(
            q, epsilon_bg=epsilon_bg, kappa=kappa, well_width=well_width
        )
        wt = (
            w2[None, :, None, None]
            * w3[None, None, :, None]
            * wphi[None, None, None, :]
        ) * (T**2)

        acc = torch.zeros(
            (X1.shape[0], len(x2), len(x3), len(phi3)), **dd
        )
        for sgn in (+1.0, -1.0):
            phi2 = phiP + sgn * dlt
            k4x = Px + k2 * torch.cos(phi2)
            k4y = Py + k2 * torch.sin(phi2)
            phi4 = torch.atan2(k4y, k4x)
            sin42 = torch.sin(phi4 - phi2).abs()
            jac = 1.0 / torch.clamp(k2 * k4 * sin42, min=1e-14)

            f1_0, f2_0 = f0(X1).expand_as(phi2), f0(X2).expand_as(phi2)
            f3_0, f4_0 = f0(X3).expand_as(phi2), f0(X4).expand_as(phi2)
            d1 = delta_f(X1.expand_as(phi2), P1.expand_as(phi2))
            d2 = delta_f(X2.expand_as(phi2), phi2)
            d3 = delta_f(X3.expand_as(phi2), P3.expand_as(phi2))
            d4 = delta_f(X4.expand_as(phi2), phi4)
            if linearize:
                # delta(B - F) = W (Phi3 + Phi4 - Phi1 - Phi2),
                # Phi_i = d_i / (f0_i (1 - f0_i)):
                W = f1_0 * f2_0 * (1 - f3_0) * (1 - f4_0)
                bmf = W * (
                    d3 / (f3_0 * (1 - f3_0))
                    + d4 / (f4_0 * (1 - f4_0))
                    - d1 / (f1_0 * (1 - f1_0))
                    - d2 / (f2_0 * (1 - f2_0))
                )
            else:
                f1, f2 = f1_0 + d1, f2_0 + d2
                f3, f4 = f3_0 + d3, f4_0 + d4
                bmf = f3 * f4 * (1 - f1) * (1 - f2) - f1 * f2 * (1 - f3) * (
                    1 - f4
                )
            acc = acc + torch.where(
                root_ok, Msq * jac * bmf, torch.zeros_like(jac)
            )
        out[sl] = pref * (wt * acc).sum(dim=(1, 2, 3))
    return out


# ----------------------------------------------------------------------------
# Energy-moment (radial-tower) matrix from exact kinematics, for Nr > 1
# ----------------------------------------------------------------------------
def L_blocks(
    *,
    x_nodes: torch.Tensor,
    psi_coeff: torch.Tensor,
    m_list: Sequence[int],
    kF: float,
    m_star: float,
    T: float,
    epsilon_bg: float,
    kappa: float,
    well_width: float = 0.0,
    n_xi: int = 32,
    xi_cut: float = 10.0,
    n_phi: int = 1024,
) -> torch.Tensor:
    """Linearized collision matrix in a radial polynomial basis.

    For each angular harmonic ``m`` in ``m_list``, computes the matrix
    ``R^(m)[r, l']``: minus the rate of change of the solver field ``Phi``
    at radial collocation node ``x_r`` per unit coefficient of the basis
    function ``psi_l'(x) e^{i m phi}``, from the exact kinematic reduction
    (linearized in amplitude, exact in temperature -- includes the
    collinear-log enhancement of the energy/heat modes).  The caller
    projects rows to coefficient space with its radial transform, giving
    ``a_dot_ml = -L^(m)_{l l'} a_ml'`` with
    ``L^(m) = T_to_radial_modes @ R^(m)``.

    ``psi_coeff[p, l]`` are power-basis coefficients: ``psi_l(x) =
    sum_p psi_coeff[p, l] x^p`` in ``x = xi/T``.  Solver-field
    convention: ``delta_f = w_eq Phi``, ``w_eq = sech^2(x/2)/(4 T)``.

    Returns tensor of shape ``(len(m_list), len(x_nodes), n_basis)``.
    """
    E_F = 0.5 * kF**2 / m_star
    t = T / E_F
    dd = dict(dtype=torch.float64)
    Nr = len(x_nodes)
    n_basis = psi_coeff.shape[1]

    # Energy grid restricted to the band (see exact_collision_reference):
    lo = -min(xi_cut, 0.8 / t)
    xg, xw = np.polynomial.legendre.leggauss(n_xi)
    x2 = torch.tensor(0.5 * (xi_cut + lo) + 0.5 * (xi_cut - lo) * xg, **dd)
    w2 = torch.tensor(0.5 * (xi_cut - lo) * xw, **dd)
    pg, pw = np.polynomial.legendre.leggauss(n_phi)
    phi3 = torch.tensor(np.pi * (pg + 1.0), **dd)
    wphi = torch.tensor(np.pi * pw, **dd)

    def f0(x):
        return torch.sigmoid(-x)

    def k_of(x):
        return kF * torch.sqrt(torch.clamp(1.0 + t * x, min=0.0))

    def psi_eval(x):  # (...,) -> (..., n_basis) via Horner
        res = torch.zeros(x.shape + (n_basis,), **dd)
        for p in range(psi_coeff.shape[0] - 1, -1, -1):
            res = res * x[..., None] + psi_coeff[p]
        return res

    pref = m_star**3 / (2 * np.pi) ** 3
    m_arr = list(m_list)
    out = torch.zeros(len(m_arr), Nr, n_basis, **dd)

    # Fixed-node quantities (phi1 = 0 by isotropy):
    x1 = x_nodes.to(torch.float64)  # (Nr,)
    psi1 = psi_eval(x1)  # (Nr, n_basis)
    psi2 = psi_eval(x2)  # (n_xi, n_basis)
    k1 = k_of(x1)[:, None, None]  # (Nr, 1, 1)
    k2 = k_of(x2)[None, :, None]  # (1, n_xi, 1)
    X1b = x1[:, None, None]
    X2b = x2[None, :, None]
    f01 = f0(x1)[:, None, None]
    f02 = f0(x2)[None, :, None]
    cosP3 = torch.cos(phi3)[None, None, :]  # (1, 1, n_phi)
    sinP3 = torch.sin(phi3)[None, None, :]
    w12 = (w2[None, :, None] * wphi[None, None, :]) * T**2  # (1, n_xi, n_phi)
    cosm3 = {m: torch.cos(m * phi3) for m in m_arr}  # each (n_phi,)

    for i3 in range(n_xi):  # stream over x3 nodes to bound memory
        x3v = x2[i3]
        w3v = w2[i3]
        X4 = X1b + X2b - x3v  # (Nr, n_xi, 1)
        k3 = k_of(x3v)  # scalar tensor
        k4 = k_of(X4)  # (Nr, n_xi, 1)
        band_ok = (1.0 + t * X4) > 0.05  # exclude deep band-bottom edge
        q_sq = k1**2 + k3**2 - 2 * k1 * k3 * cosP3  # (Nr, 1, n_phi)
        q = torch.sqrt(torch.clamp(q_sq, min=1e-300))
        Px = k1 - k3 * cosP3  # (Nr, 1, n_phi)
        Py = (-k3 * sinP3).expand_as(Px)
        phiP = torch.atan2(Py, Px)
        cos_arg = (k4**2 - q_sq - k2**2) / torch.clamp(2 * q * k2, min=1e-300)
        root_ok = band_ok & (cos_arg.abs() <= 1.0 - EDGE_EPS)
        dlt = torch.arccos(torch.clamp(cos_arg, -1.0, 1.0))
        Msq = matrix_element_sq(
            q, epsilon_bg=epsilon_bg, kappa=kappa, well_width=well_width
        )
        W = f01 * f02 * (1 - f0(x3v)) * (1 - f0(X4))  # (Nr, n_xi, 1)
        psi3 = psi_eval(x3v.reshape(1))[0]  # (n_basis,)
        psi4 = psi_eval(X4.squeeze(-1))  # (Nr, n_xi, n_basis)

        for sgn in (+1.0, -1.0):
            phi2 = phiP + sgn * dlt  # (Nr, n_xi, n_phi)
            k4x = Px + k2 * torch.cos(phi2)
            k4y = Py + k2 * torch.sin(phi2)
            phi4 = torch.atan2(k4y, k4x)
            sin42 = torch.sin(phi4 - phi2).abs()
            jac = 1.0 / torch.clamp(k2 * k4 * sin42, min=1e-14)
            G = torch.where(root_ok, Msq * W * jac, torch.zeros_like(jac))
            Gw = G * (w12 * w3v)  # (Nr, n_xi, n_phi)
            for im, m in enumerate(m_arr):
                cosm4 = torch.cos(m * phi4)
                cosm2 = torch.cos(m * phi2)
                t3 = torch.einsum("rxp,p->r", Gw, cosm3[m])[:, None] * psi3
                t4 = torch.einsum("rxp,rxb->rb", Gw * cosm4, psi4)
                t1 = Gw.sum(dim=(1, 2))[:, None] * psi1
                t2 = torch.einsum("rxp,xb->rb", Gw * cosm2, psi2)
                out[im] += pref * (t3 + t4 - t1 - t2)

    # Convert to the solver-field convention.  Input coefficients multiply
    # psi_l(x) e^{i m phi} in code units (delta_f = w_eq Phi_code with
    # w_eq = sech^2(x/2)/(4T)); the linearized bracket uses the doc field
    # Phi_doc = Phi_code / T, and converting f_dot back to Phi_code_dot
    # multiplies by 1/w_eq = 4T cosh^2(x/2) -- the T's cancel:
    #   Phi_code_dot = 4 cosh^2(x1/2) * (accumulated integral).
    # Minus sign makes this the decay matrix (a_dot = -L a):
    conv = 4 * torch.cosh(x1 / 2) ** 2
    return -out * conv[None, :, None]
