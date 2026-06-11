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


# ----------------------------------------------------------------------------
# Real <-> complex angular-harmonic transforms (cos/sin basis <-> e^{i m phi})
# ----------------------------------------------------------------------------
def _real_to_complex(M: int) -> torch.Tensor:
    """Map real angular coefficients to complex harmonics.

    ``U[M + m, c]`` such that ``ghat[m] = sum_c U[m, c] a_c`` for the real
    cos/sin coefficient vector ``a = (a_0, a_1, b_1, ..., a_M, b_M)`` and
    complex harmonics ``ghat[m]`` of ``g(phi) = sum_c a_c e_c(phi)``:
    ``ghat[0] = a_0``, ``ghat[m] = (a_m - i b_m)/2``,
    ``ghat[-m] = (a_m + i b_m)/2`` for ``m > 0``.  Shape ``(2M+1, 2M+1)``,
    rows indexed by harmonic ``m = -M .. M`` (offset ``M``).
    """
    nh = 2 * M + 1
    U = torch.zeros(nh, nh, dtype=torch.complex128)
    U[M, 0] = 1.0
    for m in range(1, M + 1):
        U[M + m, 2 * m - 1] = 0.5
        U[M + m, 2 * m] = -0.5j
        U[M - m, 2 * m - 1] = 0.5
        U[M - m, 2 * m] = 0.5j
    return U


def _complex_to_real(M: int) -> torch.Tensor:
    """Map complex output harmonics to real cos/sin coefficients.

    ``R[co, M + mo]`` such that ``out_co = sum_mo R[co, mo] Fhat[mo]`` for a
    real output field (``Fhat[-mo] = conj(Fhat[mo])``): ``a_0 = Fhat[0]``,
    ``a_m = Fhat[m] + Fhat[-m]`` (``= 2 Re Fhat[m]``),
    ``b_m = i (Fhat[m] - Fhat[-m])`` (``= -2 Im Fhat[m]``).  Shape
    ``(2M+1, 2M+1)``, columns indexed by harmonic ``mo = -M .. M``
    (offset ``M``).
    """
    nh = 2 * M + 1
    R = torch.zeros(nh, nh, dtype=torch.complex128)
    R[0, M] = 1.0
    for m in range(1, M + 1):
        R[2 * m - 1, M + m] = 1.0
        R[2 * m - 1, M - m] = 1.0
        R[2 * m, M + m] = 1.0j
        R[2 * m, M - m] = -1.0j
    return R


# ----------------------------------------------------------------------------
# Shared thermal-shell kinematics for the nonlinear vertices
# ----------------------------------------------------------------------------
def _shell_quadrature(T: float, E_F: float, n_xi: int, xi_cut: float, n_phi: int):
    """Energy (x2) and azimuth (beta = phi3 - phi1) Gauss-Legendre grids.

    Identical placement to ``L_blocks`` / ``cubic_vertex`` / ``quadratic_vertex``
    (factored out so they share one definition).  Returns
    ``(x2, w2, beta, wbeta, cosB, sinB)`` in float64.
    """
    t = T / E_F
    dd = dict(dtype=torch.float64)
    lo = -min(xi_cut, 0.8 / t)
    xg, xw = np.polynomial.legendre.leggauss(n_xi)
    x2 = torch.tensor(0.5 * (xi_cut + lo) + 0.5 * (xi_cut - lo) * xg, **dd)
    w2 = torch.tensor(0.5 * (xi_cut - lo) * xw, **dd)
    pg, pw = np.polynomial.legendre.leggauss(n_phi)
    beta = torch.tensor(np.pi * (pg + 1.0), **dd)
    wbeta = torch.tensor(np.pi * pw, **dd)
    return x2, w2, beta, wbeta, torch.cos(beta), torch.sin(beta)


def _shell_geometry(
    *, x1v, X2, k1, k2, x3v, w2, w3v, beta, wbeta, cosB, sinB, T, t, kF,
    m_star, epsilon_bg, kappa, well_width, n_xi, n_phi,
):
    """Resolve the thermal-shell kinematics at one ``(x1, x3, sgn-pair)``.

    Computes the energy-shell quantities shared by both nonlinear vertices
    (``q``, ``|M_q|^2``, the root mask and the two azimuthal roots, the
    Jacobian) once per ``(x1, x3)``.  Returns a callable ``weight_phase(sgn)``
    yielding ``(Wk, phi2, phi4)`` for ``sgn = +-1`` -- minus duplicating the
    geometry between the cubic and quadratic builders and between the two
    leg-triple/pair accumulations.  ``Wk`` carries the full quadrature weight
    ``w2 wbeta w3 T^2 pref`` (energy x2, azimuth, energy x3, thermal phase
    space, kinematic prefactor) just as the original inline builders did.
    """
    X4 = x1v + X2 - x3v  # (n_xi, 1)
    k3 = k_fermi(x3v, kF, t)
    k4 = k_fermi(X4, kF, t)
    band_ok = (1.0 + t * X4) > 0.05
    q_sq = k1**2 + k3**2 - 2 * k1 * k3 * cosB[None, :]
    q = torch.sqrt(torch.clamp(q_sq, min=1e-300))
    Px = (k1 - k3 * cosB)[None, :].expand(n_xi, n_phi)
    Py = (-k3 * sinB)[None, :].expand(n_xi, n_phi)
    phiP = torch.atan2(Py, Px)
    cos_arg = (k4**2 - q_sq - k2**2) / torch.clamp(2 * q * k2, min=1e-300)
    root_ok = band_ok & (cos_arg.abs() <= 1.0 - EDGE_EPS)
    dlt = torch.arccos(torch.clamp(cos_arg, -1.0, 1.0))
    Msq = matrix_element_sq(
        q, epsilon_bg=epsilon_bg, kappa=kappa, well_width=well_width
    ).expand(n_xi, n_phi)
    # Full quadrature weight x kinematic prefactor (per (n_xi, n_phi) grid cell):
    wt = (w2[:, None] * wbeta[None, :]) * (T**2)
    wt_pref = wt * (m_star**3 / (2 * np.pi) ** 3) * w3v

    def weight_phase(sgn: float):
        phi2 = phiP + sgn * dlt
        phi4 = torch.atan2(Py + k2 * torch.sin(phi2), Px + k2 * torch.cos(phi2))
        sin42 = torch.sin(phi4 - phi2).abs()
        jac = 1.0 / torch.clamp(k2 * k4 * sin42, min=1e-14)
        Wk = torch.where(root_ok, Msq * jac, torch.zeros_like(jac))
        return Wk * wt_pref, phi2, phi4

    return weight_phase, X4


def k_fermi(x: torch.Tensor, kF: float, t: float) -> torch.Tensor:
    """Fermi-shell wavevector ``kF sqrt(1 + t x)`` (clamped at the band bottom)."""
    return kF * torch.sqrt(torch.clamp(1.0 + t * x, min=0.0))


# ----------------------------------------------------------------------------
# Exact nonlinear vertices (cubic + quadratic) as precomputed harmonic tensors
# ----------------------------------------------------------------------------
def _cubic_complex(
    *,
    x_nodes: torch.Tensor,
    psi_coeff: torch.Tensor,
    M: int,
    kF: float,
    m_star: float,
    T: float,
    epsilon_bg: float,
    kappa: float,
    well_width: float = 0.0,
    n_xi: int = 24,
    xi_cut: float = 10.0,
    n_phi: int = 512,
):
    """Cubic e-e vertex accumulated as compact complex harmonic tensors.

    Returns ``(Tc_full, Tc_leg1, conv)`` for the cubic part
    ``C3 = d1 d2 (d3+d4) - d3 d4 (d1+d2)`` of ``(B-F)``, BEFORE any radial
    Galerkin/null projection and BEFORE the ``-conv`` solver-field scaling
    (the caller applies those).  Both tensors are indexed by the output-energy
    collocation node ``f``:

    * ``Tc_full[f, la, A, lb, B, lc, C]`` (complex, shape
      ``(Nx1, Nr, 2M+1, Nr, 2M+1, Nr, 2M+1)``): the single ``(2,3,4)-`` leg
      triple whose three legs all carry a nontrivial azimuthal phase, kept at
      full rank-3 angular resolution;
    * ``Tc_leg1[f, la, lp, P, lq, Q]`` (complex, shape
      ``(Nx1, Nr, Nr, 2M+1, Nr, 2M+1)``): the SUM of the three leg-triples that
      contain the output leg 1 -- ``(1,2,3)+``, ``(1,2,4)+``, ``(1,3,4)-`` --
      with leg 1's angular-harmonic axis ELIDED.  Leg 1 sits at ``phi1 = 0`` so
      its phase ``e^{i p . 0} == 1`` is constant in its harmonic index; the
      three triples therefore contribute no leg-1 angular structure and are
      accumulated over the radial index ``la`` alone (a ``(2M+1)x`` smaller and
      cheaper accumulation than the full triple).  At apply time leg 1 acts as a
      convolution: its harmonic is fixed to ``mo - P - Q`` by the output
      selection ``mo = pA + pB + pC``.

    The full complex vertex of the dense path is recovered as ``Tc_full +
    Tc_leg1`` broadcast over leg 1's harmonic axis (verified bit-for-bit).
    """
    E_F = 0.5 * kF**2 / m_star
    t = T / E_F
    dd = dict(dtype=torch.float64)
    Nx1 = len(x_nodes)
    nharm = 2 * M + 1
    Nr = psi_coeff.shape[1]
    ps = torch.arange(-M, M + 1)

    x2, w2, beta, wbeta, cosB, sinB = _shell_quadrature(
        T, E_F, n_xi, xi_cut, n_phi
    )

    def weq(x: torch.Tensor) -> torch.Tensor:
        return 0.25 / torch.cosh(x / 2) ** 2 / T

    def psi_eval(x: torch.Tensor) -> torch.Tensor:
        res = torch.zeros(x.shape + (Nr,), **dd)
        for p in range(psi_coeff.shape[0] - 1, -1, -1):
            res = res * x[..., None] + psi_coeff[p]
        return res

    def leg_factor(x: torch.Tensor) -> torch.Tensor:  # w_eq psi_l: (..., Nr)
        return weq(x)[..., None] * psi_eval(x)

    def phase(dphi: torch.Tensor) -> torch.Tensor:
        return torch.exp(1j * dphi[..., None] * ps)

    nrh = Nr * nharm
    ng = n_xi * n_phi
    # (2,3,4)- triple at full rank-3 angular; leg-1 triples with leg-1 harmonic
    # elided (constant phase at phi1 = 0):
    Tc_full = torch.zeros(Nx1, nrh, nrh, nrh, dtype=torch.complex128)
    Tc_leg1 = torch.zeros(Nx1, Nr, nrh, nrh, dtype=torch.complex128)
    x1t = x_nodes.to(torch.float64)

    for ix1 in range(Nx1):  # phi1 = 0 canonical (isotropy); stream over x1
        x1v = x1t[ix1]
        k1 = k_fermi(x1v, kF, t)
        lf1 = leg_factor(x1v.reshape(1))[0]  # (Nr,) radial factor at leg 1
        X2 = x2[:, None]  # (n_xi, 1)
        k2 = k_fermi(X2, kF, t)
        ph3 = phase(beta)[None, :, :]  # (1, n_phi, nharm), leg-3 phase e^{i p beta}
        for i3 in range(n_xi):  # stream over x3 to bound memory
            x3v = x2[i3]
            w3v = w2[i3]
            weight_phase, X4 = _shell_geometry(
                x1v=x1v, X2=X2, k1=k1, k2=k2, x3v=x3v, w2=w2, w3v=w3v,
                beta=beta, wbeta=wbeta, cosB=cosB, sinB=sinB, T=T, t=t, kF=kF,
                m_star=m_star, epsilon_bg=epsilon_bg, kappa=kappa,
                well_width=well_width, n_xi=n_xi, n_phi=n_phi,
            )
            lf2 = leg_factor(X2.reshape(n_xi))  # (n_xi, Nr)
            lf3 = leg_factor(x3v.reshape(1))[0]  # (Nr,)
            lf4 = leg_factor(X4.reshape(n_xi))  # (n_xi, Nr)
            for sgn in (+1.0, -1.0):
                Wk, phi2, phi4 = weight_phase(sgn)
                ph2 = phase(phi2)  # (n_xi, n_phi, nharm)
                ph4 = phase(phi4)
                # Combined (radial, harmonic) factors on the grid for legs 2-4:
                p2 = (lf2[:, None, :, None] * ph2[..., None, :]).reshape(ng, nrh)
                p3 = (
                    lf3[None, None, :, None]
                    * ph3.expand(n_xi, n_phi, nharm)[..., None, :]
                ).reshape(ng, nrh)
                p4 = (lf4[:, None, :, None] * ph4[..., None, :]).reshape(ng, nrh)
                Wkf = Wk.reshape(ng)
                # (2,3,4)- triple, full rank-3 angular:
                wa = p2 * (-Wkf)[:, None]
                Tc_full[ix1] += torch.einsum("ga,gb,gc->abc", wa, p3, p4)
                # leg-1 triples (1,2,3)+,(1,2,4)+,(1,3,4)-: leg 1 contributes
                # only its radial factor lf1[la] (phase == 1), accumulated over
                # the radial index; the two partner legs keep full (radial,
                # harmonic) structure:
                lf1g = lf1[None, :].expand(ng, Nr)  # (ng, Nr)
                for s3, pp, pq in (
                    (+1.0, p2, p3),
                    (+1.0, p2, p4),
                    (-1.0, p3, p4),
                ):
                    wla = lf1g * (Wkf * s3)[:, None]  # (ng, Nr)
                    Tc_leg1[ix1] += torch.einsum("gl,gp,gq->lpq", wla, pp, pq)

    Tc_full = Tc_full.reshape(Nx1, Nr, nharm, Nr, nharm, Nr, nharm)
    Tc_leg1 = Tc_leg1.reshape(Nx1, Nr, Nr, nharm, Nr, nharm)
    conv = 4 * torch.cosh(x1t / 2) ** 2
    return Tc_full, Tc_leg1, conv


def cubic_vertex(
    *,
    x_nodes: torch.Tensor,
    psi_coeff: torch.Tensor,
    M: int,
    kF: float,
    m_star: float,
    T: float,
    epsilon_bg: float,
    kappa: float,
    well_width: float = 0.0,
    n_xi: int = 24,
    xi_cut: float = 10.0,
    n_phi: int = 512,
    gate_odd: bool = False,
) -> torch.Tensor:
    """Finite-T cubic e-e vertex on the radial-output collocation nodes.

    Builds the cubic part ``C3 = d1 d2 (d3+d4) - d3 d4 (d1+d2)`` of ``(B-F)``
    (the ``f0``-independent, manifestly temperature-finite nonlinearity) over
    the exact thermal-shell kinematics, projected to real output coefficients.
    The angular structure is accumulated as the validated complex harmonic
    tensor (leg-triple signs/phases of the derivation), then folded to the
    real cos/sin basis and binned by the additive output harmonic
    ``mo = ma + mb + mc``.

    SCOPE: the three input legs run over the FULL modal basis -- every
    radial/energy mode ``l`` and angular harmonic ``m`` -- so the vertex is
    complete to cubic order (not restricted to the ``l = 0`` Fermi-surface
    deformation).  Output is projected onto all radial nodes and angular
    harmonics.  ``delta_f`` at each leg is the full modal field

        ``delta_f(x_leg, phi_leg) = w_eq(x_leg)
            sum_l psi_l(x_leg) [ sum_m a_{l,m} e_m(phi_leg) ]``

    with ``w_eq = sech^2(x/2)/(4T)`` and ``psi_coeff[p, l]`` the power-basis
    coefficients ``psi_l(x) = sum_p psi_coeff[p, l] x^p`` (as in ``L_blocks``
    and ``quadratic_vertex``); each leg therefore carries its own radial factor
    ``psi_l`` evaluated at that leg's energy (``x2``, ``x3`` or ``x4`` for the
    integration legs, ``x1`` for the output-energy leg, with
    ``x4 = x1 + x2 - x3`` resolved on the shell).

    Returns the real tensor ``V[i, co, (la, a), (lb, b), (lc, c)]`` reshaped as
    ``(len(x_nodes), 2M+1, Nr, 2M+1, Nr, 2M+1, Nr, 2M+1)`` giving minus the rate
    of change of the solver field ``Phi`` at node ``x_nodes[i]`` and output
    angular mode ``co`` per unit product of input modal coefficients
    ``a_{la,a} a_{lb,b} a_{lc,c}`` (``Phi_dot = -V[i,co] : a a a``, with the
    ``conv = 4 cosh^2(x_i/2)`` and decay-sign convention of ``L_blocks``).  The
    caller Galerkin-projects the node axis onto output radial modes.

    ``gate_odd`` (default ``False``) optionally forces odd output harmonics to
    exactly zero.  This is NOT needed and is OFF by default: the exact kinematic
    integral already enforces the angular parity selection on its own -- a purely
    even input field produces purely even output to machine precision (verified
    against the brute-force reference) -- while a field carrying odd angular
    content genuinely populates odd output channels (``1 + 1 + 1 = 3``), which
    the reference confirms are nonzero.  Gating those to zero would contradict
    the exact operator, so it is disabled; the flag is retained only for
    diagnostics on purely-even fields.

    This dense rank-8 form is retained for the reference-oracle acceptance
    tests; the production operator stores the compact complex tensors of
    ``_cubic_complex`` and applies them by convolution (``(2M+1)x`` less
    storage / work, bit-for-bit identical).
    """
    nharm = 2 * M + 1
    Nr = psi_coeff.shape[1]
    ps = torch.arange(-M, M + 1)
    # Compact complex accumulation (leg-1 reduction), then reconstruct the full
    # complex vertex Tc[i, la,A, lb,B, lc,C] = Tc_full + Tc_leg1 (broadcast over
    # leg-1's harmonic axis A, whose phase is constant at phi1 = 0):
    Tc_full, Tc_leg1, conv = _cubic_complex(
        x_nodes=x_nodes, psi_coeff=psi_coeff, M=M, kF=kF, m_star=m_star, T=T,
        epsilon_bg=epsilon_bg, kappa=kappa, well_width=well_width, n_xi=n_xi,
        xi_cut=xi_cut, n_phi=n_phi,
    )
    Tc = Tc_full + Tc_leg1[:, :, None, :, :, :, :]

    # Fold complex harmonics -> real basis on each leg, bin by output harmonic
    # mo = ma + mb + mc, and (optionally) gate odd output harmonics to zero.
    U = _real_to_complex(M)
    R = _complex_to_real(M)
    mo_idx = ps[:, None, None] + ps[None, :, None] + ps[None, None, :]
    Rfull = torch.zeros(nharm, nharm, nharm, nharm, dtype=torch.complex128)
    for ia in range(nharm):
        for ib in range(nharm):
            for ic in range(nharm):
                mo = int(mo_idx[ia, ib, ic].item())
                keep = (-M <= mo <= M) and (mo % 2 == 0 or not gate_odd)
                if keep:
                    Rfull[:, ia, ib, ic] = R[:, M + mo]
    # Fold to the real basis: contract the three complex input harmonic axes
    # with U (real->complex) and the output binning Rfull (complex->real).  The
    # leg radial axes (la, lb, lc) pass through untouched.  Result indexed
    # (ix1, co, la, a, lb, b, lc, c):
    V = torch.einsum(
        "ZXAYBWC,oABC,Ai,Bj,Ck->ZoXiYjWk", Tc, Rfull, U, U, U
    ).real
    return -V * conv[:, None, None, None, None, None, None, None]


def _quadratic_complex(
    *,
    x_nodes: torch.Tensor,
    psi_coeff: torch.Tensor,
    M: int,
    kF: float,
    m_star: float,
    T: float,
    epsilon_bg: float,
    kappa: float,
    well_width: float = 0.0,
    n_xi: int = 24,
    xi_cut: float = 10.0,
    n_phi: int = 512,
):
    """Quadratic (thermoelectric) e-e vertex as compact complex harmonic tensors.

    Returns ``(Qc_full, Qc_leg1, conv)`` for ``Q2`` (see ``quadratic_vertex``),
    BEFORE radial Galerkin/null projection and BEFORE the ``-conv`` scaling.
    Both are indexed by the output-energy node ``f``:

    * ``Qc_full[f, la, A, lb, B]`` (complex, shape
      ``(Nx1, Nr, 2M+1, Nr, 2M+1)``): the three pair-terms NOT containing the
      output leg 1 -- ``d2 d3``, ``d2 d4``, ``d3 d4`` -- at full rank-2 angular;
    * ``Qc_leg1[f, la, lb, B]`` (complex, shape ``(Nx1, Nr, Nr, 2M+1)``): the
      SUM of the three pair-terms that DO contain leg 1 -- ``d1 d2``, ``d1 d3``,
      ``d1 d4`` -- with leg 1's angular-harmonic axis elided (its phase is
      constant at ``phi1 = 0``); ``la`` is leg 1's radial index and ``(lb, B)``
      the partner leg's (radial, harmonic).  At apply time leg 1 is a
      convolution: its harmonic is fixed to ``mo - B`` by ``mo = ma + mb``.

    The dense complex ``Qc[f, la, lb, ma, mb]`` of the reference path is
    recovered as ``Qc_full + Qc_leg1`` broadcast over leg 1's harmonic ``ma``.
    """
    E_F = 0.5 * kF**2 / m_star
    t = T / E_F
    dd = dict(dtype=torch.float64)
    Nx1 = len(x_nodes)
    nharm = 2 * M + 1
    Nr = psi_coeff.shape[1]
    ps = torch.arange(-M, M + 1)

    x2, w2, beta, wbeta, cosB, sinB = _shell_quadrature(
        T, E_F, n_xi, xi_cut, n_phi
    )

    def weq(x: torch.Tensor) -> torch.Tensor:
        return 0.25 / torch.cosh(x / 2) ** 2 / T

    def f0(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-x)

    def psi_eval(x: torch.Tensor) -> torch.Tensor:  # (...,) -> (..., Nr)
        res = torch.zeros(x.shape + (Nr,), **dd)
        for p in range(psi_coeff.shape[0] - 1, -1, -1):
            res = res * x[..., None] + psi_coeff[p]
        return res

    def phase(dphi: torch.Tensor) -> torch.Tensor:
        return torch.exp(1j * dphi[..., None] * ps)

    # Qc_full[f, la, A, lb, B] (non-leg-1 pairs); Qc_leg1[f, la, lb, B] (leg-1
    # pairs with leg-1 harmonic elided):
    Qc_full = torch.zeros(Nx1, Nr, nharm, Nr, nharm, dtype=torch.complex128)
    Qc_leg1 = torch.zeros(Nx1, Nr, Nr, nharm, dtype=torch.complex128)
    x1t = x_nodes.to(torch.float64)

    for ix1 in range(Nx1):
        x1v = x1t[ix1]
        k1 = k_fermi(x1v, kF, t)
        f1 = f0(x1v)
        we1 = weq(x1v)
        psi1 = psi_eval(x1v.reshape(1))[0]  # (Nr,)
        A1r = we1 * psi1  # (Nr,) leg-1 radial amplitude (phase == 1)
        X2 = x2[:, None]
        k2 = k_fermi(X2, kF, t)
        ph3 = phase(beta)  # (n_phi, nharm) leg-3 phase
        for i3 in range(n_xi):
            x3v = x2[i3]
            w3v = w2[i3]
            weight_phase, X4 = _shell_geometry(
                x1v=x1v, X2=X2, k1=k1, k2=k2, x3v=x3v, w2=w2, w3v=w3v,
                beta=beta, wbeta=wbeta, cosB=cosB, sinB=sinB, T=T, t=t, kF=kF,
                m_star=m_star, epsilon_bg=epsilon_bg, kappa=kappa,
                well_width=well_width, n_xi=n_xi, n_phi=n_phi,
            )
            f2 = f0(X2).expand(n_xi, n_phi)
            f3 = f0(x3v)
            f4 = f0(X4).expand(n_xi, n_phi)
            we2 = weq(X2)
            we3 = weq(x3v)
            we4 = weq(X4)
            psi2 = psi_eval(X2.reshape(n_xi))  # (n_xi, Nr)
            psi3 = psi_eval(x3v.reshape(1))[0]  # (Nr,)
            psi4 = psi_eval(X4.reshape(n_xi))  # (n_xi, Nr)
            for sgn in (+1.0, -1.0):
                Wk, phi2, phi4 = weight_phase(sgn)  # (n_xi, n_phi)
                gshape = (n_xi, n_phi)
                # partner-leg (radial, harmonic) amplitudes on the grid:
                A2 = (we2 * psi2)[:, None, :].expand(*gshape, Nr)
                A3 = (we3 * psi3)[None, None, :].expand(*gshape, Nr)
                A4 = (we4 * psi4)[:, None, :].expand(*gshape, Nr)
                ph2 = phase(phi2)
                ph3g = ph3[None, :, :].expand(*gshape, nharm)
                ph4 = phase(phi4)
                ng = n_xi * n_phi
                # leg-1 pairs: leg 1 contributes radial A1r only; accumulate the
                # partner leg's (radial, harmonic).  cf is the f0 coefficient.
                for AB, phB, cf in (
                    (A2, ph2, f3 + f4 - 1.0),
                    (A3, ph3g, f2 - f4),
                    (A4, ph4, f2 - f3),
                ):
                    wgt = (Wk * cf).reshape(ng)
                    # leg-1 radial weighted by the term weight (cast real legs
                    # to complex to match the complex phase factor):
                    LA = (A1r[None, :].expand(ng, Nr)
                          * wgt[:, None]).to(torch.complex128)  # (ng, Nr)
                    LB = AB.reshape(ng, Nr).to(torch.complex128)
                    PB = phB.reshape(ng, nharm)
                    Qc_leg1[ix1] += torch.einsum("gl,gp,gb->lpb", LA, LB, PB)
                # non-leg-1 pairs (2,3),(2,4),(3,4): full rank-2 angular.
                for AA, phA, AB, phB, cf in (
                    (A2, ph2, A3, ph3g, f1 - f4),
                    (A2, ph2, A4, ph4, f1 - f3),
                    (A3, ph3g, A4, ph4, -(f1 + f2 - 1.0)),
                ):
                    wgt = (Wk * cf).reshape(ng)
                    LA = AA.reshape(ng, Nr) * wgt[:, None]
                    LB = AB.reshape(ng, Nr)
                    PA = phA.reshape(ng, nharm)
                    PB = phB.reshape(ng, nharm)
                    Qc_full[ix1] += torch.einsum(
                        "gx,ga,gy,gb->xayb", LA, PA, LB, PB
                    )

    conv = 4 * torch.cosh(x1t / 2) ** 2
    return Qc_full, Qc_leg1, conv


def quadratic_vertex(
    *,
    x_nodes: torch.Tensor,
    psi_coeff: torch.Tensor,
    M: int,
    kF: float,
    m_star: float,
    T: float,
    epsilon_bg: float,
    kappa: float,
    well_width: float = 0.0,
    n_xi: int = 24,
    xi_cut: float = 10.0,
    n_phi: int = 512,
) -> torch.Tensor:
    """Finite-T particle-hole-odd quadratic e-e vertex (thermoelectric).

    Builds the quadratic part of ``(B-F)``,

        ``Q2 = d1 d2 (f0_3+f0_4-1) + d1 d3 (f0_2-f0_4) + d1 d4 (f0_2-f0_3)
             + d2 d3 (f0_1-f0_4) + d2 d4 (f0_1-f0_3) - d3 d4 (f0_1+f0_2-1)``,

    over the exact thermal-shell kinematics.  Its coefficients are
    particle-hole-odd (``proportional to f0 - 1/2``), so it VANISHES on the
    Fermi surface and is ``O(T/E_F)``; it genuinely couples opposite energy
    parities, hence inputs and outputs run over ALL radial modes.

    ``delta_f`` at each leg is ``w_eq(x_leg) sum_{l,c} a_{l,c} psi_l(x_leg)
    e_c(phi_leg)`` with the full modal field (radial ``l``, angular ``c``);
    ``psi_coeff[p, l]`` are the power-basis coefficients ``psi_l(x) = sum_p
    psi_coeff[p, l] x^p`` (as in ``L_blocks``).

    Returns the real rank-5 tensor ``V[i, co, (la, a), (lb, b)]`` reshaped as
    ``(len(x_nodes), 2M+1, Nr, 2M+1, Nr, 2M+1)`` -- minus the rate of change of
    ``Phi`` at node ``x_nodes[i]`` and output angular mode ``co`` per unit
    product of input modal coefficients ``a_{la,a} a_{lb,b}`` (with the
    ``conv``/decay-sign convention of ``L_blocks``).  The caller Galerkin-
    projects the node axis onto output radial modes.

    This dense form is retained for the reference-oracle acceptance tests; the
    production operator stores the compact complex tensors of
    ``_quadratic_complex`` and applies them by convolution.
    """
    nharm = 2 * M + 1
    ps = torch.arange(-M, M + 1)
    # Compact complex accumulation (leg-1 reduction), then reconstruct the dense
    # complex Qc[i, la, lb, ma, mb] = Qc_full + Qc_leg1 (broadcast over leg-1's
    # harmonic ma, whose phase is constant at phi1 = 0):
    Qc_full, Qc_leg1, conv = _quadratic_complex(
        x_nodes=x_nodes, psi_coeff=psi_coeff, M=M, kF=kF, m_star=m_star, T=T,
        epsilon_bg=epsilon_bg, kappa=kappa, well_width=well_width, n_xi=n_xi,
        xi_cut=xi_cut, n_phi=n_phi,
    )
    # Qc_full[f, la, A, lb, B] -> Qc[f, la, lb, A, B]; add leg-1 (ma broadcast):
    Qc = Qc_full.permute(0, 1, 3, 2, 4).contiguous()
    Qc = Qc + Qc_leg1[:, :, :, None, :]  # insert ma axis (size 1 -> broadcast)

    # Fold complex harmonics -> real basis, bin by output harmonic mo=ma+mb.
    U = _real_to_complex(M)
    R = _complex_to_real(M)
    mo_idx = ps[:, None] + ps[None, :]
    Rfull = torch.zeros(nharm, nharm, nharm, dtype=torch.complex128)
    for ia in range(nharm):
        for ib in range(nharm):
            mo = int(mo_idx[ia, ib].item())
            if -M <= mo <= M:
                Rfull[:, ia, ib] = R[:, M + mo]
    tmp = Rfull[None, None, None] * Qc[:, :, :, None]  # (ix1,la,lb,co,ma,mb)
    V = torch.einsum("zxyoAB,Ai,Bj->zxyoij", tmp, U, U).real
    # reorder to (ix1, co, la, a, lb, b):
    V = V.permute(0, 3, 1, 4, 2, 5).contiguous()
    return -V * conv[:, None, None, None, None, None]
