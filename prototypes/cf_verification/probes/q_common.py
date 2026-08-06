"""Shared field / projection for the Q(1,4) triage.

ONE input field, ONE projection, used identically by every evaluator so that the
numbers are directly comparable:

    Phi_in = psi_{n_in}(x) cos(m_in phi),  scaled so peak |delta-f| = AMP_DF
    delta_f = w_eq Phi,   w_eq = 0.25 sech^2(x/2) / T

The amplitude is fixed by the SAME recipe q_sigma_ladder.py used (its 13-point
grid on [-9, 9]) so the production number is bit-comparable with the previously
measured 6.180242e-10; Q is quadratic in the amplitude, so ratios are invariant
anyway.

Angular extraction is an EXACT DFT: a single cos(m_in phi) input generates only
the harmonics {m_in} (L), {0, 2 m_in} (Q), {m_in, 3 m_in} (C), so N_AZ = 16
uniform azimuths on the full circle resolve every one of them with no
quadrature error.  Radial extraction is a Galerkin solve against the model's own
psi_n under the w_eq measure on a uniform grid (trapezoid = spectrally accurate
for this exponentially-decaying analytic integrand).
"""
import numpy as np
import torch

M_STAR, EPS_B, KF, T0 = 0.067, 12.9, 7.5e-3, 1.33e-5
E_F = 0.5 * KF**2 / M_STAR
KAPPA = 2 * M_STAR / EPS_B
M_THETA, NR = 6, 3
N_IN, M_IN = 0, 2
AMP_DF = 0.30

# amplitude convention (frozen: reproduces q_sigma_ladder's amp exactly)
AMP_NX, AMP_XC = 13, 9.0

# output harmonics: L -> 2, Q -> 0 and 4, C -> 2 and 6
M_OUT = [0, 2, 4, 6]
N_AZ = 16                      # >= 2*6+2, exact DFT for a band-limit-6 signal


def w_occ(x):
    return 0.25 / torch.cosh(x / 2) ** 2


def psi_eval(x, psi_coeff, Nr=NR):
    """psi_n(x) in the model's OWN hybrid basis {1, x} + tanh powers."""
    v = torch.tanh(0.5 * x)
    cols = [torch.ones_like(x), x]
    pe, po = 2, 1
    while len(cols) < Nr:
        cols.append(v ** pe); pe += 2
        if len(cols) < Nr:
            cols.append(v ** po); po += 2
    return torch.stack(cols[:Nr], dim=-1) @ psi_coeff


def amplitude(psi_coeff):
    xg = torch.tensor(np.linspace(-AMP_XC, AMP_XC, AMP_NX), dtype=torch.float64)
    pg = psi_eval(xg, psi_coeff)
    return AMP_DF / float((w_occ(xg) * pg[:, N_IN]).abs().max()) * T0


def grids(nx, xi_grid, n_az=None):
    n_az = N_AZ if n_az is None else n_az
    xg = torch.tensor(np.linspace(-xi_grid, xi_grid, nx), dtype=torch.float64)
    ph = torch.tensor(np.linspace(0.0, 2 * np.pi, n_az, endpoint=False),
                      dtype=torch.float64)
    X, P = torch.meshgrid(xg, ph, indexing="ij")
    return xg, ph, X.reshape(-1), P.reshape(-1)


def field(psi_coeff, amp):
    def df(x, phi):
        return (w_occ(x) * psi_eval(x, psi_coeff)[..., N_IN]
                * (amp * torch.cos(M_IN * phi)) / T0)
    return df


def orders_from_stencil(fd):
    """B - F terminates at cubic and its order-0 part vanishes on shell, so the
    signed-amplitude stencil is EXACT:
        o_s = [F(s) - F(-s)]/2 = s L + s^3 C
        L = (8 o1 - o2)/6,  Q = [F(1) + F(-1)]/2,  C = (o2 - 2 o1)/6
    """
    o1 = 0.5 * (fd[1.0] - fd[-1.0])
    o2 = 0.5 * (fd[2.0] - fd[-2.0])
    return {"L": (8.0 * o1 - o2) / 6.0,
            "Q": 0.5 * (fd[1.0] + fd[-1.0]),
            "C": (o2 - 2.0 * o1) / 6.0}


def project(arr, xg, ph, psi_coeff, null0=None, m_out=None):
    """(nx, N_AZ) f_dot field -> (Nr, len(M_OUT)) modal coefficients.

    Exactly the operator's own reduction: divide out w_eq to get the solver
    field Phi_dot = f_dot / w_eq = 4 T cosh^2(x/2) f_dot, DFT in phi, then
    Galerkin-project the energy axis onto psi_n under the w_eq measure.
    """
    nx, n_az = len(xg), len(ph)
    m_out = M_OUT if m_out is None else m_out
    psi_g = psi_eval(xg, psi_coeff)                       # (nx, Nr)
    dx = float(xg[1] - xg[0])
    W = w_occ(xg)
    Ginv = torch.linalg.inv(torch.einsum("xn,xm,x->nm", psi_g, psi_g, W) * dx)
    Wm = torch.stack([(1.0 if m == 0 else 2.0) / n_az * torch.cos(m * ph)
                      for m in m_out], dim=1)             # (n_az, n_m)
    Am = arr.reshape(nx, n_az) @ Wm
    Phi = Am * (4 * T0 * torch.cosh(xg / 2) ** 2)[:, None]
    proj = torch.einsum("nm,xm,x,xk->nk", Ginv, psi_g, W * dx, Phi)
    if null0 is not None:                                 # m = 0 conservation
        Qn, _ = torch.linalg.qr(null0)
        proj[:, 0] = proj[:, 0] - Qn @ (Qn.T @ proj[:, 0])
    return proj
