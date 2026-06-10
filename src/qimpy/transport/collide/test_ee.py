"""Tests for the microscopic 2D Fermi-liquid e-e collision operator.

Reference values: GaAs 2DEG of the derivation notes (atomic units):
m* = 0.067, kF = 7.5e-3 (E_F = 4.198e-4), T = 1.33e-5 (4.2 K),
eps_b = 12.9, kappa = 2 m*/eps_b.  Doc targets: K2 = 5.108e3,
K4 = 7.930e3, K6 = 9.811e3, gamma_2/T^2 = 1086.8 (closed form).
The exact thermal-shell (Galerkin) rates at T/E_F = 0.032 are larger:
gamma_2 = 1.299x, gamma_4 = 1.156x closed form (verified independently
against the unreduced collision integral by Monte-Carlo quadratic form).
"""
import numpy as np
import torch
import pytest

from qimpy import rc
from qimpy.mpi import ProcessGrid
from . import _kernels

M_STAR, EPS_B, KF, T0 = 0.067, 12.9, 7.5e-3, 1.33e-5
E_F = 0.5 * KF**2 / M_STAR
KAPPA = 2 * M_STAR / EPS_B


def make_fs(M_theta=8, Nr=1, ee=None, **kwargs):
    from qimpy.transport.material import FermiSurface

    process_grid = ProcessGrid(rc.comm, "rk", (-1, 1))
    return FermiSurface(
        kF=KF, vF=KF / M_STAR, M_theta=M_theta, Nr=Nr, T=T0,
        process_grid=process_grid, ee=ee, **kwargs,
    )


def test_K_table():
    K = _kernels.K_table(8, kF=KF, epsilon_bg=EPS_B, kappa=KAPPA, n_alpha=4096)
    assert torch.allclose(
        K[torch.tensor([2, 4, 6])],
        torch.tensor([5.1084e3, 7.9297e3, 9.8115e3], dtype=K.dtype),
        rtol=1e-3,
    )
    assert K[0] == 0.0 and K[1] == 0.0 and K[3] == 0.0  # parity gate
    K2 = _kernels.K_table(8, kF=KF, epsilon_bg=EPS_B, kappa=KAPPA, n_alpha=8192)
    assert (K2[2:] - K[2:]).abs().max() / K2[2:].abs().max() < 1e-10


def test_closed_form_rate():
    K = _kernels.K_table(2, kF=KF, epsilon_bg=EPS_B, kappa=KAPPA)
    gam = _kernels.gamma_linear(K, m_star=M_STAR, T=T0, E_F=E_F)
    assert abs(gam[2] / T0**2 - 1086.8) < 1.0


def test_form_factor():
    q = torch.tensor([1e-9, 0.03, 0.3, 1.0, 3.0, 8.5, 30.0, 200.0])
    F = _kernels.well_form_factor(q)
    assert abs(F[0] - 1.0) < 1e-6  # F(0) = 1
    assert abs(F[-1] - 3.0 / 200.0) / (3.0 / 200.0) < 0.05  # F -> 3/(qW)
    # against direct double integral:
    z = torch.linspace(0, 1, 801, dtype=torch.float64)[1:-1]
    rho = 2 * torch.sin(np.pi * z) ** 2
    dz = float(z[1] - z[0])
    for x in (0.3, 3.0, 8.5):
        kern = torch.exp(-x * (z[:, None] - z[None, :]).abs())
        Fn = (rho[:, None] * rho[None, :] * kern).sum() * dz**2
        Fc = _kernels.well_form_factor(torch.tensor([x], dtype=torch.float64))[0]
        assert abs(Fc - Fn) / Fn < 1e-3


def test_vertex_identities():
    """Cubic vertex vs analytic mode-form predictions (machine precision)."""
    fs = make_fs(ee=dict(epsilon_bg=EPS_B, rates="closed_form", nonlinear=True))
    dim = fs.angular.dim
    K = fs.ee.K
    A = M_STAR**2 * T0**2 / (8 * np.pi * E_F)
    gam2 = fs.ee.L_coeff[3, 0, 0]
    g_amp = 0.05
    c = 4 * T0 * g_amp  # code units (energy): Phi = 4 T g_occ
    a = torch.zeros(2, dim, dtype=fs.v.dtype, device=rc.device)
    a[:, 3] = c  # cos(2 phi)
    ad = fs.ee.a_dot(a)[0]
    pred2 = -gam2 * c + 0.5 * A * (K[4] - 3 * K[2]) * g_amp**2 * c
    pred6 = 0.5 * A * (K[4] - K[2]) * g_amp**2 * c
    assert abs(ad[3] - pred2) < 1e-12 * abs(pred2) + 1e-30
    assert abs(ad[11] - pred6) < 1e-12 * abs(pred6) + 1e-30
    # number & momentum nulls:
    assert ad[0].abs() < 1e-25 and ad[1].abs() < 1e-25 and ad[2].abs() < 1e-25
    # pure odd deformation: inert (no even partner to couple through):
    a_odd = torch.zeros(1, dim, dtype=fs.v.dtype, device=rc.device)
    a_odd[0, 5] = 0.3 * 4 * T0
    assert fs.ee.a_dot(a_odd).abs().max() < 1e-25
    # mixed odd+even: odd modes pump even output (3+3-2=4 channel):
    a_mix = a_odd.clone()
    a_mix[0, 3] = 0.2 * 4 * T0
    ad_mix = fs.ee.a_dot(a_mix)[0]
    assert ad_mix[7].abs() > 0  # cos(4 phi) output present
    assert ad_mix[5].abs() < 1e-25  # odd modes still do not relax


def test_exact_rates_nr1():
    """Exact thermal-shell rates: corner ratios vs verified reference."""
    fs = make_fs(
        M_theta=4,
        ee=dict(epsilon_bg=EPS_B, rates="exact", nonlinear=False,
                n_xi=24, n_phi=512, n_xi_proj=12),
    )
    K = _kernels.K_table(4, kF=KF, epsilon_bg=EPS_B, kappa=KAPPA)
    gam = _kernels.gamma_linear(K, m_star=M_STAR, T=T0, E_F=E_F)
    r2 = fs.ee.L_coeff[3, 0, 0].item() / gam[2].item()
    r4 = fs.ee.L_coeff[7, 0, 0].item() / gam[4].item()
    assert abs(r2 - 1.299) < 0.03  # reference evaluator: 1.2985
    assert abs(r4 - 1.156) < 0.03  # reference evaluator: 1.155
    assert fs.ee.L_coeff[1, 0, 0] == 0.0  # momentum: exact null projection
    assert fs.ee.L_coeff[0, 0, 0] == 0.0  # number
    # small genuine odd-m relaxation, positive:
    assert 0.0 <= fs.ee.L_coeff[5, 0, 0] < 0.3 * fs.ee.L_coeff[3, 0, 0]


def test_exact_rates_radial_tower():
    """Nr > 1: conservation nulls, PSD, and the hydrodynamic hierarchy."""
    fs = make_fs(
        M_theta=2, Nr=3,
        ee=dict(epsilon_bg=EPS_B, rates="exact", nonlinear=False,
                n_xi=24, n_phi=512, n_xi_proj=12),
    )
    L = fs.ee.L_coeff.to(torch.float64)
    # symmetry + PSD:
    for c in range(L.shape[0]):
        assert torch.allclose(L[c], L[c].T, atol=1e-18)
        assert torch.linalg.eigvalsh(L[c]).min() > -1e-18
    # exact nulls in the code's discrete radial measure:
    Ttm = fs.radial.T_to_modes.to(torch.float64).cpu()
    ones_c = Ttm @ torch.ones(3, dtype=torch.float64)
    x_c = Ttm @ fs.radial.xi.to(torch.float64).cpu()
    t_ratio = T0 / E_F
    k_c = Ttm @ torch.sqrt(1 + t_ratio * fs.radial.xi.to(torch.float64).cpu())
    scale = L[3].diag().max()
    assert (L[0] @ ones_c).abs().max() < 1e-12 * scale  # number
    assert (L[0] @ x_c).abs().max() < 1e-12 * scale  # energy
    assert (L[1] @ k_c).abs().max() < 1e-12 * scale  # momentum
    # hierarchy: energy modes fastest, shear slowest nonzero:
    e_energy = torch.linalg.eigvalsh(L[0])[-1]
    e_heat = sorted(torch.linalg.eigvalsh(L[1]).tolist())[1]
    e_shear = sorted(torch.linalg.eigvalsh(L[3]).tolist())[0]
    assert e_energy > e_heat > e_shear > 0


def test_material_integration():
    """FermiSurface.rho_dot with ee: shapes, decay, density conservation."""
    fs = make_fs(
        M_theta=6, tau_p=np.inf,
        ee=dict(epsilon_bg=EPS_B, rates="closed_form", nonlinear=True),
    )
    Nk = fs.angular.N_theta
    torch.manual_seed(0)
    rho = 1e-4 * torch.randn(5, 7, Nk, dtype=fs.v.dtype, device=rc.device)
    rho_dot = fs.rho_dot(rho, 0.0, 0)
    assert rho_dot.shape == rho.shape
    # density (m=0) exactly conserved at every spatial point:
    n_dot = rho_dot.mean(dim=-1)
    assert n_dot.abs().max() < 1e-22
    # total free-energy-like norm decays (H theorem, linear part dominant):
    a = fs.to_modes(rho)
    a_dot = fs.to_modes(rho_dot)
    assert (a * a_dot).sum() < 0
    # tau_ee conflict is rejected:
    with pytest.raises(Exception):
        make_fs(tau_ee=1.0, ee=dict(epsilon_bg=EPS_B, rates="closed_form"))


def test_L_blocks_pointwise_vs_reference():
    """L_blocks (init path) against the independent reference evaluator."""
    x_chk = torch.tensor([-2.0, 1.0], dtype=torch.float64)
    common = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B, kappa=KAPPA)
    R = _kernels.L_blocks(
        x_nodes=x_chk, psi_coeff=torch.ones(1, 1, dtype=torch.float64),
        m_list=[2], n_xi=16, xi_cut=9.0, n_phi=256, **common,
    )
    w_occ = lambda x: 0.25 / torch.cosh(x / 2) ** 2
    df = lambda x, phi: w_occ(x) * torch.cos(2 * phi) / T0
    fdot = _kernels.exact_collision_reference(
        df, x_chk, torch.zeros(2, dtype=torch.float64), linearize=True,
        n_xi=16, xi_cut=9.0, n_phi=256, chunk=2, **common,
    )
    phidot = fdot * 4 * T0 * torch.cosh(x_chk / 2) ** 2
    assert torch.allclose(R[0, :, 0], -phidot, rtol=1e-10)
