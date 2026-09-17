"""The FFT unitary-involution wall reflector (QIMPY_REFL_FFT=1).

Exactness here is a property of the REPRESENTATION, not of the stencil: at
dk = T/(3 v_F) on an odd grid the reflected drifted Fermi-Dirac matches the
analytic mirror to ~1e-10 at ANY wall angle, the operator is its own inverse to
the same level, all four discrete flux moments balance to roundoff, and the
occupancy stays in [0, 1] to the same accuracy.  See _cartesian_fft.py.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import pytest

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from qimpy.transport.material.fermi_surface import Cartesian
from qimpy.transport.material.fermi_surface._cartesian_fft import _CartesianFFTReflector

KF, VF, T = 7.5e-3, 0.11194, 1.3301e-5
TE_MAX, KD_MAX, XI_MAX = 5.69, 0.15 * KF, 12.0


def _pg() -> ProcessGrid:
    return ProcessGrid("rk", (1, 1))


def _make(cells: float = 3.0) -> FermiSurface:
    torch.set_default_dtype(torch.float64)
    dk = T / (cells * VF)
    k_max, n_k = Cartesian.recommended_grid(
        KF, VF, T, xi_max=XI_MAX, kD_max=KD_MAX, te_fac_max=TE_MAX, dk=dk,
        safety_cells=8)
    n_k += 1 - n_k % 2                              # odd: no Nyquist bin
    k_max = 0.5 * n_k * dk
    return FermiSurface(
        kF=KF, vF=VF, M_theta=8, Nr=2, T=T, xi_max=XI_MAX, tau_p=np.inf,
        specularity=1.0,
        cartesian=dict(annulus_xi=0.0, te_fac_max=TE_MAX, kD_max=KD_MAX,
                       dmu_max=0.0, k_max=k_max, n_k=n_k),
        process_grid=_pg())


def _state(rep, Te_fac: float, kD: torch.Tensor, k: torch.Tensor | None = None):
    k = rep.k if k is None else k
    m, mu = float(rep.m_star), float(rep.mu)
    eps_p = ((k - kD) ** 2).sum(-1) / (2 * m)
    return torch.special.expit(-(eps_p - mu) / (T * Te_fac))


def _moments(rep, n, u_in, u_out):
    """Max relative defect over the four discrete wall flux moments."""
    k = rep.k; v = k / rep.m_star
    vn = (v * n).sum(-1); t = torch.stack([-n[1], n[0]])
    vt = (v * t).sum(-1)
    eps = k.square().sum(-1) / (2 * rep.m_star)
    worst = 0.0
    for mu_ in (torch.ones_like(vn), vt, eps - rep.mu, vn.abs()):
        F_out = (vn.clamp(min=0) * mu_ * u_in).sum()
        F_in = (vn.abs() * (vn < 0) * mu_ * u_out).sum()
        worst = max(worst, float((F_out - F_in).abs() / (vn.abs() * mu_ * u_in).abs().sum()))
    return worst


@pytest.fixture(scope="module")
def fs3():
    return _make(3.0)


@pytest.mark.parametrize("deg", [0.0, 17.0, 45.0, 63.0, 88.3, 137.0])
@pytest.mark.parametrize("Te_fac", [1.0, 5.69])
def test_fft_reflector_exact_any_angle(fs3, deg: float, Te_fac: float) -> None:
    rep = fs3.representation
    k = rep.k; f0 = rep._f0_lab
    kD = KD_MAX * torch.tensor([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
                               dtype=k.dtype, device=k.device)
    u_in = _state(rep, Te_fac, kD) - f0
    phi = np.deg2rad(deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=k.dtype, device=k.device)
    refl = _CartesianFFTReflector(rep, n, 1.0)
    out = refl(u_in[None, None, :])[0, 0]
    kn = (k * n[0]).sum(-1, keepdim=True)
    kstar = k - 2.0 * kn * n[0][None, :]
    exact = _state(rep, Te_fac, kD, kstar) - f0
    scale = float(exact.abs().max())
    err = float((out - exact).abs().max()) / scale
    back = float((refl(out[None, None, :])[0, 0] - u_in).abs().max()) / scale
    mom = _moments(rep, n[0], u_in, out)
    fmin = float((f0 + out).min()); fmax = float((f0 + out).max())
    # cold/warm: resolution-limited (~1e-10) plus the bound repair, which spends
    # up to ~100x a deep-tail violation (6e-10) on the shell to restore the
    # energy moment (measured 4.8e-8 at 88.3 deg); hot: box truncation (~1e-5)
    tol = 1e-7 if Te_fac < 3 else 5e-5
    assert err < tol, f"pointwise {err:.2e}"
    assert back < tol, f"involution {back:.2e}"
    assert mom < 1e-12, f"flux moments {mom:.2e}"
    assert fmin >= -tol and fmax <= 1.0 + tol, f"bounds {fmin:.2e} {fmax-1:.2e}"


def test_fft_reflector_needs_odd_grid() -> None:
    torch.set_default_dtype(torch.float64)
    fs = FermiSurface(kF=KF, vF=VF, M_theta=8, Nr=2, T=T, xi_max=6.0,
                      tau_p=np.inf, specularity=1.0,
                      cartesian=dict(annulus_xi=0.0, te_fac_max=6.0,
                                     kD_max=1.2e-3, dmu_max=1.2e-4,
                                     k_max=0.0132557160008, n_k=112),
                      process_grid=_pg())
    n = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=rc.device)
    with pytest.raises(Exception):
        _CartesianFFTReflector(fs.representation, n, 1.0)


def test_fft_reflector_selected_by_env(fs3, monkeypatch) -> None:
    monkeypatch.setenv("QIMPY_REFL_FFT", "1")
    n = torch.tensor([[0.0, 1.0]], dtype=torch.float64, device=rc.device)
    assert isinstance(fs3.get_reflector(n), _CartesianFFTReflector)


@pytest.mark.parametrize("deg", [17.0, 63.0])
def test_fft_reflector_bound_repair(fs3, deg: float) -> None:
    """A ghost that leaves [0, 1] (here: a hot drifted state, whose rim is
    truncated) is clipped and its four flux moments restored on the headroom:
    f in [0, 1] to roundoff AND moments exact, per edge, no cross-cell coupling."""
    rep = fs3.representation
    k = rep.k; f0 = rep._f0_lab
    kD = KD_MAX * torch.tensor([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
                               dtype=k.dtype, device=k.device)
    u_in = _state(rep, 5.69, kD) - f0
    phi = np.deg2rad(deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=k.dtype, device=k.device)
    refl = _CartesianFFTReflector(rep, n, 1.0)
    refl._bound = False
    raw = refl(u_in[None, None, :])[0, 0]
    refl._bound = True
    out = refl(u_in[None, None, :])[0, 0]
    assert float((f0 + raw).min()) < -1e-9          # the raw ghost does violate
    assert float((f0 + out).min()) >= -1e-15
    assert float((f0 + out).max()) <= 1.0 + 1e-15
    assert _moments(rep, n[0], u_in, out) < 1e-12
    # the repair is no larger than the violation it removes
    assert float((out - raw).abs().max()) < 10 * float((f0 + raw).clamp(max=0).abs().max())
