"""Unit tests for FermiSurface (unified Fermi-surface / Fermi-circle material).

Covers:
- AngularBasis: exact round-trip and orthonormal projection of constants.
- RadialBasis: discrete orthonormality (T_to_modes @ T_from_modes = I) at Nr>1.
- FermiSurface transforms: tensor-product round-trip across Nr and M_theta.
- rho_dot in modes: collision is diagonal in (l, n), cyclotron is omega_c * G
  where G is the block-skew Fourier generator; compare against a hand-built
  reference (no dependency on legacy materials).
- _FermiSurfaceContactor: voltage + drift modal structure round-trips.
- _FermiSurfaceReflector: specular at axis-aligned wall; mass conservation at
  arbitrary normals and specularities (the discrete-quadrature leak is folded
  into D so the net mass flux at the wall is zero to roundoff).
- realizability_floor: defaults to None (no-op limiter for delta-f).
"""
from __future__ import annotations
import numpy as np
import torch
import pytest

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from qimpy.transport.material._fermi_surface import (
    AngularBasis, RadialBasis, _FermiSurfaceReflector,
)


def _pg() -> ProcessGrid:
    return ProcessGrid(rc.comm, "rk", (1, 1))


def _make(M_theta: int, Nr: int = 1, *, T_temp: float = 1.0,
          tau_p: float = np.inf, tau_ee: float = np.inf,
          r_c: float = np.inf, specularity: float = 1.0) -> FermiSurface:
    return FermiSurface(
        kF=1.0, vF=1.5, M_theta=M_theta, Nr=Nr, T=T_temp,
        tau_p=tau_p, tau_ee=tau_ee, r_c=r_c, specularity=specularity,
        process_grid=_pg(),
    )


# ----------------------------------------------------------------------------
# Bases
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("M", [4, 8, 16, 32])
def test_angular_basis_roundtrip(M: int) -> None:
    """T_to_modes @ T_from_modes = I for the Fourier basis at N_q = 2M+1."""
    torch.set_default_dtype(torch.float64)
    ab = AngularBasis(M, n_quad=2 * M + 1)
    eye = ab.T_to_modes @ ab.T_from_modes
    assert torch.allclose(eye, torch.eye(2 * M + 1, dtype=torch.float64),
                          atol=1e-12)


@pytest.mark.parametrize("Nr", [2, 4, 8])
@pytest.mark.parametrize("T_temp", [0.1, 1.0, 5.0])
def test_radial_basis_orthonormal(Nr: int, T_temp: float) -> None:
    """T_to_modes @ T_from_modes = I_Nr under the discrete measure."""
    torch.set_default_dtype(torch.float64)
    rb = RadialBasis(Nr, T_temp=T_temp)
    eye = rb.T_to_modes @ rb.T_from_modes
    assert torch.allclose(eye, torch.eye(Nr, dtype=torch.float64), atol=1e-10)


@pytest.mark.parametrize("Nr", [1, 2, 4])
@pytest.mark.parametrize("M", [4, 16])
def test_fermisurface_transform_roundtrip(Nr: int, M: int) -> None:
    """nodal -> modal -> nodal preserves the state to roundoff."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=M, Nr=Nr)
    N_k = Nr * fs.angular.N_theta
    f = torch.randn(4, N_k, dtype=torch.float64, device=rc.device)
    f2 = fs.from_modes(fs.to_modes(f))
    assert torch.allclose(f, f2, atol=1e-12)


def test_fermisurface_constant_projects_to_n0m0() -> None:
    """A constant nodal field projects to exactly the (n=0, m=0) mode."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1)
    f = torch.full((1, fs.angular.N_theta), 0.7, dtype=torch.float64,
                    device=rc.device)
    a = fs.to_modes(f).reshape(fs.Nr, fs.angular.dim)
    assert a[0, 0].abs() > 1e-6
    others = a.clone(); others[0, 0] = 0.0
    assert others.abs().max() < 1e-12


# ----------------------------------------------------------------------------
# rho_dot: cyclotron + collision in modes, identity in delta-k
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("tau_p, tau_ee, r_c", [
    (np.inf, np.inf, np.inf),           # ballistic, no field
    (0.7,    np.inf, np.inf),           # tau_p only
    (0.7,    1.3,    np.inf),           # tau_p + tau_ee
    (np.inf, np.inf, 2.0),              # cyclotron only
    (0.7,    1.3,    2.0),              # full
])
def test_rho_dot_matches_hand_built_modal(
    tau_p: float, tau_ee: float, r_c: float,
) -> None:
    """At Nr=1 the rho_dot is the diagonal collision plus omega_c * G acting in
    modal space.  Compare FermiSurface.rho_dot against the hand-built modal
    operator applied to the same input.
    """
    torch.set_default_dtype(torch.float64)
    M, vF = 8, 1.5
    fs = _make(M, Nr=1, tau_p=tau_p, tau_ee=tau_ee, r_c=r_c)
    # Hand-built modal rates: m=0 conserved, m=1 -> 1/tau_p, m>=2 -> 1/tau_p + 1/tau_ee
    rates = np.zeros(2 * M + 1)
    rp  = 0.0 if not np.isfinite(tau_p)  else 1.0 / tau_p
    ree = 0.0 if not np.isfinite(tau_ee) else 1.0 / tau_ee
    for m in range(1, M + 1):
        rates[2 * m - 1] = rates[2 * m] = rp if m == 1 else (rp + ree)
    rates_t = torch.as_tensor(rates, dtype=torch.float64, device=rc.device)
    omega_c = (vF / r_c) if np.isfinite(r_c) else 0.0
    a = torch.randn(4, fs.angular.dim, dtype=torch.float64, device=rc.device)
    rho = fs.from_modes(a)
    rd_fs = fs.rho_dot(rho, 0.0, 0)
    ref_modal = -rates_t * a + omega_c * torch.einsum("dc,...c->...d",
                                                       fs.angular.G, a)
    rd_ref = fs.from_modes(ref_modal)
    nrm = max(float(rd_ref.abs().max()), 1e-30)
    assert float((rd_fs - rd_ref).abs().max()) / nrm < 1e-12


# ----------------------------------------------------------------------------
# Contactor: voltage + drift in modal -> delta-k
# ----------------------------------------------------------------------------
def test_contactor_voltage_drift_at_Nr1() -> None:
    """The Nr=1 contactor produces the same modal coefficients as ModalContactor."""
    torch.set_default_dtype(torch.float64)
    M = 8; vF = 1.5
    fs = _make(M, Nr=1)
    # Wall normals at three angles
    phi = torch.tensor([0.3, 1.1, -2.4], dtype=torch.float64)
    n = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    contactor = fs.get_contactor(n, dmu=0.07, vD=0.13)
    rho_dk = contactor(0.0)                               # (3, N_theta)
    a = fs.to_modes(rho_dk)
    # Expected: a_0 = dmu; a_1 = -(vD/vF) cos phi; b_1 = -(vD/vF) sin phi
    assert torch.allclose(a[:, 0], torch.full_like(a[:, 0], 0.07), atol=1e-12)
    assert torch.allclose(a[:, 1], -(0.13 / vF) * torch.cos(phi), atol=1e-12)
    assert torch.allclose(a[:, 2], -(0.13 / vF) * torch.sin(phi), atol=1e-12)


# ----------------------------------------------------------------------------
# Reflector: specular axis-aligned, mass conservation everywhere
# ----------------------------------------------------------------------------
def test_reflector_specular_axis_aligned() -> None:
    """At n=(0,1) specular maps b_m -> -b_m, a_m unchanged."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1)
    n = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    refl = _FermiSurfaceReflector(fs, n, specularity=1.0)
    a_in = torch.randn(1, fs.angular.dim, dtype=torch.float64)
    u_in = fs.from_modes(a_in)
    a_out = fs.to_modes(refl(u_in))
    expected = a_in.clone()
    for m in range(1, fs.M_theta + 1):
        expected[..., 2 * m] = -expected[..., 2 * m]
    assert torch.allclose(a_out, expected, atol=1e-12)


@pytest.mark.parametrize("phi_deg", [0.0, 23.7, 45.0, 90.0, 137.0])
@pytest.mark.parametrize("s", [0.0, 0.3, 0.7, 1.0])
def test_reflector_mass_conservation(phi_deg: float, s: float) -> None:
    """Net normal mass flux at the wall is zero for any specularity, any wall
    angle.  The (D, T) formulation absorbs the discrete-quadrature artifacts
    of the kinked (v.n)_+/_- weights, so mass conservation holds exactly."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _FermiSurfaceReflector(fs, n, specularity=s)
    rng = torch.Generator().manual_seed(0)
    u_in = torch.randn(1, 1, fs.angular.N_theta, dtype=torch.float64,
                       generator=rng)
    u_out = refl(u_in)
    v_dot_n = fs.vF * (
        n[0, 0] * torch.cos(fs.angular.theta) + n[0, 1] * torch.sin(fs.angular.theta)
    )
    out_pos = v_dot_n.clamp(min=0); out_neg = v_dot_n.clamp(max=0)
    F_out = (out_pos * u_in[0, 0]).sum()
    F_in  = (out_neg * u_out[0, 0]).sum()
    net = float((F_out + F_in).abs())
    scale = max(float(F_out.abs()), 1e-30)
    assert net / scale < 1e-12


@pytest.mark.parametrize("phi_deg", [0.0, 23.7, 45.0, 90.0, 137.0])
@pytest.mark.parametrize("s", [0.0, 0.3, 0.7, 1.0])
def test_reflector_tang_momentum_conservation(phi_deg: float, s: float) -> None:
    """At a partially-specular wall the discrete tangential-momentum flux into
    the wall equals the continuum value  (1 - s) * F_out_tang^M  to roundoff
    (the gas keeps the specular fraction, the diffuse fraction is absorbed).
    At s=1 (pure specular) the flux is identically zero.  This is what the
    (D, T) coefficients in the diffuse correction enforce."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _FermiSurfaceReflector(fs, n, specularity=s)
    rng = torch.Generator().manual_seed(0)
    u_in = torch.randn(1, 1, fs.angular.N_theta, dtype=torch.float64,
                       generator=rng)
    u_out = refl(u_in)
    theta = fs.angular.theta
    v_dot_n = fs.vF * (n[0, 0] * torch.cos(theta) + n[0, 1] * torch.sin(theta))
    v_tang  = fs.vF * (-n[0, 1] * torch.cos(theta) + n[0, 0] * torch.sin(theta))
    out_pos = v_dot_n.clamp(min=0); out_neg = v_dot_n.clamp(max=0)
    F_out_tang = (out_pos * v_tang * u_in[0, 0]).sum()
    F_in_tang  = (out_neg * v_tang * u_out[0, 0]).sum()
    F_total = F_out_tang + F_in_tang
    expected = (1.0 - s) * F_out_tang                       # continuum identity
    scale = max(float(F_out_tang.abs()), 1e-30)
    err = float((F_total - expected).abs()) / scale
    assert err < 1e-12


@pytest.mark.parametrize("phi_deg", [0.0, 23.7, 45.0])
@pytest.mark.parametrize("s", [0.0, 1.0])
def test_reflector_energy_conservation(phi_deg: float, s: float) -> None:
    """On the Fermi circle |v| = vF is constant, so the kinetic-energy flux is
    proportional to the mass flux.  Wall conservation of energy is therefore
    inherited from mass conservation at every (phi, s)."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _FermiSurfaceReflector(fs, n, specularity=s)
    rng = torch.Generator().manual_seed(0)
    u_in = torch.randn(1, 1, fs.angular.N_theta, dtype=torch.float64,
                       generator=rng)
    u_out = refl(u_in)
    theta = fs.angular.theta
    v_dot_n = fs.vF * (n[0, 0] * torch.cos(theta) + n[0, 1] * torch.sin(theta))
    out_pos = v_dot_n.clamp(min=0); out_neg = v_dot_n.clamp(max=0)
    energy = 0.5 * fs.vF ** 2                                # constant on Fermi circle
    F_out = (out_pos * energy * u_in[0, 0]).sum()
    F_in  = (out_neg * energy * u_out[0, 0]).sum()
    net = float((F_out + F_in).abs())
    scale = max(float(F_out.abs()), 1e-30)
    assert net / scale < 1e-12


# ----------------------------------------------------------------------------
# Realizability floor: default None -> limiter is a no-op (delta-f safe)
# ----------------------------------------------------------------------------
def test_realizability_floor_default_is_none() -> None:
    fs = _make(M_theta=4, Nr=1)
    assert fs.realizability_floor() is None
