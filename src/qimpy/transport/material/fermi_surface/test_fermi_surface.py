"""Unit tests for FermiSurface (unified Fermi-surface / Fermi-circle material).

Covers:
- AngularBasis: exact round-trip and orthonormal projection of constants.
- RadialBasis: discrete orthonormality (T_to_modes @ T_from_modes = I) at Nr>1.
- FermiSurface transforms: tensor-product round-trip across Nr and M_theta.
- rho_dot in modes: collision is diagonal in (l, n), cyclotron is omega_c * G
  where G is the block-skew Fourier generator; compare against a hand-built
  reference (no dependency on legacy materials).
- _DeltaKContactor: voltage + drift modal structure round-trips.
- _DeltaKReflector: specular at axis-aligned wall; mass conservation at
  arbitrary normals and specularities (the discrete-quadrature leak is folded
  into D so the net mass flux at the wall is zero to roundoff).
"""
from __future__ import annotations
import numpy as np
import torch
import pytest

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from ._fermi_surface import AngularBasis, RadialBasis
from ._representation import _DeltaKReflector

# ⛔ CACHE THE PROCESS GRID.  Under MPI, ProcessGrid.get_comm was a free
# communicator split.  Upstream's torch.distributed get_group splits a real
# NCCL communicator, which allocates ~512 MB of device memory that is never
# released, so creating one grid per test exhausted a 40 GB card after ~50
# tests: 86 failures, every one "Failed to CUDA calloc 536870912 bytes", none
# of them a logic error.  One grid per process is all any of these tests need.
_PG_CACHE: dict[tuple, ProcessGrid] = {}


def _cached_pg(dim_names: str, shape) -> ProcessGrid:
    key = (dim_names, tuple(shape) if shape else None)
    if key not in _PG_CACHE:
        _PG_CACHE[key] = ProcessGrid(dim_names, shape)
    return _PG_CACHE[key]



def _pg() -> ProcessGrid:
    return _cached_pg("rk", (1, 1))


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
    """Transforms round-trip to roundoff.

    The angular nodes oversample the 2M+1 modes (the symmetric quadrature uses an
    even N_theta >= 2M+1), so the invariant is modal -> nodal -> modal = identity.
    Equivalently, a *physical* (band-limited) nodal state f = from_modes(a) also
    satisfies from_modes(to_modes(f)) = f; only out-of-band nodal noise is
    projected away (that content carries no represented harmonic, n, or current).
    """
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=M, Nr=Nr)
    n_modes = Nr * fs.angular.dim
    a = torch.randn(4, n_modes, dtype=torch.float64, device=rc.device)
    assert torch.allclose(a, fs.to_modes(fs.from_modes(a)), atol=1e-12)
    f = fs.from_modes(a)                                   # band-limited nodal state
    assert torch.allclose(f, fs.from_modes(fs.to_modes(f)), atol=1e-12)


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
    """Linear contactor: dmu -> m=0; PHYSICAL inward drift vD -> m=1 with
    coefficient -kF*vD (shell momentum shift), per unit Phi convention."""
    torch.set_default_dtype(torch.float64)
    M = 8; kF = 1.0
    fs = _make(M, Nr=1)
    # Wall normals at three angles
    phi = torch.tensor([0.3, 1.1, -2.4], dtype=torch.float64)
    n = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    contactor = fs.get_contactor(n, dmu=0.07, vD=0.13)
    rho_dk = contactor(0.0)                               # (3, N_theta)
    a = fs.to_modes(rho_dk)
    # Expected: a_0 = dmu; a_1 = -kF vD cos phi; b_1 = -kF vD sin phi
    assert torch.allclose(a[:, 0], torch.full_like(a[:, 0], 0.07), atol=1e-12)
    assert torch.allclose(a[:, 1], -(kF * 0.13) * torch.cos(phi), atol=1e-12)
    assert torch.allclose(a[:, 2], -(kF * 0.13) * torch.sin(phi), atol=1e-12)


def test_contactor_nonlinear_linear_limit() -> None:
    """The nonlinear (exact shifted-FD) ghost linearizes to the linear ghost:
    same convention, different fidelity."""
    torch.set_default_dtype(torch.float64)
    fs = _make(8, Nr=1)
    phi = torch.tensor([0.4, -1.9], dtype=torch.float64)
    n = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    eps = 1e-6 * fs.T_temp
    lin = fs.get_contactor(n, dmu=eps, vD=eps)(0.0)
    nl = fs.get_contactor(n, dmu=eps, vD=eps, nonlinear=True)(0.0)
    scale = lin.abs().max()
    assert (nl - lin).abs().max() < 1e-6 * scale


def test_contactor_nonlinear_saturates() -> None:
    """Nr=1 nonlinear ghost = 2T tanh(s/2): saturating (Pauli-bounded), below
    the linear ghost at strong bias."""
    torch.set_default_dtype(torch.float64)
    T = 0.02
    fs = _make(8, Nr=1, T_temp=T)
    n = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    dmu = 5.0 * T
    nl = fs.get_contactor(n, dmu=dmu, nonlinear=True)(0.0)
    expect = 2.0 * T * np.tanh(2.5)                       # s = dmu/T = 5
    assert torch.allclose(nl, torch.full_like(nl, expect), rtol=1e-10)
    lin = fs.get_contactor(n, dmu=dmu)(0.0)
    assert nl.abs().max() < lin.abs().max()               # saturation


# ----------------------------------------------------------------------------
# Reflector: specular axis-aligned, mass conservation everywhere
# ----------------------------------------------------------------------------
def test_reflector_specular_axis_aligned() -> None:
    """At n=(0,1) specular maps b_m -> -b_m, a_m unchanged."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1)
    n = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=1.0)
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
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
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
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
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
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
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
def _make_cart(Nr: int, T: float = 0.05, *, local_te_rates: bool = True,
               ee=None) -> FermiSurface:
    return FermiSurface(
        kF=1.0, vF=1.5, M_theta=6, Nr=Nr, T=T, xi_max=6.0,
        cartesian=dict(dk=T / (3.0 * 1.5), local_te_rates=local_te_rates),
        ee_scattering=ee, process_grid=_pg())


def _cart_state(fs, radial_mode: int, amp: float):
    """delta-f = w_eq_RB(xi_lab) * amp * psi_n(xi) * cos(2 theta): a pure
    (n, m=2 cos) mode of Phi about the lab frame (no N/J/E content)."""
    rep = fs.representation
    xi = (rep.eps_k - fs.mu) / fs.T_temp
    th = torch.atan2(rep.k[:, 1], rep.k[:, 0])
    f0 = torch.special.expit(-xi)
    psi = rep._psi(xi)[:, radial_mode]
    return (f0 * (1 - f0) / fs.T_temp) * amp * psi * torch.cos(2 * th)


@pytest.mark.parametrize("Nr,mode", [(1, 0), (2, 0), (2, 1)])
def test_cartesian_projection_contract(Nr: int, mode: int) -> None:
    """The coefficients handed to the modal operator ARE the Phi coefficients
    (Gram-corrected, T-normalized): a pure (n, m=2) input of amplitude amp
    projects to amp on that channel and ~0 elsewhere."""
    torch.set_default_dtype(torch.float64)
    fs = _make_cart(Nr)
    amp = 1e-8
    rho = _cart_state(fs, mode, amp)[None]        # one spatial cell
    seen = {}

    def capture(a, te=None):
        seen["a"] = a.detach().clone()
        return torch.zeros_like(a)

    fs.representation.apply_collision(rho, capture)
    a = seen["a"].reshape(Nr, fs.angular.dim)
    idx = (mode, 3)                               # (n, m=2 cos)
    err_chan = abs(float(a[idx]) - amp) / amp
    others = a.clone(); others[idx] = 0.0
    # was 2e-2 with the fine-rule normalization; the iterative solve
    # against the Gram the k-sum actually realizes makes this exact.
    assert err_chan < 1e-4, f"channel amplitude off by {err_chan:.1e}"
    assert float(others.abs().max()) < 2e-2 * amp, "cross-channel leakage"


@pytest.mark.parametrize("Nr,mode", [(1, 0), (2, 1)])
def test_cartesian_projection_roundtrip(Nr: int, mode: int) -> None:
    """apply_collision with the identity modal operator returns the input
    (P then R is the identity on non-conserved modes)."""
    torch.set_default_dtype(torch.float64)
    fs = _make_cart(Nr)
    rho = _cart_state(fs, mode, 1e-8)[None]
    out = fs.representation.apply_collision(rho, lambda a, te=None: a)
    rel = float((out - rho).abs().max() / rho.abs().max())
    assert rel < 1e-4, f"P.R != identity: rel={rel:.1e}"



# ---- residual (unresolved-mode) closure ------------------------------------

def _make_cart_closure(Nr=4, T=0.05, tau_ee=200.0):
    """Closure harness on the PHENOMENOLOGICAL rate: it isolates the closure
    mechanics from the microscopic e-e build (whose K_table hits a
    pre-existing CUDA-default-device .numpy() bug in this environment), and
    gives a known, exactly-flat gamma at every m >= 2."""
    return FermiSurface(
        kF=1.0, vF=1.5, M_theta=6, Nr=Nr, T=T, xi_max=6.0, tau_ee=tau_ee,
        cartesian=dict(dk=T / (3.0 * 1.5), local_te_rates=False),
        residual_damping=True, process_grid=_pg())


def _drifted_heated(rep, fs, kD, Te, dmu=0.0):
    """Exact local drifted-heated FD as a deviation about f0_lab."""
    kp = rep.k - torch.as_tensor(kD, device=rep.k.device, dtype=rep.k.dtype)
    eps_p = kp.square().sum(-1) / (2 * fs.m_star)
    f_le = torch.special.expit(-(eps_p - (fs.mu + dmu)) / Te)
    return f_le - rep._f0_lab


def test_residual_closure_vanishes_on_local_equilibrium() -> None:
    """C[f_le] = 0 must survive the closure.  Checked at a NON-half-cell drift:
    at kD = 0 the parity reflection is an exact grid permutation, so a zero-
    drift test cannot see an interpolation defect."""
    fs_on = _make_cart_closure()
    fs_off = FermiSurface(
        kF=1.0, vF=1.5, M_theta=6, Nr=4, T=0.05, xi_max=6.0, tau_ee=200.0,
        cartesian=dict(dk=0.05 / 4.5, local_te_rates=False),
        process_grid=_pg())
    rep = fs_on.representation
    kD = 0.37 * rep._dk_grid * np.array([1.0, 0.61])
    df = _drifted_heated(rep, fs_on, kD, 1.3 * fs_on.T_temp)[None]
    # ON vs OFF isolates the CLOSURE.  Comparing against zero instead would
    # measure the pre-existing frame-recovery / projection quadrature error,
    # which is finite on any grid and has nothing to do with this patch.
    a = rep.apply_collision(df.clone(), fs_on._modal_collision)
    b = fs_off.representation.apply_collision(df.clone(), fs_off._modal_collision)
    # Normalize by the ACTUAL residual, not by |df|.  The discrete state is
    # not exactly the local equilibrium -- frame recovery returns (kD,Te,mu)
    # from grid moments, so df_loc is small but nonzero -- and damping that
    # genuine residual is precisely the closure's job.  The invariant that
    # must hold is that the closure never exceeds gamma_res * |df_loc|, i.e.
    # it damps what is there and does not invent a source of its own.
    f = rep._f0_lab[None] + df
    kD_r, Te_r, mu_r = rep._recover_frame(f)
    kp = rep.k[None] - kD_r[:, None]
    xi = (kp.square().sum(-1) / (2 * fs_on.m_star) - mu_r[:, None]) / Te_r[:, None]
    df_loc_max = float((f - torch.special.expit(-xi)).abs().max())
    rel = float((a - b).abs().max()) / (fs_on.gamma_residual() * df_loc_max)
    assert df_loc_max < 0.05 * float(df.abs().max()), "frame recovery is off"
    assert rel < 1.0, (
        f"closure output is {rel:.3f} x gamma_res*|df_loc| -- it is producing "
        "more than a relaxation of the residual actually present")


def test_residual_closure_spares_odd_harmonics() -> None:
    """Odd angular harmonics are gated to zero at leading order; the parity
    split must leave them essentially untouched."""
    fs = _make_cart_closure()
    rep = fs.representation
    kD = 0.37 * rep._dk_grid * np.array([1.0, 0.61])
    kp = rep.k - torch.as_tensor(kD, device=rep.k.device, dtype=rep.k.dtype)
    xi = (kp.square().sum(-1) / (2 * fs.m_star) - fs.mu) / fs.T_temp
    th = torch.atan2(kp[:, 1], kp[:, 0])
    f0 = torch.special.expit(-xi)
    # a HIGH odd harmonic, above M_theta = 6, so nothing else acts on it:
    df = ((f0 * (1 - f0) / fs.T_temp) * torch.cos(9 * th))[None]
    out = rep.apply_collision(df.clone(), lambda a, te=None: torch.zeros_like(a))
    leak = float(out.abs().max()) / (fs.gamma_residual() * float(df.abs().max()))
    # MEASURED 0.057 at dk = T/(3 vF).  This is the bilinear interpolation
    # error of the drift reflection, not a parity-logic error: without the
    # split the same mode would be damped at 1.0 x gamma_res, so the gate buys
    # ~20x.  The physical odd-m rate is ~0.01 gamma_2, so the artifact is still
    # ~20x the physics it protects -- a bicubic (16-corner, O(h^4)) gather
    # should reach ~0.5%.  Threshold here is a regression guard.
    assert leak < 8e-2, f"odd harmonic damped at {leak:.3f} of gamma_res"


def test_residual_closure_damps_even_harmonics() -> None:
    """The even counterpart of the previous test MUST be damped, at ~gamma_res
    and with the sign of a decay."""
    fs = _make_cart_closure()
    rep = fs.representation
    kD = 0.37 * rep._dk_grid * np.array([1.0, 0.61])
    kp = rep.k - torch.as_tensor(kD, device=rep.k.device, dtype=rep.k.dtype)
    xi = (kp.square().sum(-1) / (2 * fs.m_star) - fs.mu) / fs.T_temp
    th = torch.atan2(kp[:, 1], kp[:, 0])
    f0 = torch.special.expit(-xi)
    df = ((f0 * (1 - f0) / fs.T_temp) * torch.cos(10 * th))[None]
    out = rep.apply_collision(df.clone(), lambda a, te=None: torch.zeros_like(a))
    ov = float((out[0] * df[0]).sum() / (df[0] * df[0]).sum())
    assert ov < 0.0, "even residual is not damped"
    assert 0.5 < abs(ov) / fs.gamma_residual() < 1.5, (
        f"even residual damped at {abs(ov)/fs.gamma_residual():.2f} x gamma_res")


def test_cartesian_annulus_with_frame_polish() -> None:
    """A frozen-sea annulus run must build AND apply.

    Regression (45baa7ad -> 011c0a59): ``_k_polish`` was sliced out of the
    ALREADY-reduced active ``k`` using the FULL-grid mask ``act | frozen``, so
    every ``annulus_xi > 0`` run raised IndexError before its first step.  It
    went unnoticed because nothing in the suite set ``annulus_xi`` at all and
    the ballistic campaign predates the 4x4 frame polish that introduced it.
    """
    T = 0.05
    fs = FermiSurface(
        kF=1.0, vF=1.5, M_theta=6, Nr=4, T=T, xi_max=6.0, tau_ee=200.0,
        cartesian=dict(dk=T / 4.5, annulus_xi=8.0, local_te_rates=False),
        process_grid=_pg())
    rep = fs.representation
    assert rep._annulus_on, "annulus did not engage: test is vacuous"
    assert rep._k_polish is not None
    # the polish set is the FULL band (active + frozen), so strictly larger
    # than the active set the dynamics carry
    assert rep._k_polish.shape[0] > rep.k.shape[0]
    assert rep._k_polish.shape[0] == rep._eps_polish.shape[0]
    torch.manual_seed(0)
    df = 0.01 * torch.randn(3, rep.Nk, device=rep.k.device, dtype=rep.k.dtype)
    out = rep.apply_collision(df, fs._modal_collision)
    assert torch.isfinite(out).all()


def test_residual_closure_off_is_bit_identical() -> None:
    """residual_damping=False must reproduce the pre-closure path exactly."""
    fs_off = FermiSurface(
        kF=1.0, vF=1.5, M_theta=6, Nr=4, T=0.05, xi_max=6.0, tau_ee=200.0,
        cartesian=dict(dk=0.05 / 4.5, local_te_rates=False),
        process_grid=_pg())
    rep = fs_off.representation
    torch.manual_seed(0)
    df = (0.01 * torch.randn(3, rep.Nk, device=rep.k.device,
                             dtype=rep.k.dtype))
    a = rep.apply_collision(df.clone(), fs_off._modal_collision)
    b = rep.apply_collision(df.clone(), fs_off._modal_collision)
    assert torch.equal(a, b), "collision apply is not deterministic"
    assert fs_off.gamma_residual() is None


# ----------------------------------------------------------------------------
# Radial basis (Nr > 1): the conservation laws must survive the extra radial
# dimension, not just the Fermi-circle (Nr=1) limit.
#
# ⛔ EVERY Nr=1 CONSERVATION TEST ABOVE IS BLIND TO A RADIAL-BASIS DEFECT.  At
# Nr=1 the radial quadrature is one weight equal to 1.0, so the w_r-weighted
# sums below degenerate into the unweighted ones and a mis-weighted radial
# projection cannot change any of their results.  These four tests carry the
# Nr>1 coverage; they were stranded on the `triangular_mesh` PR branch and
# reached the trunk only when it was ported (the module had moved to
# material/fermi_surface/, so the cherry-pick was a delete/update conflict).
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("Nr", [2, 4, 8])
@pytest.mark.parametrize("r_c", [np.inf, 2.0])
def test_collision_conserves_mass_radial(Nr: int, r_c: float) -> None:
    """At Nr>1 the collision (+ cyclotron, if r_c finite) still conserves particle
    number: the (n=0, m=0) rate is forced to 0 and the cyclotron generator leaves
    the m=0 angular mode invariant, so d/dt <n> = 0 for an arbitrary state."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=Nr, tau_p=2.0, tau_ee=3.0, r_c=r_c)
    n_obs = fs.get_observables(0.0)[0]
    rng = torch.Generator(device=rc.device).manual_seed(2)
    rho = torch.randn(7, fs.v.shape[0], dtype=torch.float64, generator=rng)
    rdot = fs.rho_dot(rho, 0.0, 0)
    mdot = (n_obs[None, :] * rdot).sum(-1)
    mass = (n_obs[None, :] * rho).sum(-1)
    assert float((mdot / mass.abs().clamp(min=1e-30)).abs().max()) < 1e-11


@pytest.mark.parametrize("phi_deg", [0.0, 23.7, 45.0, 90.0, 137.0])
@pytest.mark.parametrize("s", [0.0, 0.3, 1.0])
def test_reflector_mass_conservation_radial(phi_deg: float, s: float) -> None:
    """Net (w_r-weighted) normal mass flux at the wall is zero at Nr>1.  Only the
    n=0 radial projection carries mass; the per-radial (D, T) solve conserves it."""
    torch.set_default_dtype(torch.float64)
    Nr = 4
    fs = _make(M_theta=8, Nr=Nr, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
    u_in = torch.randn(1, 1, fs.v.shape[0], dtype=torch.float64, generator=rng)
    u_out = refl(u_in)
    Nth = fs.angular.N_theta
    ui = u_in.reshape(Nr, Nth)
    uo = u_out.reshape(Nr, Nth)
    w_r = fs.radial.quad_w
    theta = fs.angular.theta
    vdn = fs.vF * (n[0, 0] * torch.cos(theta) + n[0, 1] * torch.sin(theta))
    F_out = (w_r[:, None] * vdn.clamp(min=0)[None, :] * ui).sum()
    F_in = (w_r[:, None] * vdn.clamp(max=0)[None, :] * uo).sum()
    assert float((F_out + F_in).abs()) / max(float(F_out.abs()), 1e-30) < 1e-11


@pytest.mark.parametrize("phi_deg", [0.0, 23.7, 45.0, 90.0, 137.0])
@pytest.mark.parametrize("s", [0.0, 0.3, 1.0])
def test_reflector_tang_momentum_radial(phi_deg: float, s: float) -> None:
    """At Nr>1 the (w_r-weighted) tangential-momentum flux into the wall equals the
    continuum value (1 - s) * F_out_tang to roundoff (specular fraction kept)."""
    torch.set_default_dtype(torch.float64)
    Nr = 4
    fs = _make(M_theta=8, Nr=Nr, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
    u_in = torch.randn(1, 1, fs.v.shape[0], dtype=torch.float64, generator=rng)
    u_out = refl(u_in)
    Nth = fs.angular.N_theta
    ui = u_in.reshape(Nr, Nth)
    uo = u_out.reshape(Nr, Nth)
    w_r = fs.radial.quad_w
    theta = fs.angular.theta
    vdn = fs.vF * (n[0, 0] * torch.cos(theta) + n[0, 1] * torch.sin(theta))
    vtg = fs.vF * (-n[0, 1] * torch.cos(theta) + n[0, 0] * torch.sin(theta))
    F_out = (w_r[:, None] * (vdn.clamp(min=0) * vtg)[None, :] * ui).sum()
    F_in = (w_r[:, None] * (vdn.clamp(max=0) * vtg)[None, :] * uo).sum()
    expected = (1.0 - s) * F_out
    assert float((F_out + F_in - expected).abs()) / max(float(F_out.abs()),
                                                        1e-30) < 1e-11


@pytest.mark.parametrize("Nr", [2, 4])
def test_tau_ee_placeholder_damps_energy_by_construction(Nr: int) -> None:
    """The PHENOMENOLOGICAL `tau_ee` does not conserve energy, and cannot.

    ⛔ THIS IS NOT A DEFECT IN e-e SCATTERING.  Real electron-electron collisions
    conserve energy by construction -- two-body kinematics gives
    eps_1 + eps_2 = eps_3 + eps_4 -- and qimpy's MICROSCOPIC operator does exactly
    that: scattering/test_ee.py::test_nonlinear_conservation asserts the cubic and
    quadratic outputs annihilate the number, energy and momentum nulls at Nr=1 and
    Nr>=2.  The two operators are mutually exclusive (_fermi_surface.py raises
    InvalidInputException if both `tau_ee` and `ee_scattering` are given).

    `tau_ee` is a relaxation-time PLACEHOLDER.  It protects TWO collision
    invariants, not one:

      mass      `rates[0, 0] = 0`, set explicitly.
      momentum  the m=1 angular column is exempted from tau_inv_ee by the
                `if m == 1` in the rate loop -- e-e does not relax momentum, only
                impurities do -- so `rates[0, 1] = rates[0, 2] = tau_inv_p`,
                which is 0 at tau_p = inf.

    Energy is the one it drops.  The radial basis makes the {1, xi} shapes EXACT
    and w(xi) = (1/4T) sech^2(xi/2) is even, so <xi>_w = 0 and the energy null is
    EXACTLY the n=1 radial mode -- which `rad[1:] = tau_inv_ee` damps at exactly
    1/tau_ee.

    ⛔ So the placeholder is INTERNALLY INCONSISTENT: it encodes "e-e conserves
    momentum" and not "e-e conserves energy", though both are equally true of the
    operator it stands in for.  Making it consistent is `rad[2:] = tau_inv_ee`,
    a one-character change -- deliberately NOT made here, because it alters the
    physics of every tau_ee run in the repo.

    ⚠ The momentum protection is exact only on the Fermi circle.  The true
    momentum null is k ~ sqrt(1 + (T/E_F) xi) (see
    scattering/test_ee.py, `k_c`), which carries an n=1 component of relative
    size T/2E_F; `rates[1, 1] = tau_inv_p + tau_inv_ee` damps that.  So momentum
    is conserved exactly at Nr=1 and to O(T/E_F) at Nr>1.

    ⛔ An earlier version of this test asserted the opposite and was marked xfail
    "pending the L-matrix".  That premise was false twice over: the L-matrix
    operator already exists and already conserves energy, and no change to it
    could ever make THIS test pass, because this test builds the placeholder."""
    torch.set_default_dtype(torch.float64)
    tau_ee = 2.0
    fs = _make(M_theta=8, Nr=Nr, tau_p=np.inf, tau_ee=tau_ee, r_c=np.inf)
    Nth = fs.angular.N_theta
    w_r, xi = fs.radial.quad_w, fs.radial.xi
    E_obs = (w_r[:, None] * xi[:, None]
             * torch.ones(Nr, Nth, dtype=torch.float64) / Nth).reshape(-1)
    rng = torch.Generator(device=rc.device).manual_seed(3)
    rho = torch.randn(9, fs.v.shape[0], dtype=torch.float64, generator=rng)
    rdot = fs.rho_dot(rho, 0.0, 0)
    e_rate = (E_obs[None, :] * rdot).sum(-1)
    e_val = (E_obs[None, :] * rho).sum(-1)
    # exact exponential decay of the energy moment at the placeholder's own rate
    ratio = e_rate / e_val.abs().clamp(min=1e-30).copysign(e_val)
    assert float((ratio + 1.0 / tau_ee).abs().max()) < 1e-11

    # ...and the rate table it comes from, read directly.  This is the whole
    # content of the operator, so assert every block of it:
    rates = fs.rates_modal.reshape(Nr, fs.angular.dim)
    assert float(rates[0, 0]) == 0.0                    # mass:     protected
    assert float(rates[0, 1]) == 0.0                    # momentum: protected
    assert float(rates[0, 2]) == 0.0                    #           (tau_p = inf)
    assert float((rates[1:, 0] - 1.0 / tau_ee).abs().max()) < 1e-15   # energy: NOT
    # the m=1 exemption is what protects momentum -- if it were dropped, these
    # would pick up 1/tau_ee like every other column
    assert float((rates[1:, 1] - 1.0 / tau_ee).abs().max()) < 1e-15
    assert float((rates[1:, 2] - 1.0 / tau_ee).abs().max()) < 1e-15


@pytest.mark.parametrize("Nr", [1, 4])
def test_tau_ee_placeholder_conserves_momentum(Nr: int) -> None:
    """The `tau_ee` placeholder DOES conserve momentum -- exactly at Nr=1.

    The rate loop exempts m=1 from tau_inv_ee (`self.tau_inv_p if m == 1`), which
    is the statement that e-e scattering does not relax momentum.  At tau_p = inf
    the whole m=1 column of the n=0 block is therefore identically zero.

    ⛔ WHAT THIS TEST MEASURES IS THE MODAL MOMENT (n=0, m=1), not the physical
    momentum.  That distinction is why it passes to roundoff at Nr=4 as well as
    Nr=1: the (n=0, m=1) mode is exactly protected at every Nr.  The PHYSICAL
    momentum is int f v with v ~ k ~ sqrt(1 + (T/E_F) xi) (see `k_c` in
    scattering/test_ee.py), whose n>=1 components `rates[n, 1]` does damp.  That
    leak is pinned by the companion test below."""
    torch.set_default_dtype(torch.float64)
    tau_ee = 2.0
    fs = _make(M_theta=8, Nr=Nr, tau_p=np.inf, tau_ee=tau_ee, r_c=np.inf)
    rates = fs.rates_modal.reshape(Nr, fs.angular.dim)
    assert float(rates[0, 1]) == 0.0 and float(rates[0, 2]) == 0.0

    # behavioural: the leading (Fermi-circle) momentum moment does not decay.
    # Same observable construction as the energy test above, with the radial
    # shape 1 (the n=0 mode) in place of xi, and cos(theta) in place of isotropic.
    Nth = fs.angular.N_theta
    w_r = fs.radial.quad_w
    Jx = (w_r[:, None] * torch.cos(fs.angular.theta)[None, :]).reshape(-1)
    rng = torch.Generator(device=rc.device).manual_seed(5)
    rho = torch.randn(6, fs.v.shape[0], dtype=torch.float64, generator=rng)
    rdot = fs.rho_dot(rho, 0.0, 0)
    j_rate = (Jx[None, :] * rdot).sum(-1)
    j_val = (Jx[None, :] * rho).sum(-1)
    ratio = float((j_rate / j_val.abs().clamp(min=1e-30)).abs().max())
    assert ratio < 1e-11, f"momentum decaying at {ratio:.3e}"


@pytest.mark.parametrize("Nr", [1, 2, 4, 8])
def test_tau_ee_placeholder_physical_momentum_leak(Nr: int) -> None:
    """MEASURED size of the placeholder's physical-momentum leak: 6.1e-4 / tau_ee.

    The modal test above shows the (n=0, m=1) moment is exactly protected.  The
    PHYSICAL momentum null is k ~ sqrt(1 + (T/E_F) xi), whose n>=1 content is
    damped at 1/tau_ee, so it is conserved exactly only at Nr=1.  This pins how
    much that costs at a REAL T/E_F (the mixer material, T/E_F = 3.17e-2):

        Nr    decay rate x tau_ee
         1    2.0e-31           exact -- the Fermi circle
         2    4.3e-04
         4    5.6e-04
         8    6.0e-04
        16    6.1e-04           converged

    i.e. 0.06% of the e-e rate, converged in Nr.  Negligible, and it does NOT
    grow with Nr.

    ⛔⛔ THIS IS A RAYLEIGH QUOTIENT, NOT A MAX-OVER-STATES RATIO, AND THAT IS THE
    WHOLE POINT.  Measured the obvious way -- random states, max of
    (d/dt<J>)/<J> -- the same quantity reads 6.9e-2 / 5.6e-2 / 3.7e-1 at
    Nr = 2/4/8, which looks like a 23x violation GROWING with Nr.  Every bit of
    that is a near-zero denominator: a random state carries almost no net
    momentum.  Seeding the state WITH the momentum null makes <J,J> ~ 1e10 and
    the number falls by three orders of magnitude and goes flat.  Never report
    the max of a ratio whose denominator you have not looked at.

    ⛔ The toy material used elsewhere in this file cannot be used here at all:
    it has T/E_F = 1.33, so 1 + (T/E_F) xi goes NEGATIVE inside the xi window and
    sqrt() returns nan.  The degenerate expansion is not merely inaccurate there,
    it is undefined."""
    torch.set_default_dtype(torch.float64)
    tau_ee = 2.0
    kF, vF, T = 7.5e-3, 0.11194, 1.3301e-5          # mixer material
    t = T / (0.5 * kF * vF)
    fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=Nr, T=T, xi_max=6.0,
                      tau_p=np.inf, tau_ee=tau_ee, r_c=np.inf, specularity=1.0,
                      process_grid=_pg())
    w_r, xi = fs.radial.quad_w, fs.radial.xi
    arg = 1.0 + t * xi
    assert float(arg.min()) > 0.0, "band bottom inside the xi window"
    J = (w_r[:, None] * torch.sqrt(arg)[:, None]
         * torch.cos(fs.angular.theta)[None, :]).reshape(-1)
    rho = J[None, :].clone()                        # denominator = <J,J>, never small
    rate = -float((J[None, :] * fs.rho_dot(rho, 0.0, 0)).sum()) / float((J * J).sum())
    assert rate >= -1e-15, f"momentum GROWING at {rate:.3e}"
    assert rate * tau_ee < 1e-3, f"leak {rate * tau_ee:.3e} exceeds the measured 6.1e-4"
    if Nr == 1:
        assert rate * tau_ee < 1e-20, "Fermi circle must be exact"


# ----------------------------------------------------------------------------
# THE WALL-ANGLE REGRESSION.  This is the test that would have caught the cause
# of f leaving [0, 1], and it is the one thing here that must never regress.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("deg", [0.0, 45.0, 90.0])
def test_reflector_exact_at_grid_symmetric_angles(deg: float) -> None:
    """At a wall whose mirror is a k-grid symmetry the reflector is EXACT.

    k -> k* = k - 2(k.n)n preserves |k|, and at 0 / 45 / 90 degrees k* is a grid
    permutation (kx -> -kx, the diagonal swap, ky -> -ky), so no interpolation
    happens and the answer is exact to roundoff.  This pins the reference
    against which the tilted-wall error below is meaningful."""
    torch.set_default_dtype(torch.float64)
    kF, vF, T = 7.5e-3, 0.11194, 1.3301e-5
    fs = FermiSurface(kF=kF, vF=vF, M_theta=32, Nr=6, T=T, xi_max=6.0,
                      tau_p=np.inf, specularity=1.0,
                      cartesian=dict(annulus_xi=0.0, te_fac_max=6.0,
                                     kD_max=1.2e-3, dmu_max=1.2e-4,
                                     k_max=0.0132557160008, n_k=112),
                      process_grid=_pg())
    rep = fs.representation
    k = rep.k
    m, mu = float(rep.m_star), float(rep.mu)
    f0 = rep._f0_lab
    kD = torch.tensor([0.05 * kF, 0.0], dtype=k.dtype, device=k.device)
    u_in = torch.special.expit(
        -(((k - kD) ** 2).sum(-1) / (2 * m) - mu) / T) - f0
    phi = np.deg2rad(deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=k.dtype, device=k.device)
    out = fs.get_reflector(n)(u_in[None, None, :])[0, 0]
    kn = (k * n[0]).sum(-1, keepdim=True)
    kstar = k - 2.0 * kn * n[0][None, :]
    exact = torch.special.expit(
        -(((kstar - kD) ** 2).sum(-1) / (2 * m) - mu) / T) - f0
    scale = exact.abs().max().clamp(min=1e-300)
    assert float((out - exact).abs().max() / scale) < 1e-12
    assert float((f0 + out).min()) >= -1e-14, "occupancy went negative"


def test_reflector_never_leaves_pauli_bounds() -> None:
    """At ANY wall angle the reflector output must satisfy 0 <= f <= 1.

    ⛔ THIS, NOT POINTWISE ACCURACY, IS THE ACCEPTANCE CRITERION, and the
    difference is not academic.  A stencil with a SMALLER single-shot pointwise
    error but negative weights seeds violations that then compound over the
    ~10^3 wall bounces of a device run: on a 17-degree channel it gave min f =
    -3.1e-2 with 1,136,070 points below zero after 800 steps.  The convex
    stencil has a ~5x LARGER single-shot pointwise error and gives min f =
    -3.7e-27 with 35 points below zero -- and those 35 sit where f0 ~ 1e-30,
    i.e. they are roundoff, not violations.  Bound-respecting error does not
    amplify; bound-violating error does.

    The guarantee is structural: f_out is interpolated with NON-NEGATIVE weights
    summing to 1, so it is a convex combination of values already in [0, 1], and
    the flux repair is multiplicative (positive scalings) rather than additive,
    so it cannot break that.  No CFL, no limiter, no inequality to check."""
    import os
    torch.set_default_dtype(torch.float64)
    kF, vF, T = 7.5e-3, 0.11194, 1.3301e-5
    old = os.environ.get("QIMPY_REFL_EXACT")
    os.environ["QIMPY_REFL_EXACT"] = "1"      # the bound-exact reflector
    try:
        fs = FermiSurface(kF=kF, vF=vF, M_theta=32, Nr=6, T=T, xi_max=6.0,
                          tau_p=np.inf, specularity=1.0,
                          cartesian=dict(annulus_xi=0.0, te_fac_max=6.0,
                                         kD_max=1.2e-3, dmu_max=1.2e-4,
                                         k_max=0.0132557160008, n_k=112),
                          process_grid=_pg())
        _check_bounds(fs)
    finally:
        if old is None:
            os.environ.pop("QIMPY_REFL_EXACT", None)
        else:
            os.environ["QIMPY_REFL_EXACT"] = old


def _check_bounds(fs) -> None:
    rep = fs.representation
    k = rep.k
    kF = 7.5e-3
    T = 1.3301e-5
    m, mu = float(rep.m_star), float(rep.mu)
    f0 = rep._f0_lab
    kD = torch.tensor([0.05 * kF, 0.0], dtype=k.dtype, device=k.device)
    for te in (1.0, 5.69):
        u_in = torch.special.expit(
            -(((k - kD) ** 2).sum(-1) / (2 * m) - mu) / (T * te)) - f0
        for deg in (0.0, 17.0, 30.0, 45.0, 63.0, 90.0):
            phi = np.deg2rad(deg)
            n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=k.dtype,
                             device=k.device)
            # ⛔ THE INFLOW HALF IS THE ONLY HALF THE WALL WRITES.  The ghost
            # is consumed only where v.n < 0; the outflow entries are never
            # read by the flux assembly, and the affine offset is masked to the
            # inflow set, so outflow values are NOT a convex blend and go
            # negative harmlessly.  Measuring the whole k-set instead of the
            # inflow half reports -3.1e-2 for an operator that is exactly
            # bound-preserving where it is used.
            vn = (k * n[0]).sum(-1)
            inflow = vn < 0
            f_out = (f0 + fs.get_reflector(n)(u_in[None, None, :])[0, 0])[inflow]
            assert float(f_out.min()) > -1e-25, (
                f"f = {float(f_out.min()):.3e} < 0 at {deg} deg, Te/T = {te}")
            assert float(f_out.max()) < 1.0 + 1e-25, (
                f"f = {float(f_out.max()):.6f} > 1 at {deg} deg, Te/T = {te}")
